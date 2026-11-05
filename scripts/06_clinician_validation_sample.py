"""
06_clinician_validation_sample.py
Physician validation sample and blind review workbook.

This script does not perform the physician review itself. It:
  1. Draws a department-stratified random sample of 120 notes from isolates
     that have a discharge-summary note, so that no single specialty
     dominates (base sample).
  2. Adds a purposive ENRICHMENT SUPPLEMENT for rare flags: for any flag
     with fewer than ENRICH_MIN rule-based-positive notes in the base
     sample, additional notes the rule-based pipeline calls positive for
     that flag are drawn from the remaining pool until ENRICH_MIN positives
     are reached (or the supply is exhausted). Each added note is tagged
     with the flag(s) that motivated its inclusion (enrichment_for column).
     Because the enrichment depends on the rule-based output, enrichment
     notes are excluded from all accuracy metrics and analysed separately
     as error analysis (19c_validation_probability_sample.py).
  3. Marks a random 20% of the base sample for a second, independent
     physician review (second_reviewer_required column), for inter-rater
     agreement.
  4. Builds a physician-facing Excel review workbook that is BLIND to both
     extraction methods: for each flag, the reviewer sees only the
     de-identified note text and records their own independent judgment
     (Present / Absent / Unsure). Neither the rule-based nor the LLM flag
     value appears anywhere in the workbook; both are held in the
     restricted sample key and brought in only at scoring time
     (19c_validation_probability_sample.py).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

COHORT_PARQUET = Path('data/processed/cohort_master.parquet')
NLP_PARQUET = Path('data/processed/nlp_features.parquet')          # rule-based (comparison)
NLP_LLM_PARQUET = Path('data/processed/nlp_features_llm.parquet')  # LLM-derived (primary)
OUT_XLSX = Path('paper/clinician_validation_review_sheet.xlsx')
OUT_SAMPLE_KEY = Path('data/deid/clinician_validation_sample_key_RESTRICTED.csv')  # links review_id back to isolate_id AND both pipelines' flag values -- internal only, never shown to a reviewer

N_TARGET = 120  # within the requested 100-150 range
RANDOM_STATE = 42

REVIEW_FLAGS = [
    'comorbid_diabetes', 'comorbid_ckd_dialysis', 'comorbid_malignancy',
    'comorbid_immunosuppression', 'comorbid_cardio_cerebrovascular',
    'prior_antibiotic_any', 'device_urinary_catheter', 'device_central_line',
    'device_ventilator_tracheal', 'device_surgical_drain', 'icu_stay',
    'recent_hospitalization', 'recent_surgery',
]
FLAG_LABELS = {
    'comorbid_diabetes': 'Diabetes mellitus',
    'comorbid_ckd_dialysis': 'CKD / dialysis',
    'comorbid_malignancy': 'Malignancy',
    'comorbid_immunosuppression': 'Immunosuppression',
    'comorbid_cardio_cerebrovascular': 'Prior stroke / cardiac disease',
    'prior_antibiotic_any': 'Prior antibiotic exposure (any)',
    'device_urinary_catheter': 'Urinary catheter',
    'device_central_line': 'Central line',
    'device_ventilator_tracheal': 'Ventilator / tracheal tube',
    'device_surgical_drain': 'Surgical drain',
    'icu_stay': 'ICU stay',
    'recent_hospitalization': 'Recent hospitalization',
    'recent_surgery': 'Recent surgery',
}

# NOTE (bug fix, same root cause as script 05): DISCHARGE_SUMMARY_CONTENT_DEID
# must be read directly from cohort_master.parquet's own column list, NOT via
# a separate deidentified_master.parquet read + `.join()` -- both parquet
# files are saved with index=False so a positional join between differently
# filtered/ordered tables silently mismatches text to the wrong isolate (see
# script 05 for the full verification of this bug). cohort_master.parquet
# already carries DISCHARGE_SUMMARY_CONTENT_DEID natively.
cohort = pd.read_parquet(COHORT_PARQUET, columns=['study_id', 'isolate_id', 'drpDept', 'DISCHARGE_SUMMARY_CONTENT_DEID'])
nlp = pd.read_parquet(NLP_PARQUET)
nlp_llm = pd.read_parquet(NLP_LLM_PARQUET)

cohort['dept_clean'] = cohort['drpDept'].fillna('Unknown').replace('--Select--', 'Unknown')
# cohort, nlp, and nlp_llm are all read directly (and in full, no
# filtering/reordering) from cohort_master.parquet / the script-05 and
# script-05b outputs derived from it in the same row order, so they line up
# positionally -- reset all three to a clean 0..N-1 index and verify lengths
# and study_id order match before relying on that alignment.
cohort = cohort.reset_index(drop=True)
nlp = nlp.reset_index(drop=True)
nlp_llm = nlp_llm.reset_index(drop=True)
assert len(cohort) == len(nlp) == len(nlp_llm), (
    f"cohort ({len(cohort)}), nlp ({len(nlp)}), nlp_llm ({len(nlp_llm)}) row counts differ -- alignment assumption broken")
assert (cohort['study_id'].values == nlp['study_id'].values).all(), "cohort/nlp study_id mismatch -- row order assumption broken"
assert (cohort['study_id'].values == nlp_llm['study_id'].values).all(), "cohort/nlp_llm study_id mismatch -- row order assumption broken"
nlp.index = cohort.index
nlp_llm.index = cohort.index

has_note = cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].notna() & (cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].str.len() > 50)
pool = cohort[has_note].copy()
print(f"Eligible notes for sampling: {len(pool)}")
print(pool['dept_clean'].value_counts().head(15))

# Stratified sample proportional to department size, capped so no dept
# exceeds 25% of the sample (guards against one large dept dominating),
# minimum of 1 from any dept with at least 1 eligible note among the top
# departments, using a simple proportional-with-cap allocation.
dept_counts = pool['dept_clean'].value_counts()
top_depts = dept_counts.index.tolist()

alloc = {}
remaining = N_TARGET
cap = int(N_TARGET * 0.25)
# proportional first pass
total_pool = len(pool)
for d in top_depts:
    prop = int(round(N_TARGET * dept_counts[d] / total_pool))
    alloc[d] = min(prop, cap, dept_counts[d])

# adjust to hit N_TARGET exactly (add/remove from largest depts with room)
diff = N_TARGET - sum(alloc.values())
depts_sorted_by_room = sorted(top_depts, key=lambda d: dept_counts[d] - alloc[d], reverse=True)
i = 0
while diff != 0 and i < 10000:
    d = depts_sorted_by_room[i % len(depts_sorted_by_room)]
    if diff > 0 and alloc[d] < dept_counts[d] and alloc[d] < cap:
        alloc[d] += 1
        diff -= 1
    elif diff < 0 and alloc[d] > 0:
        alloc[d] -= 1
        diff += 1
    i += 1

print("\nStratified allocation by department:")
for d, n in sorted(alloc.items(), key=lambda x: -x[1]):
    if n > 0:
        print(f"  {d}: {n}")

sampled_frames = []
rng = np.random.RandomState(RANDOM_STATE)
for d, n in alloc.items():
    if n <= 0:
        continue
    sub = pool[pool['dept_clean'] == d]
    take = min(n, len(sub))
    sampled_frames.append(sub.sample(n=take, random_state=rng.randint(0, 1_000_000)))

base_sample = pd.concat(sampled_frames).sample(frac=1.0, random_state=RANDOM_STATE).reset_index()
base_sample = base_sample.rename(columns={'index': 'orig_row_index'})
print(f"\nBase stratified sample size: {len(base_sample)}")

# attach BOTH pipelines' flags for the base sample -- kept only in the
# internal key and in-memory dataframe, never written to the reviewer-facing
# workbook.
flag_cols = REVIEW_FLAGS
base_rb_flags = nlp.loc[base_sample['orig_row_index'], flag_cols].reset_index(drop=True)
base_rb_flags.columns = [f'rb__{c}' for c in base_rb_flags.columns]
base_llm_flags = nlp_llm.loc[base_sample['orig_row_index'], flag_cols].reset_index(drop=True)
base_llm_flags.columns = [f'llm__{c}' for c in base_llm_flags.columns]
base_sample = pd.concat([base_sample, base_rb_flags, base_llm_flags], axis=1)
base_sample['sampling_stratum'] = 'base_random'
base_sample['enrichment_for'] = ''

# ---------------------------------------------------------------------------
# Enrichment supplement: for any flag with fewer than ENRICH_MIN
# rule-based-pipeline-positive notes in the base sample, draw additional
# notes the rule-based pipeline calls positive for that flag from the
# remaining pool, until ENRICH_MIN positives are reached (or the corpus is
# exhausted). A single added note can satisfy more than one rare flag at
# once (enrichment_for lists all of them, comma-separated), so the
# supplement is smaller than the naive per-flag total would suggest.
# ---------------------------------------------------------------------------
ENRICH_MIN = 15
used_idx = set(base_sample['orig_row_index'].tolist())
enrichment_rows = []
enrichment_log = {}
for flag in REVIEW_FLAGS:
    n_have = int(base_sample[f'rb__{flag}'].sum())
    n_need = max(0, ENRICH_MIN - n_have)
    enrichment_log[flag] = {'n_in_base_sample': n_have, 'n_enriched_added': 0}
    if n_need == 0:
        continue
    candidates = pool[(nlp.loc[pool.index, flag] == 1) & (~pool.index.isin(used_idx))]
    take = min(n_need, len(candidates))
    if take == 0:
        continue
    drawn = candidates.sample(n=take, random_state=rng.randint(0, 1_000_000))
    enrichment_rows.append(drawn)
    used_idx.update(drawn.index.tolist())
    enrichment_log[flag]['n_enriched_added'] = take

if enrichment_rows:
    enriched = pd.concat(enrichment_rows).drop_duplicates()
    enriched = enriched.reset_index().rename(columns={'index': 'orig_row_index'})
    enriched_rb_flags = nlp.loc[enriched['orig_row_index'], flag_cols].reset_index(drop=True)
    enriched_rb_flags.columns = [f'rb__{c}' for c in enriched_rb_flags.columns]
    enriched_llm_flags = nlp_llm.loc[enriched['orig_row_index'], flag_cols].reset_index(drop=True)
    enriched_llm_flags.columns = [f'llm__{c}' for c in enriched_llm_flags.columns]
    enriched = pd.concat([enriched, enriched_rb_flags, enriched_llm_flags], axis=1)
    enriched['sampling_stratum'] = 'enrichment'
    # which rare flag(s) this row was drawn for (a row can satisfy several)
    def which_flags(row):
        return ','.join(f for f in REVIEW_FLAGS if row[f'rb__{f}'] == 1
                         and enrichment_log[f]['n_enriched_added'] > 0)
    enriched['enrichment_for'] = enriched.apply(which_flags, axis=1)
else:
    enriched = base_sample.iloc[0:0].copy()

print("\nEnrichment supplement (rare flags brought up to "
      f"{ENRICH_MIN} rule-based-pipeline-positive cases where the corpus allows):")
for flag, info in enrichment_log.items():
    if info['n_enriched_added'] > 0 or info['n_in_base_sample'] < ENRICH_MIN:
        print(f"  {flag}: base={info['n_in_base_sample']}, added={info['n_enriched_added']}, "
              f"total={info['n_in_base_sample'] + info['n_enriched_added']}")
print(f"Total enrichment notes added (deduplicated, some notes satisfy multiple rare flags): {len(enriched)}")

sample = pd.concat([base_sample, enriched], ignore_index=True)
sample['note_length_chars'] = sample['DISCHARGE_SUMMARY_CONTENT_DEID'].str.len()
sample.insert(0, 'review_id', [f'R{i+1:03d}' for i in range(len(sample))])

# ---------------------------------------------------------------------------
# Second-reviewer sub-check: mark a random 20% of the BASE (non-enrichment)
# sample for independent review by a second physician, so that inter-rater
# agreement can be quantified (19c_validation_probability_sample.py).
# ---------------------------------------------------------------------------
sample['second_reviewer_required'] = False
base_mask = sample['sampling_stratum'] == 'base_random'
n_second = max(1, round(0.20 * base_mask.sum()))
second_review_ids = sample.loc[base_mask, 'review_id'].sample(n=n_second, random_state=RANDOM_STATE)
sample.loc[sample['review_id'].isin(second_review_ids), 'second_reviewer_required'] = True

print(f"\nFinal sample size (base + enrichment): {len(sample)}")
print(f"Second-reviewer sub-check: {n_second} of {int(base_mask.sum())} base-sample notes "
      f"({100*n_second/base_mask.sum():.0f}%) marked for independent second review.")

# ---------------------------------------------------------------------------
# Save restricted key (review_id -> isolate_id / study_id / BOTH pipelines'
# flag values) -- internal only, NEVER shown to a reviewer, so the sheet the
# physician sees carries no extraction-method output at all, only the
# de-identified note text itself.
# ---------------------------------------------------------------------------
OUT_SAMPLE_KEY.parent.mkdir(parents=True, exist_ok=True)
key_cols = (['review_id', 'study_id', 'isolate_id', 'orig_row_index', 'dept_clean',
             'sampling_stratum', 'enrichment_for', 'second_reviewer_required']
            + [f'rb__{f}' for f in REVIEW_FLAGS] + [f'llm__{f}' for f in REVIEW_FLAGS])
sample[key_cols].to_csv(OUT_SAMPLE_KEY, index=False)
print(f"Saved restricted sample key: {OUT_SAMPLE_KEY}")

# ---------------------------------------------------------------------------
# Build the Excel review sheet -- BLIND to both extraction methods. Only the
# de-identified note text and a per-flag Present/Absent/Unsure judgment
# column are shown; no 'pipeline said' value appears anywhere in this
# workbook.
# ---------------------------------------------------------------------------
wb = Workbook()

# --- Instructions sheet ---
ws0 = wb.active
ws0.title = 'Instructions'
instructions = [
    ("NLP Extraction Clinician Validation - Instructions", True),
    ("", False),
    ("Purpose: establish independent ground truth for 13 clinical flags, from the "
     "de-identified discharge-summary text alone, for a random sample of "
     f"{len(sample)} notes stratified by department. Your judgment will be compared "
     "afterward -- by the study team, not by you -- against two automated extraction "
     "methods (a rule-based pipeline and a large-language-model pipeline). This is a "
     "BLIND review: this workbook does not show you either method's output anywhere.", False),
    ("", False),
    ("How to complete (should take ~1-2 minutes per note):", True),
    ("1. Go to the 'Review Sheet' tab.", False),
    ("2. For each row (one note per row), read the note text in column D.", False),
    ("3. For EACH of the 13 flag columns, decide for yourself, from the note text only, "
     "whether that clinical finding is documented as present. Use the dropdown: "
     "'Present' = the note documents this, 'Absent' = the note does not document this "
     "(or documents its explicit absence), 'Unsure' = the note is genuinely ambiguous.", False),
    ("4. Do not try to guess or infer what an automated system might have extracted -- "
     "just record your own clinical read of the note.", False),
    ("5. Optional free-text comments go in the last column.", False),
    ("6. When done, save the file and return it to the study team.", False),
    ("", False),
    ("Column key:", True),
    ("- 'YOUR JUDGMENT' columns are where you record your own independent call: "
     "Present / Absent / Unsure. No extraction-method output is shown anywhere in this file.", False),
    ("- 'Sampling stratum' shows whether a note is part of the base random sample or was added by the "
     "enrichment supplement (to ensure enough positive cases of some rare flags exist to score "
     "precision reliably); this does not change how you review it -- review every row the same way.", False),
    ("", False),
    ("Two sheets in this workbook:", True),
    ("- 'Review Sheet': ALL sampled notes -- the primary reviewer completes this one in full.", False),
    ("- 'Second Reviewer Subset': a random 20% of the base sample, repeated here for an INDEPENDENT "
     "second physician to review separately, without seeing the first reviewer's answers -- this lets "
     "the study team measure how often two reviewers agree with each other (inter-rater reliability), "
     "not just how often a reviewer agrees with either automated method. If only one reviewer is "
     "available, this tab can be left blank -- the primary precision/recall/F1/kappa results against "
     "both extraction methods do not depend on it.", False),
    ("", False),
    ("This sample is de-identified. If you notice any residual identifying information that the "
     "de-identification pass missed, please note it in the comments column and flag it to the "
     "study team immediately -- do not forward this file further until that is resolved.", False),
]
for i, (text, bold) in enumerate(instructions, start=1):
    c = ws0.cell(row=i, column=1, value=text)
    if bold:
        c.font = Font(bold=True, size=13 if i == 1 else 11)
    ws0.row_dimensions[i].height = 18
ws0.column_dimensions['A'].width = 130

# --- Review sheet ---
ws = wb.create_sheet('Review Sheet')
header_fill = PatternFill(start_color='1F4E78', end_color='1F4E78', fill_type='solid')
header_font = Font(bold=True, color='FFFFFF')

cols = ['review_id', 'sampling_stratum', 'dept_clean', 'note_length_chars', 'note_text']
for flag in REVIEW_FLAGS:
    cols.append(f'your_judgment__{flag}')
cols.append('comments')

def write_header(ws_target, cols_list):
    for j, col_name in enumerate(cols_list, start=1):
        if col_name == 'sampling_stratum':
            display = 'Sampling stratum'
        elif col_name.startswith('your_judgment__'):
            flag = col_name.split('__', 1)[1]
            display = 'YOUR JUDGMENT - ' + FLAG_LABELS.get(flag, flag)
        else:
            display = col_name
        cell = ws_target.cell(row=1, column=j, value=display)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    ws_target.freeze_panes = 'F2'
    ws_target.row_dimensions[1].height = 40

def write_rows(ws_target, df, dv_local):
    for i, (_, row) in enumerate(df.iterrows(), start=2):
        ws_target.cell(row=i, column=1, value=row['review_id'])
        ws_target.cell(row=i, column=2, value=row['sampling_stratum'])
        ws_target.cell(row=i, column=3, value=row['dept_clean'])
        ws_target.cell(row=i, column=4, value=int(row['note_length_chars']))
        note_cell = ws_target.cell(row=i, column=5, value=row['DISCHARGE_SUMMARY_CONTENT_DEID'])
        note_cell.alignment = Alignment(wrap_text=True, vertical='top')
        col = 6
        for flag in REVIEW_FLAGS:
            judgment_cell = ws_target.cell(row=i, column=col)
            dv_local.add(judgment_cell)
            col += 1
        ws_target.row_dimensions[i].height = 30

def set_widths(ws_target, cols_list):
    ws_target.column_dimensions['A'].width = 10
    ws_target.column_dimensions['B'].width = 14
    ws_target.column_dimensions['C'].width = 18
    ws_target.column_dimensions['D'].width = 10
    ws_target.column_dimensions['E'].width = 70
    for j in range(6, len(cols_list)):
        ws_target.column_dimensions[get_column_letter(j)].width = 16
    ws_target.column_dimensions[get_column_letter(len(cols_list))].width = 30

write_header(ws, cols)
dv = DataValidation(type='list', formula1='"Present,Absent,Unsure"', allow_blank=True, showDropDown=False)
ws.add_data_validation(dv)
write_rows(ws, sample, dv)
set_widths(ws, cols)

# --- Second Reviewer Subset sheet: the 20% second-review subset, laid out
# identically but on its own sheet so a second, independent physician can
# complete it without seeing the first reviewer's answers (neither sheet
# shows any extraction-method output or another reviewer's judgment).
# ---------------------------------------------------------------------------
second_subset = sample[sample['second_reviewer_required']].reset_index(drop=True)
ws2 = wb.create_sheet('Second Reviewer Subset')
write_header(ws2, cols)
dv2 = DataValidation(type='list', formula1='"Present,Absent,Unsure"', allow_blank=True, showDropDown=False)
ws2.add_data_validation(dv2)
write_rows(ws2, second_subset, dv2)
set_widths(ws2, cols)

OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUT_XLSX)
print(f"\nSaved physician review sheet: {OUT_XLSX}  ({len(sample)} notes x {len(REVIEW_FLAGS)} flags, blind to both "
      f"extraction methods; {len(second_subset)} notes duplicated on 'Second Reviewer Subset' for inter-rater reliability)")
