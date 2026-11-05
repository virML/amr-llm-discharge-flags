"""
05_nlp_features.py
Rule-based NLP feature extraction (the comparison extractor) -- Model B
additive features. LLM extraction (the primary extractor) is in
05a_llm_extraction.py and 05b_llm_nlp_features.py.

Method: rule-based / dictionary extraction (regex + curated clinical term
lists) run on the DE-IDENTIFIED discharge-summary text.

Output: one row per isolate (joinable by index / study_id to the structured
feature table), with binary flags for:
  - comorbidities: diabetes, CKD/dialysis, malignancy, immunosuppression,
    prior stroke/cardiac disease
  - prior antibiotic exposure (any-class flag + a few individual classes)
  - device exposure: urinary catheter, central line, ventilator/tracheal
    tube, surgical drain
  - ICU stay (text mention OR ward/unit cross-reference)
  - recent hospitalization/surgery mentioned in history

TEMPORAL AVAILABILITY OF THESE FEATURES (read before interpreting Model B):
the source document is a discharge summary, which by definition is
completed at or after discharge, i.e. after the culture that defines the
isolate under study in the great majority of admissions -- and it carries
no internal timestamps a rule-based extractor could use to attribute a
given mention to before-versus-after the culture. tables/
discharge_note_timing.json (computed below) quantifies the gap directly:
among isolates with a note, the recorded discharge date is on or after the
testing date for most of them, with a multi-day median gap. Consequently
these flags cannot be shown, and are not claimed, to reflect only
information that was available at the point the empiric-therapy decision
would actually have been made; some flags (e.g. device exposure, ICU stay,
antibiotics started) may describe events that happened during the
admission but after the culture was collected. Model B should therefore be
read as testing whether hospitalization-course narrative content adds
predictive value at all (an information-content upper bound), not as a
validated real-time predictor -- see Methods and Limitations.
"""
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path

COHORT_PARQUET = Path('data/processed/cohort_master.parquet')
OUT_PARQUET = Path('data/processed/nlp_features.parquet')
TIMING_JSON = Path('tables/discharge_note_timing.json')

# NOTE (bug fix): earlier versions of this script re-read
# DISCHARGE_SUMMARY_CONTENT_DEID from deidentified_master.parquet and joined
# it onto cohort_master.parquet via a plain pandas `.join()`. Both parquet
# files are written with index=False elsewhere in this pipeline, so on
# reload each gets a fresh, meaningless 0..N-1 RangeIndex; joining two
# DIFFERENT-length, differently-filtered tables on that fresh RangeIndex
# silently matches row i of one file to row i of the other -- NOT the same
# isolate -- rather than raising an error. Verified empirically: fewer than
# 0.03% of positional matches were correct. cohort_master.parquet already
# carries DISCHARGE_SUMMARY_CONTENT_DEID natively (script 03 does not drop
# it), so the fix is simply to read it directly from cohort_master and never
# re-join against deidentified_master.parquet by index/position again.

cohort = pd.read_parquet(COHORT_PARQUET, columns=['study_id', 'isolate_id', 'testing_date', 'Ward', 'Unit', 'DISCHARGE_SUMMARY_CONTENT_DEID', 'txtDiscDt'])

print(f"Cohort with text: {cohort.shape}, n non-null text: {cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].notna().sum()}")

# ---------------------------------------------------------------------------
# Quantify the temporal-availability gap documented above: for isolates with
# a note, how does the recorded discharge date relate to the testing
# (culture) date?
# ---------------------------------------------------------------------------
disc_dt = pd.to_datetime(cohort['txtDiscDt'], format='%d/%m/%Y %H:%M', errors='coerce')
has_note = cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].notna()
has_both_dates = has_note & disc_dt.notna() & cohort['testing_date'].notna()
gap_days = (disc_dt - cohort['testing_date']).dt.total_seconds() / 86400.0
gap_sub = gap_days[has_both_dates]

timing_summary = {
    'n_isolates_with_note': int(has_note.sum()),
    'n_with_note_and_usable_discharge_date': int(has_both_dates.sum()),
    'pct_discharge_date_before_testing_date': round(100 * float((gap_sub < 0).mean()), 1),
    'pct_discharge_date_same_day_as_testing_date': round(100 * float(((gap_sub >= 0) & (gap_sub < 1)).mean()), 1),
    'pct_discharge_date_1plus_days_after_testing_date': round(100 * float((gap_sub >= 1).mean()), 1),
    'median_gap_days_discharge_minus_testing': round(float(gap_sub.median()), 2),
    'iqr_gap_days': [round(float(gap_sub.quantile(0.25)), 2), round(float(gap_sub.quantile(0.75)), 2)],
    'note': ('Positive gap = discharge date after testing date, i.e. the note that Model B features are '
             'extracted from was compiled after the culture. A negative gap (discharge before testing) is '
             'itself a data-quality signal -- most plausibly a subsequent admission\'s culture matched to a '
             'prior discharge record via patient id -- rather than evidence the note predates the culture '
             'in any usable sense.'),
}
Path('tables').mkdir(exist_ok=True)
with open(TIMING_JSON, 'w') as f:
    json.dump(timing_summary, f, indent=2)
print("\nDischarge-note timing relative to culture (temporal-availability check for Model B):")
print(json.dumps(timing_summary, indent=2))

# ---------------------------------------------------------------------------
# Term dictionaries (compiled once, case-insensitive)
# ---------------------------------------------------------------------------
def rx(pattern):
    return re.compile(pattern, re.IGNORECASE)

PATTERNS = {
    # --- comorbidities ---
    'comorbid_diabetes': rx(r'\bdiabet\w*|\bt1dm\b|\bt2dm\b|\biddm\b|\bniddm\b|\bdm\s*[-:]?\s*(type|ii|2|i|1)\b'),
    'comorbid_ckd_dialysis': rx(r'\bckd\b|chronic kidney disease|renal failure|end[\s-]?stage renal|\besrd\b|\bdialysis\b|haemodialysis|hemodialysis|renal transplant|nephropathy'),
    'comorbid_malignancy': rx(r'carcinoma|malignan\w*|\bneoplasm\w*|\btumou?r\w*|metasta\w*|chemotherapy|oncolog\w*|leukemia|leukaemia|lymphoma|sarcoma|carcinomatosis'),
    'comorbid_immunosuppression': rx(r'immunosuppress\w*|immunocompromis\w*|on steroids|chronic steroid|long[\s-]?term steroid|transplant recipient|\bhiv\b|\baids\b|neutropeni\w*'),
    'comorbid_cardio_cerebrovascular': rx(r'\bcva\b|cerebrovascular accident|\bstroke\b|hemiplegi\w*|ischemic heart disease|ischaemic heart disease|\bihd\b|myocardial infarction|\bcad\b|coronary artery disease|heart failure|\bchf\b|cardiomyopathy'),

    # --- prior antibiotic exposure ---
    'prior_antibiotic_any': rx(r'\bantibiotic\w*|\bantimicrobial\w*|received\s+\w*\s*(inj\.?|tab\.?)|on\s+iv\s+antibiotics'),
    'prior_antibiotic_named': rx(
        r'ceftriaxone|cefotaxime|ceftazidime|cefepime|cefixime|cefuroxime|meropenem|imipenem|ertapenem|'
        r'piperacillin|tazobactam|pip[\s-]?taz|vancomycin|linezolid|teicoplanin|daptomycin|'
        r'ciprofloxacin|levofloxacin|moxifloxacin|ofloxacin|norfloxacin|'
        r'amikacin|gentamicin|tobramycin|colistin|polymyxin|'
        r'azithromycin|erythromycin|clarithromycin|'
        r'amoxicillin|augmentin|amoxyclav|ampicillin|'
        r'clindamycin|metronidazole|doxycycline|minocycline|tigecycline|'
        r'co[\s-]?trimoxazole|trimethoprim|sulfamethoxazole|nitrofurantoin|fosfomycin'
    ),

    # --- device exposure ---
    'device_urinary_catheter': rx(r'\bfoley\b|urinary catheter|indwelling catheter|per[\s-]?urethral catheter|\bpuc\b|catheteri[sz]ation'),
    'device_central_line': rx(r'central line|central venous catheter|\bcvc\b|picc line|femoral line|internal jugular line|subclavian line'),
    'device_ventilator_tracheal': rx(r'ventilat\w*|intubat\w*|tracheostomy|endotracheal|\bett\b|mechanical ventilation'),
    'device_surgical_drain': rx(r'\bdrain\b|\bjp drain\b|surgical drain|abdominal drain|chest tube|\bicd\b(?!\s*-?\s*10)'),

    # --- ICU stay ---
    'icu_stay_text': rx(r'\bicu\b|intensive care unit|critical care unit|\bmicu\b|\bsicu\b|\bnicu\b|\bpicu\b'),

    # --- recent hospitalization / surgery ---
    'recent_hospitalization': rx(r'recent(ly)?\s+(hospitali[sz]ed|admitted|admission)|previous admission|readmission|readmitted|multiple admissions'),
    'recent_surgery': rx(r'underwent surgery|post[\s-]?op(erative)?|\bs/p\s+surg|previous surgery|prior surgery|recent surgery|surgical history'),
}

# ---------------------------------------------------------------------------
# Negation handling: a simplified NegEx-style pre-trigger-window check
# (Chapman et al. 2001, "A Simple Algorithm for Identifying Negated Findings
# and Diseases in Discharge Summaries"). A match is discarded if a negation
# cue appears within a short character window immediately before it, e.g.
# "NO DYSPLASIA / MALIGNANCY" or "NEGATIVE FOR ... MALIGNANCY" should NOT
# count as a positive malignancy-history flag. This is a known, standard,
# transparent rule-based technique -- not a statistical/LLM step -- that
# meaningfully improves precision (spot-checked during development: several
# raw regex hits were negated pathology-report mentions, e.g. "NEGATIVE FOR
# INTRAEPITHELIAL LESION OR MALIGNANCY").
NEGATION_CUES = re.compile(
    r'\b(no|not|non|negative for|denies|denied|without|absence of|absent|'
    r'rule(d)?\s*out|r/o|free of|no evidence of|no history of|no h/o)\b',
    re.IGNORECASE,
)
NEGATION_WINDOW_CHARS = 40

def apply_patterns(text):
    if not isinstance(text, str) or text.strip() == '':
        return {k: 0 for k in PATTERNS}
    out = {}
    for k, p in PATTERNS.items():
        positive = 0
        for m in p.finditer(text):
            window_start = max(0, m.start() - NEGATION_WINDOW_CHARS)
            preceding = text[window_start:m.start()]
            if NEGATION_CUES.search(preceding):
                continue  # negated mention -- does not count
            positive = 1
            break
        out[k] = positive
    return out

flag_records = cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].apply(apply_patterns)
flags_df = pd.DataFrame(list(flag_records), index=cohort.index)

# ICU flag also cross-referenced against Ward/Unit field
ward_icu = cohort['Ward'].fillna('').str.contains('ICU|Critical Care', case=False, regex=True)
unit_icu = cohort['Unit'].fillna('').str.contains('ICU', case=False, regex=True)
flags_df['icu_stay'] = ((flags_df['icu_stay_text'] == 1) | ward_icu | unit_icu).astype(int)
flags_df = flags_df.drop(columns=['icu_stay_text'])

# whether text was even available (so downstream modeling can distinguish
# "no note" from "note present but no mention" -- the two are NOT the same)
flags_df['note_available'] = cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].notna().astype(int)
flags_df['note_length_chars'] = cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].fillna('').str.len()

out = pd.concat([cohort[['study_id', 'isolate_id', 'testing_date']], flags_df], axis=1)
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PARQUET, index=False)

print(f"\nSaved NLP feature table: {OUT_PARQUET} shape={out.shape}")
print("\nFlag prevalence (% of ALL isolates, including those with no note -- so these are floor estimates):")
flag_cols = [c for c in flags_df.columns if c not in ('note_available', 'note_length_chars')]
print((flags_df[flag_cols].mean() * 100).round(1).sort_values(ascending=False))

print("\nFlag prevalence AMONG ISOLATES WITH A NOTE AVAILABLE:")
has_note = flags_df['note_available'] == 1
print((flags_df.loc[has_note, flag_cols].mean() * 100).round(1).sort_values(ascending=False))
