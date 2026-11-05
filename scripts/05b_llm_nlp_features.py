"""
05b_llm_nlp_features.py

Builds the LLM-derived Model B feature table (nlp_features_llm.parquet),
matching the exact schema of the rule-based nlp_features.parquet so it is a
drop-in replacement for 09_modeling.py.

Per Methods 2.5.2 of the LLM-primary manuscript:
  - The 13 LLM-extracted flags come from the full-corpus extraction
    (scripts_new/17_llm_extraction.py output).
  - note_available and note_length_chars are UNCHANGED from the
    rule-based-feature analysis (they are structural, not extraction-method
    dependent).
  - prior_antibiotic_named (a rule-based-only, more granular field) is
    RETAINED unchanged from the rule-based pipeline, since no LLM-derived
    equivalent was extracted for it.
"""
import pandas as pd
from pathlib import Path

COHORT_PARQUET = Path('data/processed/cohort_master.parquet')
RULEBASED_NLP_PARQUET = Path('data/processed/nlp_features.parquet')
LLM_KEY_CSV = Path('data/llm/full_corpus_key.csv')
LLM_EXTRACTION_CSV = Path('data/llm/full_corpus_llm_extraction.csv')
OUT_PARQUET = Path('data/processed/nlp_features_llm.parquet')

FLAGS = [
    "comorbid_diabetes", "comorbid_ckd_dialysis", "comorbid_malignancy",
    "comorbid_immunosuppression", "comorbid_cardio_cerebrovascular",
    "prior_antibiotic_any", "device_urinary_catheter", "device_central_line",
    "device_ventilator_tracheal", "device_surgical_drain", "icu_stay",
    "recent_hospitalization", "recent_surgery",
]

cohort = pd.read_parquet(COHORT_PARQUET).reset_index(drop=True)
rb = pd.read_parquet(RULEBASED_NLP_PARQUET).reset_index(drop=True)
assert len(cohort) == len(rb), "cohort/rule-based nlp row count mismatch"
assert (cohort['study_id'].values == rb['study_id'].values).all(), "row order mismatch"

key = pd.read_csv(LLM_KEY_CSV)  # review_id, study_id, isolate_id, DISCHARGE_SUMMARY_CONTENT_DEID (per-row, same order as has_note subset)
llm_raw = pd.read_csv(LLM_EXTRACTION_CSV)

n_expected = len(key)
n_got = llm_raw['review_id'].nunique()
missing_ids = set(key['review_id']) - set(llm_raw['review_id'])
print(f"LLM extraction coverage: {n_got}/{n_expected} notes extracted ({100*n_got/n_expected:.1f}%)")
if missing_ids:
    print(f"WARNING: {len(missing_ids)} notes missing from LLM extraction output (will be left as null/0 -- "
          f"re-run scripts_new/17_llm_extraction.py on the same input/output paths to resume and fill gaps).")

merged_key = key.merge(llm_raw, on='review_id', how='left')

# Start from the rule-based table's structure (study_id, isolate_id, testing_date,
# note_available, note_length_chars, prior_antibiotic_named are retained unchanged)
out = rb[['study_id', 'isolate_id', 'testing_date', 'note_available', 'note_length_chars',
          'prior_antibiotic_named']].copy()

# Merge in LLM flags by cohort row POSITION, not by (study_id, isolate_id) --
# isolate_id is not guaranteed unique (see cohort audit), so a key-based join
# can fan out. `key` was built directly from the has_note subset of
# cohort_master in the same row order, so positional alignment is exact.
has_note = cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].notna() & (cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].str.len() > 50)
note_rows = cohort.loc[has_note].reset_index(drop=True)
assert len(note_rows) == len(key), "has_note row count doesn't match LLM key table"
# key was built with reset_index(drop=True) directly from this same has_note subset, so
# row i of `key` corresponds exactly to row i of `note_rows` / the i-th True in has_note.
flags_full = pd.DataFrame(0, index=cohort.index, columns=FLAGS)
note_idx = cohort.index[has_note]
flags_for_notes = merged_key.set_index(merged_key.index)[FLAGS].fillna(0).astype(int)
flags_for_notes.index = note_idx
flags_full.loc[note_idx, FLAGS] = flags_for_notes

out = pd.concat([out.reset_index(drop=True), flags_full.reset_index(drop=True)], axis=1)

# column order matches rule-based table exactly, for drop-in compatibility
out = out[['study_id', 'isolate_id', 'testing_date',
           'comorbid_diabetes', 'comorbid_ckd_dialysis', 'comorbid_malignancy',
           'comorbid_immunosuppression', 'comorbid_cardio_cerebrovascular',
           'prior_antibiotic_any', 'prior_antibiotic_named',
           'device_urinary_catheter', 'device_central_line', 'device_ventilator_tracheal',
           'device_surgical_drain', 'recent_hospitalization', 'recent_surgery', 'icu_stay',
           'note_available', 'note_length_chars']]

OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PARQUET, index=False)
print(f"Saved LLM-derived NLP feature table: {OUT_PARQUET} shape={out.shape}")
print("\nFlag prevalence among isolates WITH a note (LLM-derived):")
print((out.loc[out['note_available'] == 1, FLAGS].mean() * 100).round(1).sort_values(ascending=False))
