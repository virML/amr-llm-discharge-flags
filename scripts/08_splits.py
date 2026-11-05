"""
08_splits.py
Patient-level train/test splits.

Both splits are enforced at the PATIENT level: every isolate belonging to a
given patient is entirely in train OR entirely in test, never both.

  - PRIMARY (temporal): a patient's split assignment is determined by the
    Testing Date of their EARLIEST isolate in the cohort relative to a
    cutoff date chosen so ~80% of patients (by first-isolate date) fall
    before it. Patient-level exclusivity is enforced, but NOT by letting a
    train-assigned patient's later isolates travel with them regardless of
    date: any isolate dated on/after the cutoff belonging to a
    train-assigned patient is dropped from the temporal-split modeling
    population entirely (marked 'excluded_straddling'), so that every row
    used as 'train' chronologically precedes every row used as 'test'.
    Assigning such isolates to 'test' instead is not an option -- the same
    patient's earlier isolates are already in 'train', and mixing would
    reintroduce patient leakage. This costs some isolates but is the only
    way to have both a strictly calendar-ordered split and zero patient
    leakage at the same time.
  - SENSITIVITY (random): a random 80/20 patient-level split, stratified by
    each patient's most frequent organism_group (a shared covariate across
    all 3 targets, so one consistent random split can serve MRSA/ESBL/CRE
    all at once rather than needing 3 separate incompatible splits). This
    split does not claim calendar ordering, so straddling isolates are not
    dropped from it.

Both splits are verified for zero patient leakage before being saved; the
temporal split is additionally verified for zero calendar overlap (max
train date < cutoff <= min test date).
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

IN_PARQUET = Path('data/processed/structured_features.parquet')
OUT_PARQUET = Path('data/processed/splits.parquet')
OUT_JSON = Path('tables/split_summary.json')

RANDOM_STATE = 42
TRAIN_FRAC = 0.80

df = pd.read_parquet(IN_PARQUET)
print(f"Loaded structured features: {df.shape}, unique patients: {df['study_id'].nunique()}")

# ---------------------------------------------------------------------------
# 1. Temporal split
# ---------------------------------------------------------------------------
patient_first_date = df.groupby('study_id')['testing_date'].min().sort_values()
n_patients = len(patient_first_date)
cutoff_idx = int(np.floor(TRAIN_FRAC * n_patients))
cutoff_date = patient_first_date.iloc[cutoff_idx]
print(f"\nTemporal split cutoff date (80th percentile of patients' first-isolate date): {cutoff_date}")

temporal_split = pd.Series(
    np.where(patient_first_date.values < cutoff_date, 'train', 'test'),
    index=patient_first_date.index, name='split_temporal'
)

# ---------------------------------------------------------------------------
# 2. Random patient-level stratified split
# ---------------------------------------------------------------------------
patient_dominant_group = df.groupby('study_id')['organism_group'].agg(lambda s: s.mode().iloc[0])
patients = np.array(patient_dominant_group.index.tolist(), dtype=object)
strata = np.array(patient_dominant_group.values.tolist(), dtype=object)

train_pids, test_pids = train_test_split(
    patients, test_size=1 - TRAIN_FRAC, random_state=RANDOM_STATE, stratify=strata
)
random_split = pd.Series('test', index=patients, name='split_random')
random_split.loc[train_pids] = 'train'

# ---------------------------------------------------------------------------
# 3. Combine + verify zero leakage
# ---------------------------------------------------------------------------
splits = pd.DataFrame({'split_temporal': temporal_split, 'split_random': random_split})
df = df.merge(splits, left_on='study_id', right_index=True, how='left')

# ---------------------------------------------------------------------------
# 3b. Enforce strict calendar ordering on the temporal split: drop isolates
#     dated on/after the cutoff that belong to a train-assigned patient (see
#     module docstring). This does NOT touch split_random.
# ---------------------------------------------------------------------------
straddling = (df['split_temporal'] == 'train') & (df['testing_date'] >= cutoff_date)
n_straddling = int(straddling.sum())
n_straddling_patients = int(df.loc[straddling, 'study_id'].nunique())
print(f"\nTemporal split: dropping {n_straddling} isolates ({n_straddling_patients} patients also have "
      f"earlier, retained isolates) dated on/after the cutoff ({cutoff_date}) from a train-assigned "
      f"patient, so no training row postdates any test row.")
df.loc[straddling, 'split_temporal'] = 'excluded_straddling'

for col in ['split_temporal', 'split_random']:
    train_pids_check = set(df.loc[df[col] == 'train', 'study_id'])
    test_pids_check = set(df.loc[df[col] == 'test', 'study_id'])
    overlap = train_pids_check & test_pids_check
    assert len(overlap) == 0, f"LEAKAGE DETECTED in {col}: {len(overlap)} patients in both train and test!"
    print(f"\n[{col}] verified zero patient overlap. "
          f"train patients={len(train_pids_check)}, test patients={len(test_pids_check)}")

max_train_date = df.loc[df['split_temporal'] == 'train', 'testing_date'].max()
min_test_date = df.loc[df['split_temporal'] == 'test', 'testing_date'].min()
assert max_train_date < cutoff_date <= min_test_date, (
    f"Temporal split is not strictly calendar-ordered: max train date {max_train_date}, "
    f"cutoff {cutoff_date}, min test date {min_test_date}"
)
print(f"\n[split_temporal] verified strict calendar ordering: max train date {max_train_date} "
      f"< cutoff {cutoff_date} <= min test date {min_test_date}")

# ---------------------------------------------------------------------------
# 4. Summary stats: isolate counts, date ranges, target prevalence per side
# ---------------------------------------------------------------------------
summary = {}
for col in ['split_temporal', 'split_random']:
    summary[col] = {}
    for side in ['train', 'test']:
        sub = df[df[col] == side]
        s = {
            'n_isolates': int(len(sub)),
            'n_patients': int(sub['study_id'].nunique()),
            'date_min': str(sub['testing_date'].min()),
            'date_max': str(sub['testing_date'].max()),
        }
        for target, cohort_flag in [('MRSA', 'in_MRSA_target_cohort'), ('ESBL', 'in_ESBL_target_cohort'), ('CRE', 'in_CRE_target_cohort')]:
            tsub = sub[sub[cohort_flag]]
            lab = tsub[f'label_{target}']
            s[f'{target}_n_target_cohort'] = int(len(tsub))
            s[f'{target}_n_labeled'] = int(lab.notna().sum())
            s[f'{target}_prevalence_pct'] = round(100 * (lab == 1).sum() / max(lab.notna().sum(), 1), 1)
        summary[col][side] = s

print("\n=== SPLIT SUMMARY ===")
print(json.dumps(summary, indent=2))

OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)
with open(OUT_JSON, 'w') as f:
    json.dump({
        'cutoff_date_temporal': str(cutoff_date),
        'n_isolates_excluded_straddling': n_straddling,
        'n_patients_with_excluded_straddling_isolates': n_straddling_patients,
        'summary': summary,
    }, f, indent=2, default=str)

print(f"\nSaved: {OUT_PARQUET}, {OUT_JSON}")
