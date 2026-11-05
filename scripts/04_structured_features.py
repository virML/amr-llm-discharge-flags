"""
04_structured_features.py
Structured feature engineering -- Model A feature table.

Leakage safety: "prior isolate" features for a given row use ONLY isolates
from the same patient with a strictly earlier Testing Date, computed over
the patient's FULL microbiology history in the de-identified master table
(not just the narrower modeling cohort), because a clinician deciding on
empiric therapy would see all prior culture reports for that patient, not
only the subset that happens to satisfy this study's organism/ID-confidence
inclusion filters. The "prior resistant organism" flag uses the
Organism-Name resistance suffix tags (MRSA/MDR/MDRO/CRE), which are always
available on any prior report a clinician would have seen, rather than the
narrower triangulated target labels (which are only defined within
target-specific sub-cohorts and would otherwise silently drop information
that a clinician-facing signal should retain).
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path

FULL_HISTORY_PARQUET = Path('data/processed/deidentified_master.parquet')
COHORT_PARQUET = Path('data/processed/cohort_master.parquet')
OUT_PARQUET = Path('data/processed/structured_features.parquet')

cohort = pd.read_parquet(COHORT_PARQUET)
full = pd.read_parquet(FULL_HISTORY_PARQUET)

print(f"Cohort: {cohort.shape}, full history universe: {full.shape}")

# ---------------------------------------------------------------------------
# 1. Age parsing: 'XXY XXM XXD' (any subset of components) -> numeric years
# ---------------------------------------------------------------------------
AGE_RE = re.compile(r'(?:(\d+)\s*Y)?\s*(?:(\d+)\s*M)?\s*(?:(\d+)\s*D)?', re.IGNORECASE)

def parse_age_years(raw):
    if not isinstance(raw, str) or raw.strip() == '':
        return np.nan
    m = AGE_RE.search(raw)
    if not m:
        return np.nan
    y, mo, d = m.groups()
    if y is None and mo is None and d is None:
        return np.nan
    y = int(y) if y else 0
    mo = int(mo) if mo else 0
    d = int(d) if d else 0
    if y == 0 and mo == 0 and d == 0:
        return np.nan
    return round(y + mo / 12.0 + d / 365.25, 3)

cohort['age_years'] = cohort['Age'].apply(parse_age_years)
n_age_missing = int(cohort['age_years'].isna().sum())
print(f"Age parsed: {cohort['age_years'].notna().sum()}/{len(cohort)} "
      f"(missing/unparseable: {n_age_missing})")

# ---------------------------------------------------------------------------
# 2. Gender cleanup
# ---------------------------------------------------------------------------
def clean_gender(g):
    if not isinstance(g, str):
        return 'Unknown'
    g = g.strip()
    if g in ('Male', 'Male Child'):
        return 'Male'
    if g in ('Female', 'Female Child'):
        return 'Female'
    return 'Unknown'

cohort['gender_clean'] = cohort['Gender'].apply(clean_gender)
cohort['is_pediatric_flag'] = cohort['Gender'].isin(['Male Child', 'Female Child']).astype(int)
# supplement pediatric flag with parsed age < 18 where age is known
cohort.loc[cohort['age_years'] < 18, 'is_pediatric_flag'] = 1

# ---------------------------------------------------------------------------
# 3. Ward / Unit / Department cleanup
# ---------------------------------------------------------------------------
for c, newc in [('Ward', 'ward_clean'), ('Unit', 'unit_clean'), ('drpDept', 'dept_clean')]:
    cohort[newc] = cohort[c].fillna('Unknown')
    cohort[newc] = cohort[newc].replace('--Select--', 'Unknown')

# ---------------------------------------------------------------------------
# 4. Admission-to-culture-collection interval
# ---------------------------------------------------------------------------
admit_dt = pd.to_datetime(cohort['txtAdmitDt'], format='%d/%m/%Y %H:%M', errors='coerce')
testing_dt = cohort['testing_date']  # already parsed as datetime (date only) in script 03
# testing_date has no time-of-day component; treat as midnight of that day
delta_hours = (testing_dt - admit_dt).dt.total_seconds() / 3600.0

# Invalid: admit date after testing date (data entry error) -> treat as missing
delta_hours = delta_hours.where(delta_hours >= 0, np.nan)
cohort['admission_to_culture_hours'] = delta_hours

def onset_bucket(h):
    if pd.isna(h):
        return 'Unknown'
    return 'Community-onset (<48h)' if h < 48 else 'Hospital-onset (>=48h)'

cohort['onset_bucket'] = cohort['admission_to_culture_hours'].apply(onset_bucket)
print("\nOnset bucket counts:\n", cohort['onset_bucket'].value_counts())

# ---------------------------------------------------------------------------
# 5. Prior-isolate history (leakage-safe, computed on FULL patient history)
# ---------------------------------------------------------------------------
full_testing_dt = pd.to_datetime(full['Testing Date'], format='%d/%m/%y', errors='coerce')
full = full.assign(testing_date=full_testing_dt)
full = full[full['study_id'].notna() & full['testing_date'].notna()].copy()

suffix_pattern = re.compile(r'\((MRSA|MDR|MDRO|CRE)\)')
full['has_resistance_suffix'] = full['Organism Name'].fillna('').apply(lambda s: bool(suffix_pattern.search(s))).astype(int)

full_sorted = full.sort_values(['study_id', 'testing_date'])

prior_count_map = {}
prior_resist_map = {}
days_since_last_map = {}

for pid, grp in full_sorted.groupby('study_id', sort=False):
    dates = grp['testing_date'].values
    resist = grp['has_resistance_suffix'].values
    idxs = grp.index.values
    n = len(grp)
    for i in range(n):
        # strictly earlier dates only (ties on the same date are NOT "prior")
        earlier_mask = dates[:i] < dates[i] if i > 0 else np.array([], dtype=bool)
        # handle possible out-of-order ties by comparing directly (sorted already, but same-date rows adjacent)
        strictly_prior = np.sum(dates[:i] < dates[i])
        prior_count_map[idxs[i]] = int(strictly_prior)
        if strictly_prior > 0:
            prior_dates = dates[:i][dates[:i] < dates[i]]
            prior_resist_map[idxs[i]] = int(np.any(resist[:i][dates[:i] < dates[i]]))
            last_prior_date = prior_dates.max()
            days_since_last_map[idxs[i]] = float((dates[i] - last_prior_date) / np.timedelta64(1, 'D'))
        else:
            prior_resist_map[idxs[i]] = 0
            days_since_last_map[idxs[i]] = np.nan

full_sorted['prior_isolate_count'] = full_sorted.index.map(prior_count_map)
full_sorted['prior_resistant_organism_flag'] = full_sorted.index.map(prior_resist_map)
full_sorted['days_since_last_isolate'] = full_sorted.index.map(days_since_last_map)

prior_feats = full_sorted[['prior_isolate_count', 'prior_resistant_organism_flag', 'days_since_last_isolate']]
cohort = cohort.join(prior_feats, how='left')

print("\nPrior isolate count distribution:\n", cohort['prior_isolate_count'].describe())
print("Any prior resistant organism (suffix-tagged): ", cohort['prior_resistant_organism_flag'].mean())

# ---------------------------------------------------------------------------
# 6. ID-confidence / percent-probability data-quality covariate
# ---------------------------------------------------------------------------
cohort['percent_probability_num'] = pd.to_numeric(cohort['Percent Probability'], errors='coerce')
cohort['id_confidence_known'] = cohort['ID Confidence'].notna().astype(int)
cohort['id_confidence_clean'] = cohort['ID Confidence'].fillna('Not recorded')

# ---------------------------------------------------------------------------
# 7. Specimen / infection site cleanup
# ---------------------------------------------------------------------------
cohort['specimen_type_clean'] = cohort['Specimen Type'].fillna('Unknown')
cohort['infection_site_clean'] = cohort['Selected BP Infection Site'].fillna('Not specified')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
keep_meta = ['study_id', 'isolate_id', 'testing_date', '__source_csv']
keep_targets = ['label_MRSA', 'label_ESBL', 'label_CRE',
                 'in_MRSA_target_cohort', 'in_ESBL_target_cohort', 'in_CRE_target_cohort']
keep_structured = [
    'organism_group', 'organism_base', 'species_ecoli_klebsiella',
    'specimen_type_clean', 'infection_site_clean',
    'age_years', 'gender_clean', 'is_pediatric_flag',
    'ward_clean', 'unit_clean', 'dept_clean',
    'admission_to_culture_hours', 'onset_bucket',
    'prior_isolate_count', 'prior_resistant_organism_flag', 'days_since_last_isolate',
    'percent_probability_num', 'id_confidence_known', 'id_confidence_clean',
]
out = cohort[keep_meta + keep_targets + keep_structured].copy()
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
out.to_parquet(OUT_PARQUET, index=False)
print(f"\nSaved structured feature table: {OUT_PARQUET} shape={out.shape}")
print(out.dtypes)
