"""08b_fix_admission_dates.py  -- admission/discharge date handling and record linkage (run after 08_splits.py).

Problems found in the original feature construction (04_structured_features.py, 05_nlp_features.py):
  1. Admission and discharge timestamps were parsed only with a four-digit-year format (dd/mm/YYYY HH:MM).
     One export file (2025, second half) used two-digit years (dd/mm/yy H:MM), so these dates were silently
     set to missing: the admission-to-testing interval and onset category became 'Unknown' for these isolates
     (concentrated in the temporal test period), and the note-timing summary excluded them.
  2. The admission record is linked to isolates by patient, so for some isolates the testing date falls outside
     the linked admission. The interval was then computed against an unrelated admission (large values,
     classed as hospital onset) or was negative (set to missing).
  3. For isolates whose testing date preceded the linked admission by more than one day, the linked discharge
     summary was written at a later admission, i.e. it contains future information relative to the culture.

Corrections:
  - Parse dates with four-digit years first, then two-digit years.
  - admission_to_culture_hours (testing date at 00:00 minus admission timestamp, as before) is kept only when the
    testing date lies within the linked admission (from the admission day to one day after discharge); values
    between -24 h and 0 (testing on the admission day) are set to 0; otherwise missing. onset_bucket recomputed
    with the unchanged 48-hour rule.
  - Notes linked to a later admission (testing date more than one day before admission) are removed from the
    note features (note_available = 0, flags = 0, note length = 0) for both extraction methods; notes from an
    earlier admission (discharged before the culture) are kept, because they were available before the culture.
  - Ward, unit and department (taken from the linked admission) are set to 'Unknown' when the testing date lies outside
    the linked admission, because they then describe a different encounter.
  - The note-timing summary is recomputed.
Originals are kept as *_pre_datefix.parquet. Writes tables/date_fix_summary.json.
"""
import json, numpy as np, pandas as pd
from pathlib import Path
P = Path('data/processed')
FL = ['comorbid_diabetes', 'comorbid_ckd_dialysis', 'comorbid_malignancy', 'comorbid_immunosuppression', 'comorbid_cardio_cerebrovascular',
      'prior_antibiotic_any', 'prior_antibiotic_named', 'device_urinary_catheter', 'device_central_line', 'device_ventilator_tracheal',
      'device_surgical_drain', 'recent_hospitalization', 'recent_surgery', 'icu_stay']
def parse(x):
    a = pd.to_datetime(x, format='%d/%m/%Y %H:%M', errors='coerce'); b = pd.to_datetime(x, format='%d/%m/%y %H:%M', errors='coerce')
    return a.fillna(b)
cohort = pd.read_parquet(P / 'cohort_master.parquet').reset_index(drop=True)
for name in ['structured_features', 'splits', 'nlp_features', 'nlp_features_llm']:
    src = P / f'{name}.parquet'; bak = P / f'{name}_pre_datefix.parquet'
    if not bak.exists(): pd.read_parquet(src).to_parquet(bak, index=False)
sf = pd.read_parquet(P / 'structured_features_pre_datefix.parquet').reset_index(drop=True)
sp = pd.read_parquet(P / 'splits_pre_datefix.parquet').reset_index(drop=True)
rb = pd.read_parquet(P / 'nlp_features_pre_datefix.parquet').reset_index(drop=True)
ll = pd.read_parquet(P / 'nlp_features_llm_pre_datefix.parquet').reset_index(drop=True)
for d in (sf, sp, rb, ll):
    assert (d['study_id'].values == cohort['study_id'].values).all() and (d['isolate_id'].values == cohort['isolate_id'].values).all()
old_admit = pd.to_datetime(cohort['txtAdmitDt'], format='%d/%m/%Y %H:%M', errors='coerce')
admit = parse(cohort['txtAdmitDt']); disc = parse(cohort['txtDiscDt'])
test_dt = pd.to_datetime(cohort['testing_date'])
td = test_dt.dt.normalize(); ad = admit.dt.normalize(); dd = disc.dt.normalize()
day = pd.Timedelta(days=1)
within = admit.notna() & (td >= ad) & (disc.isna() | (td <= dd + day))
before_adm = admit.notna() & (td < ad - day)          # culture well before the linked admission
after_disc = disc.notna() & (td > dd + day)           # culture after the linked discharge
hours = (test_dt - admit).dt.total_seconds() / 3600.0
hours = hours.where(within, np.nan).clip(lower=0)
onset = np.where(hours.isna(), 'Unknown', np.where(hours < 48, 'Community-onset (<48h)', 'Hospital-onset (>=48h)'))
summary = dict(
    admission_dates_parsed_before=int(old_admit.notna().sum()), admission_dates_parsed_after=int(admit.notna().sum()),
    two_digit_year_admission_dates_recovered=int((admit.notna() & old_admit.isna()).sum()),
    testing_within_linked_admission=int(within.sum()), testing_after_linked_discharge=int(after_disc.sum()),
    testing_before_linked_admission=int(before_adm.sum()),
    onset_before=sf['onset_bucket'].value_counts().to_dict(), onset_after=pd.Series(onset).value_counts().to_dict())
outside = (before_adm | after_disc).values
for d in (sf, sp):
    d['admission_to_culture_hours'] = hours.values; d['onset_bucket'] = onset
    # ward, unit and department come from the linked admission; when the culture was taken outside that admission
    # they describe a different encounter (for cultures before the admission, a future one) and are set to missing
    for col in ['ward_clean', 'unit_clean', 'dept_clean']:
        d.loc[outside, col] = 'Unknown'
summary['ward_unit_department_set_missing'] = int(outside.sum())
has_note = ll['note_available'] == 1
later_note = has_note & before_adm
summary['notes_from_later_admission_removed'] = int(later_note.sum())
summary['notes_from_earlier_admission_kept'] = int((has_note & after_disc).sum())
for d in (rb, ll):
    d.loc[later_note.values, FL] = 0; d.loc[later_note.values, 'note_available'] = 0; d.loc[later_note.values, 'note_length_chars'] = 0
# (the rule-based ICU flag also uses the ward code of the linked admission; for these isolates that admission is a later one, so it stays 0)
summary['isolates_with_note_after_fix'] = int((ll['note_available'] == 1).sum())
for t in ['MRSA', 'ESBL', 'CRE']:
    for s_ in ['train', 'test']:
        m = (sp[f'in_{t}_target_cohort'] & sp[f'label_{t}'].notna() & (sp['split_temporal'] == s_)).values
        summary[f'{t}_{s_}_onset_after'] = pd.Series(onset[m]).value_counts().to_dict()
        summary[f'{t}_{s_}_later_notes_removed'] = int(later_note.values[m].sum())
sf.to_parquet(P / 'structured_features.parquet', index=False); sp.to_parquet(P / 'splits.parquet', index=False)
rb.to_parquet(P / 'nlp_features.parquet', index=False); ll.to_parquet(P / 'nlp_features_llm.parquet', index=False)
# note timing (isolates with a note used in modelling and a usable discharge date)
hn = (ll['note_available'] == 1)
both = hn & disc.notna()
gap = (disc - td).dt.total_seconds() / 86400.0
g = gap[both]
timing = dict(n_isolates_with_note=int(hn.sum()), n_with_note_and_usable_discharge_date=int(both.sum()),
              pct_discharge_date_before_testing_date=round(100 * float((dd[both] < td[both]).mean()), 1),
              pct_discharge_date_same_day_as_testing_date=round(100 * float((dd[both] == td[both]).mean()), 1),
              pct_discharge_date_1plus_days_after_testing_date=round(100 * float((dd[both] > td[both]).mean()), 1),
              pct_discharge_on_or_after_testing_date=round(100 * float((dd[both] >= td[both]).mean()), 1),
              median_gap_days_discharge_minus_testing=round(float(g.median()), 2), iqr_gap_days=[round(float(g.quantile(.25)), 2), round(float(g.quantile(.75)), 2)],
              note='Positive gap = discharge after the testing date. Notes linked to a later admission were removed before this summary; '
                   'a discharge before the testing date means the note comes from an earlier admission and predates the culture.')
for out in ['tables/discharge_note_timing.json', 'tables/llm/discharge_note_timing.json']:
    json.dump(timing, open(out, 'w'), indent=2)
json.dump(summary, open('tables/date_fix_summary.json', 'w'), indent=1, default=int)
print(json.dumps(summary, indent=1, default=int)); print(json.dumps(timing, indent=1))
