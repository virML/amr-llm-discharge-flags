"""Table 1: characteristics of the analytic cohort. Writes tables/table1_final.json and tables/table1_final.csv.

Every categorical section sums to the full cohort (checked with assertions below).
Setting at testing: an isolate is an outpatient isolate when no inpatient admission covers its testing date, i.e. the
linked admission record has no admission date and no ward, or the testing date falls more than one day before the
linked admission or more than one day after its discharge (the rules of 08b_fix_admission_dates.py). Ward,
department and the admission-to-testing interval do not apply to outpatient isolates. A few admitted isolates had no
ward, department or admission time recorded; they are shown separately.
Run from the project root after 08b_fix_admission_dates.py.
"""
import json
import numpy as np
import pandas as pd

c = pd.read_parquet('data/processed/cohort_master.parquet').reset_index(drop=True)
s = pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
ll_pre = pd.read_parquet('data/processed/nlp_features_llm_pre_datefix.parquet').reset_index(drop=True)
ll = pd.read_parquet('data/processed/nlp_features_llm.parquet').reset_index(drop=True)
for d in (s, ll_pre, ll):
    assert (d['isolate_id'].values == c['isolate_id'].values).all()
N = len(s)

# ---- setting at testing (same date rules as 08b) --------------------------------------------------------------
def parse(x):
    a = pd.to_datetime(x, format='%d/%m/%Y %H:%M', errors='coerce'); b = pd.to_datetime(x, format='%d/%m/%y %H:%M', errors='coerce')
    return a.fillna(b)
admit, disc = parse(c['txtAdmitDt']), parse(c['txtDiscDt'])
td, day = pd.to_datetime(c['testing_date']).dt.normalize(), pd.Timedelta(days=1)
before_adm = admit.notna() & (td < admit.dt.normalize() - day)
after_disc = disc.notna() & (td > disc.dt.normalize() + day)
no_admission = admit.isna() & c['Ward'].isna()
outpatient = no_admission | before_adm | after_disc
# cross-checks against the model features
assert (outpatient == ((s['ward_clean'] == 'Unknown') & s['admission_to_culture_hours'].isna())).all()
assert (s.loc[outpatient, 'dept_clean'] == 'Unknown').all() and (s.loc[outpatient, 'onset_bucket'] == 'Unknown').all()

rows = []
def section(title): rows.append({'section': title})
def item(label, n, denom=N, pct=True, note=None):
    rows.append({'label': label, 'n': int(n), 'pct': round(100 * n / denom, 1) if pct else None, 'note': note})
def check(start, total=N):
    tot = sum(r['n'] for r in rows[start:] if 'n' in r)
    assert tot == total, (rows[start - 1], tot, total)

# ---- demographics -----------------------------------------------------------------------------------------------
age = s['age_years'].astype(float)
rows.append({'label': 'Isolates, n', 'text': str(N)})
rows.append({'label': 'Patients, n', 'text': str(s['study_id'].nunique())})
rows.append({'label': 'Age, years, mean (SD)', 'text': f'{age.mean():.1f} ({age.std():.1f})'})
q = np.nanpercentile(age, [50, 25, 75])
rows.append({'label': 'Age, years, median (IQR)', 'text': f'{q[0]:.1f} ({q[1]:.1f} to {q[2]:.1f})'})
item('Age not recorded, n (%)', age.isna().sum())
g0 = len(rows)
item('Gender: male, n (%)', (s['gender_clean'] == 'Male').sum())
item('Gender: female, n (%)', (s['gender_clean'] == 'Female').sum())
item('Gender: not recorded, n (%)', (s['gender_clean'] == 'Unknown').sum())
check(g0)
item('Paediatric (<18 years or coded as child), n (%)', (s['is_pediatric_flag'] == 1).sum())

# ---- ward and department ------------------------------------------------------------------------------------------
WARD = {'Special Ward': 'Special Ward (General Medicine)', 'OBS': 'Obstetrics'}
DEPT = {'Neuro Surgery': 'Neurosurgery', 'Orthopedics': 'Orthopaedics'}
def setting_section(col, title, names, k=9, what='ward'):
    section(title); st = len(rows)
    item('Outpatient (not admitted at testing)', outpatient.sum(), note='a')
    adm = s.loc[~outpatient, col]
    vc = adm[adm != 'Unknown'].value_counts()
    for name, n in vc.head(k).items():
        item(names.get(name, name), n)
    rest = vc.iloc[k:]
    item(f'Other {what}s ({len(rest)})', rest.sum())
    item(f'Admitted, {what} not recorded', (adm == 'Unknown').sum())
    check(st)
setting_section('ward_clean', 'Ward at testing', WARD, what='ward')
setting_section('dept_clean', 'Department at testing', DEPT, what='department')

# ---- organism and specimen ----------------------------------------------------------------------------------------
section('Organism group'); st = len(rows)
for g, lab in [('Enterobacterales', 'Enterobacterales'), ('Enterococcus', '_Enterococcus_'), ('Pseudomonas', '_Pseudomonas_'),
               ('Acinetobacter', '_Acinetobacter_'), ('Staph_aureus', '_Staphylococcus aureus_')]:
    item(lab, (s['organism_group'] == g).sum())
check(st)
section('Specimen type'); st = len(rows)
sp = s['specimen_type_clean']
named = sp[sp != 'OTHERS'].value_counts().head(7)
for name, n in named.items():
    item(name.capitalize(), n)
item('Other specimen types', (~sp.isin(named.index)).sum(), note='b')
check(st)
spec_other = dict(n_specimen_coded_others=int((sp == 'OTHERS').sum()), n_specimen_other_named=int((~sp.isin(named.index) & (sp != 'OTHERS')).sum()),
                  n_specimen_other_named_types=int(sp[~sp.isin(named.index) & (sp != 'OTHERS')].nunique()))

# ---- onset ----------------------------------------------------------------------------------------------------------
section('Admission-to-testing interval'); st = len(rows)
h = s['admission_to_culture_hours']
item('Under 48 hours', (h < 48).sum())
item('48 hours or later', (h >= 48).sum())
item('Outpatient (not admitted at testing)', outpatient.sum(), note='a')
item('Admitted, admission time not recorded', (h.isna() & ~outpatient).sum())
check(st)
assert (h < 48).sum() == (s['onset_bucket'] == 'Community-onset (<48h)').sum()

# ---- history and notes ----------------------------------------------------------------------------------------------
section('Prior microbiology and notes')
item('Prior isolate from the same patient, n (%)', (s['prior_isolate_count'].fillna(0) > 0).sum())
item('Linked discharge summary, n (%)', (ll_pre['note_available'] == 1).sum())
item('Note features available (summary not from a later admission), n (%)', (ll['note_available'] == 1).sum())

# ---- resistance -----------------------------------------------------------------------------------------------------
section('Resistance among labelled isolates, n/N (%)')
for t, lab in [('MRSA', 'MRSA (_S. aureus_)'), ('ESBL', 'ESBL (_E. coli_ or _Klebsiella_)'), ('CRE', 'CRE (Enterobacterales)')]:
    x = s.loc[s[f'in_{t}_target_cohort'] == 1, f'label_{t}'].dropna()
    rows.append({'label': lab, 'text': f'{int(x.sum())}/{len(x)} ({100 * x.mean():.1f}%)'})

facts = {'n_outpatient': int(outpatient.sum()), 'n_outpatient_no_admission_record': int(no_admission.sum()),
         'n_outpatient_outside_linked_admission': int((before_adm | after_disc).sum()),
         'n_age_gender_not_recorded': int(age.isna().sum()),
         'n_age_gender_not_recorded_outpatient': int((age.isna() & outpatient).sum()),
         'n_admitted_ward_not_recorded': int(((s['ward_clean'] == 'Unknown') & ~outpatient).sum()),
         'n_admitted_department_not_recorded': int(((s['dept_clean'] == 'Unknown') & ~outpatient).sum()),
         'n_admitted_admission_time_not_recorded': int((h.isna() & ~outpatient).sum())}
facts.update(spec_other)
for t in ['MRSA', 'ESBL', 'CRE']:
    m = (s['split_temporal'] == 'test') & (s[f'in_{t}_target_cohort'] == 1) & s[f'label_{t}'].notna()
    facts[f'test_{t}_age_gender_not_recorded'] = int((m & age.isna()).sum())
    facts[f'test_{t}_age_gender_not_recorded_outpatient'] = int((m & age.isna() & outpatient).sum())
    facts[f'test_{t}_outpatient'] = int((m & outpatient).sum())
json.dump({'rows': rows, 'facts': facts}, open('tables/table1_final.json', 'w'), indent=1, ensure_ascii=False)
pd.DataFrame([{'characteristic': r.get('label', '## ' + r.get('section', '')),
               'value': r.get('text') if 'text' in r else (f"{r['n']} ({r['pct']:.1f}%)" if 'n' in r else '')} for r in rows]
             ).to_csv('tables/table1_final.csv', index=False)
for r in rows:
    print(('## ' + r['section']) if 'section' in r else f"   {r['label']:<75} {r.get('text') or str(r['n']) + ' (' + format(r['pct'], '.1f') + '%)'}")
print(json.dumps(facts, indent=1))
