"""Data-quality and reproducibility checks defined after the primary analysis (Additional file 1 Note S7).

  1. Unlabelled target isolates: why no label could be assigned, and labelled versus unlabelled characteristics.
  2. LLM batching: batch membership reconstructed from the archived output (batches of 25 consecutive review_ids,
     review_id // 25, confirmed by the output row order); repeat agreement for identical documents processed in the
     same versus different calls; and a within-document test for cross-note contamination (for discordant repeat
     extractions in different calls, is the positive call more often made in the call whose other notes had more
     positives for that flag?).
  3. Species-frequency rule recomputed with training-period isolates only.
  4. Development sample size: encoded candidate parameters, events per parameter and the Riley et al. criteria.
  5. SHA-256 hashes of the archived LLM input key, LLM output and extraction code.
Writes tables/reviewer/data_quality.json and CSVs. Run from the project root.
Inputs: data/processed/*, data/llm/full_corpus_key.csv (review_id, study_id, isolate_id, note text; RESTRICTED),
data/llm/full_corpus_llm_extraction.csv (review_id + 13 flags, in the order batches were checkpointed).
"""
import hashlib
import json
import os
import re
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

OUT = 'tables/reviewer'
os.makedirs(OUT, exist_ok=True)
R = {}
FL = ['comorbid_diabetes', 'comorbid_ckd_dialysis', 'comorbid_malignancy', 'comorbid_immunosuppression', 'comorbid_cardio_cerebrovascular',
      'prior_antibiotic_any', 'device_urinary_catheter', 'device_central_line', 'device_ventilator_tracheal', 'device_surgical_drain',
      'icu_stay', 'recent_hospitalization', 'recent_surgery']
coh = pd.read_parquet('data/processed/cohort_master.parquet').reset_index(drop=True)
sp = pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
assert (coh.study_id.values == sp.study_id.values).all()
MIC = [c for c in coh.columns if re.match(r'^[A-Z0-9]{2,6}-', c)]
BL = [c for c in coh.columns if c.startswith('Family -BETA-LACTAMS')]
FAM = [c for c in coh.columns if c.startswith('Family -')]

# ---- 1. unlabelled isolates ----------------------------------------------------------------------------------------
outpatient = (sp.ward_clean == 'Unknown') & sp.admission_to_culture_hours.isna()
no_ast = coh[MIC].isna().all(axis=1) & coh[FAM].isna().all(axis=1)
reason_rows, cmp_rows = [], []
need = {'MRSA': ['OXSF-Cefoxitin Screen', 'OX1-Oxacillin'], 'ESBL': ['CRO-Ceftriaxone', 'CTX-Cefotaxime', 'CAZ-Ceftazidime'],
        'CRE': ['MEM-Meropenem', 'IPM-Imipenem']}
for t in ['MRSA', 'ESBL', 'CRE']:
    inc = sp[f'in_{t}_target_cohort'] == 1; un = inc & sp[f'label_{t}'].isna(); lab = inc & sp[f'label_{t}'].notna()
    r = dict(target=t, in_cohort=int(inc.sum()), unlabelled=int(un.sum()), no_AST_result_at_all=int((un & no_ast).sum()),
             no_beta_lactam_AES_call=int((un & coh[BL].isna().all(axis=1)).sum()),
             no_relevant_MIC=int((un & coh[need[t]].isna().all(axis=1)).sum()))
    if t == 'CRE':
        prot = coh.organism_base.fillna('').str.startswith(('Proteus', 'Prov.', 'Morg.'))
        r['proteeae_imipenem_only'] = int((un & prot & coh['MEM-Meropenem'].isna() & coh['IPM-Imipenem'].notna()).sum())
    r['card_type_ID_only'] = int((un & (coh['Card Type'] == 'ID') & coh['Card Type.1'].isna()).sum())
    reason_rows.append(r)
    for nm, m in [('labelled', lab), ('unlabelled', un)]:
        s = sp[m]
        cmp_rows.append(dict(target=t, group=nm, n=int(m.sum()), test_set_pct=100 * float((s.split_temporal == 'test').mean()),
                             outpatient_pct=100 * float(outpatient[m].mean()), urine_pct=100 * float((s.specimen_type_clean == 'URINE').mean()),
                             top_organism=str(s.organism_base.value_counts().index[0]), top_organism_pct=100 * float(s.organism_base.value_counts(normalize=True).iloc[0]),
                             year_2025_pct=100 * float((pd.to_datetime(s.testing_date).dt.year == 2025).mean())))
R['unlabelled'] = reason_rows
pd.DataFrame(reason_rows).to_csv(f'{OUT}/unlabelled_reasons.csv', index=False)
pd.DataFrame(cmp_rows).to_csv(f'{OUT}/unlabelled_vs_labelled.csv', index=False)
print(pd.DataFrame(reason_rows).to_string(index=False)); print(pd.DataFrame(cmp_rows).round(1).to_string(index=False))

# ---- 2. batching ----------------------------------------------------------------------------------------------------
key = pd.read_csv('data/llm/full_corpus_key.csv')
docs = key['DISCHARGE_SUMMARY_CONTENT_DEID'].astype(str)
out = pd.read_csv('data/llm/full_corpus_llm_extraction.csv')
idx = out.review_id.str[1:].astype(int).values
starts = np.r_[0, np.where(np.diff(idx) != 1)[0] + 1]
assert (idx[starts] % 25 == 0).all(), 'checkpointed runs do not start on 25-note batch boundaries'
kk = key[['review_id']].copy(); kk['doc'] = [hashlib.sha256(x.encode()).hexdigest() for x in docs]
kk['batch'] = kk.review_id.str[1:].astype(int) // 25
m = kk.merge(out, on='review_id', how='left'); assert m[FL].notna().all().all()
m['pos'] = m.review_id.str[1:].astype(int)
bat = dict(n_batches=int(m.batch.nunique()), batch_size_max=int(m.groupby('batch').size().max()))
rep = []
first = m.sort_values('pos').groupby('doc').head(1).set_index('doc')
for doc, g in m.sort_values('pos').groupby('doc'):
    if len(g) < 2: continue
    f0 = g.iloc[0]
    for _, r in g.iloc[1:].iterrows():
        rep.append(dict(doc=doc, same_batch=bool(r.batch == f0.batch), batch_first=int(f0.batch), batch_rep=int(r.batch),
                        **{f: int(r[f] == f0[f]) for f in FL}, **{f'v0_{f}': int(f0[f]) for f in FL}, **{f'v1_{f}': int(r[f]) for f in FL}))
rp = pd.DataFrame(rep)
bat['documents_with_repeats'] = int(rp.doc.nunique()); bat['repeat_extractions'] = int(len(rp))
bat['repeats_in_same_call'] = int(rp.same_batch.sum()); bat['repeats_in_different_calls'] = int((~rp.same_batch).sum())
for lab, sub in [('same_call', rp[rp.same_batch]), ('different_call', rp[~rp.same_batch]), ('all', rp)]:
    bat[f'agreement_pct_{lab}'] = 100 * float(sub[FL].values.mean()) if len(sub) else None
    bat[f'agreement_by_flag_{lab}'] = {f: 100 * float(sub[f].mean()) for f in FL} if len(sub) else None
# contamination test among different-call discordant pairs
bsum = m.groupby('batch')[FL].sum(); bn = m.groupby('batch').size()
docs_in_batch = m.groupby(['batch', 'doc']).size()
def neighbour_prev(b, doc, f):
    own = m[(m.batch == b) & (m.doc == doc)][f].sum(); n_own = int(docs_in_batch.loc[(b, doc)])
    n = bn.loc[b] - n_own
    return (bsum.loc[b, f] - own) / n if n > 0 else np.nan
wins = ties = n = 0
for _, r in rp[~rp.same_batch].iterrows():
    for f in FL:
        if r[f'v0_{f}'] != r[f'v1_{f}']:
            bpos, bneg = (r.batch_first, r.batch_rep) if r[f'v0_{f}'] == 1 else (r.batch_rep, r.batch_first)
            a, b_ = neighbour_prev(bpos, r.doc, f), neighbour_prev(bneg, r.doc, f)
            if np.isnan(a) or np.isnan(b_): continue
            n += 1
            if a > b_: wins += 1
            elif a == b_: ties += 1
bat['contamination_test'] = dict(discordant_calls=n, positive_call_in_batch_with_more_neighbour_positives=wins, ties=ties,
                                 sign_test_p=float(binomtest(wins, n - ties, 0.5).pvalue) if n - ties > 0 else None)
R['batching'] = bat
print(json.dumps({k: v for k, v in bat.items() if not k.startswith('agreement_by_flag')}, indent=1, default=float))

# ---- 3. species rule from training period ----------------------------------------------------------------------------
cut = pd.Timestamp('2025-06-28')
td = pd.to_datetime(coh.testing_date)
cnt_all = coh['Organism Name'].value_counts(); cnt_tr = coh.loc[td < cut, 'Organism Name'].value_counts()
low = [o for o in cnt_all.index if cnt_tr.get(o, 0) < 20]
aff = coh['Organism Name'].isin(low)
R['species_rule'] = dict(species_retained=int(len(cnt_all)), species_below20_in_training_period=len(low),
                         isolates_affected=int(aff.sum()), test_isolates_affected=int((aff & (sp.split_temporal == 'test')).sum()),
                         affected_target_labelled={t: int((aff & (sp[f'in_{t}_target_cohort'] == 1) & sp[f'label_{t}'].notna() & (sp.split_temporal == 'test')).sum()) for t in ['MRSA', 'ESBL', 'CRE']},
                         note='counts after 7-day de-duplication (the original rule counted records before de-duplication), so this over-counts affected species')
print(R['species_rule'])

# ---- 4. development sample size ----------------------------------------------------------------------------------
NUM_A = ['age_years', 'admission_to_culture_hours', 'prior_isolate_count', 'days_since_last_isolate', 'percent_probability_num']
CAT_A = ['organism_base', 'specimen_type_clean', 'infection_site_clean', 'gender_clean', 'ward_clean', 'dept_clean', 'onset_bucket', 'id_confidence_clean']
BIN_A = ['is_pediatric_flag', 'prior_resistant_organism_flag', 'id_confidence_known']
nlp = pd.read_parquet('data/processed/nlp_features_llm.parquet').reset_index(drop=True)
NLPC = [c for c in nlp.columns if c not in ('study_id', 'isolate_id', 'testing_date')]
data = pd.concat([sp, nlp[NLPC]], axis=1)
EPS = 1e-12


def nr(y, X):
    X = np.column_stack([np.ones(len(y)), X]); beta = np.zeros(X.shape[1])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ beta))); W = mu * (1 - mu) + EPS
        step = np.linalg.solve((X * W[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]), X.T @ (y - mu)); beta += step
        if np.max(np.abs(step)) < 1e-9: break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    return float(np.sum(y * np.log(mu + EPS) + (1 - y) * np.log(1 - mu + EPS)))


def r2cs_from_c(C, phi, n=200000, seed=7):
    """Riley, Van Calster & Collins (Stat Med 2021): simulate a linear predictor with the given C statistic."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed); mu = np.sqrt(2) * norm.ppf(C); ne = int(n * phi)
    y = np.r_[np.ones(ne), np.zeros(n - ne)]; lp = np.r_[rng.normal(mu, 1, ne), rng.normal(0, 1, n - ne)]
    ll1 = nr(y, lp[:, None]); ll0 = n * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi))
    return 1 - np.exp(2 * (ll0 - ll1) / n)


perf = pd.read_csv('tables/llm/table4_model_performance_full.csv')
best = json.load(open('tables/llm/best_model_per_target.json'))
ss = []
for t in ['MRSA', 'ESBL', 'CRE']:
    d = data[(data[f'in_{t}_target_cohort'] == 1) & data[f'label_{t}'].notna()]; tr = d[d.split_temporal == 'train']; y = tr[f'label_{t}'].astype(int)
    phi = float(y.mean()); ev = int(min(y.sum(), (1 - y).sum()))
    for fs, num, bin_ in [('A', NUM_A, BIN_A), ('B', NUM_A + ['note_length_chars'], BIN_A + [c for c in NLPC if c != 'note_length_chars'])]:
        ct = ColumnTransformer([('num', SimpleImputer(strategy='median', add_indicator=True), num),
                                ('cat', Pipeline([('i', SimpleImputer(strategy='constant', fill_value='Missing')), ('o', OneHotEncoder(handle_unknown='ignore'))]), CAT_A),
                                ('bin', SimpleImputer(strategy='constant', fill_value=0), bin_)])
        p = ct.fit_transform(tr).shape[1]
        C = float(perf[(perf.target == t) & (perf.split == 'split_temporal') & (perf.feature_set == fs) & (perf.model == best[t])].AUROC.iloc[0])
        r2 = r2cs_from_c(C, phi); maxr2 = 1 - np.exp(2 * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi)))
        S = 0.9; n1 = p / ((S - 1) * np.log(1 - r2 / S))
        Svh = r2 / (r2 + 0.05 * maxr2); n2 = p / ((Svh - 1) * np.log(1 - r2 / Svh))
        n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)
        ss.append(dict(target=t, feature_set=fs, n_train=len(tr), events_minority=ev, prevalence=phi, encoded_parameters=p, events_per_parameter=ev / p,
                       C_used=C, R2cs_anticipated=r2, max_R2cs=maxr2, n_required_shrinkage090=n1, n_required_optimism=n2, n_required_intercept=n3,
                       n_required=max(n1, n2, n3), adequate=len(tr) >= max(n1, n2, n3)))
ssd = pd.DataFrame(ss); ssd.to_csv(f'{OUT}/development_sample_size.csv', index=False); R['sample_size'] = ss
print(ssd.round(3).to_string(index=False))

# ---- 5. hashes -------------------------------------------------------------------------------------------------------
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
R['sha256'] = {p: sha(p) for p in ['data/llm/full_corpus_llm_extraction.csv', 'data/llm/full_corpus_key.csv', 'data/processed/nlp_features_llm_pre_datefix.parquet', 'data/processed/nlp_features_llm.parquet']}
for p in ['scripts/05a_llm_extraction.py']:
    if os.path.exists(p): R['sha256'][p] = sha(p)
print(json.dumps(R['sha256'], indent=1))
json.dump(R, open(f'{OUT}/data_quality.json', 'w'), indent=1, default=float)
print('saved')
