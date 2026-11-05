"""Extraction validation in the 120-note probability (base) sample.

Every metric for a flag comes from ONE confusion matrix of method output against the physician reference standard in
the 120 department-stratified random notes:
    precision = TP/(TP+FP), recall = TP/(TP+FN), specificity = TN/(TN+FP), F1 = 2TP/(2TP+FP+FN), Cohen's kappa.
Pooled (micro) metrics stack the note x flag judgments across all 13 flags (1560 judgments).
95% CIs: bootstrap resampling notes within department strata (the sampling design), 2000 reps, seed 2026; paired
LLM - rule-based differences from the same resamples.
Design-weighted sensitivity analysis: each base note weighted by N_frame(dept)/n_sample(dept) (frame = notes >50
characters, as in 06_clinician_validation_sample.py), undoing the 25% department cap.
The 53 enrichment notes (drawn because the RULE-BASED pipeline was positive for a rare flag) are NOT used for any
metric above; they are reported separately as error analysis (precision among method-positive enrichment notes).
Also: inter-rater agreement (pooled over the 13 flags), each reviewer as the reference standard on the 24 doubly reviewed notes,
and an integrity check that the method outputs in the sample key equal the extraction outputs used for modelling.
Run from the project root. Writes tables/validation_v2/.
"""
import json
import os
import numpy as np
import openpyxl
import pandas as pd

FL = ['comorbid_diabetes', 'comorbid_ckd_dialysis', 'comorbid_malignancy', 'comorbid_immunosuppression',
      'comorbid_cardio_cerebrovascular', 'prior_antibiotic_any', 'device_urinary_catheter', 'device_central_line',
      'device_ventilator_tracheal', 'device_surgical_drain', 'icu_stay', 'recent_hospitalization', 'recent_surgery']
KEY = 'data/deid/clinician_validation_sample_key_RESTRICTED.csv'
FINAL = 'paper/clinician_validation_review_sheet_completed.xlsx'  # primary reviewer and second-reviewer sheets
OUT = 'tables/validation_v2'
os.makedirs(OUT, exist_ok=True)
B = 2000

key = pd.read_csv(KEY).set_index('review_id')


def load(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True)); h = rows[0]
    jc = [i for i, x in enumerate(h) if x and str(x).startswith('YOUR JUDGMENT')]
    assert len(jc) == 13
    d, vals = {}, set()
    for r in rows[1:]:
        if r and r[0]:
            v = [(r[i] if i < len(r) else None) for i in jc]; vals |= set(v)
            d[r[0]] = [1 if x == 'Present' else 0 for x in v]
    assert vals <= {'Present', 'Absent', None}, vals
    return pd.DataFrame.from_dict(d, orient='index', columns=FL)


J = load(FINAL, 'Review Sheet')            # reference standard (primary reviewer)
J2 = load(FINAL, 'Second Reviewer Subset')  # second reviewer
assert set(J.index) == set(key.index)
base = np.array(key.index[key.sampling_stratum == 'base_random'], dtype=object)
enr = np.array(key.index[key.sampling_stratum == 'enrichment'], dtype=object)
assert len(base) == 120 and len(enr) == 53 and set(J2.index) <= set(base) and len(J2) == 24
dept = key.loc[base, 'dept_clean']

# ---- integrity: key outputs equal the modelling extraction outputs ------------------------------------------------
c = pd.read_parquet('data/processed/cohort_master.parquet', columns=['isolate_id', 'drpDept', 'DISCHARGE_SUMMARY_CONTENT_DEID'])
llm_pre = pd.read_parquet('data/processed/nlp_features_llm_pre_datefix.parquet')
rb_pre = pd.read_parquet('data/processed/nlp_features_pre_datefix.parquet')
integrity = {}
rix = key['orig_row_index'].values  # positional row in cohort_master (isolate_id is not unique across organisms)
assert (c['isolate_id'].values[rix] == key['isolate_id'].values).all()
for m, src in [('llm', llm_pre), ('rb', rb_pre)]:
    s = src[FL].values[rix]
    integrity[m] = int((s != key[[f'{m}__{f}' for f in FL]].values).sum())
txt = pd.Series(c['DISCHARGE_SUMMARY_CONTENT_DEID'].values[rix], index=key.index)
integrity['distinct_documents_all173'] = int(txt.nunique())
integrity['distinct_documents_base120'] = int(txt.loc[base].nunique())
integrity['distinct_patients_base120'] = int(key.loc[base, 'study_id'].nunique())
print('integrity (mismatching key cells; must be 0):', integrity)
assert integrity['llm'] == 0 and integrity['rb'] == 0

# ---- sampling frame and design weights ------------------------------------------------------------------------------
c['dept_clean'] = c['drpDept'].fillna('Unknown').replace('--Select--', 'Unknown')
frame = c[c['DISCHARGE_SUMMARY_CONTENT_DEID'].notna() & (c['DISCHARGE_SUMMARY_CONTENT_DEID'].str.len() > 50)]
Nd = frame['dept_clean'].value_counts(); nd = dept.value_counts()
assert len(frame) == 12930 and set(nd.index) <= set(Nd.index)
w_dept = (Nd.loc[nd.index] / nd).to_dict()
alloc = pd.DataFrame({'frame_notes': Nd.loc[nd.index], 'sampled': nd, 'weight': pd.Series(w_dept)}).sort_values('frame_notes', ascending=False)
alloc['frame_share_pct'] = (100 * alloc['frame_notes'] / len(frame)).round(1)
alloc['sample_share_pct'] = (100 * alloc['sampled'] / 120).round(1)
alloc.to_csv(f'{OUT}/base_sample_allocation_and_weights.csv')
print(alloc)
W = dept.map(w_dept).astype(float)


# ---- metric helpers -------------------------------------------------------------------------------------------------
def cm(y, p, w=None):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    y = np.asarray(y); p = np.asarray(p)
    return dict(TP=float(w[(y == 1) & (p == 1)].sum()), FP=float(w[(y == 0) & (p == 1)].sum()),
                FN=float(w[(y == 1) & (p == 0)].sum()), TN=float(w[(y == 0) & (p == 0)].sum()))


def from_cm(d):
    TP, FP, FN, TN = d['TP'], d['FP'], d['FN'], d['TN']; n = TP + FP + FN + TN
    P = TP / (TP + FP) if TP + FP else np.nan
    R = TP / (TP + FN) if TP + FN else np.nan
    S = TN / (TN + FP) if TN + FP else np.nan
    F = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else np.nan
    po = (TP + TN) / n; py = (TP + FN) / n; pp = (TP + FP) / n; pe = py * pp + (1 - py) * (1 - pp)
    K = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return dict(precision=P, recall=R, specificity=S, F1=F, kappa=K)


MET = ['precision', 'recall', 'specificity', 'F1', 'kappa']


def evaluate(ref, ids, w=None):
    """per-flag and pooled metrics for both methods on note ids (ids may repeat in bootstrap)."""
    out = {}
    for m in ['llm', 'rb']:
        per = {}; tot13 = dict(TP=0, FP=0, FN=0, TN=0)
        for f in FL:
            d = cm(ref.loc[ids, f].values, key.loc[ids, f'{m}__{f}'].values, w)
            per[f] = (d, from_cm(d))
            for k in d:
                tot13[k] += d[k]
        per['POOLED_13'] = (tot13, from_cm(tot13))
        out[m] = per
    return out


def strat_resample(rng):
    return np.concatenate([rng.choice(np.array(dept.index[dept == d], dtype=object), n, replace=True) for d, n in nd.items()])


def boot(ref, weighted=False, seed=2026):
    rng = np.random.default_rng(seed)
    res = {m: {f: {k: [] for k in MET} for f in FL + ['POOLED_13']} for m in ['llm', 'rb']}
    dif = {g: {k: [] for k in MET} for g in ['POOLED_13']}
    for _ in range(B):
        ids = strat_resample(rng)
        e = evaluate(ref, ids, W.loc[ids].values if weighted else None)
        for m in e:
            for f in e[m]:
                for k in MET: res[m][f][k].append(e[m][f][1][k])
        for g in dif:
            for k in MET: dif[g][k].append(e['llm'][g][1][k] - e['rb'][g][1][k])
    return res, dif


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]


def table(ref, label, weighted=False, seed=2026):
    pt = evaluate(ref, base, W.loc[base].values if weighted else None)
    bs, dif = boot(ref, weighted, seed)
    rows = []
    for m in ['llm', 'rb']:
        for f in FL + ['POOLED_13']:
            d, mt = pt[m][f]
            r = dict(analysis=label, method=m, flag=f, **{k: round(v, 3) if weighted else int(v) for k, v in d.items()})
            r['method_pos'] = r['TP'] + r['FP']; r['ref_pos'] = r['TP'] + r['FN']
            for k in MET:
                r[k] = mt[k]; lo, hi = ci(bs[m][f][k]); r[k + '_lo'] = lo; r[k + '_hi'] = hi
            rows.append(r)
    diffs = {g: {k: dict(delta=pt['llm'][g][1][k] - pt['rb'][g][1][k], lo=ci(dif[g][k])[0], hi=ci(dif[g][k])[1])
                 for k in MET} for g in dif}
    return pd.DataFrame(rows), diffs


prim, prim_d = table(J, 'final_reference_unweighted')
wtd, wtd_d = table(J, 'final_reference_design_weighted', weighted=True)
allm = pd.concat([prim, wtd])
allm.to_csv(f'{OUT}/validation_metrics_base120.csv', index=False)


def kb(df, m):
    ks = df[(df.method == m) & df.flag.isin(FL)].set_index('flag')['kappa']
    return {'>=0.80': int((ks >= 0.8).sum()), '0.60-0.79': int(((ks >= 0.6) & (ks < 0.8)).sum()),
            '0.40-0.59': int(((ks >= 0.4) & (ks < 0.6)).sum()), '<0.40': int((ks < 0.4).sum())}


# ---- enrichment notes: error analysis only ------------------------------------------------------------------------
EF = {i: set(str(key.loc[i, 'enrichment_for']).split(',')) for i in enr}
enr_rows = []
for f in FL:
    own = np.array([i for i in enr if f in EF[i]], dtype=object)
    r = dict(flag=f, enrichment_notes_drawn_for_flag=len(own))
    for m in ['llm', 'rb']:
        pos = own[key.loc[own, f'{m}__{f}'].values == 1] if len(own) else own
        r[f'{m}_positive'] = int(len(pos)); r[f'{m}_true_positive'] = int(J.loc[pos, f].sum()) if len(pos) else 0
        r[f'{m}_precision'] = (r[f'{m}_true_positive'] / len(pos)) if len(pos) else None
    r['reference_positive'] = int(J.loc[own, f].sum()) if len(own) else 0
    r['llm_recall_among_reference_positive'] = (int(((J.loc[own, f] == 1) & (key.loc[own, f'llm__{f}'] == 1)).sum()) / r['reference_positive']) if r['reference_positive'] else None
    if len(own): enr_rows.append(r)
enr_df = pd.DataFrame(enr_rows); enr_df.to_csv(f'{OUT}/enrichment_error_analysis.csv', index=False)
print(enr_df.to_string(index=False))

# ---- inter-rater agreement ------------------------------------------------------------------------------------
def kap(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel(); po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


ids2 = np.array(J2.index, dtype=object)
irr = {}
for lab, R1 in [('final', J)]:
    for g, fl in [('13', FL)]:
        a = R1.loc[ids2, fl].values; b = J2.loc[ids2, fl].values
        rng = np.random.default_rng(2027); bs = []
        for _ in range(B):
            s = rng.choice(ids2, len(ids2), replace=True); bs.append(kap(R1.loc[s, fl].values, J2.loc[s, fl].values))
        irr[f'{lab}_{g}'] = dict(kappa=float(kap(a, b)), ci=ci(bs), pct_agreement=float(100 * (a == b).mean()), n_judgments=int(a.size))
irr_flag = []
for f in FL:
    a1, b = J.loc[ids2, f].values, J2.loc[ids2, f].values
    irr_flag.append(dict(flag=f, r1_final_pos=int(a1.sum()), r2_pos=int(b.sum()), agree_final_pct=round(100 * float((a1 == b).mean()), 1),
                         kappa_final=(None if a1.sum() + b.sum() == 0 else float(kap(a1, b)))))
pd.DataFrame(irr_flag).to_csv(f'{OUT}/interrater_by_flag.csv', index=False)

# second reviewer as the reference standard on the 24-note subset (robustness of the method comparison)
r2ref = {}
for m in ['llm', 'rb']:
    for g, fl in [('13', FL)]:
        y = J2.loc[ids2, fl].values.ravel(); p = key.loc[ids2, [f'{m}__{f}' for f in fl]].values.ravel()
        r2ref[f'{m}_{g}'] = from_cm(cm(y, p))
# primary reviewer as the reference standard on the same 24 notes (like-for-like comparator for r2ref)
r1ref24 = {}
for m in ['llm', 'rb']:
    for g, fl in [('13', FL)]:
        y = J.loc[ids2, fl].values.ravel(); p = key.loc[ids2, [f'{m}__{f}' for f in fl]].values.ravel()
        r1ref24[f'{m}_{g}'] = from_cm(cm(y, p))


# ---- summary ---------------------------------------------------------------------------------------------------------
def pooled(df, g):
    return {m: {k: dict(est=float(df[(df.method == m) & (df.flag == g)][k].iloc[0]), lo=float(df[(df.method == m) & (df.flag == g)][k + '_lo'].iloc[0]),
                        hi=float(df[(df.method == m) & (df.flag == g)][k + '_hi'].iloc[0])) for k in MET} for m in ['llm', 'rb']}


ref_pos = {f: int(J.loc[base, f].sum()) for f in FL}
method_pos = {m: {f: int(key.loc[base, f'{m}__{f}'].sum()) for f in FL} for m in ['llm', 'rb']}
summary = dict(
    design=dict(n_base=120, n_enrichment=53, n_second_reviewer=24, judgments_13=1560, bootstrap=B, seed=2026),
    primary=dict(pooled_13=pooled(prim, 'POOLED_13'), llm_minus_rb=prim_d,
                 kappa_bands={m: kb(prim, m) for m in ['llm', 'rb']}),
    design_weighted=dict(pooled_13=pooled(wtd, 'POOLED_13'), llm_minus_rb=wtd_d),
    reference_positives_base120=ref_pos, method_positives_base120=method_pos,
    interrater=irr, second_reviewer_as_reference_24notes=r2ref, primary_reviewer_as_reference_24notes=r1ref24, integrity=integrity,
    macro_mean_13={m: {k: float(prim[(prim.method == m) & prim.flag.isin(FL)][k].mean()) for k in MET} for m in ['llm', 'rb']})
json.dump(summary, open(f'{OUT}/validation_summary.json', 'w'), indent=1, default=float)

pd.set_option('display.width', 250)
show = ['method', 'flag', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'specificity', 'F1', 'kappa', 'kappa_lo', 'kappa_hi']
print(prim[show].round(3).to_string(index=False))
print(json.dumps({k: summary[k] for k in ['primary', 'interrater']}, indent=1, default=float)[:6000])
print('weighted:', json.dumps(summary['design_weighted']['pooled_13'], default=float))
print('r2 as reference:', json.dumps(r2ref, default=float))
print('r1 as reference (same 24 notes):', json.dumps(r1ref24, default=float))
