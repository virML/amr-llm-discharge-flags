"""Decomposition and sensitivity analyses defined after the primary analysis (Table 5; Additional file 1 Note S6).

Every model is fitted exactly as in 09_modeling_llm_final.py (same preprocessing, hyperparameters, class weighting,
random_state 42, temporal split) and every Model-versus-Model difference uses the paired patient-cluster bootstrap of
10_delta_ci_llm_final.py (1000 resamples of patients, numpy RandomState(123)). The script first checks that it
reproduces the archived primary test-set predictions exactly.

Feature sets (LLM flags = the 13 physician-validated flags):
  A          structured predictors (Model A)
  AM         A + text metadata: note availability, note length and the rule-based 'named antibiotic' indicator
  B          AM + 13 LLM flags (primary Model B)
  A_LLM      A + note availability + 13 LLM flags (no note length, no rule-based field)
  AM_hist    AM + 6 pre-admission history flags (5 comorbidities, prior hospitalization)
  AM_course  AM + 7 hospital-course flags (4 devices, ICU stay, prior antibiotic exposure, recent surgery)
Scenarios:
  S0  primary data: decomposition A -> AM -> B, A -> A_LLM, AM -> AM_hist, AM -> AM_course
  S3  without the admission-to-testing interval (numeric hours and the <48 h / >=48 h category) in A and B
  S4  prospectively available notes only: note features kept only when the linked discharge date is before the
      testing date (all other isolates treated as having no note), in training and test
  S5  all-future temporal test: the 134 post-cutoff isolates of training-period patients added to the test set
  S6  without class weighting (A and B), with Brier score, calibration intercept and slope, and net benefit
  S7  CRE redefined: (a) CDC-like - resistance to meropenem, imipenem (non-Proteeae) or ertapenem (MIC >= 2 mg/L) or
      an AES carbapenemase call; (b) carbapenem MIC only (the 13 AES-only labels set to unlabelled)
  S8  episode definition: (a) 30-day instead of 7-day de-duplication window; (b) first labelled isolate per patient
  S9  admitted isolates only (outpatient isolates excluded from training and test)
  S10 without species-level identification (organism, identification probability and confidence removed)
  S11 note-present versus note-absent test isolates: characteristics, and Model B - Model A difference in each subset
      from the primary models, with the difference between subsets (interaction) from the same paired resamples
  S12 matched operating point: specificity of A and B at exactly 90% test-set sensitivity
  S13 species-frequency rule (>= 20 isolates) recomputed from training-period isolates only
Outputs: tables/reviewer/*.csv and summary_<scenarios>.json; run with scenario names as arguments (e.g. S0 S3) or none for all.
Run from the project root.
"""
import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')
OUT = 'tables/reviewer'
os.makedirs(OUT, exist_ok=True)
RS, NB = 42, 1000
ONLY = sys.argv[1:]  # optional list of scenarios to run
TAG = '_'.join(ONLY) if ONLY else 'all'
FN = f'{OUT}/sensitivity_contrasts_{TAG}.csv'

struct = pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
nlp = pd.read_parquet('data/processed/nlp_features_llm.parquet').reset_index(drop=True)
coh = pd.read_parquet('data/processed/cohort_master.parquet').reset_index(drop=True)
assert (struct.study_id.values == nlp.study_id.values).all() and (coh.study_id.values == struct.study_id.values).all()
NLPC = [c for c in nlp.columns if c not in ('study_id', 'isolate_id', 'testing_date')]
data = pd.concat([struct, nlp[NLPC]], axis=1)

NUM_A = ['age_years', 'admission_to_culture_hours', 'prior_isolate_count', 'days_since_last_isolate', 'percent_probability_num']
CAT_A = ['organism_base', 'specimen_type_clean', 'infection_site_clean', 'gender_clean', 'ward_clean', 'dept_clean', 'onset_bucket', 'id_confidence_clean']
BIN_A = ['is_pediatric_flag', 'prior_resistant_organism_flag', 'id_confidence_known']
B_BIN = [c for c in NLPC if c != 'note_length_chars']           # order exactly as in 09
LLM13 = ['comorbid_diabetes', 'comorbid_ckd_dialysis', 'comorbid_malignancy', 'comorbid_immunosuppression', 'comorbid_cardio_cerebrovascular',
         'prior_antibiotic_any', 'device_urinary_catheter', 'device_central_line', 'device_ventilator_tracheal', 'device_surgical_drain',
         'recent_hospitalization', 'recent_surgery', 'icu_stay']
HIST = ['comorbid_diabetes', 'comorbid_ckd_dialysis', 'comorbid_malignancy', 'comorbid_immunosuppression', 'comorbid_cardio_cerebrovascular', 'recent_hospitalization']
COURSE = ['device_urinary_catheter', 'device_central_line', 'device_ventilator_tracheal', 'device_surgical_drain', 'icu_stay', 'prior_antibiotic_any', 'recent_surgery']
META_BIN = ['prior_antibiotic_named', 'note_available']
assert sorted(B_BIN) == sorted(LLM13 + META_BIN)
FS = {
    'A': (NUM_A, CAT_A, BIN_A),
    'B': (NUM_A + ['note_length_chars'], CAT_A, BIN_A + B_BIN),
    'AM': (NUM_A + ['note_length_chars'], CAT_A, BIN_A + META_BIN),
    'A_LLM': (NUM_A, CAT_A, BIN_A + LLM13 + ['note_available']),
    'AM_hist': (NUM_A + ['note_length_chars'], CAT_A, BIN_A + META_BIN + HIST),
    'AM_course': (NUM_A + ['note_length_chars'], CAT_A, BIN_A + META_BIN + COURSE),
}
TARGETS = {'MRSA': ('in_MRSA_target_cohort', 'label_MRSA'), 'ESBL': ('in_ESBL_target_cohort', 'label_ESBL'), 'CRE': ('in_CRE_target_cohort', 'label_CRE')}
BEST = json.load(open('tables/llm/best_model_per_target.json'))
FAMS = ['logreg', 'rf', 'xgb']


def drop_cols(fs, drop):
    n, c, b = FS[fs]
    return ([x for x in n if x not in drop], [x for x in c if x not in drop], [x for x in b if x not in drop])


def prep(spec, mk):
    num, cat, bin_ = spec
    nump = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True))] + ([('scale', StandardScaler())] if mk == 'logreg' else []))
    catp = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    binp = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0))])
    tr = []
    if num: tr.append(('num', nump, num))
    if cat: tr.append(('cat', catp, cat))
    if bin_: tr.append(('bin', binp, bin_))
    return ColumnTransformer(tr, remainder='drop')


def model(mk, pw, weighted=True):
    if mk == 'logreg':
        return LogisticRegression(max_iter=2000, class_weight='balanced' if weighted else None, C=1.0, random_state=RS)
    if mk == 'rf':
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight='balanced' if weighted else None, n_jobs=-1, random_state=RS)
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=pw if weighted else 1.0, eval_metric='logloss', random_state=RS, n_jobs=-1)


def fit_predict(tr, te, lab, spec, mk, weighted=True):
    pw = (tr[lab] == 0).sum() / max((tr[lab] == 1).sum(), 1)
    p = Pipeline([('prep', prep(spec, mk)), ('clf', model(mk, pw, weighted))])
    p.fit(tr, tr[lab])
    return p.predict_proba(te)[:, 1]


def groups_of(pid):
    g = {}
    for i, p in enumerate(pid): g.setdefault(p, []).append(i)
    return g, np.array(list(g.keys()), dtype=object)


def paired(y, pa, pb, pid):
    g, keys = groups_of(pid); rng = np.random.RandomState(123); da, dp = [], []
    for _ in range(NB):
        idx = np.concatenate([g[k] for k in keys[rng.randint(0, len(keys), len(keys))]])
        if len(np.unique(y[idx])) < 2: continue
        da.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]))
        dp.append(average_precision_score(y[idx], pb[idx]) - average_precision_score(y[idx], pa[idx]))
    return dict(dAUROC=roc_auc_score(y, pb) - roc_auc_score(y, pa), dAUROC_lo=np.percentile(da, 2.5), dAUROC_hi=np.percentile(da, 97.5),
                dAUPRC=average_precision_score(y, pb) - average_precision_score(y, pa), dAUPRC_lo=np.percentile(dp, 2.5), dAUPRC_hi=np.percentile(dp, 97.5))


def target_frame(df, t):
    f, lab = TARGETS[t]
    d = df[df[f] & df[lab].notna()].copy(); d[lab] = d[lab].astype(int)
    return d, lab


ROWS = []


def contrast(scn, t, mk, a_name, b_name, y, pa, pb, pid, n_train, extra=None):
    r = dict(scenario=scn, target=t, model=mk, headline=(BEST[t] == mk), from_set=a_name, to_set=b_name, n_train=n_train, n_test=len(y),
             events_test=int(y.sum()), AUROC_from=roc_auc_score(y, pa), AUROC_to=roc_auc_score(y, pb),
             AUPRC_from=average_precision_score(y, pa), AUPRC_to=average_precision_score(y, pb))
    r.update(paired(y, pa, pb, pid))
    if extra: r.update(extra)
    ROWS.append(r)
    print(f"  {scn:14s} {t:4s} {mk:6s} {a_name:>12s} -> {b_name:<12s} AUROC {r['AUROC_from']:.4f} -> {r['AUROC_to']:.4f}  d={r['dAUROC']:+.4f} ({r['dAUROC_lo']:+.4f},{r['dAUROC_hi']:+.4f})", flush=True)
    return r


def run_pairs(scn, df, pairs, targets=('MRSA', 'ESBL', 'CRE'), fams=FAMS, specs=None, weighted=True, test_mask=None, train_mask=None, keep=None):
    """fit every feature set needed by pairs on train, evaluate on test; returns dict of predictions."""
    specs = specs or {}
    store = {}
    for t in targets:
        d, lab = target_frame(df, t)
        tr = d[d.split_temporal == 'train'] if train_mask is None else d[train_mask(d)]
        te = d[d.split_temporal == 'test'] if test_mask is None else d[test_mask(d)]
        y = te[lab].values; pid = te.study_id.values
        for mk in fams:
            preds = {}
            for fs in sorted({x for p in pairs for x in p}):
                preds[fs] = fit_predict(tr, te, lab, specs.get(fs, FS.get(fs)), mk, weighted)
            for a, b in pairs:
                contrast(scn, t, mk, a, b, y, preds[a], preds[b], pid, len(tr))
            store[(t, mk)] = dict(y=y, pid=pid, te=te, preds=preds, lab=lab)
    return store


def want(s): return not ONLY or s in ONLY


summary = {}
t0 = time.time()
# ===================================================================================================================
# S0: replication + decomposition
# ===================================================================================================================
if want('S0'):
    print('S0 decomposition', flush=True)
    pr = json.load(open('data/processed/model_outputs/llm/predictions.json'))
    st = run_pairs('S0_decomposition', data, [('A', 'B'), ('A', 'AM'), ('AM', 'B'), ('A', 'A_LLM'), ('AM', 'AM_hist'), ('AM', 'AM_course')])
    rep = {}
    for (t, mk), v in st.items():
        for fs in ['A', 'B']:
            ref = np.array(pr[f'{t}__split_temporal__{fs}__{mk}']['y_proba'])
            rep[f'{t}_{fs}_{mk}'] = float(np.max(np.abs(ref - v['preds'][fs])))
    print('max abs difference from archived predictions:', rep, flush=True)
    assert max(rep.values()) < 1e-6, 'primary predictions not reproduced'
    summary['replication_max_abs_diff'] = rep
    pd.DataFrame(ROWS).to_csv(FN, index=False)

# ===================================================================================================================
# S3: without admission-to-testing interval
# ===================================================================================================================
if want('S3'):
    print('S3 no interval', flush=True)
    drop = ['admission_to_culture_hours', 'onset_bucket']
    sp = {'A_noint': drop_cols('A', drop), 'B_noint': drop_cols('B', drop)}
    st = run_pairs('S3_no_interval', data, [('A_noint', 'B_noint'), ('A', 'A_noint')], specs=sp)
    pd.DataFrame(ROWS).to_csv(FN, index=False)

# ===================================================================================================================
# S4: prospectively available notes only
# ===================================================================================================================
def parse(x):
    a = pd.to_datetime(x, format='%d/%m/%Y %H:%M', errors='coerce'); b = pd.to_datetime(x, format='%d/%m/%y %H:%M', errors='coerce')
    return a.fillna(b)


disc = parse(coh['txtDiscDt']); td = pd.to_datetime(coh['testing_date']).dt.normalize()
prospective = (data['note_available'] == 1) & disc.notna() & (disc.dt.normalize() < td)
if want('S4'):
    print('S4 prospective notes', flush=True)
    dp = data.copy()
    nonp = (dp['note_available'] == 1) & ~prospective
    dp.loc[nonp, B_BIN] = 0; dp.loc[nonp, 'note_length_chars'] = 0
    facts = dict(n_prospective_notes=int(prospective.sum()), n_note_features=int((data.note_available == 1).sum()))
    for t in TARGETS:
        d, lab = target_frame(data, t)
        for s in ['train', 'test']:
            m = d.split_temporal == s
            facts[f'{t}_{s}_prospective'] = int(prospective.loc[d.index[m]].sum()); facts[f'{t}_{s}_n'] = int(m.sum())
    summary['S4_facts'] = facts; print(facts, flush=True)
    run_pairs('S4_prospective_notes', dp, [('A', 'B'), ('AM', 'B'), ('A', 'A_LLM')])
    pd.DataFrame(ROWS).to_csv(FN, index=False)

# ===================================================================================================================
# S5: all future isolates (returning patients included)
# ===================================================================================================================
if want('S5'):
    print('S5 all-future test', flush=True)
    st = run_pairs('S5_all_future', data, [('A', 'B')], test_mask=lambda d: d.split_temporal.isin(['test', 'excluded_straddling']))
    f5 = {}
    for (t, mk), v in st.items():
        ret = (v['te'].split_temporal == 'excluded_straddling').values
        f5[f'{t}_added'] = int(ret.sum()); f5[f'{t}_added_events'] = int(v['y'][ret].sum()); f5[f'{t}_added_patients'] = int(len(set(v['pid'][ret])))
        if mk == BEST[t] and ret.sum() >= 10 and len(np.unique(v['y'][ret])) == 2:
            f5[f'{t}_returning_only_AUROC_A'] = roc_auc_score(v['y'][ret], v['preds']['A'][ret]); f5[f'{t}_returning_only_AUROC_B'] = roc_auc_score(v['y'][ret], v['preds']['B'][ret])
    summary['S5_facts'] = f5; print(f5, flush=True)
    pd.DataFrame(ROWS).to_csv(FN, index=False)

# ===================================================================================================================
# S6: no class weighting
# ===================================================================================================================
EPS = 1e-6


def logit(p): p = np.clip(np.asarray(p, float), EPS, 1 - EPS); return np.log(p / (1 - p))


def nr(y, x, offset=False):
    y = np.asarray(y, float); X = np.ones((len(y), 1)) if offset else np.column_stack([np.ones(len(y)), x]); off = x if offset else 0.0
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta + off; mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu) + 1e-12
        step = np.linalg.solve((X * W[:, None]).T @ X, X.T @ (y - mu)); beta += step
        if np.max(np.abs(step)) < 1e-10: break
    return beta


def nb(y, p, t): n = len(y); return np.sum((p >= t) & (y == 1)) / n - np.sum((p >= t) & (y == 0)) / n * t / (1 - t)


if want('S6'):
    print('S6 unweighted', flush=True)
    st = run_pairs('S6_unweighted', data, [('A', 'B')], weighted=False)
    cal = []
    for (t, mk), v in st.items():
        y = v['y']
        for fs in ['A', 'B']:
            p = v['preds'][fs]; lp = logit(p)
            cal.append(dict(target=t, model=mk, feature_set=fs, weighting='none', AUROC=roc_auc_score(y, p), Brier=brier_score_loss(y, p),
                            cal_intercept=nr(y, lp, True)[0], cal_slope=nr(y, lp)[1], mean_pred=float(p.mean()), prevalence=float(y.mean()),
                            nb_10=nb(y, p, .10), nb_20=nb(y, p, .20), nb_30=nb(y, p, .30)))
        g, keys = groups_of(v['pid']); rng = np.random.RandomState(42); dd = []
        for _ in range(NB):
            idx = np.concatenate([g[k] for k in keys[rng.randint(0, len(keys), len(keys))]])
            dd.append(nb(y[idx], v['preds']['B'][idx], .20) - nb(y[idx], v['preds']['A'][idx], .20))
        cal[-1].update(dNB20_B_minus_A=nb(y, v['preds']['B'], .2) - nb(y, v['preds']['A'], .2), dNB20_lo=np.percentile(dd, 2.5), dNB20_hi=np.percentile(dd, 97.5))
    pd.DataFrame(cal).to_csv(f'{OUT}/unweighted_calibration.csv', index=False)
    pd.DataFrame(ROWS).to_csv(FN, index=False)

# ===================================================================================================================
# S7: CRE definitions
# ===================================================================================================================
if want('S7'):
    print('S7 CRE definitions', flush=True)

    def mic(s):
        s = s.astype(str).str.strip().str.replace('<=', '', regex=False).str.replace('>=', '', regex=False).str.replace('<', '', regex=False).str.replace('>', '', regex=False)
        return pd.to_numeric(s, errors='coerce')
    ent = data.organism_group == 'Enterobacterales'
    mem, ipm, etp = mic(coh['MEM-Meropenem']), mic(coh['IPM-Imipenem']), mic(coh['ETP-Ertapenem'])
    proteeae = coh['organism_base'].fillna('').str.startswith(('Proteus', 'Prov.', 'Morg.'))
    ipm = ipm.where(~proteeae)
    bl = [c for c in coh.columns if c.startswith('Family -BETA-LACTAMS')]
    aes = coh[bl].isin(['CARBAPENEMASE (+ OR - ESBL)', 'CARBAPENEMASE']).any(axis=1).astype(float).where(coh[bl].notna().any(axis=1))
    anyR = (mem >= 4) | (ipm >= 4) | (etp >= 2) | (aes == 1)
    anyTested = mem.notna() | ipm.notna() | etp.notna() | aes.notna()
    cdc = pd.Series(np.where(anyTested, anyR.astype(float), np.nan), index=data.index).where(ent)
    memx = (mem >= 4).astype(float).where(mem.notna()); ipmx = (ipm >= 4).astype(float).where(ipm.notna())
    pheno = memx.where(memx.notna(), ipmx).where(ent)
    assert ((pheno.where(pheno.notna(), aes.where(ent)) == data.label_CRE) | data.label_CRE.isna()).all()
    f7 = dict(primary_pos=int((data.label_CRE == 1).sum()), primary_n=int(data.label_CRE.notna().sum()),
              cdc_pos=int((cdc == 1).sum()), cdc_n=int(cdc.notna().sum()), pheno_pos=int((pheno == 1).sum()), pheno_n=int(pheno.notna().sum()),
              ertapenem_tested=int((etp.notna() & ent).sum()), ertapenem_R=int(((etp >= 2) & ent).sum()),
              ertapenem_R_but_primary_negative=int(((etp >= 2) & (data.label_CRE == 0)).sum()),
              aes_only_labels=int((data.label_CRE.notna() & pheno.isna() & ent).sum()),
              aes_carbapenemase_but_primary_negative=int(((aes == 1) & (data.label_CRE == 0)).sum()))
    for nm, lab in [('cdc', cdc), ('pheno', pheno)]:
        m = data.split_temporal == 'test'
        f7[f'{nm}_test_pos'] = int(((lab == 1) & m).sum()); f7[f'{nm}_test_n'] = int((lab.notna() & m).sum())
    summary['S7_facts'] = f7; print(f7, flush=True)
    for nm, lab in [('S7a_CRE_CDC_like', cdc), ('S7b_CRE_MIC_only', pheno)]:
        d7 = data.copy(); d7['label_CRE'] = lab
        run_pairs(nm, d7, [('A', 'B')], targets=('CRE',))
    pd.DataFrame(ROWS).to_csv(FN, index=False)

# ===================================================================================================================
# S8: episode definition
# ===================================================================================================================
if want('S8'):
    print('S8 episode definition', flush=True)
    g = pd.DataFrame({'sid': coh.study_id, 'org': coh.organism_base, 'spec': coh['Specimen Type'], 'td': pd.to_datetime(coh.testing_date)})
    g = g.sort_values(['sid', 'org', 'spec', 'td'])
    keep = pd.Series(True, index=g.index); last = {}
    for i, r in zip(g.index, g.itertuples(index=False)):
        k = (r.sid, r.org, r.spec)
        if k in last and (r.td - last[k]).days < 30: keep[i] = False
        else: last[k] = r.td
    keep = keep.reindex(data.index)
    d8 = data[keep.values].copy()
    summary['S8_facts'] = dict(removed_30d=int((~keep).sum()), remaining=int(keep.sum()))
    print(summary['S8_facts'], flush=True)
    run_pairs('S8a_30day_episodes', d8, [('A', 'B')])
    # first labelled isolate per patient within each target cohort
    for t in TARGETS:
        d, lab = target_frame(data, t)
        d = d.assign(_td=pd.to_datetime(d.testing_date)).sort_values(['study_id', '_td'])
        first = d.groupby('study_id').head(1).index
        d1 = data.loc[first]
        summary.setdefault('S8_facts', {})[f'{t}_first_isolate_n'] = int(len(first))
        run_pairs('S8b_first_isolate', d1, [('A', 'B')], targets=(t,))
    pd.DataFrame(ROWS).to_csv(FN, index=False)

# ===================================================================================================================
# S9: admitted isolates only;  S10: without species-level identification
# ===================================================================================================================
outpatient = (struct.ward_clean == 'Unknown') & struct.admission_to_culture_hours.isna()
if want('S9'):
    print('S9 admitted only', flush=True)
    run_pairs('S9_admitted_only', data[~outpatient.values], [('A', 'B')])
    pd.DataFrame(ROWS).to_csv(FN, index=False)
if want('S10'):
    print('S10 no species identity', flush=True)
    drop = ['organism_base', 'percent_probability_num', 'id_confidence_clean', 'id_confidence_known']
    sp = {'A_noid': drop_cols('A', drop), 'B_noid': drop_cols('B', drop)}
    run_pairs('S10_no_species_id', data, [('A_noid', 'B_noid'), ('A', 'A_noid')], specs=sp)
    pd.DataFrame(ROWS).to_csv(FN, index=False)

# ===================================================================================================================
# S13: species-frequency rule applied to training-period isolates only
# ===================================================================================================================
if want('S13'):
    print('S13 species rule from training period', flush=True)
    tdt = pd.to_datetime(coh.testing_date)
    cnt_tr = coh.loc[tdt < pd.Timestamp('2025-06-28'), 'Organism Name'].value_counts()
    low = [o for o in coh['Organism Name'].unique() if cnt_tr.get(o, 0) < 20]
    keep13 = ~coh['Organism Name'].isin(low)
    summary['S13_facts'] = dict(species_excluded=len(low), isolates_excluded=int((~keep13).sum()))
    run_pairs('S13_species_rule_training', data[keep13.values], [('A', 'B')])
    pd.DataFrame(ROWS).to_csv(FN, index=False)

# ===================================================================================================================
# S11: note-present versus note-absent (primary predictions); S12: matched operating points
# ===================================================================================================================
if want('S11') or want('S12'):
    pr = json.load(open('data/processed/model_outputs/llm/predictions.json'))
    char_rows, inter_rows, op_rows = [], [], []
    for t in TARGETS:
        d, lab = target_frame(data, t); te = d[d.split_temporal == 'test']
        for mk in FAMS:
            ka, kb = f'{t}__split_temporal__A__{mk}', f'{t}__split_temporal__B__{mk}'
            y = np.array(pr[ka]['y_test']); pa = np.array(pr[ka]['y_proba']); pb = np.array(pr[kb]['y_proba']); pid = np.array(pr[ka]['study_id'])
            assert np.array_equal(y, te[lab].values) and np.array_equal(pid, te.study_id.values)
            note = (te.note_available == 1).values
            # interaction: d(B-A) in note-present minus d(B-A) in note-absent, same patient resamples
            g, keys = groups_of(pid); rng = np.random.RandomState(123); D1, D0, DI = [], [], []
            def dd(idx): return roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx])
            for _ in range(NB):
                idx = np.concatenate([g[k] for k in keys[rng.randint(0, len(keys), len(keys))]])
                i1, i0 = idx[note[idx]], idx[~note[idx]]
                if len(np.unique(y[i1])) < 2 or len(np.unique(y[i0])) < 2: continue
                a1, a0 = dd(i1), dd(i0); D1.append(a1); D0.append(a0); DI.append(a1 - a0)
            o1, o0 = dd(np.where(note)[0]), dd(np.where(~note)[0])
            q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            inter_rows.append(dict(target=t, model=mk, headline=BEST[t] == mk, n_note=int(note.sum()), n_no_note=int((~note).sum()),
                                   prev_note=float(y[note].mean()), prev_no_note=float(y[~note].mean()),
                                   AUROC_A_note=roc_auc_score(y[note], pa[note]), AUROC_B_note=roc_auc_score(y[note], pb[note]),
                                   AUROC_A_no_note=roc_auc_score(y[~note], pa[~note]), AUROC_B_no_note=roc_auc_score(y[~note], pb[~note]),
                                   d_note=o1, d_note_lo=q(D1)[0], d_note_hi=q(D1)[1], d_no_note=o0, d_no_note_lo=q(D0)[0], d_no_note_hi=q(D0)[1],
                                   interaction=o1 - o0, interaction_lo=q(DI)[0], interaction_hi=q(DI)[1]))
            print('  S11', inter_rows[-1], flush=True)
            if mk == BEST[t]:
                for grp, m in [('note present', note), ('note absent', ~note)]:
                    s = te[m]; outp = outpatient.loc[s.index]
                    char_rows.append(dict(target=t, group=grp, n=int(m.sum()), patients=int(s.study_id.nunique()), resistant_pct=100 * float(y[m].mean()),
                                          outpatient_pct=100 * float(outp.mean()), age_recorded_pct=100 * float(s.age_years.notna().mean()),
                                          age_median=float(s.age_years.median()), female_pct=100 * float((s.gender_clean == 'Female').mean()),
                                          hospital_onset_pct=100 * float((s.admission_to_culture_hours >= 48).mean()),
                                          icu_ward_pct=100 * float(s.ward_clean.str.contains('ICU', case=False, na=False).mean()),
                                          urine_pct=100 * float((s.specimen_type_clean == 'URINE').mean()), blood_pct=100 * float((s.specimen_type_clean == 'BLOOD').mean()),
                                          prior_isolate_pct=100 * float((s.prior_isolate_count.fillna(0) > 0).mean()),
                                          prior_resistant_pct=100 * float((s.prior_resistant_organism_flag == 1).mean())))
                # S12: specificity at exactly 90% test-set sensitivity, and at common thresholds
                def spec_at(yy, pp, sens=0.90):
                    fpr, tpr, th = roc_curve(yy, pp); i = np.where(tpr >= sens)[0][0]; return 1 - fpr[i], tpr[i], th[i]
                sa, ta, tha = spec_at(y, pa); sb, tb, thb = spec_at(y, pb)
                rng = np.random.RandomState(123); DS = []
                for _ in range(NB):
                    idx = np.concatenate([g[k] for k in keys[rng.randint(0, len(keys), len(keys))]])
                    if len(np.unique(y[idx])) < 2: continue
                    DS.append(spec_at(y[idx], pb[idx])[0] - spec_at(y[idx], pa[idx])[0])
                op_rows.append(dict(target=t, model=mk, operating_point='90% test-set sensitivity', sens_A=ta, spec_A=sa, sens_B=tb, spec_B=sb,
                                    fp_A=int(round((1 - sa) * (y == 0).sum())), fp_B=int(round((1 - sb) * (y == 0).sum())),
                                    d_spec=sb - sa, d_spec_lo=q(DS)[0], d_spec_hi=q(DS)[1]))
                print('  S12', op_rows[-1], flush=True)
    pd.DataFrame(char_rows).to_csv(f'{OUT}/note_present_vs_absent_characteristics.csv', index=False)
    pd.DataFrame(inter_rows).to_csv(f'{OUT}/note_present_interaction.csv', index=False)
    pd.DataFrame(op_rows).to_csv(f'{OUT}/matched_operating_point.csv', index=False)

json.dump(summary, open(f'{OUT}/summary_{TAG}.json', 'w'), indent=1, default=float)
if ROWS:
    pd.DataFrame(ROWS).to_csv(FN, index=False)
print(f'done in {time.time() - t0:.0f} s')
