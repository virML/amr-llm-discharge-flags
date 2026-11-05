"""
17_sensitivity_analyses.py
Four targeted robustness/sensitivity analyses, defined after the primary
results and each addressing a specific, falsifiable concern about them:

  (A) Note-present sensitivity analysis. The primary comparison
      (Model A vs Model B) is run on the full modeling cohort, in which
      only 65.9% of isolates have a discharge-summary note at all -- so a
      null result there could reflect incomplete text coverage rather than
      an absence of signal in the text itself. Here we restrict to the
      12,930 isolates that DO have a note and repeat the identical
      Model A-vs-B procedure (same headline model family per target, same
      inner-train/validation threshold selection, same patient-cluster
      bootstrap) on that restricted population, to ask the sharper
      question: when clinical text actually exists, does information
      extracted from it add predictive value?

  (B) ESBL label-definition sensitivity analysis. The primary ESBL label
      falls back to a cephalosporin-resistance cross-check for isolates
      with no direct VITEK AES ESBL phenotype call (Methods); cephalosporin
      resistance is sensitive but not specific for ESBL production
      (agreement with the AES call was only 63.4%, kappa=0.276). Here we
      repeat the ESBL Model A/B comparison and the organism-only baseline
      restricted to isolates with a DIRECT AES ESBL phenotype call only
      (label_ESBL_source_primary not null), excluding every
      cephalosporin-fallback-derived label, to check whether the primary
      ESBL result is an artifact of the less specific fallback labels.

  (C) Unweighted-threshold sensitivity check. The primary pipeline uses
      class-balanced training throughout (class_weight='balanced' /
      scale_pos_weight) so the ~90%-sensitivity threshold procedure remains
      well behaved under class imbalance, at the cost of the raw-probability
      distortion documented in Results (Calibration) and corrected by
      post-hoc recalibration. Here we refit each target's headline Feature
      Set A model WITHOUT class weighting to check whether discrimination
      (AUROC) and the achievable operating point differ materially, i.e.
      whether class weighting was doing real work or was an unnecessary
      complication.

  (D) Baseline paired-delta analysis. Two simple structured baselines
      (organism identity alone; organism+specimen+ward+interval category)
      are compared with the full Model A feature set. Here we
      compute a paired, patient-cluster bootstrap CI directly on the delta
      (Model A minus each baseline), reusing the cached test-set
      predictions from 09_modeling.py exactly as 10_delta_ci.py does for
      the Model A-vs-B comparison.

All four analyses reuse the same patient-cluster bootstrap (patients, not
isolates, resampled with replacement) used throughout this pipeline, for
the same reason: >1 in 5 patients contribute more than one isolate to
these cohorts, so an isolate-level bootstrap would understate resampling
uncertainty.
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)
RANDOM_STATE = 42
N_BOOT = 1000

TABLES_DIR = Path('tables')
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data: same structured+NLP combined table used throughout, plus the
# ESBL label-source columns (dropped after 03_cohort_and_labels.py, so
# merged back in here on isolate_id) and note_available (already present
# in the NLP feature table).
# ---------------------------------------------------------------------------
struct = pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
nlp = pd.read_parquet('data/processed/nlp_features.parquet').reset_index(drop=True)
nlp_cols_to_use = [c for c in nlp.columns if c not in ('study_id', 'isolate_id', 'testing_date')]
data = pd.concat([struct, nlp[nlp_cols_to_use]], axis=1)

cohort_master = pd.read_parquet('data/processed/cohort_master.parquet').reset_index(drop=True)
assert len(cohort_master) == len(data), "cohort_master/splits row count mismatch"
assert (cohort_master['isolate_id'].values == data['isolate_id'].values).all(), "row order mismatch: cohort_master vs splits (isolate_id not aligned)"
assert (cohort_master['study_id'].values == data['study_id'].values).all(), "row order mismatch: cohort_master vs splits (study_id not aligned)"
# isolate_id is not a unique key (2,166 isolates share an id with another row --
# distinct organisms cultured from the same accession), so we align by row
# position (verified identical order above) rather than merging on isolate_id.
data['label_ESBL_source_primary'] = cohort_master['label_ESBL_source_primary'].values

with open('tables/best_model_per_target.json') as f:
    best_model = json.load(f)

TARGETS = {
    'MRSA': ('in_MRSA_target_cohort', 'label_MRSA'),
    'ESBL': ('in_ESBL_target_cohort', 'label_ESBL'),
    'CRE': ('in_CRE_target_cohort', 'label_CRE'),
}

NUMERIC_A = ['age_years', 'admission_to_culture_hours', 'prior_isolate_count',
             'days_since_last_isolate', 'percent_probability_num']
CATEGORICAL_A = ['organism_base', 'specimen_type_clean', 'infection_site_clean',
                  'gender_clean', 'ward_clean', 'dept_clean', 'onset_bucket', 'id_confidence_clean']
BINARY_A = ['is_pediatric_flag', 'prior_resistant_organism_flag', 'id_confidence_known']
NUMERIC_B_EXTRA = ['note_length_chars']
BINARY_B_EXTRA = [c for c in nlp_cols_to_use if c not in ('note_length_chars', 'note_available')] + ['note_available']

BASELINE0 = {'numeric': [], 'categorical': ['organism_base'], 'binary': []}


def build_preprocessor(feature_set, model_kind):
    if feature_set in ('A', 'B'):
        numeric = NUMERIC_A + (NUMERIC_B_EXTRA if feature_set == 'B' else [])
        categorical = CATEGORICAL_A
        binary = BINARY_A + (BINARY_B_EXTRA if feature_set == 'B' else [])
    else:
        numeric, categorical, binary = feature_set['numeric'], feature_set['categorical'], feature_set['binary']
    num_pipe = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True))]
                         + ([('scale', StandardScaler())] if model_kind == 'logreg' else []))
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    bin_pipe = Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=0))])
    transformers = []
    if numeric:
        transformers.append(('num', num_pipe, numeric))
    if categorical:
        transformers.append(('cat', cat_pipe, categorical))
    if binary:
        transformers.append(('bin', bin_pipe, binary))
    return ColumnTransformer(transformers, remainder='drop')


def make_model(model_kind, pos_weight, weighted=True):
    if model_kind == 'logreg':
        return LogisticRegression(max_iter=2000, class_weight=('balanced' if weighted else None),
                                   C=1.0, random_state=RANDOM_STATE)
    if model_kind == 'rf':
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                       class_weight=('balanced' if weighted else None),
                                       n_jobs=-1, random_state=RANDOM_STATE)
    if model_kind == 'xgb':
        return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=(pos_weight if weighted else 1.0),
                                  eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)
    raise ValueError(model_kind)


def bootstrap_ci_clustered(y_true, y_score, patient_ids, metric_fn, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score); patient_ids = np.asarray(patient_ids)
    pid_to_rows = {}
    for i, p in enumerate(patient_ids):
        pid_to_rows.setdefault(p, []).append(i)
    pid_list = np.array(list(pid_to_rows.keys()), dtype=object)
    n_pat = len(pid_list)
    vals = []
    for _ in range(n_boot):
        sampled = pid_list[rng.randint(0, n_pat, n_pat)]
        idx = np.concatenate([pid_to_rows[p] for p in sampled])
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            vals.append(metric_fn(yt, ys))
        except Exception:
            continue
    if not vals:
        return None, None, None
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_delta(y, pa, pb, patient_ids, metric_fn, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); pa = np.asarray(pa); pb = np.asarray(pb); patient_ids = np.asarray(patient_ids)
    pid_to_rows = {}
    for i, p in enumerate(patient_ids):
        pid_to_rows.setdefault(p, []).append(i)
    pid_list = np.array(list(pid_to_rows.keys()), dtype=object)
    n_pat = len(pid_list)
    deltas = []
    for _ in range(n_boot):
        sampled = pid_list[rng.randint(0, n_pat, n_pat)]
        idx = np.concatenate([pid_to_rows[p] for p in sampled])
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            deltas.append(metric_fn(yt, pb[idx]) - metric_fn(yt, pa[idx]))
        except Exception:
            continue
    deltas = np.array(deltas)
    return float(deltas.mean()), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def choose_threshold_for_sensitivity(y_true, y_score, target_sens=0.90):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    ok = tpr >= target_sens
    if not ok.any():
        return float(thresh[np.argmax(tpr)])
    idx_candidates = np.where(ok)[0]
    idx = idx_candidates[np.argmin(tpr[idx_candidates])]
    return float(thresh[idx])


def fit_eval(train_df, test_df, label_col, feature_set, model_kind, weighted=True):
    ct = build_preprocessor(feature_set, model_kind)
    pos_weight = (train_df[label_col] == 0).sum() / max((train_df[label_col] == 1).sum(), 1)
    inner_train, inner_val = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE,
                                               stratify=train_df[label_col])
    ct_t = build_preprocessor(feature_set, model_kind)
    pipe_t = Pipeline([('prep', ct_t), ('clf', make_model(model_kind, pos_weight, weighted))])
    pipe_t.fit(inner_train, inner_train[label_col])
    val_proba = pipe_t.predict_proba(inner_val)[:, 1]
    threshold = choose_threshold_for_sensitivity(inner_val[label_col], val_proba, 0.90)

    pipe = Pipeline([('prep', ct), ('clf', make_model(model_kind, pos_weight, weighted))])
    pipe.fit(train_df, train_df[label_col])
    test_proba = pipe.predict_proba(test_df)[:, 1]
    y_test = test_df[label_col].values
    pid_test = test_df['study_id'].values

    auroc = roc_auc_score(y_test, test_proba)
    a_m, a_lo, a_hi = bootstrap_ci_clustered(y_test, test_proba, pid_test, roc_auc_score)
    auprc = average_precision_score(y_test, test_proba)
    p_m, p_lo, p_hi = bootstrap_ci_clustered(y_test, test_proba, pid_test, average_precision_score)
    brier = brier_score_loss(y_test, test_proba)
    pred_label = (test_proba >= threshold).astype(int)
    tp = int(((pred_label == 1) & (y_test == 1)).sum()); fn = int(((pred_label == 0) & (y_test == 1)).sum())
    tn = int(((pred_label == 0) & (y_test == 0)).sum()); fp = int(((pred_label == 1) & (y_test == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        'n_train': len(train_df), 'n_test': len(test_df), 'prevalence_pct': round(100 * y_test.mean(), 1),
        'AUROC': round(auroc, 4), 'AUROC_CI_lo': round(a_lo, 4) if a_lo else None, 'AUROC_CI_hi': round(a_hi, 4) if a_hi else None,
        'AUPRC': round(auprc, 4), 'AUPRC_CI_lo': round(p_lo, 4) if p_lo else None, 'AUPRC_CI_hi': round(p_hi, 4) if p_hi else None,
        'Brier': round(brier, 4), 'threshold': round(threshold, 4),
        'sensitivity': round(sens, 4), 'specificity': round(spec, 4),
        'PPV': round(ppv, 4) if not np.isnan(ppv) else None, 'NPV': round(npv, 4) if not np.isnan(npv) else None,
        '_y_test': y_test, '_proba': test_proba, '_pid': pid_test,
    }


# ===========================================================================
# (A) Note-present sensitivity analysis: Model A vs B restricted to isolates
# with a discharge-summary note available.
# ===========================================================================
print("\n" + "=" * 70 + "\n(A) NOTE-PRESENT SENSITIVITY ANALYSIS\n" + "=" * 70)
note_rows = []
for target_name, (cohort_flag, label_col) in TARGETS.items():
    model_kind = best_model[target_name]
    target_df = data[data[cohort_flag] & data[label_col].notna() & (data['note_available'] == 1)].copy()
    target_df[label_col] = target_df[label_col].astype(int)
    train_df = target_df[target_df['split_temporal'] == 'train']
    test_df = target_df[target_df['split_temporal'] == 'test']
    if train_df[label_col].nunique() < 2 or test_df[label_col].nunique() < 2:
        print(f"  {target_name}: insufficient class variation in note-present subset, skipping")
        continue
    res_a = fit_eval(train_df, test_df, label_col, 'A', model_kind)
    res_b = fit_eval(train_df, test_df, label_col, 'B', model_kind)
    d_auroc_m, d_auroc_lo, d_auroc_hi = paired_bootstrap_delta(
        res_a['_y_test'], res_a['_proba'], res_b['_proba'], res_a['_pid'], roc_auc_score)
    d_auprc_m, d_auprc_lo, d_auprc_hi = paired_bootstrap_delta(
        res_a['_y_test'], res_a['_proba'], res_b['_proba'], res_a['_pid'], average_precision_score)
    row = {
        'target': target_name, 'model': model_kind,
        'n_train': res_a['n_train'], 'n_test': res_a['n_test'], 'prevalence_pct': res_a['prevalence_pct'],
        'AUROC_A': res_a['AUROC'], 'AUROC_A_CI': f"{res_a['AUROC_CI_lo']}-{res_a['AUROC_CI_hi']}",
        'AUROC_B': res_b['AUROC'], 'AUROC_B_CI': f"{res_b['AUROC_CI_lo']}-{res_b['AUROC_CI_hi']}",
        'AUROC_delta': round(d_auroc_m, 4), 'AUROC_delta_CI_lo': round(d_auroc_lo, 4), 'AUROC_delta_CI_hi': round(d_auroc_hi, 4),
        'AUROC_delta_excludes_zero': bool(d_auroc_lo > 0 or d_auroc_hi < 0),
        'AUPRC_A': res_a['AUPRC'], 'AUPRC_B': res_b['AUPRC'],
        'AUPRC_delta': round(d_auprc_m, 4), 'AUPRC_delta_CI_lo': round(d_auprc_lo, 4), 'AUPRC_delta_CI_hi': round(d_auprc_hi, 4),
        'AUPRC_delta_excludes_zero': bool(d_auprc_lo > 0 or d_auprc_hi < 0),
    }
    note_rows.append(row)
    print(f"  {target_name} ({model_kind}, n_test={res_a['n_test']}): "
          f"AUROC A={res_a['AUROC']:.3f} B={res_b['AUROC']:.3f} delta={d_auroc_m:+.4f} ({d_auroc_lo:+.4f} to {d_auroc_hi:+.4f}) | "
          f"AUPRC delta={d_auprc_m:+.4f} ({d_auprc_lo:+.4f} to {d_auprc_hi:+.4f})")
pd.DataFrame(note_rows).to_csv(TABLES_DIR / 'table_note_present_sensitivity.csv', index=False)
print(f"Saved: tables/table_note_present_sensitivity.csv")

# ===========================================================================
# (B) ESBL label-definition sensitivity: direct AES phenotype calls only.
# ===========================================================================
print("\n" + "=" * 70 + "\n(B) ESBL LABEL-DEFINITION SENSITIVITY ANALYSIS\n" + "=" * 70)
esbl_strict_rows = []
cohort_flag, label_col = TARGETS['ESBL']
model_kind = best_model['ESBL']
strict_df = data[data[cohort_flag] & data[label_col].notna() & data['label_ESBL_source_primary'].notna()].copy()
strict_df[label_col] = strict_df[label_col].astype(int)
n_full_esbl = int((data[cohort_flag] & data[label_col].notna()).sum())
print(f"  Full ESBL-labeled cohort: n={n_full_esbl}; direct-AES-only subset: n={len(strict_df)} "
      f"({100*len(strict_df)/n_full_esbl:.1f}% of full)")
train_df = strict_df[strict_df['split_temporal'] == 'train']
test_df = strict_df[strict_df['split_temporal'] == 'test']
for feature_set in ['A', 'B', BASELINE0]:
    fs_label = feature_set if isinstance(feature_set, str) else 'Baseline0_organism_only'
    mk = model_kind if isinstance(feature_set, str) else 'logreg'
    res = fit_eval(train_df, test_df, label_col, feature_set, mk)
    esbl_strict_rows.append({
        'feature_set': fs_label, 'model': mk, 'n_train': res['n_train'], 'n_test': res['n_test'],
        'prevalence_pct': res['prevalence_pct'], 'AUROC': res['AUROC'],
        'AUROC_CI_lo': res['AUROC_CI_lo'], 'AUROC_CI_hi': res['AUROC_CI_hi'],
        'AUPRC': res['AUPRC'], 'AUPRC_CI_lo': res['AUPRC_CI_lo'], 'AUPRC_CI_hi': res['AUPRC_CI_hi'],
    })
    print(f"  {fs_label:30s} ({mk}): AUROC={res['AUROC']:.3f} ({res['AUROC_CI_lo']}-{res['AUROC_CI_hi']})  "
          f"AUPRC={res['AUPRC']:.3f} ({res['AUPRC_CI_lo']}-{res['AUPRC_CI_hi']})")
pd.DataFrame(esbl_strict_rows).to_csv(TABLES_DIR / 'table_esbl_label_sensitivity.csv', index=False)
print(f"Saved: tables/table_esbl_label_sensitivity.csv")

# ===========================================================================
# (C) Unweighted-threshold sensitivity check: headline Feature Set A models
# refit without class-balanced weighting.
# ===========================================================================
print("\n" + "=" * 70 + "\n(C) UNWEIGHTED-THRESHOLD SENSITIVITY CHECK (Feature Set A, headline family)\n" + "=" * 70)
unweighted_rows = []
with open('tables/table4_model_performance_full.csv') as f:
    perf_full = pd.read_csv(f)
for target_name, (cohort_flag, label_col) in TARGETS.items():
    model_kind = best_model[target_name]
    target_df = data[data[cohort_flag] & data[label_col].notna()].copy()
    target_df[label_col] = target_df[label_col].astype(int)
    train_df = target_df[target_df['split_temporal'] == 'train']
    test_df = target_df[target_df['split_temporal'] == 'test']
    res_w = fit_eval(train_df, test_df, label_col, 'A', model_kind, weighted=True)
    res_u = fit_eval(train_df, test_df, label_col, 'A', model_kind, weighted=False)
    row = {
        'target': target_name, 'model': model_kind,
        'AUROC_weighted': res_w['AUROC'], 'AUROC_unweighted': res_u['AUROC'],
        'AUROC_abs_diff': round(abs(res_w['AUROC'] - res_u['AUROC']), 4),
        'threshold_weighted': res_w['threshold'], 'threshold_unweighted': res_u['threshold'],
        'sensitivity_weighted': res_w['sensitivity'], 'sensitivity_unweighted': res_u['sensitivity'],
        'specificity_weighted': res_w['specificity'], 'specificity_unweighted': res_u['specificity'],
        'Brier_weighted': res_w['Brier'], 'Brier_unweighted': res_u['Brier'],
    }
    unweighted_rows.append(row)
    print(f"  {target_name} ({model_kind}): AUROC weighted={res_w['AUROC']:.3f} unweighted={res_u['AUROC']:.3f} "
          f"(|diff|={row['AUROC_abs_diff']:.4f}) | threshold weighted={res_w['threshold']:.3f} "
          f"unweighted={res_u['threshold']:.3f} | Sens/Spec weighted={res_w['sensitivity']:.2f}/{res_w['specificity']:.2f} "
          f"unweighted={res_u['sensitivity']:.2f}/{res_u['specificity']:.2f} | "
          f"Brier weighted={res_w['Brier']:.3f} unweighted={res_u['Brier']:.3f}")
pd.DataFrame(unweighted_rows).to_csv(TABLES_DIR / 'table_unweighted_sensitivity.csv', index=False)
print(f"Saved: tables/table_unweighted_sensitivity.csv")

# ===========================================================================
# (D) Baseline paired-delta CI: Model A (full, headline family) minus each
# simple baseline, using cached predictions from 09_modeling.py.
# ===========================================================================
print("\n" + "=" * 70 + "\n(D) BASELINE PAIRED-DELTA ANALYSIS (Full Model A vs simple baselines)\n" + "=" * 70)
with open('data/processed/model_outputs/predictions.json') as f:
    preds = json.load(f)
baseline_delta_rows = []
for target_name in TARGETS:
    model_kind = best_model[target_name]
    key_a = f"{target_name}__split_temporal__A__{model_kind}"
    for baseline_key_name, baseline_label in [
        ('Baseline0_organism_only', 'organism_only'),
        ('Baseline1_organism_specimen_ward_onset', 'organism_specimen_ward_onset'),
    ]:
        key_base = f"{target_name}__split_temporal__{baseline_key_name}__logreg"
        if key_a not in preds or key_base not in preds:
            continue
        ya, pa = np.array(preds[key_a]['y_test']), np.array(preds[key_a]['y_proba'])
        yb, pb = np.array(preds[key_base]['y_test']), np.array(preds[key_base]['y_proba'])
        assert np.array_equal(ya, yb), f"y_test mismatch for {target_name}/{baseline_key_name}"
        pid = np.array(preds[key_a]['study_id'])
        # paired_bootstrap_delta(y, pa, pb, ...) returns metric(pb)-metric(pa); passing
        # (baseline_proba, full_proba) as (pa, pb) gives metric(full) - metric(baseline) directly.
        d_auroc_m, d_auroc_lo, d_auroc_hi = paired_bootstrap_delta(ya, pb, pa, pid, roc_auc_score)
        row = {
            'target': target_name, 'model_family': model_kind, 'baseline': baseline_label,
            'AUROC_full': round(float(roc_auc_score(ya, pa)), 4), 'AUROC_baseline': round(float(roc_auc_score(yb, pb)), 4),
            'AUROC_delta_full_minus_baseline': round(d_auroc_m, 4),
            'AUROC_delta_CI_lo': round(d_auroc_lo, 4), 'AUROC_delta_CI_hi': round(d_auroc_hi, 4),
            'AUROC_delta_excludes_zero': bool(d_auroc_lo > 0 or d_auroc_hi < 0),
        }
        baseline_delta_rows.append(row)
        print(f"  {target_name} vs {baseline_label}: Full={row['AUROC_full']:.3f} Baseline={row['AUROC_baseline']:.3f} "
              f"delta={d_auroc_m:+.4f} ({d_auroc_lo:+.4f} to {d_auroc_hi:+.4f})  "
              f"{'EXCLUDES ZERO' if row['AUROC_delta_excludes_zero'] else 'includes zero'}")
pd.DataFrame(baseline_delta_rows).to_csv(TABLES_DIR / 'table4b_baseline_delta_ci.csv', index=False)
print(f"Saved: tables/table4b_baseline_delta_ci.csv")

print("\nAll four sensitivity analyses complete.")
