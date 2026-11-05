"""
09_modeling.py
Modeling: Model A (structured only) vs Model B
(structured + NLP) for each of MRSA / ESBL / CRE, on both the temporal
(primary) and random patient-level (sensitivity) splits, using logistic
regression, random forest, and XGBoost. Two simple clinical baselines
(organism identity alone; organism + specimen + ward + interval category) are also
fit with logistic regression, to show what a naive antibiogram-style rule
already achieves before crediting the full structured/NLP feature sets.

For every (target x split x feature-set x model) combination we compute:
  - AUROC, AUPRC with a 1,000-resample PATIENT-CLUSTER bootstrap 95% CI on
    the test set (resampling patients, then taking all of that patient's
    test-set isolates together, since 20.8% of patients contribute more
    than one isolate and an isolate-level bootstrap would treat correlated
    within-patient observations as independent, understating the true CI
    width)
  - a clinically motivated operating threshold (chosen on an inner
    train/validation split to target ~90% sensitivity -- "optimized for
    sensitivity given the clinical cost of missing a resistant organism",
    as planned -- NEVER chosen on the test set itself), and the
    resulting test-set sensitivity/specificity/PPV/NPV at that threshold
  - Brier score and calibration-curve bins (for the calibration figure)

The model family reported as each target's "best-discriminating" pipeline
(Table 4, Figs. 2-4, SHAP, subgroup analysis) is chosen by AUROC on the
inner train/validation split using Feature Set A only -- never by test-set
performance, and never by a criterion that could pick different model
families for Model A and Model B -- so the evaluation set plays no role in
model selection (written to tables/best_model_per_target.json).

All raw predictions (including each row's patient id, for downstream
cluster-bootstrap consistency) are cached to disk so the figures/SHAP
script and the verification notebook can reproduce every number without
refitting.
"""
import pandas as pd
import numpy as np
import json
import warnings
import pickle
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

STRUCT_PARQUET = Path('data/processed/splits.parquet')
NLP_PARQUET = Path('data/processed/nlp_features_llm.parquet')
OUT_DIR = Path('data/processed/model_outputs/llm')
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path('tables/llm')
MODELS_DIR = Path('data/processed/fitted_models/llm')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOT = 1000

struct = pd.read_parquet(STRUCT_PARQUET).reset_index(drop=True)
nlp = pd.read_parquet(NLP_PARQUET).reset_index(drop=True)
assert len(struct) == len(nlp), "structured/NLP feature tables have different row counts"
assert (struct['study_id'].values == nlp['study_id'].values).all(), "row order mismatch between structured/NLP tables"

nlp_cols_to_use = [c for c in nlp.columns if c not in ('study_id', 'isolate_id', 'testing_date')]
data = pd.concat([struct, nlp[nlp_cols_to_use]], axis=1)
print(f"Combined modeling table: {data.shape}")

# ---------------------------------------------------------------------------
# Feature set definitions
# ---------------------------------------------------------------------------
NUMERIC_A = ['age_years', 'admission_to_culture_hours', 'prior_isolate_count',
             'days_since_last_isolate', 'percent_probability_num']
CATEGORICAL_A = ['organism_base', 'specimen_type_clean', 'infection_site_clean',
                  'gender_clean', 'ward_clean', 'dept_clean', 'onset_bucket', 'id_confidence_clean']
BINARY_A = ['is_pediatric_flag', 'prior_resistant_organism_flag', 'id_confidence_known']

NUMERIC_B_EXTRA = ['note_length_chars']
BINARY_B_EXTRA = [c for c in nlp_cols_to_use if c not in ('note_length_chars',)]

# Simple clinical baselines (an organism-identity-only "local antibiogram"
# rule, and a minimally richer rule adding specimen/ward/interval category, to show what naive, already-available structured
# shortcuts achieve before crediting Model A's fuller feature set or Model
# B's NLP additions). Logistic regression only, by design -- the point of a
# baseline is that it is simple and interpretable, not competitive tuning.
BASELINE_FEATURE_SETS = {
    'Baseline0_organism_only': {'numeric': [], 'categorical': ['organism_base'], 'binary': []},
    'Baseline1_organism_specimen_ward_onset': {
        'numeric': [], 'categorical': ['organism_base', 'specimen_type_clean', 'ward_clean', 'onset_bucket'], 'binary': [],
    },
}

TARGETS = {
    'MRSA': ('in_MRSA_target_cohort', 'label_MRSA'),
    'ESBL': ('in_ESBL_target_cohort', 'label_ESBL'),
    'CRE': ('in_CRE_target_cohort', 'label_CRE'),
}
SPLITS = ['split_temporal', 'split_random']
FEATURE_SETS = ['A', 'B']

def build_preprocessor(feature_set, model_kind):
    if feature_set in ('A', 'B'):
        numeric = NUMERIC_A + (NUMERIC_B_EXTRA if feature_set == 'B' else [])
        categorical = CATEGORICAL_A
        binary = BINARY_A + (BINARY_B_EXTRA if feature_set == 'B' else [])
    else:
        d = BASELINE_FEATURE_SETS[feature_set]
        numeric, categorical, binary = d['numeric'], d['categorical'], d['binary']

    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
    ] + ([('scale', StandardScaler())] if model_kind == 'logreg' else []))

    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    bin_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
    ])

    transformers = []
    if numeric:
        transformers.append(('num', num_pipe, numeric))
    if categorical:
        transformers.append(('cat', cat_pipe, categorical))
    if binary:
        transformers.append(('bin', bin_pipe, binary))
    ct = ColumnTransformer(transformers, remainder='drop')
    return ct, numeric, categorical, binary

def make_model(model_kind, pos_weight):
    if model_kind == 'logreg':
        return LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0, random_state=RANDOM_STATE)
    if model_kind == 'rf':
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                       class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
    if model_kind == 'xgb':
        return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=pos_weight, eval_metric='logloss',
                                  random_state=RANDOM_STATE, n_jobs=-1)
    raise ValueError(model_kind)

def bootstrap_ci_clustered(y_true, y_score, patient_ids, metric_fn, n_boot=N_BOOT, seed=RANDOM_STATE):
    """Patient-cluster bootstrap: resample PATIENTS with replacement, then
    take all of that patient's rows (so a patient's isolates always move
    together), rather than resampling isolates independently. This is the
    statistically appropriate unit here because within-patient isolates can
    have correlated resistance status, and >1 in 5 patients contribute more
    than one isolate to these cohorts."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    patient_ids = np.asarray(patient_ids)
    pid_to_rows = {}
    for i, pid in enumerate(patient_ids):
        pid_to_rows.setdefault(pid, []).append(i)
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

def choose_threshold_for_sensitivity(y_true, y_score, target_sens=0.90):
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    # find smallest threshold achieving tpr >= target_sens (thresholds descending)
    ok = tpr >= target_sens
    if not ok.any():
        return float(thresh[np.argmax(tpr)])
    idx_candidates = np.where(ok)[0]
    idx = idx_candidates[np.argmin(tpr[idx_candidates])]  # tightest one still meeting target
    return float(thresh[idx])

results_rows = []
predictions_store = {}
best_model_per_target = {}

for target_name, (cohort_flag, label_col) in TARGETS.items():
    target_df = data[data[cohort_flag] & data[label_col].notna()].copy()
    target_df[label_col] = target_df[label_col].astype(int)
    print(f"\n{'='*70}\nTARGET: {target_name}  (n={len(target_df)}, prevalence={target_df[label_col].mean():.3f})\n{'='*70}")

    for split_col in SPLITS:
        train_df = target_df[target_df[split_col] == 'train']
        test_df = target_df[target_df[split_col] == 'test']
        if train_df[label_col].nunique() < 2 or test_df[label_col].nunique() < 2:
            print(f"  Skipping {split_col}: insufficient class variation")
            continue

        # inner split of TRAIN for threshold selection AND model-family
        # selection (never touches test)
        inner_train, inner_val = train_test_split(
            train_df, test_size=0.2, random_state=RANDOM_STATE, stratify=train_df[label_col]
        )

        # -----------------------------------------------------------------
        # Model-family selection for the primary (temporal) split: pick the
        # model family with the highest inner-validation AUROC on Feature
        # Set A. This never looks at the test set, and is computed once so
        # the SAME family is used for both Model A and Model B downstream
        # (Table 4 headline rows, Figs. 2-4, SHAP, subgroup analysis) --
        # unlike choosing separately per feature set, which could otherwise
        # let two different families "win" for A and B respectively.
        # -----------------------------------------------------------------
        if split_col == 'split_temporal':
            inner_val_auroc = {}
            for model_kind in ['logreg', 'rf', 'xgb']:
                ct_sel, *_ = build_preprocessor('A', model_kind)
                pos_weight_sel = (inner_train[label_col] == 0).sum() / max((inner_train[label_col] == 1).sum(), 1)
                pipe_sel = Pipeline([('prep', ct_sel), ('clf', make_model(model_kind, pos_weight_sel))])
                pipe_sel.fit(inner_train, inner_train[label_col])
                val_proba_sel = pipe_sel.predict_proba(inner_val)[:, 1]
                inner_val_auroc[model_kind] = float(roc_auc_score(inner_val[label_col], val_proba_sel))
            chosen_model = max(inner_val_auroc, key=inner_val_auroc.get)
            best_model_per_target[target_name] = chosen_model
            print(f"  Model-family selection (inner-validation AUROC, Feature Set A, "
                  f"test set not used): {inner_val_auroc} -> chosen: {chosen_model}")

        for feature_set in FEATURE_SETS:
            for model_kind in ['logreg', 'rf', 'xgb']:
                ct, numeric, categorical, binary = build_preprocessor(feature_set, model_kind)
                pos_weight = (train_df[label_col] == 0).sum() / max((train_df[label_col] == 1).sum(), 1)

                # --- fit on inner_train to pick threshold on inner_val ---
                pipe_thresh = Pipeline([('prep', ct), ('clf', make_model(model_kind, pos_weight))])
                pipe_thresh.fit(inner_train, inner_train[label_col])
                val_proba = pipe_thresh.predict_proba(inner_val)[:, 1]
                threshold = choose_threshold_for_sensitivity(inner_val[label_col], val_proba, target_sens=0.90)

                # --- refit on FULL train, evaluate on test ---
                ct2, *_ = build_preprocessor(feature_set, model_kind)
                pipe = Pipeline([('prep', ct2), ('clf', make_model(model_kind, pos_weight))])
                pipe.fit(train_df, train_df[label_col])
                test_proba = pipe.predict_proba(test_df)[:, 1]
                y_test = test_df[label_col].values
                pid_test = test_df['study_id'].values

                auroc = roc_auc_score(y_test, test_proba)
                auroc_mean, auroc_lo, auroc_hi = bootstrap_ci_clustered(y_test, test_proba, pid_test, roc_auc_score)
                auprc = average_precision_score(y_test, test_proba)
                auprc_mean, auprc_lo, auprc_hi = bootstrap_ci_clustered(y_test, test_proba, pid_test, average_precision_score)
                brier = brier_score_loss(y_test, test_proba)

                pred_label = (test_proba >= threshold).astype(int)
                tp = int(((pred_label == 1) & (y_test == 1)).sum())
                fn = int(((pred_label == 0) & (y_test == 1)).sum())
                tn = int(((pred_label == 0) & (y_test == 0)).sum())
                fp = int(((pred_label == 1) & (y_test == 0)).sum())
                sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
                ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

                row = {
                    'target': target_name, 'split': split_col, 'feature_set': feature_set,
                    'model': model_kind, 'n_train': len(train_df), 'n_test': len(test_df),
                    'prevalence_test_pct': round(100 * y_test.mean(), 1),
                    'AUROC': round(auroc, 4), 'AUROC_CI_lo': round(auroc_lo, 4) if auroc_lo else None,
                    'AUROC_CI_hi': round(auroc_hi, 4) if auroc_hi else None,
                    'AUPRC': round(auprc, 4), 'AUPRC_CI_lo': round(auprc_lo, 4) if auprc_lo else None,
                    'AUPRC_CI_hi': round(auprc_hi, 4) if auprc_hi else None,
                    'Brier': round(brier, 4), 'threshold': round(threshold, 4),
                    'sensitivity_at_threshold': round(sens, 4), 'specificity_at_threshold': round(spec, 4),
                    'PPV_at_threshold': round(ppv, 4) if not np.isnan(ppv) else None,
                    'NPV_at_threshold': round(npv, 4) if not np.isnan(npv) else None,
                    'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
                }
                results_rows.append(row)
                print(f"  [{split_col:16s}] {feature_set} {model_kind:7s} "
                      f"AUROC={auroc:.3f} ({auroc_lo:.3f}-{auroc_hi:.3f})  "
                      f"AUPRC={auprc:.3f} ({auprc_lo:.3f}-{auprc_hi:.3f})  "
                      f"Sens={sens:.2f} Spec={spec:.2f} PPV={ppv:.2f} NPV={npv:.2f}")

                key = f"{target_name}__{split_col}__{feature_set}__{model_kind}"
                predictions_store[key] = {
                    'y_test': y_test.tolist(), 'y_proba': test_proba.tolist(), 'threshold': threshold,
                    'study_id': pid_test.tolist(),
                }
                # persist fitted pipeline for the (target, split=temporal) primary combos only, to keep disk usage sane
                if split_col == 'split_temporal':
                    with open(MODELS_DIR / f'{key}.pkl', 'wb') as f:
                        pickle.dump({'pipeline': pipe, 'feature_lists': {'numeric': numeric, 'categorical': categorical, 'binary': binary}}, f)

        # -------------------------------------------------------------
        # Simple clinical baselines (logistic regression only): organism
        # identity alone, and organism+specimen+ward+onset. Fit/evaluated
        # exactly like the main feature sets (same inner-train threshold
        # selection, same held-out test set, same cluster bootstrap).
        # -------------------------------------------------------------
        for feature_set in BASELINE_FEATURE_SETS:
            model_kind = 'logreg'
            ct, numeric, categorical, binary = build_preprocessor(feature_set, model_kind)
            pos_weight = (train_df[label_col] == 0).sum() / max((train_df[label_col] == 1).sum(), 1)

            pipe_thresh = Pipeline([('prep', ct), ('clf', make_model(model_kind, pos_weight))])
            pipe_thresh.fit(inner_train, inner_train[label_col])
            val_proba = pipe_thresh.predict_proba(inner_val)[:, 1]
            threshold = choose_threshold_for_sensitivity(inner_val[label_col], val_proba, target_sens=0.90)

            ct2, *_ = build_preprocessor(feature_set, model_kind)
            pipe = Pipeline([('prep', ct2), ('clf', make_model(model_kind, pos_weight))])
            pipe.fit(train_df, train_df[label_col])
            test_proba = pipe.predict_proba(test_df)[:, 1]
            y_test = test_df[label_col].values
            pid_test = test_df['study_id'].values

            auroc = roc_auc_score(y_test, test_proba)
            auroc_mean, auroc_lo, auroc_hi = bootstrap_ci_clustered(y_test, test_proba, pid_test, roc_auc_score)
            auprc = average_precision_score(y_test, test_proba)
            auprc_mean, auprc_lo, auprc_hi = bootstrap_ci_clustered(y_test, test_proba, pid_test, average_precision_score)
            brier = brier_score_loss(y_test, test_proba)

            pred_label = (test_proba >= threshold).astype(int)
            tp = int(((pred_label == 1) & (y_test == 1)).sum())
            fn = int(((pred_label == 0) & (y_test == 1)).sum())
            tn = int(((pred_label == 0) & (y_test == 0)).sum())
            fp = int(((pred_label == 1) & (y_test == 0)).sum())
            sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

            row = {
                'target': target_name, 'split': split_col, 'feature_set': feature_set,
                'model': model_kind, 'n_train': len(train_df), 'n_test': len(test_df),
                'prevalence_test_pct': round(100 * y_test.mean(), 1),
                'AUROC': round(auroc, 4), 'AUROC_CI_lo': round(auroc_lo, 4) if auroc_lo else None,
                'AUROC_CI_hi': round(auroc_hi, 4) if auroc_hi else None,
                'AUPRC': round(auprc, 4), 'AUPRC_CI_lo': round(auprc_lo, 4) if auprc_lo else None,
                'AUPRC_CI_hi': round(auprc_hi, 4) if auprc_hi else None,
                'Brier': round(brier, 4), 'threshold': round(threshold, 4),
                'sensitivity_at_threshold': round(sens, 4), 'specificity_at_threshold': round(spec, 4),
                'PPV_at_threshold': round(ppv, 4) if not np.isnan(ppv) else None,
                'NPV_at_threshold': round(npv, 4) if not np.isnan(npv) else None,
                'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            }
            results_rows.append(row)
            print(f"  [{split_col:16s}] {feature_set:38s} {model_kind:7s} "
                  f"AUROC={auroc:.3f} ({auroc_lo:.3f}-{auroc_hi:.3f})  AUPRC={auprc:.3f} ({auprc_lo:.3f}-{auprc_hi:.3f})")

            key = f"{target_name}__{split_col}__{feature_set}__{model_kind}"
            predictions_store[key] = {
                'y_test': y_test.tolist(), 'y_proba': test_proba.tolist(), 'threshold': threshold,
                'study_id': pid_test.tolist(),
            }

results_df = pd.DataFrame(results_rows)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
results_df.to_csv(TABLES_DIR / 'table4_model_performance_full.csv', index=False)
with open(OUT_DIR / 'predictions.json', 'w') as f:
    json.dump(predictions_store, f)
with open(TABLES_DIR / 'best_model_per_target.json', 'w') as f:
    json.dump(best_model_per_target, f, indent=2)

print(f"\nSaved full results table: {TABLES_DIR/'table4_model_performance_full.csv'}  ({len(results_df)} rows)")
print(f"Saved predictions cache: {OUT_DIR/'predictions.json'}")
print(f"Saved fitted primary (temporal-split) pipelines to: {MODELS_DIR}")
print(f"Saved model-family selection (inner-validation AUROC, Feature Set A only): {best_model_per_target}")

# ---------------------------------------------------------------------------
# Model A vs Model B delta table (primary = temporal split, XGBoost best-in-class typical)
# ---------------------------------------------------------------------------
delta_rows = []
for target_name in TARGETS:
    for model_kind in ['logreg', 'rf', 'xgb']:
        sub = results_df[(results_df['target'] == target_name) & (results_df['split'] == 'split_temporal') & (results_df['model'] == model_kind)]
        sub = sub[sub['feature_set'].isin(['A', 'B'])]
        if len(sub) != 2:
            continue
        a = sub[sub['feature_set'] == 'A'].iloc[0]
        b = sub[sub['feature_set'] == 'B'].iloc[0]
        delta_rows.append({
            'target': target_name, 'model': model_kind,
            'AUROC_A': a['AUROC'], 'AUROC_B': b['AUROC'], 'AUROC_delta': round(b['AUROC'] - a['AUROC'], 4),
            'AUPRC_A': a['AUPRC'], 'AUPRC_B': b['AUPRC'], 'AUPRC_delta': round(b['AUPRC'] - a['AUPRC'], 4),
        })
delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(TABLES_DIR / 'table4_model_A_vs_B_delta.csv', index=False)
print(f"\n{delta_df.to_string(index=False)}")
print(f"\nSaved: {TABLES_DIR/'table4_model_A_vs_B_delta.csv'}")

# ---------------------------------------------------------------------------
# Baseline vs Model A/B summary (temporal split, chosen model family per
# target) -- how much of Model A's apparent discrimination is already
# achieved by an organism-only or organism+specimen+ward+onset rule.
# ---------------------------------------------------------------------------
baseline_rows = []
for target_name, model_kind in best_model_per_target.items():
    sub = results_df[(results_df['target'] == target_name) & (results_df['split'] == 'split_temporal')]
    b0 = sub[sub['feature_set'] == 'Baseline0_organism_only']
    b1 = sub[sub['feature_set'] == 'Baseline1_organism_specimen_ward_onset']
    a = sub[(sub['feature_set'] == 'A') & (sub['model'] == model_kind)]
    if len(b0) and len(b1) and len(a):
        baseline_rows.append({
            'target': target_name, 'best_model_family': model_kind,
            'AUROC_Baseline0_organism_only': b0.iloc[0]['AUROC'],
            'AUROC_Baseline1_organism_specimen_ward_onset': b1.iloc[0]['AUROC'],
            'AUROC_ModelA_full_structured': a.iloc[0]['AUROC'],
        })
baseline_df = pd.DataFrame(baseline_rows)
baseline_df.to_csv(TABLES_DIR / 'table4b_baseline_comparison.csv', index=False)
print(f"\n{baseline_df.to_string(index=False)}")
print(f"\nSaved: {TABLES_DIR/'table4b_baseline_comparison.csv'}")
