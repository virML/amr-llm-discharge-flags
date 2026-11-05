"""
15_recalibration.py
Post-hoc probability recalibration (Methods, Results: Calibration).

Class-balanced training -- used throughout for principled threshold
selection under class imbalance (Methods) -- distorts predicted
probabilities away from true population prevalence (Results: Calibration).
This script applies leakage-safe, cross-fitted isotonic regression
(sklearn's CalibratedClassifierCV, cv=5) to each target's headline pipeline
on BOTH Feature Set A (structured only) and Feature Set B (structured+NLP),
temporal split, per tables/best_model_per_target.json.

Model A is recalibrated alongside Model B -- not only Model B -- because
the manuscript's clinical recommendation is to validate/deploy the
structured-only model (Model A), given the primary result; reporting
calibrated probabilities only for Model B would leave the actually
recommended model's probability estimates uncalibrated. The calibration
mapping is learned ONLY from 5-fold cross-fitted out-of-fold predictions on
the TRAINING set -- a fresh, unfit clone of each pipeline is refit inside
each fold, so the calibrator never sees the test set during fitting, and
the already-fit pipeline used for the "before" comparison is left
untouched. Isotonic regression is a monotonic transform, so it does not
change AUROC/AUPRC rank-based discrimination in principle; because
CalibratedClassifierCV's default cv-ensemble prediction slightly perturbs
this (it averages 5 refit base models rather than reusing the single
full-training-set fit), we recompute and report AUROC after recalibration
too, to confirm discrimination is materially unchanged.

Produces figures/figure3b_recalibration.png (before/after reliability
plots, 3 targets x 2 feature sets) and tables/table_recalibration.json
(Brier scores before/after, vs. the same constant-prevalence baseline used
in Results: Calibration, plus post-recalibration AUROC as a discrimination
sanity check, for A and B side by side).
"""
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.base import clone

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

FIG_DIR = Path('figures/llm')
TAB_DIR = Path('tables/llm')
MODELS_DIR = Path('data/processed/fitted_models/llm')

struct = pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
nlp = pd.read_parquet('data/processed/nlp_features_llm.parquet').reset_index(drop=True)
nlp_cols_to_use = [c for c in nlp.columns if c not in ('study_id', 'isolate_id', 'testing_date')]
data = pd.concat([struct, nlp[nlp_cols_to_use]], axis=1)

best = json.load(open(TAB_DIR / 'best_model_per_target.json'))

TARGET_INFO = {
    'MRSA': ('in_MRSA_target_cohort', 'label_MRSA'),
    'ESBL': ('in_ESBL_target_cohort', 'label_ESBL'),
    'CRE': ('in_CRE_target_cohort', 'label_CRE'),
}

results = {}
RECAL_PRED = {}
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for col, target in enumerate(['MRSA', 'ESBL', 'CRE']):
    cohort_flag, label_col = TARGET_INFO[target]
    model_kind = best[target]
    target_df = data[data[cohort_flag] & data[label_col].notna()].copy()
    train_df = target_df[target_df['split_temporal'] == 'train'].reset_index(drop=True)
    test_df = target_df[target_df['split_temporal'] == 'test'].reset_index(drop=True)
    y_train = train_df[label_col].astype(int).values
    y_test = test_df[label_col].astype(int).values
    results[target] = {'model': model_kind, 'n_train': int(len(train_df)), 'n_test': int(len(test_df))}

    for row, feature_set in enumerate(['A', 'B']):
        ax = axes[row, col]
        key = f"{target}__split_temporal__{feature_set}__{model_kind}"
        with open(MODELS_DIR / f"{key}.pkl", 'rb') as f:
            saved = pickle.load(f)
        base_pipe = saved['pipeline']

        # "Before": the already-fit (on 100% of training data) pipeline used
        # everywhere else in the manuscript -- untouched by this script.
        p_uncal = base_pipe.predict_proba(test_df)[:, 1]
        auroc_uncal = roc_auc_score(y_test, p_uncal)

        # "After": leakage-safe 5-fold cross-fitted isotonic recalibration.
        fresh_pipe = clone(base_pipe)
        calib = CalibratedClassifierCV(fresh_pipe, method='isotonic', cv=5)
        calib.fit(train_df, y_train)
        p_cal = calib.predict_proba(test_df)[:, 1]
        auroc_cal = roc_auc_score(y_test, p_cal)
        RECAL_PRED[f'{target}__{feature_set}__{model_kind}'] = {'y_test': y_test.tolist(), 'p_uncal': p_uncal.tolist(), 'p_cal': p_cal.tolist()}

        brier_uncal = brier_score_loss(y_test, p_uncal)
        brier_cal = brier_score_loss(y_test, p_cal)
        prevalence = float(y_test.mean())
        brier_baseline = brier_score_loss(y_test, np.full_like(p_uncal, prevalence, dtype=float))

        frac_pos_u, mean_pred_u = calibration_curve(y_test, p_uncal, n_bins=10, strategy='quantile')
        frac_pos_c, mean_pred_c = calibration_curve(y_test, p_cal, n_bins=10, strategy='quantile')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
        ax.plot(mean_pred_u, frac_pos_u, 'o-', color='#d62728', label=f'Before (Brier={brier_uncal:.3f})')
        ax.plot(mean_pred_c, frac_pos_c, 's-', color='#2ca02c', label=f'After isotonic (Brier={brier_cal:.3f})')
        ax.set_title(f'{target} Model {feature_set} ({model_kind.upper()})')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Observed frequency')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        results[target][f'model_{feature_set}'] = {
            'auroc_uncalibrated': float(auroc_uncal),
            'auroc_after_recalibration': float(auroc_cal),
            'brier_uncalibrated': float(brier_uncal),
            'brier_isotonic_recalibrated': float(brier_cal),
            'brier_constant_prevalence_baseline': float(brier_baseline),
            'brier_relative_improvement_pct': float(100 * (brier_uncal - brier_cal) / brier_uncal),
            'test_prevalence': prevalence,
        }
        print(f"{target} Model {feature_set} ({model_kind}): AUROC before={auroc_uncal:.4f}, after={auroc_cal:.4f} | "
              f"Brier before={brier_uncal:.4f}, after={brier_cal:.4f}, baseline={brier_baseline:.4f}, "
              f"rel. improvement={results[target][f'model_{feature_set}']['brier_relative_improvement_pct']:.1f}%")

(lambda *a, **k: None)('Figure 3b (Supplementary). Post-hoc isotonic recalibration, before vs. after -- Model A (top) and Model B (bottom)', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_DIR / 'figure3b_recalibration.png')
plt.close(fig)

with open('data/processed/model_outputs/llm/recalibration_predictions.json', 'w') as f:
    json.dump(RECAL_PRED, f)
with open(TAB_DIR / 'table_recalibration.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved: figures/figure3b_recalibration.png, tables/table_recalibration.json")
