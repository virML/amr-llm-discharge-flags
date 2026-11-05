"""
13_multiple_testing_correction.py
Multiple-testing correction for the 36 paired-bootstrap delta comparisons
(3 targets x 2 splits x 3 models x 2 metrics) computed in 10_delta_ci.py.

This script:

  1. Recomputes an approximate two-sided bootstrap p-value for each of the
     36 delta comparisons (percentile-bootstrap p-value: twice the smaller
     tail proportion of the bootstrap distribution on the wrong side of
     zero), using a larger resample count (5,000) than the headline CI
     computation (1,000) for more stable p-values in the tails.
  2. Applies Bonferroni correction (family-wise error rate) and
     Benjamini-Hochberg correction (false discovery rate) across all 36
     comparisons.
  3. Reports which comparisons survive each correction level.

Uses the same predictions cache as 10_delta_ci.py and the same RANDOM_STATE
so this is fully reproducible from data/processed/model_outputs/predictions.json.

Resampling is done at the PATIENT level (matching 09_modeling.py and
10_delta_ci.py), not the isolate level, for the same reason: >1 in 5
patients contribute multiple isolates, so isolate-level resampling would
understate resampling uncertainty and inflate apparent significance.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_JSON = Path('data/processed/model_outputs/predictions.json')
OUT_CSV = Path('tables/table4_multiple_testing_correction.csv')

with open(PRED_JSON) as f:
    preds = json.load(f)

N_BOOT = 5000
RANDOM_STATE = 123
ALPHA = 0.05

targets = ['MRSA', 'ESBL', 'CRE']
splits = ['split_temporal', 'split_random']
models = ['logreg', 'rf', 'xgb']

def two_sided_bootstrap_p(deltas):
    p_pos = (deltas <= 0).mean()
    p_neg = (deltas >= 0).mean()
    return min(2 * min(p_pos, p_neg), 1.0)

rows = []
for target in targets:
    for split in splits:
        for model in models:
            key_a = f"{target}__{split}__A__{model}"
            key_b = f"{target}__{split}__B__{model}"
            if key_a not in preds or key_b not in preds:
                continue
            ya, pa = np.array(preds[key_a]['y_test']), np.array(preds[key_a]['y_proba'])
            yb, pb = np.array(preds[key_b]['y_test']), np.array(preds[key_b]['y_proba'])
            assert len(ya) == len(yb) and np.array_equal(ya, yb), f"y_test mismatch for {target}/{split}/{model}"
            y = ya
            n = len(y)
            pid = np.array(preds[key_a]['study_id'])
            assert pid.tolist() == np.array(preds[key_b]['study_id']).tolist(), f"study_id mismatch for {target}/{split}/{model}"

            pid_to_rows = {}
            for i, p in enumerate(pid):
                pid_to_rows.setdefault(p, []).append(i)
            pid_list = np.array(list(pid_to_rows.keys()), dtype=object)
            n_pat = len(pid_list)

            rng = np.random.RandomState(RANDOM_STATE)
            auroc_deltas, auprc_deltas = [], []
            for _ in range(N_BOOT):
                sampled = pid_list[rng.randint(0, n_pat, n_pat)]
                idx = np.concatenate([pid_to_rows[p] for p in sampled])
                yt = y[idx]
                if len(np.unique(yt)) < 2:
                    continue
                pa_s, pb_s = pa[idx], pb[idx]
                auroc_deltas.append(roc_auc_score(yt, pb_s) - roc_auc_score(yt, pa_s))
                auprc_deltas.append(average_precision_score(yt, pb_s) - average_precision_score(yt, pa_s))
            auroc_deltas = np.array(auroc_deltas)
            auprc_deltas = np.array(auprc_deltas)

            for metric, deltas in [('AUROC', auroc_deltas), ('AUPRC', auprc_deltas)]:
                rows.append({
                    'target': target, 'split': split, 'model': model, 'metric': metric,
                    'delta_mean': round(float(deltas.mean()), 4),
                    'ci_lo': round(float(np.percentile(deltas, 2.5)), 4),
                    'ci_hi': round(float(np.percentile(deltas, 97.5)), 4),
                    'p_boot': round(float(two_sided_bootstrap_p(deltas)), 5),
                })

df = pd.DataFrame(rows)
m = len(df)
assert m == 36, f"expected 36 comparisons, got {m}"

df = df.sort_values('p_boot').reset_index(drop=True)
df['rank'] = np.arange(1, m + 1)

# Bonferroni (family-wise error rate control)
df['bonferroni_sig'] = df['p_boot'] < (ALPHA / m)

# Benjamini-Hochberg (false discovery rate control)
df['bh_threshold'] = df['rank'] / m * ALPHA
df['bh_pass_raw'] = df['p_boot'] <= df['bh_threshold']
sig_idx = df.index[df['bh_pass_raw']].tolist()
if sig_idx:
    k_max = max(sig_idx)
    df['bh_sig'] = df.index <= k_max
else:
    df['bh_sig'] = False

df = df.drop(columns=['bh_pass_raw'])
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print(df.to_string(index=False))
print(f"\nSaved: {OUT_CSV}")
print(f"\nOf {m} comparisons:")
print(f"  Nominal (uncorrected) 95% CI excludes zero: {(df['p_boot'] < ALPHA).sum()}")
print(f"  Bonferroni-significant (alpha={ALPHA}/{m}={ALPHA/m:.5f}): {df['bonferroni_sig'].sum()}")
print(f"  Benjamini-Hochberg FDR-significant (alpha={ALPHA}): {df['bh_sig'].sum()}")
print("\nRows surviving Benjamini-Hochberg FDR correction:")
print(df[df['bh_sig']].to_string(index=False))
print("\nRows surviving Bonferroni correction:")
print(df[df['bonferroni_sig']].to_string(index=False))
