"""
10_delta_ci.py
Proper PAIRED bootstrap confidence interval on the Model A vs Model B delta.

Independent 95% CIs on AUROC_A and AUROC_B separately (computed in
09_modeling.py) are the wrong tool for testing whether the A-vs-B
difference itself is statistically distinguishable from zero, because A
and B are evaluated on the IDENTICAL test set and their errors are
correlated. This script resamples the same bootstrap indices for A and B
simultaneously (paired resampling) and reports the CI directly on the
delta, which is the statistically appropriate approach and typically much
tighter / more informative than eyeballing overlap between two independent
CIs.

Resampling is done at the PATIENT level (using each row's cached study_id),
not the isolate level: 20.8% of patients contribute more than one isolate
to these cohorts, so treating isolates as independent draws understates the
true resampling uncertainty. A patient is drawn with replacement and all of
their test-set isolates move together, exactly as in 09_modeling.py.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

PRED_JSON = Path('data/processed/model_outputs/llm/predictions.json')
OUT_CSV = Path('tables/llm/table4_paired_delta_ci.csv')

with open(PRED_JSON) as f:
    preds = json.load(f)

N_BOOT = 1000
RANDOM_STATE = 123

targets = ['MRSA', 'ESBL', 'CRE']
splits = ['split_temporal', 'split_random']
models = ['logreg', 'rf', 'xgb']

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
            row = {
                'target': target, 'split': split, 'model': model,
                'AUROC_delta_mean': round(float(auroc_deltas.mean()), 4),
                'AUROC_delta_CI_lo': round(float(np.percentile(auroc_deltas, 2.5)), 4),
                'AUROC_delta_CI_hi': round(float(np.percentile(auroc_deltas, 97.5)), 4),
                'AUROC_delta_excludes_zero': bool(np.percentile(auroc_deltas, 2.5) > 0 or np.percentile(auroc_deltas, 97.5) < 0),
                'AUPRC_delta_mean': round(float(auprc_deltas.mean()), 4),
                'AUPRC_delta_CI_lo': round(float(np.percentile(auprc_deltas, 2.5)), 4),
                'AUPRC_delta_CI_hi': round(float(np.percentile(auprc_deltas, 97.5)), 4),
                'AUPRC_delta_excludes_zero': bool(np.percentile(auprc_deltas, 2.5) > 0 or np.percentile(auprc_deltas, 97.5) < 0),
            }
            rows.append(row)

df = pd.DataFrame(rows)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(df.to_string(index=False))
print(f"\nSaved: {OUT_CSV}")

n_sig = df['AUROC_delta_excludes_zero'].sum() + df['AUPRC_delta_excludes_zero'].sum()
print(f"\nOf {2*len(df)} delta CIs computed (AUROC + AUPRC across all target/split/model combos), "
      f"{n_sig} exclude zero at the 95% level.")
