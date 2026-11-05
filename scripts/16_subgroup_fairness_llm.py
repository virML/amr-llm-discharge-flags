"""
16_subgroup_fairness.py
Pre-specified subgroup performance analysis for the BMC Medical Informatics
and Decision Making manuscript variant, addressing TRIPOD+AI Item 14
(fairness considerations) and Item 23a (performance estimates by subgroup).

For each target's best-discriminating Model B pipeline (temporal split,
model family read from tables/best_model_per_target.json -- never
hardcoded, so this always matches whichever family Table 4 reports), we
recompute AUROC/AUPRC with bootstrap 95% CIs overall and within two
pre-specified demographic subgroups: gender (male vs female; 'Unknown'
excluded --- see rationale in text) and age group (pediatric, <18y, vs
adult). This uses the exact same cached test-set predictions
(data/processed/model_outputs/llm/predictions.json) already used for Table 4,
so no model is refit and no number here can be inconsistent with the
main-text results.

Bootstrap resampling is done at the PATIENT level (using each row's cached
study_id), matching 09/10/13, for the same reason: within-patient isolates
are not independent draws.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

RANDOM_STATE = 42
N_BOOT = 1000
MIN_N_PER_GROUP = 30
MIN_POS_PER_GROUP = 10

struct = pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
nlp = pd.read_parquet('data/processed/nlp_features_llm.parquet').reset_index(drop=True)
nlp_cols_to_use = [c for c in nlp.columns if c not in ('study_id', 'isolate_id', 'testing_date')]
data = pd.concat([struct, nlp[nlp_cols_to_use]], axis=1)

with open('tables/llm/best_model_per_target.json') as f:
    best_model = json.load(f)

TARGETS = {
    'MRSA': ('in_MRSA_target_cohort', 'label_MRSA', best_model['MRSA']),
    'ESBL': ('in_ESBL_target_cohort', 'label_ESBL', best_model['ESBL']),
    'CRE':  ('in_CRE_target_cohort', 'label_CRE', best_model['CRE']),
}

with open('data/processed/model_outputs/llm/predictions.json') as f:
    preds = json.load(f)


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


rows = []
for target_name, (cohort_flag, label_col, model_kind) in TARGETS.items():
    target_df = data[data[cohort_flag] & data[label_col].notna()].copy()
    target_df[label_col] = target_df[label_col].astype(int)
    test_df = target_df[target_df['split_temporal'] == 'test'].reset_index(drop=True)

    key = f"{target_name}__split_temporal__B__{model_kind}"
    y_test = np.array(preds[key]['y_test'])
    y_proba = np.array(preds[key]['y_proba'])
    pid_test = np.array(preds[key]['study_id'])
    assert len(y_test) == len(test_df)
    assert (test_df[label_col].values == y_test).all()

    def add_row(subgroup_dim, subgroup_level, mask):
        n = int(mask.sum())
        n_pos = int(y_test[mask].sum())
        if n < MIN_N_PER_GROUP or n_pos < MIN_POS_PER_GROUP or (n - n_pos) < MIN_POS_PER_GROUP:
            rows.append({'target': target_name, 'model': model_kind, 'subgroup_dim': subgroup_dim,
                         'subgroup_level': subgroup_level, 'n': n, 'n_pos': n_pos,
                         'prevalence_pct': round(100 * n_pos / n, 1) if n else None,
                         'AUROC': None, 'AUROC_CI_lo': None, 'AUROC_CI_hi': None,
                         'AUPRC': None, 'AUPRC_CI_lo': None, 'AUPRC_CI_hi': None,
                         'note': 'insufficient n or events for stable estimate'})
            return
        auroc = roc_auc_score(y_test[mask], y_proba[mask])
        a_m, a_lo, a_hi = bootstrap_ci_clustered(y_test[mask], y_proba[mask], pid_test[mask], roc_auc_score)
        auprc = average_precision_score(y_test[mask], y_proba[mask])
        p_m, p_lo, p_hi = bootstrap_ci_clustered(y_test[mask], y_proba[mask], pid_test[mask], average_precision_score)
        rows.append({'target': target_name, 'model': model_kind, 'subgroup_dim': subgroup_dim,
                     'subgroup_level': subgroup_level, 'n': n, 'n_pos': n_pos,
                     'prevalence_pct': round(100 * n_pos / n, 1),
                     'AUROC': round(auroc, 4), 'AUROC_CI_lo': round(a_lo, 4) if a_lo else None,
                     'AUROC_CI_hi': round(a_hi, 4) if a_hi else None,
                     'AUPRC': round(auprc, 4), 'AUPRC_CI_lo': round(p_lo, 4) if p_lo else None,
                     'AUPRC_CI_hi': round(p_hi, 4) if p_hi else None, 'note': ''})

    # overall (identical to Table 4, included here as the reference row)
    add_row('overall', 'overall', np.ones(len(y_test), dtype=bool))

    # gender
    gender = test_df['gender_clean'].values
    add_row('gender', 'male', gender == 'Male')
    add_row('gender', 'female', gender == 'Female')
    n_unknown_gender = int((~np.isin(gender, ['Male', 'Female'])).sum())
    rows.append({'target': target_name, 'model': model_kind, 'subgroup_dim': 'gender',
                 'subgroup_level': 'unknown_excluded', 'n': n_unknown_gender, 'n_pos': None,
                 'prevalence_pct': None, 'AUROC': None, 'AUROC_CI_lo': None, 'AUROC_CI_hi': None,
                 'AUPRC': None, 'AUPRC_CI_lo': None, 'AUPRC_CI_hi': None,
                 'note': 'excluded from gender comparison (gender not recorded)'})

    # age group
    pediatric = test_df['is_pediatric_flag'].values.astype(bool)
    age_known = test_df['age_years'].notna().values
    add_row('age_group', 'pediatric_<18y', pediatric)
    add_row('age_group', 'adult_>=18y', (~pediatric) & age_known)   # recorded age >= 18 years
    add_row('age_group', 'age_not_recorded', (~pediatric) & (~age_known))  # no linked admission record (age, gender and ward missing)

out = pd.DataFrame(rows)
Path('tables/llm').mkdir(exist_ok=True)
out.to_csv('tables/llm/table_subgroup_fairness.csv', index=False)
print(out.to_string(index=False))
