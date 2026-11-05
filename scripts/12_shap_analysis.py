"""
12_shap_analysis.py
SHAP interpretability for the best-performing model per
target (selected in 11_tables_figures.py), using the Model B (structured +
NLP) pipeline fitted on the temporal-split training set.
"""
import pandas as pd
import numpy as np
import json
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from pathlib import Path

warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

FIG_DIR = Path('figures')
TAB_DIR = Path('tables')
MODELS_DIR = Path('data/processed/fitted_models')

with open(TAB_DIR / 'best_model_per_target.json') as f:
    best_model = json.load(f)

struct = pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
nlp = pd.read_parquet('data/processed/nlp_features.parquet').reset_index(drop=True)
nlp_cols_to_use = [c for c in nlp.columns if c not in ('study_id', 'isolate_id', 'testing_date')]
data = pd.concat([struct, nlp[nlp_cols_to_use]], axis=1)

TARGETS = {
    'MRSA': ('in_MRSA_target_cohort', 'label_MRSA'),
    'ESBL': ('in_ESBL_target_cohort', 'label_ESBL'),
    'CRE': ('in_CRE_target_cohort', 'label_CRE'),
}

def get_feature_names(ct):
    """Extract post-transform feature names from a fitted ColumnTransformer."""
    names = []
    for name, trans, cols in ct.transformers_:
        if name == 'remainder':
            continue
        if hasattr(trans, 'get_feature_names_out'):
            try:
                fn = trans.get_feature_names_out(cols)
            except Exception:
                try:
                    fn = trans.get_feature_names_out()
                except Exception:
                    fn = list(cols)
            names.extend(list(fn))
        else:
            names.extend(cols)
    return names

for target, model_kind in best_model.items():
    print(f"\n=== SHAP for {target} (best model: {model_kind}) ===")
    key_b = f"{target}__split_temporal__B__{model_kind}"
    pkl_path = MODELS_DIR / f"{key_b}.pkl"
    if not pkl_path.exists():
        print(f"  Missing fitted pipeline: {pkl_path}, skipping.")
        continue
    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)
    pipe = saved['pipeline']
    prep = pipe.named_steps['prep']
    clf = pipe.named_steps['clf']

    cohort_flag, label_col = TARGETS[target]
    target_df = data[data[cohort_flag] & data[label_col].notna()].copy()
    test_df = target_df[target_df['split_temporal'] == 'test']
    # sample for SHAP speed/plot clarity if large
    sample_df = test_df.sample(n=min(500, len(test_df)), random_state=42)

    X_trans = prep.transform(sample_df)
    if hasattr(X_trans, 'toarray'):
        X_trans = X_trans.toarray()
    feature_names = get_feature_names(prep)
    # guard against length mismatch (e.g. numeric add_indicator adds cols dynamically)
    if len(feature_names) != X_trans.shape[1]:
        feature_names = [f'f{i}' for i in range(X_trans.shape[1])]

    try:
        if model_kind in ('rf', 'xgb'):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):
                sv = sv[1]  # positive class
        else:
            explainer = shap.LinearExplainer(clf, X_trans)
            sv = explainer.shap_values(X_trans)
    except Exception as e:
        print(f"  TreeExplainer/LinearExplainer failed ({e}), falling back to generic Explainer")
        explainer = shap.Explainer(clf.predict_proba, X_trans)
        sv_full = explainer(X_trans)
        sv = sv_full.values[..., 1] if sv_full.values.ndim == 3 else sv_full.values

    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]

    # supplementary: raw per-dummy-level beeswarm (standard SHAP output, but
    # note in text that rare-category one-hot levels can show high-variance
    # attributions -- this is why the PRIMARY reported figure aggregates to
    # the original-variable level, below)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, X_trans, feature_names=feature_names, show=False, max_display=20)
    plt.title(f'Supplementary: raw per-category SHAP beeswarm -- {target} '
              f'({model_kind.upper()}, Model B, temporal-split test set)', fontsize=10)
    plt.tight_layout()
    outpath_supp = FIG_DIR / f'figure4_supp_shap_beeswarm_{target}.png'
    plt.savefig(outpath_supp, dpi=150)
    plt.close()
    print(f"  Saved {outpath_supp}")

    # -------------------------------------------------------------------
    # PRIMARY: aggregate one-hot dummy levels back to their ORIGINAL
    # variable (sum signed SHAP across a group's columns per row, THEN take
    # mean(|.|) across rows) -- avoids a handful of tiny/rare categories
    # (e.g. a dept or ward with a few isolates) dominating the headline
    # figure purely due to high-variance, low-n tree splits.
    # -------------------------------------------------------------------
    feat_lists = saved['feature_lists']
    categorical_cols = feat_lists['categorical']
    numeric_cols = feat_lists['numeric']

    def original_feature_of(fname):
        if fname.startswith('missingindicator_'):
            return fname[len('missingindicator_'):] + ' (missing flag)'
        for c in sorted(categorical_cols, key=len, reverse=True):
            if fname.startswith(c + '_'):
                return c
        if fname in numeric_cols:
            return fname
        return fname  # binary / already-original

    groups = {}
    for i, fname in enumerate(feature_names):
        groups.setdefault(original_feature_of(fname), []).append(i)

    agg_importance = {}
    for orig_name, idxs in groups.items():
        row_sum = sv[:, idxs].sum(axis=1)
        agg_importance[orig_name] = float(np.abs(row_sum).mean())

    agg_df = pd.DataFrame(sorted(agg_importance.items(), key=lambda x: -x[1]),
                           columns=['feature', 'mean_abs_shap']).head(20)
    agg_df.to_csv(TAB_DIR / f'shap_top_features_{target}.csv', index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(agg_df['feature'][::-1], agg_df['mean_abs_shap'][::-1], color='#4C72B0')
    plt.xlabel('Mean |SHAP value| (aggregated across one-hot levels of the same variable)')
    plt.title(f'Figure 4. SHAP Feature Importance: {target}\n'
              f'(best model = {model_kind.upper()}, Model B, temporal-split test set)', fontsize=10)
    plt.tight_layout()
    outpath = FIG_DIR / f'figure4_shap_{target}.png'
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved {outpath}")
    print(agg_df.to_string(index=False))

print("\nSHAP analysis complete.")
