"""
14_shap_cre_all_models.py
Cross-family SHAP comparison for the CRE Model B (Additional file 1, Fig. S4).
The Model B minus Model A AUROC differences for CRE were seen in more than
one model family, whereas the headline SHAP figure uses the XGBoost
pipeline. This script computes the same aggregated-SHAP analysis (same
aggregation-to-original-variable logic as 12_shap_analysis.py) for the
logistic-regression and random-forest Model B pipelines as well, to check
whether the note-derived features that rank highly are specific to one
model family or present across families, and produces the three-panel
comparison figure.
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

FIG_DIR = Path('figures/llm')
TAB_DIR = Path('tables/llm')
MODELS_DIR = Path('data/processed/fitted_models/llm')

struct = pd.read_parquet('data/processed/splits.parquet').reset_index(drop=True)
nlp = pd.read_parquet('data/processed/nlp_features_llm.parquet').reset_index(drop=True)
nlp_cols_to_use = [c for c in nlp.columns if c not in ('study_id', 'isolate_id', 'testing_date')]
data = pd.concat([struct, nlp[nlp_cols_to_use]], axis=1)

cohort_flag, label_col = 'in_CRE_target_cohort', 'label_CRE'
target_df = data[data[cohort_flag] & data[label_col].notna()].copy()
test_df = target_df[target_df['split_temporal'] == 'test']

def get_feature_names(ct):
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

results_summary = {}

for model_kind in ['logreg', 'rf', 'xgb']:
    key_b = f"CRE__split_temporal__B__{model_kind}"
    pkl_path = MODELS_DIR / f"{key_b}.pkl"
    if not pkl_path.exists():
        print(f"  Missing fitted pipeline: {pkl_path}, skipping {model_kind}.")
        continue
    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)
    pipe = saved['pipeline']
    prep = pipe.named_steps['prep']
    clf = pipe.named_steps['clf']

    # Use the FULL temporal-split test set (n=2,170), not a subsample,
    # since TreeExplainer/LinearExplainer are fast enough at this size and
    # subsampling would add ranking-stability noise.
    sample_df = test_df

    X_trans = prep.transform(sample_df)
    if hasattr(X_trans, 'toarray'):
        X_trans = X_trans.toarray()
    feature_names = get_feature_names(prep)
    if len(feature_names) != X_trans.shape[1]:
        feature_names = [f'f{i}' for i in range(X_trans.shape[1])]

    try:
        if model_kind in ('rf', 'xgb'):
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):
                sv = sv[1]
        else:
            explainer = shap.LinearExplainer(clf, X_trans)
            sv = explainer.shap_values(X_trans)
    except Exception as e:
        print(f"  Explainer failed for {model_kind} ({e}), using generic Explainer")
        explainer = shap.Explainer(clf.predict_proba, X_trans)
        sv_full = explainer(X_trans)
        sv = sv_full.values[..., 1] if sv_full.values.ndim == 3 else sv_full.values

    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]

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
        return fname

    groups = {}
    for i, fname in enumerate(feature_names):
        groups.setdefault(original_feature_of(fname), []).append(i)

    agg_importance = {}
    for orig_name, idxs in groups.items():
        row_sum = sv[:, idxs].sum(axis=1)
        agg_importance[orig_name] = float(np.abs(row_sum).mean())

    agg_df = pd.DataFrame(sorted(agg_importance.items(), key=lambda x: -x[1]),
                           columns=['feature', 'mean_abs_shap'])
    agg_df['rank'] = np.arange(1, len(agg_df) + 1)
    agg_df.head(20).to_csv(TAB_DIR / f'shap_top_features_CRE_{model_kind}.csv', index=False)

    device_rank = agg_df[agg_df['feature'] == 'device_ventilator_tracheal']
    organism_rank = agg_df[agg_df['feature'] == 'organism_base']
    print(f"\n=== CRE {model_kind.upper()} (n_test={len(sample_df)}) ===")
    print(agg_df.head(10).to_string(index=False))
    if len(device_rank):
        print(f"  device_ventilator_tracheal: rank {device_rank['rank'].values[0]} "
              f"of {len(agg_df)}, mean|SHAP|={device_rank['mean_abs_shap'].values[0]:.4f}")
    if len(organism_rank):
        print(f"  organism_base: rank {organism_rank['rank'].values[0]} "
              f"of {len(agg_df)}, mean|SHAP|={organism_rank['mean_abs_shap'].values[0]:.4f}")

    results_summary[model_kind] = {
        'n_test': len(sample_df),
        'device_ventilator_tracheal_rank': int(device_rank['rank'].values[0]) if len(device_rank) else None,
        'device_ventilator_tracheal_shap': float(device_rank['mean_abs_shap'].values[0]) if len(device_rank) else None,
        'organism_base_rank': int(organism_rank['rank'].values[0]) if len(organism_rank) else None,
        'organism_base_shap': float(organism_rank['mean_abs_shap'].values[0]) if len(organism_rank) else None,
        'top10': agg_df.head(10)[['feature', 'mean_abs_shap']].to_dict('records'),
    }

with open(TAB_DIR / 'shap_cre_cross_model_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY: does the device-exposure signal replicate across CRE model families?")
print("=" * 70)
for model_kind, r in results_summary.items():
    print(f"  {model_kind.upper():8s}: device_ventilator_tracheal rank {r['device_ventilator_tracheal_rank']}, "
          f"organism_base rank {r['organism_base_rank']}")

# ---------------------------------------------------------------------------
# Supplementary Figure S2: side-by-side SHAP top-10 bar charts for all three
# CRE model families, with the device-exposure flag and organism identity
# highlighted, so the cross-model comparison above is visually inspectable
# rather than only reported as ranks in a JSON summary.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
titles = {
    'logreg': 'Logistic Regression\n(A-vs-B AUROC delta: borderline, p~0.002-0.003)',
    'rf': 'Random Forest\n(A-vs-B AUROC delta: p~0.0004-0.0017, closest to surviving correction)',
    'xgb': 'XGBoost\n(A-vs-B AUROC delta: not significant, p~0.75)',
}
for ax, model_kind in zip(axes, ['logreg', 'rf', 'xgb']):
    if model_kind not in results_summary:
        continue
    top10 = results_summary[model_kind]['top10']
    feats = [t['feature'] for t in top10][::-1]
    vals = [t['mean_abs_shap'] for t in top10][::-1]
    colors = ['#d62728' if f == 'device_ventilator_tracheal' else ('#2ca02c' if f == 'organism_base' else '#4C72B0') for f in feats]
    ax.barh(feats, vals, color=colors)
    ax.set_title(titles[model_kind], fontsize=9)
    ax.set_xlabel('Mean |SHAP value|')
    ax.tick_params(axis='y', labelsize=8)

fig.suptitle(
    'Supplementary Figure S2: CRE SHAP feature importance (Model B) across all three model families\n'
    'Red = NLP-derived ventilator/device-exposure flag; Green = organism identity',
    fontsize=11, fontweight='bold',
)
plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig(FIG_DIR / 'figure4_supp_cre_cross_model_shap.png', dpi=150)
plt.close(fig)
print(f"\nSaved: {FIG_DIR / 'figure4_supp_cre_cross_model_shap.png'}")
