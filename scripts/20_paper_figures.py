"""Publication figures 2-7 and supplementary figures for the LLM-primary manuscript (no embedded figure titles; titles/legends live in the manuscript).
Inputs are the saved outputs of the final-pipeline runs (09/10/15c/17/18/19)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve
from pathlib import Path

OUT = Path('figures/paper'); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                     'legend.fontsize': 7.2, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
C_A, C_LLM, C_RB = '#3B6FB6', '#C0392B', '#7F7F7F'
TARGET_LABEL = {'MRSA': 'MRSA', 'ESBL': 'ESBL', 'CRE': 'CRE'}
FAM = {'logreg': 'Logistic regression', 'rf': 'Random forest', 'xgb': 'XGBoost'}
best = json.load(open('tables/llm/best_model_per_target.json'))
pred_llm = json.load(open('data/processed/model_outputs/llm/predictions.json'))
pred_rb = json.load(open('data/processed/model_outputs/predictions.json'))

# ---------------------------------------------------------------- Figure 1
# Figure 1 (study design) is drawn by 20a_figure1_study_design.py.

# ---------------------------------------------------------------- Figure 2: extraction validation (kappa, F1)
# All metrics from the 120-note probability sample, one confusion matrix per flag (19c_validation_probability_sample.py).
t3 = pd.read_csv('tables/validation_v2/validation_metrics_base120.csv')
t3 = t3[t3.analysis == 'final_reference_unweighted']
FLAG_LABEL = {'comorbid_diabetes': 'Diabetes', 'comorbid_ckd_dialysis': 'CKD / dialysis', 'comorbid_malignancy': 'Malignancy',
              'comorbid_immunosuppression': 'Immunosuppression', 'comorbid_cardio_cerebrovascular': 'Cardio-/cerebrovascular disease',
              'device_urinary_catheter': 'Urinary catheter', 'device_central_line': 'Central line',
              'device_ventilator_tracheal': 'Ventilation / tracheal tube', 'device_surgical_drain': 'Surgical drain', 'icu_stay': 'ICU stay',
              'recent_hospitalization': 'Prior hospitalization',
              'prior_antibiotic_any': 'Prior antibiotic exposure', 'recent_surgery': 'Recent surgery', 'POOLED_13': 'Pooled, all 13 flags'}
order = list(FLAG_LABEL.keys())
ypos = np.arange(len(order))[::-1].astype(float)
gap_after = order.index('POOLED_13') - 1  # visual gap before the pooled row
ypos = np.array([v + (0.8 if i <= gap_after else 0.0) for i, v in enumerate(ypos)])
fig, axes = plt.subplots(1, 2, figsize=(6.9, 4.8), sharey=True)
for ax, (metric, lab) in zip(axes, [('kappa', "Cohen's κ vs physician"), ('F1', 'F1 score vs physician')]):
    for j, (m, col, off, name) in enumerate([('rb', C_RB, -0.17, 'Rule-based'), ('llm', C_LLM, 0.17, 'LLM')]):
        sub = t3[t3.method == m].set_index('flag').loc[order]
        yv = ypos + off
        val = sub[metric].values; lo = sub[metric + '_lo'].values; hi = sub[metric + '_hi'].values
        ax.errorbar(val, yv, xerr=[val - lo, hi - val], fmt='o' if m == 'llm' else 's', ms=3.6, color=col, ecolor=col,
                    elinewidth=0.9, capsize=1.6, label=name)
    ax.axvline(0, color='#999999', lw=0.6)
    for xv in [0.4, 0.6, 0.8]:
        ax.axvline(xv, color='#DDDDDD', lw=0.6, zorder=0)
    ax.set_xlim(-0.2, 1.02); ax.set_xlabel(lab)
    ax.axhline((ypos[gap_after] + ypos[gap_after + 1]) / 2, color='#555555', lw=0.6)
axes[0].set_yticks(ypos); axes[0].set_yticklabels([FLAG_LABEL[f] for f in order])
for i, f in enumerate(order):
    if f.startswith('POOLED'): axes[0].get_yticklabels()[i].set_fontweight('bold')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.62, 1.03))
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / 'Figure2_extraction_validation.png'); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: ROC and PR curves
fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.9))
for c, t in enumerate(['MRSA', 'ESBL', 'CRE']):
    m = best[t]; k = lambda fs: f'{t}__split_temporal__{fs}__{m}'
    y = np.array(pred_llm[k('A')]['y_test'])
    assert pred_rb[k('B')]['y_test'] == pred_llm[k('A')]['y_test'] == pred_llm[k('B')]['y_test']
    series = [('Model A (structured features)', 'A', np.array(pred_llm[k('A')]['y_proba']), C_A, '-'),
              ('Model B with LLM-extracted flags', 'B-LLM', np.array(pred_llm[k('B')]['y_proba']), C_LLM, '-'),
              ('Model B with rule-based flags', 'B-RB', np.array(pred_rb[k('B')]['y_proba']), C_RB, '--')]
    for i, (name, short, p, col, ls) in enumerate(series):
        fpr, tpr, _ = roc_curve(y, p); axes[0, c].plot(fpr, tpr, color=col, ls=ls, lw=1.1, label=name)
        pr, rc_, _ = precision_recall_curve(y, p); axes[1, c].plot(rc_, pr, color=col, ls=ls, lw=1.1)
        axes[0, c].text(0.97, 0.05 + 0.075 * (2 - i), f'{short}  {roc_auc_score(y, p):.3f}', color=col, ha='right', va='bottom', fontsize=6.6, transform=axes[0, c].transAxes)
        axes[1, c].text(0.97, 0.80 + 0.065 * (2 - i), f'{short}  {average_precision_score(y, p):.3f}', color=col, ha='right', va='bottom', fontsize=6.6, transform=axes[1, c].transAxes)
    axes[0, c].text(0.97, 0.05 + 0.075 * 3, 'AUROC', ha='right', va='bottom', fontsize=6.6, fontweight='bold', transform=axes[0, c].transAxes)
    axes[1, c].text(0.97, 0.80 + 0.065 * 3, 'AUPRC', ha='right', va='bottom', fontsize=6.6, fontweight='bold', transform=axes[1, c].transAxes)
    axes[0, c].plot([0, 1], [0, 1], color='#BBBBBB', lw=0.7, ls=':')
    axes[1, c].axhline(y.mean(), color='#BBBBBB', lw=0.7, ls=':')
    axes[0, c].set_title(f'{t} ({FAM[m]}; n = {len(y)})')
    axes[0, c].set_xlabel('1 − specificity'); axes[0, c].set_ylabel('Sensitivity')
    axes[1, c].set_xlabel('Recall (sensitivity)'); axes[1, c].set_ylabel('Precision (PPV)')
    for r in range(2):
        axes[r, c].set_xlim(0, 1); axes[r, c].set_ylim(0, 1.02)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03), fontsize=7)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / 'Figure3_roc_pr.png'); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: where the CRE gain comes from (ablation)
ab = pd.read_csv('tables/obs/ablation_obs.csv')  # observed differences (point) with paired-bootstrap percentile CIs
VAR = [('B_all', 'All 13 LLM flags + note metadata'),
       ('B_history_only', 'History flags* + note metadata'),
       ('B_history_flags_no_length', 'History flags* + note availability'),
       ('B_recent_hosp_only', 'Prior hospitalization + note metadata'),
       ('B_course_only', 'Hospital-course flags† + note metadata'),
       ('B_note_metadata_only', 'Note metadata only (availability, length)')]
def forest(target, fname):
    sub = ab[ab.target == target]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.6), sharey=True)
    for ax, (d, lo, hi, lab) in zip(axes, [('dAUROC', 'dAUROC_lo', 'dAUROC_hi', 'Δ AUROC, Model B − Model A'),
                                         ('dAUPRC', 'dAUPRC_lo', 'dAUPRC_hi', 'Δ AUPRC, Model B − Model A')]):
        yt = []; yl = []
        for i, (v, vl) in enumerate(VAR):
            for j, (fam, mk, col) in enumerate([('logreg', 's', '#6C8EBF'), ('rf', '^', '#7E9D4C'), ('xgb', 'o', C_LLM)]):
                r = sub[(sub.feature_set == v) & (sub.model == fam)].iloc[0]
                yy = (len(VAR) - i) * 1.0 + (0.22 - 0.22 * j)
                ax.errorbar(r[d], yy, xerr=[[r[d] - r[lo]], [r[hi] - r[d]]], fmt=mk, ms=3.6, color=col, ecolor=col,
                            elinewidth=0.9, capsize=1.6, label=FAM[fam] if i == 0 else None)
            yt.append(len(VAR) - i); yl.append(vl)
        ax.axvline(0, color='#555555', lw=0.7)
        ax.set_xlabel(lab)
    axes[0].set_yticks(yt); axes[0].set_yticklabels(yl)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.62, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / fname); plt.close(fig)
forest('CRE', 'Figure4_CRE_ablation.png')
forest('ESBL', 'FigureS3_ESBL_ablation.png')

# ---------------------------------------------------------------- Figure 5: calibration before/after isotonic recalibration (Model B, LLM)
rc = json.load(open('data/processed/model_outputs/llm/recalibration_predictions.json'))
def calib_fig(fs, fname):
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    for c, t in enumerate(['MRSA', 'ESBL', 'CRE']):
        d = rc[f'{t}__{fs}__{best[t]}']; y = np.array(d['y_test'])
        ax = axes[c]; ax.plot([0, 1], [0, 1], color='#BBBBBB', lw=0.7, ls=':')
        brs = []
        for key_, col, mk, name in [('p_uncal', '#8E44AD', 'o', 'Before recalibration'), ('p_cal', '#16A085', 's', 'After isotonic recalibration')]:
            p = np.array(d[key_]); fp, mp = calibration_curve(y, p, n_bins=10, strategy='quantile')
            brs.append(float(np.mean((p - y) ** 2)))
            ax.plot(mp, fp, marker=mk, ms=3, lw=1.0, color=col, label=name)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(f'{t} ({FAM[best[t]]})\nBrier {brs[0]:.3f} before, {brs[1]:.3f} after', fontsize=8)
        ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed proportion' if c == 0 else '')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(OUT / fname); plt.close(fig)
calib_fig('B', 'Figure5_calibration_modelB_llm.png')
calib_fig('A', 'FigureS4_calibration_modelA.png')

# ---------------------------------------------------------------- Figure 6: SHAP for CRE Model B (LLM)
FEAT = {'ward_clean': 'Ward', 'dept_clean': 'Department', 'specimen_type_clean': 'Specimen type', 'note_length_chars': 'Note length',
        'admission_to_culture_hours': 'Admission-to-testing interval', 'prior_resistant_organism_flag': 'Prior resistant isolate',
        'prior_isolate_count': 'Number of prior isolates', 'infection_site_clean': 'Infection site', 'organism_base': 'Organism',
        'age_years': 'Age', 'gender_clean': 'Gender', 'onset_bucket': 'Admission-to-testing interval category', 'id_confidence_clean': 'ID confidence',
        'percent_probability_num': 'ID probability', 'days_since_last_isolate': 'Days since last isolate', 'is_pediatric_flag': 'Paediatric',
        'note_available': 'Note available', 'id_confidence_known': 'ID confidence recorded', 'prior_antibiotic_named': 'Named antibiotic mentioned (rule-based)'}
NLP = ['comorbid_diabetes', 'comorbid_ckd_dialysis', 'comorbid_malignancy', 'comorbid_immunosuppression', 'comorbid_cardio_cerebrovascular',
       'prior_antibiotic_any', 'device_urinary_catheter', 'device_central_line', 'device_ventilator_tracheal', 'device_surgical_drain',
       'icu_stay', 'recent_hospitalization', 'recent_surgery']
def nice(f):
    base = f.replace(' (missing flag)', '')
    lab = FLAG_LABEL.get(base, FEAT.get(base, base))
    if base in NLP: lab += ' (LLM)'
    if '(missing flag)' in f: lab += ' — missing'
    return lab
SCALE = {'logreg': 'log-odds', 'xgb': 'log-odds', 'rf': 'probability'}  # LinearExplainer/XGBoost TreeExplainer: margin; RF TreeExplainer: probability
def shap_bar(target, ax, n=15, fam=None, csv=None):
    fam = fam or best[target]
    s = pd.read_csv(csv or f'tables/llm/shap_top_features_{target}.csv').head(n)[::-1]
    cols = [C_LLM if (f.replace(' (missing flag)', '') in NLP or f.replace(' (missing flag)', '') in ('note_length_chars', 'note_available', 'prior_antibiotic_named')) else C_A for f in s.feature]
    ax.barh(range(len(s)), s.mean_abs_shap, color=cols, height=0.7)
    ax.set_yticks(range(len(s))); ax.set_yticklabels([nice(f) for f in s.feature]); ax.set_xlabel(f'Mean |SHAP value| ({SCALE[fam]})')
fig, ax = plt.subplots(figsize=(5.2, 3.9)); shap_bar('CRE', ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_A, label='Structured feature'), Patch(color=C_LLM, label='Note-derived feature')], loc='lower right', frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'Figure6_shap_CRE_modelB_llm.png'); plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.9))
for ax, t in zip(axes, ['ESBL', 'MRSA']):
    shap_bar(t, ax); ax.set_title(f'{t} ({FAM[best[t]]})')
axes[1].legend(handles=[Patch(color=C_A, label='Structured'), Patch(color=C_LLM, label='Note-derived')], loc='lower right', frameon=False)
fig.tight_layout(); fig.savefig(OUT / 'FigureS1_shap_ESBL_MRSA_modelB_llm.png'); plt.close(fig)
# Supplementary: CRE Model B (LLM) SHAP across the three model families
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.9))
for ax, m in zip(axes, ['logreg', 'rf', 'xgb']):
    shap_bar('CRE', ax, n=10, fam=m, csv=f'tables/llm/shap_top_features_CRE_{m}.csv'); ax.set_title(FAM[m])
fig.legend(handles=[Patch(color=C_A, label='Structured feature'), Patch(color=C_LLM, label='Note-derived feature')], loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(OUT / 'FigureS_shap_CRE_by_family_modelB_llm.png'); plt.close(fig)
# ---------------------------------------------------------------- Incremental value forest plot (main figure)
obs = json.load(open('tables/obs/core_obs.json'))
fig, axes = plt.subplots(1, 2, figsize=(6.9, 4.6), sharey=True)
rows_ = []
for t in ['MRSA', 'ESBL', 'CRE']:
    for m in ['logreg', 'rf', 'xgb']:
        rows_.append((t, m))
ypos = {}; yy = 0; yt = []; yl = []
for i, (t, m) in enumerate(rows_[::-1]):
    yy = i + (i // 3) * 0.8
    ypos[(t, m)] = yy; yt.append(yy); yl.append(f"{t}, {FAM[m]}" + (' *' if best[t] == m else ''))
for ax, metric, lab in [(axes[0], 'AUROC', 'Δ AUROC, Model B − Model A'), (axes[1], 'AUPRC', 'Δ AUPRC, Model B − Model A')]:
    for (t, m), y0 in ypos.items():
        for run, col, mk, off, name in [('llm', C_LLM, 'o', 0.17, 'LLM flags'), ('rb', C_RB, 's', -0.17, 'Rule-based flags')]:
            d = obs[f'delta_{run}'][f'{t}__split_temporal__{m}']
            v, lo, hi = d[f'd{metric}'], d[f'd{metric}_lo'], d[f'd{metric}_hi']
            ax.errorbar(v, y0 + off, xerr=[[v - lo], [hi - v]], fmt=mk, ms=3.6, color=col, ecolor=col, elinewidth=0.9, capsize=1.6,
                        label=name if (t, m) == ('MRSA', 'logreg') else None)
    ax.axvline(0, color='#555555', lw=0.7); ax.set_xlabel(lab)
    from matplotlib.ticker import MaxNLocator; ax.xaxis.set_major_locator(MaxNLocator(5))
axes[0].set_yticks(yt); axes[0].set_yticklabels(yl)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.6, 1.02))
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / 'Figure_incremental_value.png'); plt.close(fig)
print('figures written:', sorted(p.name for p in OUT.glob('*.png')))
