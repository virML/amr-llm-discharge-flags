"""Supplementary Figs. S1 (CRE) and S2 (ESBL): decomposition of the Model B - Model A difference and analyses of the
temporal availability of note information. Reads tables/reviewer/sensitivity_contrasts_*.csv written by
29_decomposition_sensitivity.py. No figure titles (legends live in the supplement)."""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('figures/paper'); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5, 'legend.fontsize': 7.2,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})
FAM = {'logreg': 'Logistic regression', 'rf': 'Random forest', 'xgb': 'XGBoost'}
C = pd.concat([pd.read_csv(p) for p in sorted(glob.glob('tables/reviewer/sensitivity_contrasts_S*.csv'))], ignore_index=True)
VAR = [('S0_decomposition', 'A', 'B', 'Primary: Model B vs Model A'),
       ('S0_decomposition', 'A', 'AM', 'Text-derived covariates only (vs Model A)'),
       ('S0_decomposition', 'AM', 'B', 'LLM flags beyond these covariates'),
       ('S0_decomposition', 'A', 'A_LLM', 'LLM flags + note availability (vs Model A)'),
       ('S0_decomposition', 'AM', 'AM_hist', 'Pre-admission history flags* beyond covariates'),
       ('S0_decomposition', 'AM', 'AM_course', 'Hospital-course flags† beyond covariates'),
       ('S4_prospective_notes', 'A', 'B', 'Notes discharged before testing only'),
       ('S3_no_interval', 'A_noint', 'B_noint', 'Without admission-to-testing interval')]


def forest(target, fname):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.9), sharey=True)
    for ax, (d, lo, hi, lab) in zip(axes, [('dAUROC', 'dAUROC_lo', 'dAUROC_hi', 'Δ AUROC'), ('dAUPRC', 'dAUPRC_lo', 'dAUPRC_hi', 'Δ AUPRC')]):
        yt, yl = [], []
        for i, (scn, a, b, vl) in enumerate(VAR):
            for j, (fam, mk, col) in enumerate([('logreg', 's', '#6C8EBF'), ('rf', '^', '#7E9D4C'), ('xgb', 'o', '#C0392B')]):
                r = C[(C.scenario == scn) & (C.target == target) & (C.model == fam) & (C.from_set == a) & (C.to_set == b)]
                assert len(r) == 1, (scn, target, fam, a, b, len(r))
                r = r.iloc[0]
                yy = (len(VAR) - i) * 1.0 + (0.22 - 0.22 * j)
                ax.errorbar(r[d], yy, xerr=[[r[d] - r[lo]], [r[hi] - r[d]]], fmt=mk, ms=3.6, color=col, ecolor=col,
                            elinewidth=0.9, capsize=1.6, label=FAM[fam] if i == 0 else None)
            yt.append(len(VAR) - i); yl.append(vl)
        ax.axvline(0, color='#555555', lw=0.7)
        ax.axhline(len(VAR) - 5.5, color='#BBBBBB', lw=0.6, ls=':')
        ax.set_xlabel(lab)
    axes[0].set_yticks(yt); axes[0].set_yticklabels(yl)
    axes[0].get_yticklabels()[0].set_fontweight('bold')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.62, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / fname); plt.close(fig)


forest('CRE', 'FigureS1_CRE_decomposition.png')
forest('ESBL', 'FigureS2_ESBL_decomposition.png')
print('written')
