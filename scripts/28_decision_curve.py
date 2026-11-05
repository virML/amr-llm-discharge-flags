"""28_decision_curve.py -- decision curve analysis (net benefit) for the headline CRE and ESBL models (temporal test set).
Uses the isotonic-recalibrated probabilities (15c), because net benefit requires calibrated risks.
Net benefit = TP/n - FP/n * pt/(1-pt); reference strategies: treat all, treat none.
Net-benefit difference (Model B minus Model A) at selected thresholds with patient-cluster bootstrap 95% CIs (seed 42, 1000).
Outputs tables/supp/decision_curve.csv, tables/supp/decision_curve_differences.csv, figures/paper/FigureS_decision_curve.png"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
rc = json.load(open('data/processed/model_outputs/llm/recalibration_predictions.json'))
pr = json.load(open('data/processed/model_outputs/llm/predictions.json'))
TH = np.round(np.arange(0.02, 0.61, 0.01), 2)
def nb(y, p, t):
    n = len(y); tp = np.sum((p >= t) & (y == 1)); fp = np.sum((p >= t) & (y == 0)); return tp / n - fp / n * t / (1 - t)
def nb_all(y, t): prev = y.mean(); return prev - (1 - prev) * t / (1 - t)
rows = []; drows = []
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.0))
for ax, (t, m, sel) in zip(axes, [('CRE', 'xgb', [0.10, 0.15, 0.20, 0.30]), ('ESBL', 'xgb', [0.20, 0.30, 0.40, 0.50])]):
    a = rc[f'{t}__A__{m}']; b = rc[f'{t}__B__{m}']; y = np.array(a['y_test']); assert np.array_equal(y, np.array(b['y_test']))
    pa = np.array(a['p_cal']); pb = np.array(b['p_cal']); pid = np.array(pr[f'{t}__split_temporal__A__{m}']['study_id'])
    for th in TH:
        rows.append(dict(target=t, threshold=th, nb_model_A=nb(y, pa, th), nb_model_B=nb(y, pb, th), nb_treat_all=nb_all(y, th), nb_treat_none=0.0))
    d = pd.DataFrame([r for r in rows if r['target'] == t])
    ax.plot(d.threshold, d.nb_treat_all, color='#999999', lw=0.9, ls='--', label='Treat all')
    ax.axhline(0, color='#555555', lw=0.7, label='Treat none')
    ax.plot(d.threshold, d.nb_model_A, color='#3B6FB6', lw=1.3, label='Model A')
    ax.plot(d.threshold, d.nb_model_B, color='#C0392B', lw=1.3, label='Model B (LLM)')
    ax.set_ylim(-0.02, max(0.05, y.mean() + 0.02)); ax.set_xlim(TH[0], 0.6); ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit'); ax.set_title(t)
    groups = {}; [groups.setdefault(p_, []).append(i) for i, p_ in enumerate(pid)]; keys = np.array(list(groups.keys()), dtype=object)
    rng = np.random.RandomState(42); boots = {th: [] for th in sel}
    for _ in range(1000):
        idx = np.concatenate([groups[k] for k in keys[rng.randint(0, len(keys), len(keys))]])
        for th in sel: boots[th].append(nb(y[idx], pb[idx], th) - nb(y[idx], pa[idx], th))
    for th in sel:
        dlt = nb(y, pb, th) - nb(y, pa, th)
        drows.append(dict(target=t, threshold=th, nb_A=nb(y, pa, th), nb_B=nb(y, pb, th), nb_treat_all=nb_all(y, th), delta_nb=dlt,
                          delta_lo=float(np.percentile(boots[th], 2.5)), delta_hi=float(np.percentile(boots[th], 97.5)),
                          net_tp_per_1000=1000 * dlt))
axes[0].legend(frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig('figures/paper/FigureS_decision_curve.png'); plt.close(fig)
pd.DataFrame(rows).to_csv('tables/supp/decision_curve.csv', index=False)
pd.DataFrame(drows).to_csv('tables/supp/decision_curve_differences.csv', index=False)
print(pd.DataFrame(drows).round(4).to_string(index=False))
