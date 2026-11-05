"""
11_tables_figures.py
Table 1 (cohort characteristics), CONSORT flow diagram, ROC/PR overlay
figures (Model A vs B), and calibration plots.
"""
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

FIG_DIR = Path('figures/llm')
TAB_DIR = Path('tables/llm')
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

# =============================================================================
# TABLE 1: Cohort characteristics
# =============================================================================
cohort = pd.read_parquet('data/processed/cohort_master.parquet')
sf = pd.read_parquet('data/processed/structured_features.parquet')

rows = []
rows.append(('N isolates (final cohort)', f"{len(cohort):,}"))
rows.append(('N unique patients', f"{cohort['study_id'].nunique():,}"))

age = sf['age_years']
rows.append(('Age, years -- mean (SD)', f"{age.mean():.1f} ({age.std():.1f})"))
rows.append(('Age, years -- median (IQR)', f"{age.median():.1f} ({age.quantile(.25):.1f}-{age.quantile(.75):.1f})"))
rows.append(('Age missing, n (%)', f"{age.isna().sum():,} ({100*age.isna().mean():.1f}%)"))

gen = sf['gender_clean'].value_counts()
for g, n in gen.items():
    rows.append((f'Gender -- {g}, n (%)', f"{n:,} ({100*n/len(sf):.1f}%)"))

rows.append(('Pediatric (<18y or coded child), n (%)', f"{sf['is_pediatric_flag'].sum():,} ({100*sf['is_pediatric_flag'].mean():.1f}%)"))

rows.append(('--- Ward (top 10) ---', ''))
for w, n in sf['ward_clean'].value_counts().head(10).items():
    rows.append((f'Ward -- {w}', f"{n:,} ({100*n/len(sf):.1f}%)"))

rows.append(('--- Department (top 10) ---', ''))
for d, n in sf['dept_clean'].value_counts().head(10).items():
    rows.append((f'Department -- {d}', f"{n:,} ({100*n/len(sf):.1f}%)"))

rows.append(('--- Organism group ---', ''))
for og, n in sf['organism_group'].value_counts().items():
    rows.append((f'Organism group -- {og}', f"{n:,} ({100*n/len(sf):.1f}%)"))

rows.append(('--- Specimen type (top 8) ---', ''))
for s, n in sf['specimen_type_clean'].value_counts().head(8).items():
    rows.append((f'Specimen -- {s}', f"{n:,} ({100*n/len(sf):.1f}%)"))

rows.append(('--- Onset (admission-to-culture) ---', ''))
for o, n in sf['onset_bucket'].value_counts().items():
    rows.append((f'Onset -- {o}', f"{n:,} ({100*n/len(sf):.1f}%)"))

rows.append(('Prior isolate for this patient, n (%)', f"{(sf['prior_isolate_count']>0).sum():,} ({100*(sf['prior_isolate_count']>0).mean():.1f}%)"))
if 'DISCHARGE_SUMMARY_CONTENT_DEID' in cohort.columns:
    _note_n = cohort['DISCHARGE_SUMMARY_CONTENT_DEID'].notna().sum()
    _note_val = f"{_note_n:,} ({100*_note_n/len(sf):.1f}%)"
else:
    _note_val = 'see NLP table'
rows.append(('Discharge-summary note available, n (%)', _note_val))

table1 = pd.DataFrame(rows, columns=['Characteristic', 'Value'])
table1.to_csv(TAB_DIR / 'table1_cohort_characteristics.csv', index=False)
print(f"Saved {TAB_DIR/'table1_cohort_characteristics.csv'}  ({len(table1)} rows)")

# =============================================================================
# CONSORT FLOW DIAGRAM
# =============================================================================
with open(TAB_DIR / 'consort_counts.json') as f:
    consort = json.load(f)

# Sequence of (main-flow box text, kind) with the exclusion reason that
# separates it from the PREVIOUS main box attached as `excl_text` (None for
# the first box, which has no preceding exclusion step).
steps = [
    (f"Total isolates in combined dataset\n(5 batch files)\nn = {consort['0_total_isolates_in_dataset']:,}", None, None),
    (f"After organism-name completeness check\nn = {consort['0_total_isolates_in_dataset'] - consort['1_excluded_missing_organism_name']:,}",
     f"Excluded: missing Organism Name\nn = {consort['1_excluded_missing_organism_name']:,}", 'main'),
    (f"After ID-confidence exclusion\nn = {consort['0_total_isolates_in_dataset'] - consort['1_excluded_missing_organism_name'] - consort['2_excluded_low_discrimination_or_unidentified']:,}",
     f"Excluded: ID Confidence = Low discrimination\nor Unidentified organism\nn = {consort['2_excluded_low_discrimination_or_unidentified']:,}", 'main'),
    (f"After organism-group restriction\nn = {consort['7_isolates_before_episode_dedup'] + consort['4_excluded_rare_species_lt20']:,}",
     f"Excluded: organism group not in scope\n(not Enterobacterales/Pseudomonas/\nAcinetobacter/Staph. aureus/Enterococcus)\nn = {consort['3_excluded_organism_group_not_in_scope']:,}", 'main'),
    (f"After rare-species restriction\nn = {consort['7_isolates_before_episode_dedup']:,}",
     f"Excluded: rare/ambiguous species\n(<20 total occurrences)\nn = {consort['4_excluded_rare_species_lt20']:,}", 'main'),
    (f"Isolates eligible for episode dedup\nn = {consort['7_isolates_before_episode_dedup']:,}",
     f"Excluded: no usable Testing Date or\npatient identifier\nn = {consort['5_excluded_no_testing_date'] + consort['6_excluded_no_patient_id']:,}", 'main'),
    (f"FINAL ANALYTIC COHORT\nn = {consort['9_final_cohort_size']:,} isolates\nfrom {consort['9_final_unique_patients']:,} unique patients",
     f"Excluded: same-episode duplicate isolate\n(same patient/organism/specimen, <7 days)\nn = {consort['8_excluded_same_episode_duplicate_isolates_within_7d']:,}", 'final'),
]

n_main = len(steps)
fig_h = 2.2 * n_main
fig, ax = plt.subplots(figsize=(9, fig_h))
ax.axis('off')
ax.set_xlim(0, 15)
ax.set_ylim(0, n_main * 10)

box_w, box_h = 6.0, 3.2
y_positions = [ (n_main - 1 - i) * 10 + 5 for i in range(n_main) ]

for i, (main_text, excl_text, kind) in enumerate(steps):
    y = y_positions[i]
    color = '#DCE6F1' if kind != 'final' else '#C6E0B4'
    box = FancyBboxPatch((1, y - box_h/2), box_w, box_h, boxstyle='round,pad=0.25',
                          facecolor=color, edgecolor='black', linewidth=1.2)
    ax.add_patch(box)
    ax.text(1 + box_w/2, y, main_text, ha='center', va='center', fontsize=8.5,
            fontweight='bold' if kind == 'final' else 'normal')

    if i > 0:
        prev_y = y_positions[i - 1]
        arrow_y_top = prev_y - box_h/2
        arrow_y_bot = y + box_h/2
        ax.annotate('', xy=(1 + box_w/2, arrow_y_bot), xytext=(1 + box_w/2, arrow_y_top),
                     arrowprops=dict(arrowstyle='-|>', color='black', lw=1.3))
        # exclusion box for the step we just came FROM (attached to steps[i-1]'s excl_text)
        prev_excl = steps[i - 1][1]
        if prev_excl:
            mid_y = (arrow_y_top + arrow_y_bot) / 2
            ax.annotate('', xy=(8.8, mid_y), xytext=(1 + box_w/2 + 0.1, mid_y),
                         arrowprops=dict(arrowstyle='-|>', color='#B00000', lw=1))
            ebox = FancyBboxPatch((9.0, mid_y - 1.6), 5.5, 3.2, boxstyle='round,pad=0.2',
                                   facecolor='#FCE4E4', edgecolor='#B00000', linewidth=0.8)
            ax.add_patch(ebox)
            ax.text(11.75, mid_y, prev_excl, ha='center', va='center', fontsize=7.3)

plt.title('Figure 1. Cohort Selection (CONSORT-style Flow Diagram)', fontsize=12, fontweight='bold', y=1.0)
plt.savefig(FIG_DIR / 'figure1_consort_flow.png', dpi=150)
plt.close()
print(f"Saved {FIG_DIR/'figure1_consort_flow.png'}")

# =============================================================================
# ROC / PR curves: Model A vs B overlay, XGBoost, temporal split, per target
# =============================================================================
with open('data/processed/model_outputs/llm/predictions.json') as f:
    preds = json.load(f)

results_df = pd.read_csv(TAB_DIR / 'table4_model_performance_full.csv')

targets = ['MRSA', 'ESBL', 'CRE']

# Model family per target is selected in 09_modeling.py by inner-validation
# AUROC on Feature Set A (never by test-set performance -- see that script's
# docstring), and written to tables/best_model_per_target.json. Read it here
# rather than recomputing from test-set numbers, so figure/SHAP/calibration
# generation can never silently drift back to test-set-based selection.
with open(TAB_DIR / 'best_model_per_target.json') as f:
    best_model_per_target = json.load(f)
print("Best model per target (inner-validation AUROC, Feature Set A, from 09_modeling.py):", best_model_per_target)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for col, target in enumerate(targets):
    model = best_model_per_target[target]
    key_a = f"{target}__split_temporal__A__{model}"
    key_b = f"{target}__split_temporal__B__{model}"

    ax_roc = axes[0, col]
    ax_pr = axes[1, col]
    for key, label, color in [(key_a, f'Model A (structured)', '#1f77b4'), (key_b, f'Model B (+NLP)', '#d62728')]:
        y = np.array(preds[key]['y_test'])
        p = np.array(preds[key]['y_proba'])
        fpr, tpr, _ = roc_curve(y, p)
        auroc = results_df[(results_df.target == target) & (results_df.split == 'split_temporal') &
                            (results_df.model == model) & (results_df.feature_set == key.split('__')[2])]['AUROC'].values[0]
        ax_roc.plot(fpr, tpr, label=f'{label} (AUROC={auroc:.3f})', color=color, lw=2)

        prec, rec, _ = precision_recall_curve(y, p)
        auprc = results_df[(results_df.target == target) & (results_df.split == 'split_temporal') &
                            (results_df.model == model) & (results_df.feature_set == key.split('__')[2])]['AUPRC'].values[0]
        ax_pr.plot(rec, prec, label=f'{label} (AUPRC={auprc:.3f})', color=color, lw=2)

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax_roc.set_xlabel('1 - Specificity (FPR)')
    ax_roc.set_ylabel('Sensitivity (TPR)')
    ax_roc.set_title(f'{target}: ROC ({model.upper()}, temporal split)')
    ax_roc.legend(loc='lower right', fontsize=8)

    prevalence = np.array(preds[key_a]['y_test']).mean()
    ax_pr.axhline(prevalence, color='gray', ls='--', lw=0.8, alpha=0.5, label=f'Prevalence={prevalence:.2f}')
    ax_pr.set_xlabel('Recall (Sensitivity)')
    ax_pr.set_ylabel('Precision (PPV)')
    ax_pr.set_title(f'{target}: Precision-Recall')
    ax_pr.legend(loc='upper right', fontsize=8)

plt.suptitle('Figure 2. ROC and Precision-Recall Curves: Model A vs Model B (best model per target, temporal split)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure2_roc_pr_curves.png', dpi=150)
plt.close()
print(f"Saved {FIG_DIR/'figure2_roc_pr_curves.png'}")

# =============================================================================
# Calibration plots (Model B, best model per target, temporal split)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for col, target in enumerate(targets):
    model = best_model_per_target[target]
    key_b = f"{target}__split_temporal__B__{model}"
    y = np.array(preds[key_b]['y_test'])
    p = np.array(preds[key_b]['y_proba'])
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy='quantile')
    ax = axes[col]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, marker='o', color='#d62728', lw=2, label=f'Model B ({model.upper()})')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(f'{target} calibration (temporal split)')
    ax.legend(fontsize=8)
plt.suptitle('Figure 3. Calibration Plots (Model B, best model per target)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'figure3_calibration.png', dpi=150)
plt.close()
print(f"Saved {FIG_DIR/'figure3_calibration.png'}")

print("\nDone.")
