"""Figure 1: study design (cohort assembly, note extraction and validation, temporal validation and analysis).

All counts are read from the saved pipeline outputs; the validation-sample sizes are fixed design constants
(Methods, Physician validation of extraction). Writes figures/paper/Figure1_study_flow.png (600 dpi) and .pdf.
Run from the project root after scripts 03, 08, 08b and 23.
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Polygon

OUT = Path('figures/paper'); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'custom', 'mathtext.rm': 'DejaVu Sans',
                     'mathtext.it': 'DejaVu Sans:italic', 'mathtext.bf': 'DejaVu Sans:bold'})

# ------------------------------------------------------------------ counts
cc = json.load(open('tables/consort_counts.json'))
sp = json.load(open('tables/split_summary.json'))
dc = json.load(open('tables/supp/document_counts.json'))
dfx = json.load(open('tables/date_fix_summary.json'))
s = pd.read_parquet('data/processed/splits.parquet')
T = ['MRSA', 'ESBL', 'CRE']

def counts(d):
    out = {}
    for t in T:
        x = d[d[f'in_{t}_target_cohort'] == 1]; lab = x[f'label_{t}'].dropna()
        out[t] = (len(x), len(lab), int(lab.sum()))
    return out
ALL, TR, TE = counts(s), counts(s[s.split_temporal == 'train']), counts(s[s.split_temporal == 'test'])
spt, spr = sp['summary']['split_temporal'], sp['summary']['split_random']
excl_inelig = sum(cc[k] for k in ['1_excluded_missing_organism_name', '2_excluded_low_discrimination_or_unidentified',
                                  '3_excluded_organism_group_not_in_scope', '4_excluded_rare_species_lt20'])
assert cc['0_total_isolates_in_dataset'] - excl_inelig == cc['7_isolates_before_episode_dedup']
assert cc['7_isolates_before_episode_dedup'] - cc['8_excluded_same_episode_duplicate_isolates_within_7d'] == cc['9_final_cohort_size']
assert spt['train']['n_isolates'] + spt['test']['n_isolates'] + sp['n_isolates_excluded_straddling'] == cc['9_final_cohort_size']
VAL_BASE, VAL_ENRICH, VAL_SECOND, N_FLAGS = 120, 53, 24, 13          # study design constants
n = lambda v: f'{v:,}'
pct = lambda a, b: f'{100 * a / b:.1f}%'

# ------------------------------------------------------------------ drawing helpers (units: mm)
W, H = 170.0, 222.0
INK, MUTED, RULE = '#1D2733', '#5B6675', '#B8C1CC'
NAVY, NAVY_T = '#1F3B5C', '#E9EFF6'
C_A, C_LLM, C_RB, C_TEST = '#3B6FB6', '#C0392B', '#7F7F7F', '#E69F00'
EXC, EXC_T = '#9C3C3C', '#FBF0F0'
fig = plt.figure(figsize=(W / 25.4, H / 25.4)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')

def rbox(x, y, w, h, fc='white', ec=NAVY, lw=0.7, r=1.4, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))
def txt(x, y, s, fs=6.6, c=INK, w='normal', ha='left', va='center', st='normal', z=5, **k):
    ax.text(x, y, s, fontsize=fs, color=c, fontweight=w, ha=ha, va=va, style=st, zorder=z, **k)
def arrow(x1, y1, x2, y2, c='#4A4F57', lw=0.7, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=3, arrowprops=dict(arrowstyle='-|>,head_length=0.45,head_width=0.22',
                lw=lw, color=c, ls=ls, shrinkA=0, shrinkB=0))
def line(xs, ys, c='#4A4F57', lw=0.7, ls='-'):
    ax.plot(xs, ys, color=c, lw=lw, ls=ls, zorder=3, solid_capstyle='butt')
def panel(label, title, y):
    txt(4, y, label, fs=10, w='bold', va='baseline')
    txt(10, y, title, fs=8, w='bold', va='baseline')
    line([4, 166], [y - 3.2, y - 3.2], c=RULE, lw=0.6)
def header_box(x, y, w, h, head, color, hh=5.2, fs=6.9):
    rbox(x, y, w, h, fc='white', ec=color, lw=0.8)
    ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh, boxstyle='round,pad=0,rounding_size=1.4', fc=color, ec=color, lw=0.8, zorder=3))
    ax.add_patch(Rectangle((x, y + h - hh), w, hh / 2, fc=color, ec=color, lw=0, zorder=3))
    txt(x + w / 2, y + h - hh / 2, head, fs=fs, c='white', w='bold', ha='center')
def row(x1, x2, y, label, value, fs=6.5, bold_val=False, c=INK):
    txt(x1, y, label, fs=fs, c=c); txt(x2, y, value, fs=fs, ha='right', w='bold' if bold_val else 'normal', c=c)
def database_icon(cx, cy, w=4.2, h=5.2, c=NAVY):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=NAVY_T, ec=c, lw=0.6, zorder=4))
    for yy in (cy - h / 2, cy + h / 2, cy):
        ax.add_patch(Ellipse((cx, yy), w, 1.5, fc=NAVY_T if yy != cy + h / 2 else 'white', ec=c, lw=0.6, zorder=4))
def document_icon(x, y, w=6.5, h=8.2, c=NAVY):
    f = 1.8
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)], closed=True, fc='white', ec=c, lw=0.6, zorder=4))
    ax.add_patch(Polygon([(x + w - f, y + h), (x + w - f, y + h - f), (x + w, y + h - f)], closed=True, fc=NAVY_T, ec=c, lw=0.5, zorder=4))
    for i, yy in enumerate([y + h - 3.0, y + h - 4.5, y + h - 6.0, y + 1.3]):
        line([x + 1.1, x + w - (1.9 if i == 0 else 1.1)], [yy, yy], c=c, lw=0.45)
def person_icon(cx, cy, c):
    ax.add_patch(Ellipse((cx, cy + 1.3), 1.9, 1.9, fc=c, ec=c, lw=0, zorder=5))
    ax.add_patch(FancyBboxPatch((cx - 1.6, cy - 1.6), 3.2, 2.0, boxstyle='round,pad=0,rounding_size=0.9', fc=c, ec=c, lw=0, zorder=5))

# ================================================================== a  Cohort assembly
panel('a', 'Cohort assembly and resistance targets', 216.5)
MX, MW = 10, 84; MC = MX + MW / 2
EX, EW = 104, 62
# source
rbox(MX, 196, MW, 13.5, fc=NAVY_T)
database_icon(MX + 6.5, 202.7)
txt(MC + 4, 206.4, 'Isolates in the VITEK 2 laboratory export', fs=6.6, ha='center')
txt(MC + 4, 202.6, 'tested 1 Oct 2022 to 24 Dec 2025', fs=6.3, c=MUTED, ha='center')
txt(MC + 4, 198.6, f'n = {n(cc["0_total_isolates_in_dataset"])} isolates', fs=7.2, w='bold', ha='center')
# exclusions 1
arrow(MC, 196, MC, 182.5); line([MC, EX], [189.5, 189.5]); arrow(EX - 0.01, 189.5, EX, 189.5)
rbox(EX, 180.2, EW, 19, fc=EXC_T, ec=EXC)
txt(EX + 2, 196.4, f'Excluded: ineligible records (n = {n(excl_inelig)})', fs=6.6, w='bold', c=EXC)
for i, (lab, k) in enumerate([('Organism name missing', '1_excluded_missing_organism_name'),
                              ('Low-discrimination or unidentified ID', '2_excluded_low_discrimination_or_unidentified'),
                              ('Organism outside the five study groups', '3_excluded_organism_group_not_in_scope'),
                              ('Species with fewer than 20 isolates', '4_excluded_rare_species_lt20')]):
    row(EX + 3.4, EX + EW - 2, 192.6 - i * 3.3, lab, n(cc[k]), fs=6.2)
# eligible
rbox(MX, 172.5, MW, 10, fc=NAVY_T)
txt(MC, 179.2, 'Eligible isolates (five clinically actionable organism groups)', fs=6.5, ha='center')
txt(MC, 175.4, f'n = {n(cc["7_isolates_before_episode_dedup"])}', fs=7.2, w='bold', ha='center')
# exclusions 2
arrow(MC, 172.5, MC, 160.5); line([MC, EX], [166.5, 166.5]); arrow(EX - 0.01, 166.5, EX, 166.5)
rbox(EX, 161.5, EW, 10, fc=EXC_T, ec=EXC)
txt(EX + 2, 168.6, f'Repeat isolates removed (n = {n(cc["8_excluded_same_episode_duplicate_isolates_within_7d"])})', fs=6.6, w='bold', c=EXC)
txt(EX + 3.4, 164.5, 'same patient, organism and specimen within 7 days', fs=6.2, c=INK)
# analytic cohort
rbox(MX, 149, MW, 11.5, fc=NAVY, ec=NAVY)
txt(MC, 156.8, 'Analytic cohort', fs=7.6, w='bold', c='white', ha='center')
txt(MC, 152.4, f'{n(cc["9_final_cohort_size"])} isolates from {n(cc["9_final_unique_patients"])} patients', fs=7.2, c='white', ha='center')
# target cards
CW, CG = 50, 6; CX = [4, 4 + CW + CG, 4 + 2 * (CW + CG)]; CC = [x + CW / 2 for x in CX]
line([MC, MC], [149, 144.2]); line([CC[0], CC[2]], [144.2, 144.2])
for c in CC: arrow(c, 144.2, c, 140.6)
ORG = {'MRSA': r'$\mathit{Staphylococcus\ aureus}$', 'ESBL': r'$\mathit{Escherichia\ coli}$ and $\mathit{Klebsiella}$ spp.',
       'CRE': 'Enterobacterales'}
NAME = {'MRSA': 'MRSA', 'ESBL': 'ESBL', 'CRE': 'CRE'}
for t, x in zip(T, CX):
    header_box(x, 118, CW, 22.6, NAME[t], NAVY)
    txt(x + CW / 2, 131.6, ORG[t], fs=6.4, ha='center', c=MUTED)
    tot, lab, ev = ALL[t]
    row(x + 3, x + CW - 3, 127.0, 'Isolates in target cohort', n(tot))
    row(x + 3, x + CW - 3, 123.6, 'Labelled', n(lab))
    row(x + 3, x + CW - 3, 120.2, 'Resistant', f'{n(ev)} ({pct(ev, lab)})', bold_val=True)

# ================================================================== b  Note extraction and validation
panel('b', 'Note-derived predictors and physician validation', 111.5)
# documents
rbox(4, 65.5, 42, 38.5, fc=NAVY_T)
document_icon(21.7, 93.0)
txt(25, 89.0, 'Linked discharge summaries', fs=6.6, w='bold', ha='center')
row(7, 43, 84.8, 'Isolates with a note', n(dc['isolates_with_note']))
row(7, 43, 81.3, 'Distinct documents', n(dc['distinct_documents']))
row(7, 43, 77.8, 'Used in models', n(dfx['isolates_with_note_after_fix']), bold_val=True)
txt(7, 72.9, f'{dfx["notes_from_later_admission_removed"]} summaries written at a later\nadmission were excluded', fs=5.9, c=MUTED, va='center', linespacing=1.25)
txt(7, 68.0, 'De-identified before extraction', fs=5.9, c=MUTED)
# extractors
XX, XW = 53.5, 51
header_box(XX, 86.0, XW, 18, 'LLM extraction (primary)', C_LLM)
txt(XX + XW / 2, 94.9, 'Claude Sonnet 5; verbatim prompt with', fs=6.2, ha='center')
txt(XX + XW / 2, 91.5, 'written flag definitions and negation', fs=6.2, ha='center')
txt(XX + XW / 2, 88.1, 'rules; one JSON object per note', fs=6.2, ha='center')
header_box(XX, 65.5, XW, 18, 'Rule-based extraction (comparison)', C_RB)
txt(XX + XW / 2, 74.4, 'Curated term lists, named-antibiotic', fs=6.2, ha='center')
txt(XX + XW / 2, 71.0, 'dictionary and NegEx-style negation', fs=6.2, ha='center')
txt(XX + XW / 2, 67.6, 'window', fs=6.2, ha='center')
line([46, 49.8], [84.75, 84.75]); line([49.8, 49.8], [74.5, 95.0])
arrow(49.8, 95.0, XX, 95.0); arrow(49.8, 74.5, XX, 74.5)
# flags
FX, FW = 110, 56
rbox(FX, 65.5, FW, 38.5, fc='white', ec=NAVY, lw=0.8)
txt(FX + FW / 2, 100.4, f'{N_FLAGS} binary flags per note', fs=6.8, w='bold', ha='center')
txt(FX + 3, 96.0, 'Pre-admission history', fs=6.3, w='bold', c=NAVY)
for i, l in enumerate(['diabetes · chronic kidney disease or dialysis', 'malignancy · immunosuppression',
                       'cardiovascular or cerebrovascular disease', 'prior hospitalization']):
    txt(FX + 3, 92.8 - i * 2.95, l, fs=6.0)
txt(FX + 3, 80.6, 'Hospital course', fs=6.3, w='bold', c=NAVY)
for i, l in enumerate(['urinary catheter · central line · surgical drain', 'ventilation or tracheal tube · ICU stay',
                       'prior antibiotic exposure · recent surgery']):
    txt(FX + 3, 77.4 - i * 2.95, l, fs=6.0)
txt(FX + 3, 68.0, '+ note availability, note length, named-antibiotic indicator', fs=6.0, c=MUTED)
arrow(XX + XW, 95.0, FX, 95.0); arrow(XX + XW, 74.5, FX, 74.5)
# validation
rbox(4, 48.4, 162, 14.6, fc='#F6F7F9', ec=NAVY, lw=0.7)
txt(8, 59.4, 'Blinded physician validation (reference standard)', fs=6.7, w='bold')
txt(8, 56.0, f'{VAL_BASE} department-stratified random notes: all accuracy metrics', fs=6.2)
txt(8, 52.9, f'+ {VAL_ENRICH} notes enriched for six rare flags: error analysis only', fs=6.2)
txt(8, 50.0, 'Both extractors scored on the same notes; reviewers saw no extractor output', fs=5.9, c=MUTED)
person_icon(97, 57.6, NAVY); txt(101, 57.6, f'Primary reviewer: surgeon (MS), all {VAL_BASE + VAL_ENRICH} notes', fs=6.2)
person_icon(97, 51.8, C_RB); txt(101, 51.8, f'Second reviewer: radiologist, {VAL_SECOND} random notes', fs=6.2)

# ================================================================== c  Temporal validation and analysis
panel('c', 'Temporal validation and analysis', 41.6)
d0, d1 = pd.Timestamp('2022-10-01'), pd.Timestamp('2025-12-24')
X0, X1 = 4, 166
xd = lambda d: X0 + (pd.Timestamp(d) - d0).days / (d1 - d0).days * (X1 - X0)
cut = pd.Timestamp(sp['cutoff_date_temporal'])
yb, hb = 26.8, 5.2
ax.add_patch(Rectangle((xd(spt['train']['date_min']), yb), xd(spt['train']['date_max']) - xd(spt['train']['date_min']), hb,
                       fc=C_A, ec='none', zorder=3))
ax.add_patch(Rectangle((xd(spt['test']['date_min']), yb), xd(spt['test']['date_max']) - xd(spt['test']['date_min']), hb,
                       fc=C_TEST, ec='none', zorder=3))
txt((xd(spt['train']['date_min']) + xd(spt['train']['date_max'])) / 2, yb + hb / 2,
    f'Training (model development):  {n(spt["train"]["n_isolates"])} isolates  ·  {n(spt["train"]["n_patients"])} patients',
    fs=6.5, c='white', w='bold', ha='center')
tx_mid = (xd(spt['test']['date_min']) + xd(spt['test']['date_max'])) / 2
txt(xd(spt['test']['date_min']) - 2.0, yb + hb + 2.8, f'Test:  {n(spt["test"]["n_isolates"])} isolates · {n(spt["test"]["n_patients"])} patients',
    fs=6.4, c='#8A5A00', w='bold', ha='right')
line([xd(spt['test']['date_min']) - 1.6, tx_mid, tx_mid], [yb + hb + 2.8, yb + hb + 2.8, yb + hb], c='#8A5A00', lw=0.5)
line([xd(cut), xd(cut)], [yb - 2.2, yb + hb + 0.8], c=INK, lw=0.7, ls=(0, (2.2, 1.4)))
line([X0, X1], [yb - 1.2, yb - 1.2], c=MUTED, lw=0.5)
for d, lab in [('2022-10-01', 'Oct 2022'), ('2023-07-01', 'Jul 2023'), ('2024-04-01', 'Apr 2024'), ('2025-01-01', 'Jan 2025'),
               ('2025-06-28', 'cut-off 28 Jun 2025'), ('2025-12-24', 'Dec 2025')]:
    line([xd(d), xd(d)], [yb - 1.2, yb - 2.2], c=MUTED, lw=0.5)
    txt(xd(d), yb - 3.9, lab, fs=5.9, c=INK if 'cut' in lab else MUTED, w='bold' if 'cut' in lab else 'normal',
        ha='right' if lab == 'Dec 2025' else ('left' if lab == 'Oct 2022' else 'center'))
txt(X0, yb + hb + 2.8, 'Calendar-ordered, patient-exclusive split by date of first isolate', fs=6.2, c=INK)
txt(X0, yb - 7.6, f'{sp["n_isolates_excluded_straddling"]} later isolates of {sp["n_patients_with_excluded_straddling_isolates"]} training-period '
    f'patients excluded.  Sensitivity analysis: random patient-level split ({n(spr["train"]["n_isolates"])} / {n(spr["test"]["n_isolates"])} isolates).',
    fs=5.8, c=MUTED)
# events table
TX = 4; ty = 14.6
txt(TX, ty, 'Events / labelled', fs=6.2, w='bold')
txt(TX + 40, ty, 'Training', fs=6.2, w='bold', c=C_A, ha='right'); txt(TX + 67, ty, 'Test', fs=6.2, w='bold', c='#8A5A00', ha='right')
line([TX, TX + 67.5], [ty - 1.8, ty - 1.8], c=RULE, lw=0.5)
for i, t in enumerate(T):
    yy = ty - 4.6 - i * 3.3
    txt(TX, yy, t, fs=6.2, w='bold')
    _, l1, e1 = TR[t]; _, l2, e2 = TE[t]
    txt(TX + 40, yy, f'{n(e1)}/{n(l1)} ({pct(e1, l1)})', fs=6.0, ha='right')
    txt(TX + 67, yy, f'{n(e2)}/{n(l2)} ({pct(e2, l2)})', fs=6.0, ha='right')
# models and evaluation
BX, BW, BH, BY = 76, 24.5, 15.4, 1.6
header_box(BX, BY, BW, BH, 'Model A', C_A, hh=4.6, fs=6.6)
for i, l in enumerate(['structured', 'microbiology and', 'admission fields']):
    txt(BX + BW / 2, BY + 8.0 - i * 2.75, l, fs=5.8, ha='center')
txt(BX + BW + 2.4, BY + 6.2, '+', fs=8, w='bold', ha='center')
header_box(BX + BW + 4.8, BY, BW, BH, 'Model B', C_LLM, hh=4.6, fs=6.6)
for i, l in enumerate(['Model A', '+ 13 note flags', '+ 3 text covariates']):
    txt(BX + BW + 4.8 + BW / 2, BY + 8.0 - i * 2.75, l, fs=5.8, ha='center')
EX0 = BX + 2 * BW + 9.4
arrow(BX + 2 * BW + 4.9, BY + 6.2, EX0, BY + 6.2)
rbox(EX0, BY, 166 - EX0, BH, fc='#F6F7F9', ec=NAVY, lw=0.7)
for i, (l, b) in enumerate([('Test-set evaluation', True), ('LR · RF · XGBoost', False), ('AUROC · AUPRC', False),
                            ('calibration · net benefit', False), ('36 comparisons, corrected', False)]):
    txt((EX0 + 166) / 2, BY + BH - 2.4 - i * 2.7, l, fs=5.8 if not b else 6.0, w='bold' if b else 'normal', ha='center')

fig.savefig(OUT / 'Figure1_study_flow.png', dpi=600)
fig.savefig(OUT / 'Figure1_study_flow.pdf')
plt.close(fig)
print('Figure 1 written:', {t: ALL[t] for t in T}, 'train', TR, 'test', TE)
