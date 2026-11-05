"""
03_cohort_and_labels.py
Cohort construction + label triangulation on the de-identified master
dataset.

Produces:
  data/processed/cohort_master.parquet   -- one row per retained EPISODE
                                             (post dedup), with organism
                                             grouping, all 3 label columns
                                             (+ per-source sub-labels used
                                             for triangulation), CONSORT
                                             bookkeeping columns.
  tables/consort_counts.json             -- flow-diagram counts
  tables/label_triangulation.json        -- disagreement rates / kappa
  tables/table2_antibiogram.csv          -- resistance prevalence by organism

DATA-DRIVEN DEFINITION CHOICES (based on field completeness and phenotype
strings in the full export, without reference to outcomes):
 1. `ID Confidence` is non-null for only ~8% of isolates (2,098/25,483) in
    the FULL dataset. Requiring a
    non-null value would discard 92% of otherwise-good isolates for a
    field that is simply not always populated by the LIS export, not
    because those isolates are unreliable. We therefore EXCLUDE only
    isolates *explicitly* flagged 'Low discrimination' or 'Unidentified
    organism'; a missing ID Confidence value is not treated as exclusion.
 2. ESBL primary evidence: besides 'ESBL (CTX-M LIKE)' and 'ESBL
    (CLAVULANATE INHIBITED)', a third,
    more common phenotype string, 'EXTENDED SPECTRUM BETA-LACTAMASE'
    (5,618 occurrences, concentrated in E. coli/Klebsiella/Proteus --
    exactly the organisms this target concerns), is included as primary
    ESBL evidence. Separately,
    'ESBL (CLAVULANATE INHIBITED)' is observed almost exclusively in
    Acinetobacter/Pseudomonas in this dataset (not E. coli/Klebsiella),
    so it contributes negligibly to the ESBL target once the target
    cohort is restricted to E. coli/Klebsiella.
 3. CRE is restricted to Enterobacterales isolates only (standard
    CDC/CLSI definition -- carbapenem-resistant Acinetobacter/Pseudomonas
    are CRAB/CRPA, a distinct category, not CRE), even though a bare
    'CARBAPENEMASE' flag (without ESBL) appears mostly in Acinetobacter/
    Pseudomonas isolates in this data.
 4. For imipenem-based MIC cross-checks of CRE, Proteus/Providencia/
    Morganella are excluded from imipenem-only resistance calls (these
    genera have intrinsically elevated imipenem MICs unrelated to
    carbapenemase production, per CDC/CLSI guidance) -- meropenem is used
    as the primary MIC cross-check carbapenem for all Enterobacterales.
 5. CRE label priority: CDC/CLSI define CRE phenotypically, as Enterobacterales
    resistant to a carbapenem, of which carbapenemase production is one
    (common but not the only) mechanism -- porin loss combined with AmpC/
    ESBL production, or efflux overexpression, also produce a resistant
    phenotype without a positive carbapenemase call. The primary evidence
    for label_CRE is therefore the CLSI-breakpoint carbapenem MIC call
    (meropenem, or imipenem for non-Proteeae genera), i.e. the phenotype
    itself; the VITEK AES 'CARBAPENEMASE' mechanism flag is used only to
    fill in isolates with no usable MIC result. An isolate with a
    carbapenem-resistant MIC but an AES call that does not report
    carbapenemase production is still labelled CRE-positive: mechanism
    absence does not overturn a directly measured resistant phenotype.
"""
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

IN_PARQUET = Path('data/processed/deidentified_master.parquet')
OUT_PARQUET = Path('data/processed/cohort_master.parquet')
CONSORT_JSON = Path('tables/consort_counts.json')
TRIANG_JSON = Path('tables/label_triangulation.json')
TABLE2_CSV = Path('tables/table2_antibiogram.csv')

df = pd.read_parquet(IN_PARQUET)
n0 = len(df)
consort = {'0_total_isolates_in_dataset': int(n0)}

# ---------------------------------------------------------------------------
# A. Organism taxonomy mapping
# ---------------------------------------------------------------------------
ENTEROBACTERALES_PREFIXES = [
    'Esch.', 'K.', 'Klebsiella', 'Citro.', 'Proteus', 'Morg.', 'Ser.', 'Salm.',
    'Ent.cloacae', 'Ent.aerogenes', 'Prov.', 'Raou.', 'Cro.', 'Pantoea',
    'Plesio.', 'Ent.cloacae complex',
]
PSEUDOMONAS_PREFIXES = ['Ps.', 'Pseudomonas']
ACINETOBACTER_PREFIXES = ['Aci.', 'Acinetobacter']
STAPH_AUREUS_NAMES_CONTAINS = ['Staph.aureus', 'Coagulase pos.Staph']
ENTEROCOCCUS_PREFIXES = ['Entero.', 'Enterococcus']

def strip_suffix(name):
    """Remove a trailing '(MRSA)'/'(MDR)'/'(MDRO)'/'(CRE)'/'(VRE)' tag, return (base, suffix)."""
    if not isinstance(name, str):
        return name, None
    m = re.search(r'\((MRSA|MDR|MDRO|CRE|VRE)\)\s*$', name)
    if m:
        return name[:m.start()].strip(), m.group(1)
    return name.strip(), None

def organism_group(base_name):
    if not isinstance(base_name, str) or base_name == '':
        return 'Unknown'
    for p in ENTEROBACTERALES_PREFIXES:
        if base_name.startswith(p):
            return 'Enterobacterales'
    for p in PSEUDOMONAS_PREFIXES:
        if base_name.startswith(p):
            return 'Pseudomonas'
    for p in ACINETOBACTER_PREFIXES:
        if base_name.startswith(p):
            return 'Acinetobacter'
    for s in STAPH_AUREUS_NAMES_CONTAINS:
        if base_name.startswith(s):
            return 'Staph_aureus'
    for p in ENTEROCOCCUS_PREFIXES:
        if base_name.startswith(p):
            return 'Enterococcus'
    return 'Other_excluded'

base_and_suffix = df['Organism Name'].apply(strip_suffix)
df['organism_base'] = base_and_suffix.apply(lambda x: x[0])
df['organism_name_suffix'] = base_and_suffix.apply(lambda x: x[1])
df['organism_group'] = df['organism_base'].apply(organism_group)

# fine species label for E.coli / Klebsiella (needed for the ESBL target cohort)
def species_label(base_name):
    if not isinstance(base_name, str):
        return None
    if base_name.startswith('Esch.coli'):
        return 'E_coli'
    if base_name.startswith('K.pneum') or base_name.startswith('K.oxytoca') or base_name == 'Klebsiella spp':
        return 'Klebsiella_spp'
    return None
df['species_ecoli_klebsiella'] = df['organism_base'].apply(species_label)

print("Organism group counts:\n", df['organism_group'].value_counts())

# ---------------------------------------------------------------------------
# B. Inclusion / exclusion flow (CONSORT)
# ---------------------------------------------------------------------------
# Step 1: non-missing Organism Name
step1 = df['Organism Name'].notna()
consort['1_excluded_missing_organism_name'] = int((~step1).sum())
df1 = df[step1].copy()

# Step 2: ID Confidence exclusion -- ONLY exclude explicit Low discrimination /
# Unidentified organism (see documented deviation #1 above)
bad_conf = df1['ID Confidence'].isin(['Low discrimination', 'Unidentified organism'])
consort['2_excluded_low_discrimination_or_unidentified'] = int(bad_conf.sum())
df2 = df1[~bad_conf].copy()

# Step 3: organism group must be one of the 5 clinically actionable groups
in_group = df2['organism_group'].isin(['Enterobacterales', 'Pseudomonas', 'Acinetobacter', 'Staph_aureus', 'Enterococcus'])
consort['3_excluded_organism_group_not_in_scope'] = int((~in_group).sum())
df3 = df2[in_group].copy()

# Step 4: drop rare/ambiguous species (<20 total occurrences) WITHIN the retained groups
name_counts = df3['Organism Name'].value_counts()
rare_names = name_counts[name_counts < 20].index.tolist()
is_rare = df3['Organism Name'].isin(rare_names)
consort['4_excluded_rare_species_lt20'] = int(is_rare.sum())
consort['4_rare_species_list'] = rare_names
df4 = df3[~is_rare].copy()

# Step 5: must have a usable Testing Date
testing_dt = pd.to_datetime(df4['Testing Date'], format='%d/%m/%y', errors='coerce')
df4['testing_date'] = testing_dt
has_date = testing_dt.notna()
consort['5_excluded_no_testing_date'] = int((~has_date).sum())
df5 = df4[has_date].copy()

# Step 6: must have a usable study_id (patient id)
has_pid = df5['study_id'].notna()
consort['6_excluded_no_patient_id'] = int((~has_pid).sum())
df6 = df5[has_pid].copy()

consort['7_isolates_before_episode_dedup'] = int(len(df6))

# Step 7: collapse same-episode duplicate isolates: same patient + same
# organism_base + same Specimen Type, testing dates within 7 days of each
# other -> keep the EARLIEST isolate in each such cluster (the one that
# would have triggered the empiric-therapy decision).
df6 = df6.sort_values(['study_id', 'organism_base', 'Specimen Type', 'testing_date'])

def mark_episode_keep(group):
    keep = np.ones(len(group), dtype=bool)
    dates = group['testing_date'].values
    last_kept_date = dates[0]
    for i in range(1, len(group)):
        gap_days = (dates[i] - last_kept_date) / np.timedelta64(1, 'D')
        if gap_days < 7:
            keep[i] = False
        else:
            last_kept_date = dates[i]
    return pd.Series(keep, index=group.index)

keep_mask = df6.groupby(['study_id', 'organism_base', 'Specimen Type'], group_keys=False).apply(mark_episode_keep)
keep_mask = keep_mask.reindex(df6.index).fillna(True).astype(bool)
n_collapsed = int((~keep_mask).sum())
consort['8_excluded_same_episode_duplicate_isolates_within_7d'] = n_collapsed
df7 = df6[keep_mask].copy()

consort['9_final_cohort_size'] = int(len(df7))
consort['9_final_unique_patients'] = int(df7['study_id'].nunique())

print("\n=== CONSORT FLOW ===")
for k, v in consort.items():
    if k != '4_rare_species_list':
        print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# C. MIC parsing + CLSI breakpoint interpretation
# ---------------------------------------------------------------------------
def parse_mic(raw):
    """Return a float MIC (using the reported boundary for censored values) or NaN."""
    if not isinstance(raw, str):
        return np.nan
    s = raw.strip()
    if s == '' or s == 'TRM' or s.lower() == 'nan':
        return np.nan
    s = s.replace('<=', '').replace('>=', '').replace('<', '').replace('>', '')
    try:
        return float(s)
    except ValueError:
        return np.nan

# (S_cutoff, R_cutoff): S if MIC<=S_cutoff, R if MIC>=R_cutoff, else I
# Values per CLSI M100 (breakpoints unchanged for these drug/organism
# combinations since the 2010 CLSI revision, through M100 Ed35, Jan 2025).
CLSI_BREAKPOINTS = {
    'MEM-Meropenem': (1, 4),
    'IPM-Imipenem': (1, 4),
    'CRO-Ceftriaxone': (1, 4),
    'CTX-Cefotaxime': (1, 4),
    'CAZ-Ceftazidime': (4, 16),
    'OX1-Oxacillin': (2, 4),
}

def sir_from_mic(series, s_cut, r_cut):
    mic = series.apply(parse_mic)
    out = pd.Series(np.nan, index=series.index, dtype=object)
    out[mic <= s_cut] = 'S'
    out[mic >= r_cut] = 'R'
    out[(mic > s_cut) & (mic < r_cut)] = 'I'
    return out

for abx, (s_cut, r_cut) in CLSI_BREAKPOINTS.items():
    if abx in df7.columns:
        df7[f'SIR_{abx}'] = sir_from_mic(df7[abx], s_cut, r_cut)

# ---------------------------------------------------------------------------
# D. Label triangulation
# ---------------------------------------------------------------------------
family_cols = [c for c in df7.columns if c.startswith('Family -')]
beta_lactam_cols = [c for c in df7.columns if c.startswith('Family -BETA-LACTAMS')]

def any_value_in(row_df_subset, values):
    return row_df_subset.isin(values).any(axis=1)

triang = {}

# --- MRSA (target cohort = Staph_aureus group only) -------------------------
staph = df7[df7['organism_group'] == 'Staph_aureus'].copy()

primary_mrsa = staph['OXSF-Cefoxitin Screen'].map({'POS': 1, 'NEG': 0})
mecA_flag = any_value_in(staph[beta_lactam_cols], ['MODIFICATION OF PBP (mecA)']).astype(int)
mecA_flag = mecA_flag.where(staph[beta_lactam_cols].notna().any(axis=1), np.nan)  # keep NaN if no mechanism data at all
oxa_sir = staph.get('SIR_OX1-Oxacillin')
oxa_cross = oxa_sir.map({'R': 1, 'I': 0, 'S': 0}) if oxa_sir is not None else pd.Series(np.nan, index=staph.index)
name_suffix_mrsa = staph['organism_name_suffix'].isin(['MRSA']).astype(int)

# final MRSA label: primary (cefoxitin) if available; else mecA flag; else oxacillin; else name suffix
mrsa_label = primary_mrsa.copy()
mrsa_label = mrsa_label.where(mrsa_label.notna(), mecA_flag)
mrsa_label = mrsa_label.where(mrsa_label.notna(), oxa_cross)
mrsa_label = mrsa_label.where(mrsa_label.notna(), name_suffix_mrsa.where(name_suffix_mrsa == 1))  # only use suffix to confirm positives, not to assert negatives
staph['label_MRSA'] = mrsa_label
staph['label_MRSA_source_primary'] = primary_mrsa
staph['label_MRSA_source_mecA'] = mecA_flag
staph['label_MRSA_source_oxacillin'] = oxa_cross
staph['label_MRSA_source_nametag'] = name_suffix_mrsa

def agreement_and_kappa(a, b):
    both = a.notna() & b.notna()
    n = int(both.sum())
    if n == 0:
        return {'n_compared': 0, 'pct_agree': None, 'kappa': None}
    aa, bb = a[both].astype(int), b[both].astype(int)
    agree = float((aa == bb).mean() * 100)
    try:
        kappa = float(cohen_kappa_score(aa, bb)) if aa.nunique() > 1 or bb.nunique() > 1 else 1.0
    except Exception:
        kappa = None
    return {'n_compared': n, 'pct_agree': round(agree, 2), 'kappa': round(kappa, 3) if kappa is not None else None}

triang['MRSA_primary_cefoxitin_vs_mecA'] = agreement_and_kappa(primary_mrsa, mecA_flag)
triang['MRSA_primary_cefoxitin_vs_oxacillin'] = agreement_and_kappa(primary_mrsa, oxa_cross)
triang['MRSA_n_labeled_positive'] = int((mrsa_label == 1).sum())
triang['MRSA_n_labeled_negative'] = int((mrsa_label == 0).sum())
triang['MRSA_n_unlabeled'] = int(mrsa_label.isna().sum())
triang['MRSA_target_cohort_n'] = int(len(staph))

# --- ESBL (target cohort = E. coli / Klebsiella only) -----------------------
ek = df7[df7['species_ecoli_klebsiella'].notna()].copy()

esbl_strings_primary = ['ESBL (CTX-M LIKE)', 'EXTENDED SPECTRUM BETA-LACTAMASE']
esbl_strings_secondary_included = ['ESBL (CLAVULANATE INHIBITED)']  # rare in E.coli/Kleb but included
primary_esbl = any_value_in(ek[beta_lactam_cols], esbl_strings_primary + esbl_strings_secondary_included).astype(int)
primary_esbl = primary_esbl.where(ek[beta_lactam_cols].notna().any(axis=1), np.nan)

ceph_cols = [c for c in ['SIR_CTX-Cefotaxime', 'SIR_CRO-Ceftriaxone', 'SIR_CAZ-Ceftazidime'] if c in ek.columns]
ceph_any_R = ek[ceph_cols].eq('R').any(axis=1) if ceph_cols else pd.Series(False, index=ek.index)
ceph_any_tested = ek[ceph_cols].notna().any(axis=1) if ceph_cols else pd.Series(False, index=ek.index)
ceph_cross = ceph_any_R.astype(int).where(ceph_any_tested, np.nan)

esbl_label = primary_esbl.copy()
esbl_label = esbl_label.where(esbl_label.notna(), ceph_cross)
ek['label_ESBL'] = esbl_label
ek['label_ESBL_source_primary'] = primary_esbl
ek['label_ESBL_source_cephalosporin'] = ceph_cross

triang['ESBL_primary_vs_3rdgen_cephalosporin'] = agreement_and_kappa(primary_esbl, ceph_cross)
triang['ESBL_n_labeled_positive'] = int((esbl_label == 1).sum())
triang['ESBL_n_labeled_negative'] = int((esbl_label == 0).sum())
triang['ESBL_n_unlabeled'] = int(esbl_label.isna().sum())
triang['ESBL_target_cohort_n'] = int(len(ek))

# --- CRE (target cohort = Enterobacterales only) -----------------------------
entb = df7[df7['organism_group'] == 'Enterobacterales'].copy()

primary_cre = any_value_in(entb[beta_lactam_cols], ['CARBAPENEMASE (+ OR - ESBL)', 'CARBAPENEMASE']).astype(int)
primary_cre = primary_cre.where(entb[beta_lactam_cols].notna().any(axis=1), np.nan)

mem_sir = entb.get('SIR_MEM-Meropenem')
mem_cross = mem_sir.map({'R': 1, 'I': 0, 'S': 0}) if mem_sir is not None else pd.Series(np.nan, index=entb.index)

# imipenem cross-check, EXCLUDING Proteus/Providencia/Morganella (intrinsic elevated MIC, not carbapenemase)
proteeae_prefixes = ('Proteus', 'Prov.', 'Morg.')
is_proteeae = entb['organism_base'].fillna('').str.startswith(proteeae_prefixes)
ipm_sir = entb.get('SIR_IPM-Imipenem')
ipm_cross = ipm_sir.map({'R': 1, 'I': 0, 'S': 0}) if ipm_sir is not None else pd.Series(np.nan, index=entb.index)
ipm_cross = ipm_cross.where(~is_proteeae, np.nan)

carbapenem_cross = mem_cross.where(mem_cross.notna(), ipm_cross)

# PRIMARY evidence = phenotypic carbapenem resistance by CLSI MIC breakpoint
# (the standard CRE definition); the AES carbapenemase mechanism flag fills
# in only isolates with no usable carbapenem MIC. See documented deviation
# #5 above -- a negative/absent carbapenemase call does not override a
# directly measured resistant MIC, since carbapenem resistance in
# Enterobacterales also arises through non-carbapenemase mechanisms.
cre_label = carbapenem_cross.copy()
cre_label = cre_label.where(cre_label.notna(), primary_cre)
entb['label_CRE'] = cre_label
entb['label_CRE_source_primary'] = carbapenem_cross
entb['label_CRE_source_carbapenemase_mechanism'] = primary_cre

triang['CRE_primary_MIC_vs_carbapenemase_mechanism'] = agreement_and_kappa(carbapenem_cross, primary_cre)
triang['CRE_n_labeled_positive'] = int((cre_label == 1).sum())
triang['CRE_n_labeled_negative'] = int((cre_label == 0).sum())
triang['CRE_n_unlabeled'] = int(cre_label.isna().sum())
triang['CRE_target_cohort_n'] = int(len(entb))
triang['CRE_n_proteeae_imipenem_excluded_as_intrinsic'] = int(is_proteeae.sum())

print("\n=== LABEL TRIANGULATION ===")
print(json.dumps(triang, indent=2))

# ---------------------------------------------------------------------------
# E. Merge target labels back onto the master cohort table (isolate-level)
# ---------------------------------------------------------------------------
df7['label_MRSA'] = np.nan
df7.loc[staph.index, 'label_MRSA'] = staph['label_MRSA']
df7['label_ESBL'] = np.nan
df7.loc[ek.index, 'label_ESBL'] = ek['label_ESBL']
df7['label_CRE'] = np.nan
df7.loc[entb.index, 'label_CRE'] = entb['label_CRE']

df7['in_MRSA_target_cohort'] = df7.index.isin(staph.index)
df7['in_ESBL_target_cohort'] = df7.index.isin(ek.index)
df7['in_CRE_target_cohort'] = df7.index.isin(entb.index)

# Also persist each target's per-source evidence columns (dropped from the
# per-group working frames above otherwise) -- needed for label-definition
# sensitivity analyses (e.g. restricting ESBL to isolates with a direct AES
# phenotype call, excluding cephalosporin-fallback-derived labels; see
# 17_sensitivity_analyses.py).
df7['label_MRSA_source_primary'] = np.nan
df7.loc[staph.index, 'label_MRSA_source_primary'] = staph['label_MRSA_source_primary']
df7['label_ESBL_source_primary'] = np.nan
df7.loc[ek.index, 'label_ESBL_source_primary'] = ek['label_ESBL_source_primary']
df7['label_ESBL_source_cephalosporin'] = np.nan
df7.loc[ek.index, 'label_ESBL_source_cephalosporin'] = ek['label_ESBL_source_cephalosporin']
df7['label_CRE_source_primary'] = np.nan
df7.loc[entb.index, 'label_CRE_source_primary'] = entb['label_CRE_source_primary']
df7['label_CRE_source_carbapenemase_mechanism'] = np.nan
df7.loc[entb.index, 'label_CRE_source_carbapenemase_mechanism'] = entb['label_CRE_source_carbapenemase_mechanism']

OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df7.to_parquet(OUT_PARQUET, index=False)
print(f"\nSaved cohort master: {OUT_PARQUET} shape={df7.shape}")

with open(CONSORT_JSON, 'w') as f:
    json.dump(consort, f, indent=2, default=str)
with open(TRIANG_JSON, 'w') as f:
    json.dump(triang, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# F. Table 2 -- resistance prevalence / local antibiogram
# ---------------------------------------------------------------------------
rows = []
for grp_name, sub, label_col in [
    ('MRSA (Staph. aureus)', staph, 'label_MRSA'),
    ('ESBL (E. coli/Klebsiella)', ek, 'label_ESBL'),
    ('CRE (Enterobacterales)', entb, 'label_CRE'),
]:
    for org, orgsub in sub.groupby('organism_base'):
        lab = orgsub[label_col]
        n_tested = int(lab.notna().sum())
        if n_tested == 0:
            continue
        n_pos = int((lab == 1).sum())
        rows.append({
            'target': grp_name, 'organism': org, 'n_isolates': int(len(orgsub)),
            'n_labeled': n_tested, 'n_positive': n_pos,
            'pct_positive_of_labeled': round(100 * n_pos / n_tested, 1),
        })
table2 = pd.DataFrame(rows).sort_values(['target', 'n_isolates'], ascending=[True, False])
table2.to_csv(TABLE2_CSV, index=False)
print(f"\nSaved {TABLE2_CSV}")
print(table2.to_string(index=False))
