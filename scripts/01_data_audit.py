"""
01_data_audit.py
Full data audit on the combined 25,468-row AMR dataset (5 batch CSVs).
Runs the data checks against the FULL dataset (all five export files) and
flags any discrepancies.

Outputs:
  data/processed/combined_raw.parquet   -- concatenated raw data (untouched values)
  tables/audit_report.json              -- machine-readable audit summary
  tables/audit_report.md                -- human-readable audit summary
"""
import pandas as pd
import numpy as np
import glob
import json
import re
from pathlib import Path

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

RAW_DIR = Path('data/raw')
OUT_PARQUET = Path('data/processed/combined_raw.parquet')
OUT_JSON = Path('tables/audit_report.json')
OUT_MD = Path('tables/audit_report.md')

report = {}

# ---------------------------------------------------------------------------
# 1. Load and concatenate all 5 batches
# ---------------------------------------------------------------------------
files = sorted(glob.glob(str(RAW_DIR / '*.csv')))
print(f"Found {len(files)} files: {files}")

dfs = []
for f in files:
    d = pd.read_csv(f, low_memory=False, dtype=str)  # read everything as string first for a clean audit
    d['__source_csv'] = Path(f).name
    dfs.append(d)
    print(f"  {f}: {d.shape}")

df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined shape: {df.shape}")
report['n_rows_total'] = int(df.shape[0])
report['n_cols_total'] = int(df.shape[1])
report['files'] = [Path(f).name for f in files]
report['rows_per_file'] = {Path(f).name: int(len(d)) for f, d in zip(files, dfs)}

# ---------------------------------------------------------------------------
# 2. Exact duplicate rows (across or within batches)
# ---------------------------------------------------------------------------
dup_mask = df.drop(columns=['__source_csv']).duplicated(keep=False)
report['n_exact_duplicate_rows'] = int(dup_mask.sum())

# ---------------------------------------------------------------------------
# 3. Identifier columns present
# ---------------------------------------------------------------------------
id_cols = ['Patient Name', 'Patient ID', 'corrected_Patient_ID', 'Address', 'UHIDNo',
           'PatientName', 'txtUHIDNo', 'txtPatientName', 'txtAddress', 'Lab ID',
           'Bar Code', 'Bar Code.1']
report['identifier_columns_present'] = [c for c in id_cols if c in df.columns]
report['identifier_columns_missing_from_data'] = [c for c in id_cols if c not in df.columns]

# ---------------------------------------------------------------------------
# 4. Patient ID field reconciliation
#    corrected_Patient_ID is preferred; check how much it
#    differs from Patient ID / UHIDNo / txtUHIDNo
# ---------------------------------------------------------------------------
pid_report = {}
for c in ['Patient ID', 'corrected_Patient_ID', 'UHIDNo', 'txtUHIDNo']:
    if c in df.columns:
        pid_report[c] = {
            'n_non_null': int(df[c].notna().sum()),
            'n_unique': int(df[c].nunique(dropna=True)),
        }
report['patient_id_fields'] = pid_report

# does corrected_Patient_ID differ from Patient ID when both present?
if 'Patient ID' in df.columns and 'corrected_Patient_ID' in df.columns:
    both = df[['Patient ID', 'corrected_Patient_ID']].dropna()
    n_diff = int((both['Patient ID'].astype(str).str.strip() != both['corrected_Patient_ID'].astype(str).str.strip()).sum())
    report['patient_id_vs_corrected_diff_count'] = n_diff
    report['patient_id_vs_corrected_total_compared'] = int(len(both))

# ---------------------------------------------------------------------------
# 5. Cross-batch patient overlap (does any patient appear in >1 source csv?)
# ---------------------------------------------------------------------------
study_pid_col = 'corrected_Patient_ID' if 'corrected_Patient_ID' in df.columns else 'Patient ID'
tmp = df[[study_pid_col, '__source_csv']].dropna(subset=[study_pid_col])
overlap = tmp.groupby(study_pid_col)['__source_csv'].nunique()
n_patients_multi_batch = int((overlap > 1).sum())
report['n_unique_patients_using_' + study_pid_col] = int(tmp[study_pid_col].nunique())
report['n_patients_appearing_in_multiple_source_files'] = n_patients_multi_batch

# ---------------------------------------------------------------------------
# 6. Isolates per patient
# ---------------------------------------------------------------------------
isolate_counts = tmp.groupby(study_pid_col).size()
report['isolates_per_patient'] = {
    'n_patients_with_gt1_isolate': int((isolate_counts > 1).sum()),
    'max_isolates_single_patient': int(isolate_counts.max()),
    'median_isolates': float(isolate_counts.median()),
}

# ---------------------------------------------------------------------------
# 7. Dates
# ---------------------------------------------------------------------------
date_report = {}
for c in ['Collection Date', 'Testing Date', 'txtAdmitDt', 'txtDiscDt']:
    if c in df.columns:
        non_null = df[c].notna() & (df[c].astype(str).str.strip() != '')
        date_report[c] = {'n_non_null': int(non_null.sum()), 'pct_non_null': round(100 * non_null.mean(), 2)}
report['date_field_completeness'] = date_report

# try parsing Testing Date (DD/MM/YY) and admit/disc (DD/MM/YYYY HH:MM)
def try_parse(series, fmt):
    return pd.to_datetime(series, format=fmt, errors='coerce')

testing_dt = try_parse(df['Testing Date'], '%d/%m/%y') if 'Testing Date' in df.columns else None
if testing_dt is not None:
    report['testing_date_parse'] = {
        'n_parsed_ok': int(testing_dt.notna().sum()),
        'n_failed_but_nonnull': int(((df['Testing Date'].notna()) & (df['Testing Date'].astype(str).str.strip() != '') & testing_dt.isna()).sum()),
        'min': str(testing_dt.min()),
        'max': str(testing_dt.max()),
    }

admit_dt = try_parse(df['txtAdmitDt'], '%d/%m/%Y %H:%M') if 'txtAdmitDt' in df.columns else None
if admit_dt is not None:
    report['admit_date_parse'] = {
        'n_parsed_ok': int(admit_dt.notna().sum()),
        'min': str(admit_dt.min()),
        'max': str(admit_dt.max()),
    }

# ---------------------------------------------------------------------------
# 8. ID Confidence / Organism / specimen distributions
# ---------------------------------------------------------------------------
if 'ID Confidence' in df.columns:
    report['id_confidence_counts'] = df['ID Confidence'].value_counts(dropna=False).to_dict()

if 'Organism Name' in df.columns:
    report['n_unique_organism_names'] = int(df['Organism Name'].nunique(dropna=True))
    report['top_30_organisms'] = df['Organism Name'].value_counts(dropna=False).head(30).to_dict()

if 'Specimen Type' in df.columns:
    report['specimen_type_counts'] = df['Specimen Type'].value_counts(dropna=False).head(20).to_dict()

# ---------------------------------------------------------------------------
# 9. Antibiotic MIC columns - check value vocab (MIC strings vs TRM vs blank)
# ---------------------------------------------------------------------------
antibiotic_cols = ['AM-Ampicillin', 'CAZ-Ceftazidime', 'CIP-Ciprofloxacin', 'MEM-Meropenem',
                    'VA-Vancomycin', 'OX1-Oxacillin', 'OXSF-Cefoxitin Screen', 'CS-Colistin',
                    'IPM-Imipenem', 'CTX-Cefotaxime', 'CRO-Ceftriaxone', 'FOX-Cefoxitin']
abx_report = {}
for c in antibiotic_cols:
    if c in df.columns:
        vc = df[c].value_counts(dropna=False)
        abx_report[c] = {
            'n_non_null': int(df[c].notna().sum()),
            'n_TRM': int((df[c] == 'TRM').sum()),
            'sample_values': vc.head(15).to_dict(),
        }
report['antibiotic_column_samples'] = abx_report

# how many antibiotic columns total (defined as columns matching PATTERN-Name or in known abx list)
all_abx_like = [c for c in df.columns if re.match(r'^[A-Z0-9]{1,6}-[A-Za-z]', c)]
report['n_antibiotic_like_columns'] = len(all_abx_like)
report['antibiotic_like_columns'] = all_abx_like

# ---------------------------------------------------------------------------
# 10. Resistance-mechanism "Family -X - N" columns: vocabulary of values
# ---------------------------------------------------------------------------
family_cols = [c for c in df.columns if c.startswith('Family -')]
report['n_family_mechanism_columns'] = len(family_cols)

mech_values = pd.Series(dtype=int)
for c in family_cols:
    vc = df[c].value_counts(dropna=True)
    mech_values = mech_values.add(vc, fill_value=0)
report['resistance_mechanism_value_vocab'] = mech_values.sort_values(ascending=False).head(60).astype(int).to_dict()

# ---------------------------------------------------------------------------
# 11. Organism-name resistance suffixes
# ---------------------------------------------------------------------------
if 'Organism Name' in df.columns:
    for tag in ['MRSA', 'MDR', 'MDRO', 'CRE']:
        report[f'organism_name_contains_{tag}'] = int(df['Organism Name'].fillna('').str.contains(tag, case=False).sum())

# ---------------------------------------------------------------------------
# 12. Discharge summary text
# ---------------------------------------------------------------------------
if 'DISCHARGE_SUMMARY_CONTENT' in df.columns:
    lens = df['DISCHARGE_SUMMARY_CONTENT'].fillna('').str.len()
    non_null = df['DISCHARGE_SUMMARY_CONTENT'].notna() & (df['DISCHARGE_SUMMARY_CONTENT'].astype(str).str.strip() != '')
    report['discharge_summary'] = {
        'n_non_null': int(non_null.sum()),
        'pct_non_null': round(100 * non_null.mean(), 2),
        'mean_len_chars': round(float(lens[non_null].mean()), 1),
        'median_len_chars': float(lens[non_null].median()),
        'max_len_chars': int(lens.max()),
        'min_len_chars_nonnull': int(lens[non_null].min()) if non_null.sum() else None,
    }

# ---------------------------------------------------------------------------
# 13. Age field format check
# ---------------------------------------------------------------------------
if 'Age' in df.columns:
    sample_ages = df['Age'].dropna().sample(min(20, df['Age'].notna().sum()), random_state=42).tolist()
    report['age_field_samples'] = sample_ages
    age_pattern = re.compile(r'(\d+)\s*Y')
    n_parseable = int(df['Age'].fillna('').apply(lambda x: bool(age_pattern.search(str(x)))).sum())
    report['age_n_parseable_years'] = n_parseable
    report['age_n_total_nonnull'] = int(df['Age'].notna().sum())

# ---------------------------------------------------------------------------
# 14. Gender values
# ---------------------------------------------------------------------------
if 'Gender' in df.columns:
    report['gender_value_counts'] = df['Gender'].value_counts(dropna=False).to_dict()

# ---------------------------------------------------------------------------
# 15. Save combined raw parquet (still contains identifiers - restricted, not shared)
# ---------------------------------------------------------------------------
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)
print(f"\nSaved combined raw data to {OUT_PARQUET}  shape={df.shape}")

# ---------------------------------------------------------------------------
# Write reports
# ---------------------------------------------------------------------------
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(report, f, indent=2, default=str)

md_lines = ["# Full Dataset Audit Report\n"]
md_lines.append(f"- Total rows: **{report['n_rows_total']}**")
md_lines.append(f"- Total columns: **{report['n_cols_total']}**")
md_lines.append(f"- Files: {report['files']}")
md_lines.append(f"- Rows per file: {report['rows_per_file']}")
md_lines.append(f"- Exact duplicate rows: {report['n_exact_duplicate_rows']}")
md_lines.append(f"- Unique patients ({study_pid_col}): {report.get('n_unique_patients_using_' + study_pid_col)}")
md_lines.append(f"- Patients appearing in >1 source file: {report['n_patients_appearing_in_multiple_source_files']}")
md_lines.append(f"- Patients with >1 isolate: {report['isolates_per_patient']['n_patients_with_gt1_isolate']}")
md_lines.append(f"- Max isolates for a single patient: {report['isolates_per_patient']['max_isolates_single_patient']}")
with open(OUT_MD, 'w') as f:
    f.write('\n'.join(md_lines))

print("\n=== KEY AUDIT FINDINGS ===")
for k in ['n_rows_total', 'n_exact_duplicate_rows', 'n_patients_appearing_in_multiple_source_files',
          'patient_id_vs_corrected_diff_count', 'patient_id_vs_corrected_total_compared']:
    print(f"  {k}: {report.get(k)}")
print("id_confidence_counts:", report.get('id_confidence_counts'))
print("date completeness:", report.get('date_field_completeness'))
print("discharge_summary:", report.get('discharge_summary'))
print("gender_value_counts:", report.get('gender_value_counts'))
