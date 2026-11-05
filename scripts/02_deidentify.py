"""
02_deidentify.py
De-identification, executed BEFORE any cohort/label/feature work touches
the data.

Steps:
 1. Load combined raw data (still has identifiers).
 2. Build a single hashed study ID (SHA-256, truncated, salted) from
    corrected_Patient_ID (fallback: Patient ID -> UHIDNo -> txtUHIDNo).
    The salt + raw-ID<->hash mapping is written ONLY to data/deid/ (kept
    separate from the working/analysis dataset and never included in any
    shared deliverable).
 3. Build a hashed isolate ID from Lab ID (same treatment).
 4. Drop direct identifiers entirely (names, address, doctor name, UHID,
    lab ID, barcodes).
 5. Rule-based de-identification pass over DISCHARGE_SUMMARY_CONTENT:
    redacts patient/doctor names (cross-referenced against the identifier
    columns themselves), dates, phone numbers, ID/UHID numbers, and
    address fragments. Redaction counts are logged for QA; the mapping is
    NOT reversible from the working file.
 6. Save the de-identified working dataset.

"""
import pandas as pd
import numpy as np
import hashlib
import re
import json
from pathlib import Path

IN_PARQUET = Path('data/processed/combined_raw.parquet')
OUT_PARQUET = Path('data/processed/deidentified_master.parquet')
MAPPING_DIR = Path('data/deid')  # RESTRICTED - never shipped to user, never committed to notebook
LOG_JSON = Path('tables/deidentification_log.json')

MAPPING_DIR.mkdir(parents=True, exist_ok=True)

# Project salt: generated once with a cryptographically secure random generator and
# persisted to the restricted folder (never shared) so that hashing is stable across
# re-runs. (Published version: the salt is random, not derived from a constant.)
import secrets
SALT_PATH = MAPPING_DIR / 'salt.txt'
if SALT_PATH.exists():
    SALT = SALT_PATH.read_text().strip()
else:
    SALT = secrets.token_hex(16)
    SALT_PATH.write_text(SALT)

def hash_id(raw_value, salt=SALT, length=16):
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return None
    s = str(raw_value).strip()
    if s == '' or s.lower() == 'nan':
        return None
    h = hashlib.sha256((salt + '|' + s).encode('utf-8')).hexdigest()
    return h[:length]

df = pd.read_parquet(IN_PARQUET)
print(f"Loaded {df.shape}")

# ---------------------------------------------------------------------------
# 1. Build hashed study ID (patient level) with priority order
# ---------------------------------------------------------------------------
def coalesce_patient_id(row):
    for c in ['corrected_Patient_ID', 'Patient ID', 'UHIDNo', 'txtUHIDNo']:
        v = row.get(c)
        if v is not None and str(v).strip() not in ('', 'nan'):
            return str(v).strip()
    return None

raw_pid = df.apply(coalesce_patient_id, axis=1)
df['study_id'] = raw_pid.apply(hash_id)

# ---------------------------------------------------------------------------
# 2. Hashed isolate ID from Lab ID (fallback: row index if Lab ID missing)
# ---------------------------------------------------------------------------
df['isolate_id'] = df['Lab ID'].apply(hash_id)
missing_isolate = df['isolate_id'].isna()
df.loc[missing_isolate, 'isolate_id'] = [
    hash_id(f"__rowidx_{i}") for i in df.index[missing_isolate]
]

n_missing_study_id = int(df['study_id'].isna().sum())
print(f"Rows with no usable patient identifier (all of corrected_Patient_ID/"
      f"Patient ID/UHIDNo/txtUHIDNo missing): {n_missing_study_id}")

# ---------------------------------------------------------------------------
# 3. Persist the RESTRICTED mapping table separately (never shipped)
# ---------------------------------------------------------------------------
mapping = df[['study_id']].copy()
mapping['raw_patient_id'] = raw_pid
mapping['isolate_id'] = df['isolate_id']
mapping['raw_lab_id'] = df['Lab ID']
mapping = mapping.dropna(subset=['study_id']).drop_duplicates()
mapping.to_csv(MAPPING_DIR / 'id_mapping_RESTRICTED_DO_NOT_SHARE.csv', index=False)
print(f"Restricted mapping table written to {MAPPING_DIR}/id_mapping_RESTRICTED_DO_NOT_SHARE.csv "
      f"({len(mapping)} unique patients) -- keep this out of any shared deliverable.")

# ---------------------------------------------------------------------------
# 4. Drop direct identifiers
# ---------------------------------------------------------------------------
identifier_cols_to_drop = [
    'Patient Name', 'PatientName', 'txtPatientName',
    'Address', 'txtAddress',
    'UHIDNo', 'txtUHIDNo', 'Patient ID', 'corrected_Patient_ID',
    'Lab ID', 'Bar Code', 'Bar Code.1',
    'drpDoctor',  # dropped (not hashed - not used as a covariate)
]
present_to_drop = [c for c in identifier_cols_to_drop if c in df.columns]
df_deid = df.drop(columns=present_to_drop)
print(f"Dropped {len(present_to_drop)} direct-identifier columns: {present_to_drop}")

# ---------------------------------------------------------------------------
# 5. Rule-based de-identification of DISCHARGE_SUMMARY_CONTENT
# ---------------------------------------------------------------------------
# Build per-row redaction targets from the identifier fields we are about to
# discard, so literal name/address occurrences inside the free text get
# scrubbed even though the columns themselves are dropped.
patient_names = df.get('PatientName', df.get('Patient Name'))
txt_patient_names = df.get('txtPatientName')
doctor_names = df.get('drpDoctor')
addresses = df.get('Address', df.get('txtAddress'))

DATE_PATTERNS = [
    re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),                      # 12/05/2023, 12-5-23
    re.compile(r'\b\d{1,2}(st|nd|rd|th)?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}\b', re.IGNORECASE),
    re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b', re.IGNORECASE),
]
PHONE_PATTERN = re.compile(r'\b(\+?91[\-\s]?)?[6-9]\d{9}\b')
LONG_DIGIT_ID_PATTERN = re.compile(r'\b\d{6,}\b')          # UHID/MRN/Aadhaar-like sequences
EMAIL_PATTERN = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
PIN_PATTERN = re.compile(r'\b\d{6}\b')                      # Indian PIN codes (also caught by 6+ digit rule)

def build_name_pattern(name):
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if len(name) < 3:
        return None
    # allow flexible whitespace between name tokens, case-insensitive, word-bounded
    tokens = [re.escape(t) for t in name.split() if len(t) >= 2]
    if not tokens:
        return None
    pattern = r'\b' + r'\s+'.join(tokens) + r'\b'
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None

def redact_text(text, patient_name, txt_patient_name, doctor_name, address):
    if not isinstance(text, str) or text.strip() == '':
        return text, {}
    counts = {'name': 0, 'date': 0, 'phone': 0, 'id_number': 0, 'email': 0, 'address': 0}
    redacted = text

    for nm in {patient_name, txt_patient_name, doctor_name}:
        pat = build_name_pattern(nm)
        if pat:
            redacted, n = pat.subn('[REDACTED-NAME]', redacted)
            counts['name'] += n

    if isinstance(address, str) and len(address.strip()) >= 6:
        addr_tokens = [re.escape(t) for t in address.split() if len(t) >= 4]
        if addr_tokens:
            addr_pat = re.compile(r'\b(' + '|'.join(addr_tokens) + r')\b', re.IGNORECASE)
            redacted, n = addr_pat.subn('[REDACTED-ADDR]', redacted)
            counts['address'] += n

    for pat in DATE_PATTERNS:
        redacted, n = pat.subn('[REDACTED-DATE]', redacted)
        counts['date'] += n

    redacted, n = EMAIL_PATTERN.subn('[REDACTED-EMAIL]', redacted)
    counts['email'] += n

    redacted, n = PHONE_PATTERN.subn('[REDACTED-PHONE]', redacted)
    counts['phone'] += n

    redacted, n = LONG_DIGIT_ID_PATTERN.subn('[REDACTED-ID]', redacted)
    counts['id_number'] += n

    return redacted, counts

texts = df['DISCHARGE_SUMMARY_CONTENT'] if 'DISCHARGE_SUMMARY_CONTENT' in df.columns else pd.Series([None]*len(df))
redacted_texts = []
total_counts = {'name': 0, 'date': 0, 'phone': 0, 'id_number': 0, 'email': 0, 'address': 0}
n_notes_with_any_redaction = 0

for i in df.index:
    pn = patient_names.loc[i] if patient_names is not None else None
    tpn = txt_patient_names.loc[i] if txt_patient_names is not None else None
    dn = doctor_names.loc[i] if doctor_names is not None else None
    ad = addresses.loc[i] if addresses is not None else None
    rtxt, counts = redact_text(texts.loc[i], pn, tpn, dn, ad)
    redacted_texts.append(rtxt)
    if isinstance(texts.loc[i], str) and sum(counts.values()) > 0:
        n_notes_with_any_redaction += 1
    for k, v in counts.items():
        total_counts[k] += v

df_deid['DISCHARGE_SUMMARY_CONTENT_DEID'] = redacted_texts
df_deid = df_deid.drop(columns=['DISCHARGE_SUMMARY_CONTENT'])

n_notes_total = int(texts.notna().sum())
print(f"\nText de-identification: {n_notes_total} non-null notes processed, "
      f"{n_notes_with_any_redaction} had >=1 redaction ({100*n_notes_with_any_redaction/max(n_notes_total,1):.1f}%)")
print("Redaction counts by category:", total_counts)

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_deid.to_parquet(OUT_PARQUET, index=False)
print(f"\nSaved de-identified working dataset: {OUT_PARQUET}  shape={df_deid.shape}")

log = {
    'n_rows': int(len(df)),
    'n_unique_study_ids': int(df['study_id'].nunique()),
    'n_rows_missing_study_id': n_missing_study_id,
    'identifier_columns_dropped': present_to_drop,
    'text_redaction': {
        'n_notes_total_nonnull': n_notes_total,
        'n_notes_with_any_redaction': n_notes_with_any_redaction,
        'pct_notes_with_any_redaction': round(100*n_notes_with_any_redaction/max(n_notes_total,1), 2),
        'total_counts_by_category': total_counts,
    },
    'note': ('Restricted ID mapping stored separately in data/deid/ and excluded from all '
             'shared deliverables (paper, notebook, tables).'),
}
LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_JSON, 'w') as f:
    json.dump(log, f, indent=2)

print("\n=== DE-ID SUMMARY ===")
print(json.dumps(log, indent=2))
