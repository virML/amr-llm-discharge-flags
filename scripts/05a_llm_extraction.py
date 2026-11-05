"""
05a_llm_extraction.py

LLM extraction of the 13 note-derived flags, as run on 6-7 September 2026 to produce the
archived extraction analysed in the paper (Additional file 4 documents the run).

Input: CSV with one row per de-identified discharge summary (columns review_id and
DISCHARGE_SUMMARY_CONTENT_DEID; see README). Output: CSV with review_id and 13 binary flags,
checkpointed after every batch so that an interrupted run can be resumed.

Each batch of up to 25 notes is sent to Claude through the Claude Code command-line interface
(`claude -p`, prompt on standard input, interface-default sampling); batches run concurrently in
a thread pool. A batch is retried up to 3 times if the response cannot be parsed or omits any
review_id. LLM outputs are not guaranteed to be identical across runs, model versions or dates.
(Only this docstring was edited for publication; the prompt text, definitions and code are unchanged.)
"""
import json
import re
import subprocess
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

FLAG_DEFS = """
1. comorbid_diabetes -- Diabetes mellitus (any type), including "known case of T2DM/T1DM", "on insulin/OHA", diabetic complications mentioned as pre-existing.
2. comorbid_ckd_dialysis -- Chronic kidney disease or dialysis dependence (not acute kidney injury alone), including "CKD stage X", "on hemodialysis/peritoneal dialysis", "ESRD".
3. comorbid_malignancy -- Any current or past malignancy/cancer, including "carcinoma", "on chemotherapy/radiotherapy", "post-mastectomy for Ca breast".
4. comorbid_immunosuppression -- Immunosuppressed state: on steroids/immunosuppressants, post-transplant, HIV/AIDS, or other explicitly stated immunocompromise.
5. comorbid_cardio_cerebrovascular -- Prior stroke (CVA) or cardiac disease (CAD, prior MI, heart failure, prior cardiac surgery) as PRE-EXISTING history, not the current admission's primary diagnosis alone.
6. prior_antibiotic_any -- Any mention of antibiotic use before or leading up to this admission (named drug or generic "antibiotics"/"antimicrobials"), from outside treatment or earlier in this same admission's course.
7. device_urinary_catheter -- Urinary catheter (Foley catheter, indwelling catheter) in place or recently placed.
8. device_central_line -- Central venous catheter / central line / PICC line in place or recently placed.
9. device_ventilator_tracheal -- Mechanical ventilation, intubation, or tracheostomy tube.
10. device_surgical_drain -- Surgical drain in place.
11. icu_stay -- Explicit ICU/ICU-equivalent (MICU, SICU, RICU, IMCU, critical care unit) stay mentioned, OR ward name itself is an ICU-type ward.
12. recent_hospitalization -- Hospitalization prior to THIS admission, including implicit/indirect mentions: e.g. "a procedure was done at [named hospital] one month back" implies a prior hospitalization even if the word "hospitalized" never appears. A readmission to the same facility for a related complaint also counts.
13. recent_surgery -- Any surgery/procedure performed recently, either before this admission (history) or earlier in the course of this same admission.
""".strip()

NEGATION_INSTRUCTIONS = """
Negation and uncertainty handling (critical, read carefully):
- Indian discharge summaries commonly use "N/K/C/O" or "N/K/C/O." as shorthand for "Not a Known Case Of" -- treat this exactly like an explicit denial/negation of whatever follows it. E.g. "N/K/C/O DM" means the patient is NOT a known diabetic -> flag = 0.
- Standard negation phrases also count as denial: "denies", "no h/o", "not a known case of", "ruled out", "no history of".
- Do NOT infer a flag purely from a related but distinct current-admission diagnosis. E.g. a patient admitted now with newly diagnosed diabetic ketoacidosis and NO prior history mentioned is a NEW diagnosis, not pre-existing comorbid_diabetes, UNLESS the note itself frames it as pre-existing.
- Implicit and indirect mentions DO count as positive when the clinical meaning is clear even without an exact keyword match (this is precisely why an LLM is being used instead of keyword regex) -- e.g. "catheterized on admission" -> device_urinary_catheter = 1; "underwent appendectomy at [Named Hospital] 2 weeks back" -> recent_surgery = 1 AND recent_hospitalization = 1.
- If the note gives no information either way on a flag, call it 0 (absent), not missing -- every flag must be a binary 0/1 for every note.
""".strip()

PROMPT_TEMPLATE = """You are extracting structured clinical flags from de-identified hospital discharge-summary text for a research study on antimicrobial resistance prediction. For EACH of the following 13 binary clinical flags, decide 1 (present) or 0 (absent) based on the note text, using clinical judgment rather than simple keyword matching.

FLAG DEFINITIONS:
{flag_defs}

{negation_instructions}

You will be given a batch of {n_notes} de-identified discharge summaries, each with a review_id. For EACH note, output one JSON object with the review_id and all 13 flags as 0/1 integers. Return ONLY a single JSON array of these objects, no other text, no markdown code fences.

Output schema per note:
{{"review_id": "...", "comorbid_diabetes": 0, "comorbid_ckd_dialysis": 0, "comorbid_malignancy": 0, "comorbid_immunosuppression": 0, "comorbid_cardio_cerebrovascular": 0, "prior_antibiotic_any": 0, "device_urinary_catheter": 0, "device_central_line": 0, "device_ventilator_tracheal": 0, "device_surgical_drain": 0, "icu_stay": 0, "recent_hospitalization": 0, "recent_surgery": 0}}

NOTES BATCH:
{notes_block}

Return the JSON array now, one object per note, in the same order as given, with the exact review_id values provided."""

FLAGS = [
    "comorbid_diabetes", "comorbid_ckd_dialysis", "comorbid_malignancy",
    "comorbid_immunosuppression", "comorbid_cardio_cerebrovascular",
    "prior_antibiotic_any", "device_urinary_catheter", "device_central_line",
    "device_ventilator_tracheal", "device_surgical_drain", "icu_stay",
    "recent_hospitalization", "recent_surgery",
]

WRITE_LOCK = threading.Lock()


def build_prompt(batch_df):
    notes_block = "\n\n".join(
        f"--- NOTE review_id={row.review_id} ---\n{row.DISCHARGE_SUMMARY_CONTENT_DEID}"
        for row in batch_df.itertuples()
    )
    return PROMPT_TEMPLATE.format(
        flag_defs=FLAG_DEFS,
        negation_instructions=NEGATION_INSTRUCTIONS,
        n_notes=len(batch_df),
        notes_block=notes_block,
    )


def extract_json_array(text):
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in output (len={len(text)}): {text[:300]}")
    return json.loads(text[start:end + 1])


def run_batch(batch_df, retries=3):
    prompt = build_prompt(batch_df)
    expected_ids = set(batch_df['review_id'])
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                ['claude', '-p'],
                input=prompt,
                capture_output=True, text=True, timeout=300,
            )
            out = result.stdout
            parsed = extract_json_array(out)
            got_ids = {r.get('review_id') for r in parsed}
            missing = expected_ids - got_ids
            if missing:
                raise ValueError(f"Missing review_ids in response: {missing}")
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"Batch failed after {retries} attempts (review_ids={sorted(expected_ids)}): {last_err}")


def checkpoint(out_path, new_rows):
    with WRITE_LOCK:
        out_df = pd.DataFrame(new_rows)
        for f in FLAGS:
            if f not in out_df.columns:
                out_df[f] = None
        out_df = out_df[['review_id'] + FLAGS]
        if out_path.exists():
            prev = pd.read_csv(out_path)
            out_df = pd.concat([prev, out_df], ignore_index=True).drop_duplicates('review_id', keep='last')
        out_df.to_csv(out_path, index=False)
        return len(out_df)


def main(input_csv, output_csv, batch_size=25, limit=None, workers=8, max_batches=None):
    df = pd.read_csv(input_csv)
    if limit:
        df = df.iloc[:limit]
    out_path = Path(output_csv)
    done_ids = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done_ids = set(prev['review_id'])
        print(f"Resuming: {len(done_ids)} notes already extracted.")

    remaining = df[~df['review_id'].isin(done_ids)].reset_index(drop=True)
    batches = [remaining.iloc[i:i + batch_size] for i in range(0, len(remaining), batch_size)]
    if max_batches:
        batches = batches[:max_batches]
    print(f"{len(remaining)} notes remaining -> running {len(batches)} batches (of up to {batch_size}) with {workers} workers.")

    failed = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_batch, b): idx for idx, b in enumerate(batches)}
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            completed += 1
            try:
                rows = fut.result()
                total = checkpoint(out_path, rows)
                print(f"[{completed}/{len(batches)}] batch {idx} OK -> {total} total rows checkpointed", flush=True)
            except Exception as e:
                failed.append(idx)
                print(f"[{completed}/{len(batches)}] batch {idx} FAILED: {e}", file=sys.stderr, flush=True)

    print(f"Done. {len(failed)} batches failed: {failed}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('input_csv')
    ap.add_argument('output_csv')
    ap.add_argument('--batch-size', type=int, default=25)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-batches', type=int, default=None)
    args = ap.parse_args()
    main(args.input_csv, args.output_csv, args.batch_size, args.limit, args.workers, args.max_batches)
