# Analysis code: LLM-extracted discharge-summary flags for antimicrobial resistance prediction

Additional file 4 for: *Incremental value of large language model-extracted discharge-summary flags for antimicrobial resistance prediction: a retrospective cohort study with temporal validation in an Indian tertiary-care hospital.*

Archived on Zenodo; the DOI is given in the article (Availability of data and materials).

This folder contains every script used to produce the results, tables and figures reported in the manuscript and Additional files 1, 2 and 3. Patient-level data are not included (see the Availability of data and materials statement); the scripts expect the directory layout described below.

Licence: MIT (see `LICENSE`).

## Environment

Python 3.13 with the package versions in `requirements.txt` (pandas 3.0.5, numpy 2.5.3, scikit-learn 1.8.0, XGBoost 3.0.0, SHAP 0.54.1, matplotlib 3.11.2, openpyxl 3.1.5, pyarrow 26.0.0). XGBoost results are reproduced exactly with versions 2.1 to 3.1; later versions give slightly different tree ensembles. LLM extraction (script 05a) additionally requires the Claude Code command-line interface (`claude`) on the path, authenticated to Anthropic.

Run all scripts from the project root (the folder that contains `scripts/` and `data/`), for example `python scripts/03_cohort_and_labels.py`. On a fresh copy, first run `python scripts/00_create_output_folders.py`, which creates the output folders that some scripts write into.

## Expected directory layout

```
data/raw/                      laboratory export CSV files (restricted; not shared)
data/deid/                     identifier mapping and salt written by 02 (restricted; never shared)
data/processed/                intermediate parquet files and model outputs written by the scripts
data/llm/full_corpus_key.csv   LLM input: review_id and de-identified note text (see 05a below)
data/llm/full_corpus_llm_extraction.csv   LLM output written by 05a
data/deid/clinician_validation_sample_key_RESTRICTED.csv   validation sample key written by 06 (restricted)
paper/clinician_validation_review_sheet_completed.xlsx     review workbook (template written by 06) with the physicians' judgments (restricted)
tables/, figures/              outputs (rule-based Model B in tables/, LLM Model B in tables/llm/)
```

## Run order

### 1. Data preparation and cohort
| Script | Purpose |
|---|---|
| `00_create_output_folders.py` | Creates the output folders used by the scripts (run once on a fresh copy) |
| `01_data_audit.py` | Audit of the raw export (completeness, field population) |
| `02_deidentify.py` | Removes direct identifiers, assigns salted SHA-256 study identifiers, redacts identifiers in note text |
| `03_cohort_and_labels.py` | Cohort inclusion and exclusions, 7-day episode de-duplication, MRSA/ESBL/CRE labels with triangulation (`tables/label_triangulation.json`, `tables/consort_counts.json`) |
| `04_structured_features.py` | Model A predictors, including prior microbiology history using only earlier isolates |

### 2. Note-derived features
| Script | Purpose |
|---|---|
| `05_nlp_features.py` | Rule-based extraction of the 13 flags, note availability and length, named-antibiotic field; note timing (`tables/discharge_note_timing.json`) |
| `05a_llm_extraction.py` | LLM extraction (Claude Sonnet 5 via `claude -p`, batches of 25 consecutive notes in review_id order, 8 parallel workers, 3 retries, 300 s timeout). Usage: `python scripts/05a_llm_extraction.py data/llm/full_corpus_key.csv data/llm/full_corpus_llm_extraction.csv` |
| `05b_llm_nlp_features.py` | Builds `data/processed/nlp_features_llm.parquet` (LLM flags; note availability, note length and the rule-based named-antibiotic field carried over unchanged) |

The LLM input file `data/llm/full_corpus_key.csv` contains one row per isolate with a linked note (note text longer than 50 characters), in cohort row order, with columns `review_id` (N000000, N000001, …), `study_id`, `isolate_id` and `DISCHARGE_SUMMARY_CONTENT_DEID`. It can be created with:

```python
import pandas as pd
c = pd.read_parquet('data/processed/cohort_master.parquet').reset_index(drop=True)
t = c['DISCHARGE_SUMMARY_CONTENT_DEID']
k = c.loc[t.notna() & (t.str.len() > 50), ['study_id', 'isolate_id', 'DISCHARGE_SUMMARY_CONTENT_DEID']].reset_index(drop=True)
k.insert(0, 'review_id', [f'N{i:06d}' for i in range(len(k))])
k.to_csv('data/llm/full_corpus_key.csv', index=False)
```

Because LLM outputs can vary between runs and model versions, re-running 05a will not necessarily reproduce the archived extraction exactly; the downstream analyses use the archived output file.

### 3. Physician validation
| Script | Purpose |
|---|---|
| `06_clinician_validation_sample.py` | Draws the 120-note department-stratified base sample, the rare-flag enrichment notes and the 24-note second-reviewer subset; writes the blinded review workbook (`paper/`) and the restricted key (`data/deid/`) |
| `19c_validation_probability_sample.py` | Physician validation: all metrics (precision, recall, specificity, F1 = 2TP/(2TP + FP + FN), Cohen's κ) from one confusion matrix per flag in the 120-note random sample, pooled over all 13 flags; bootstrap CIs within departments (seed 2026); design-weighted analysis; enrichment-note error analysis; inter-rater agreement; integrity check of the sample key against the modelling features (Table 2, Fig. 2; Additional file 1 Tables S6, S7 and S21) |

### 4. Modelling (run once for each extraction method)
Run `08b_fix_admission_dates.py` once before the modelling scripts. Scripts without a suffix use the rule-based flags (`nlp_features.parquet`, outputs in `tables/`, `figures/`, `data/processed/model_outputs/`); scripts with the suffix `_llm` are identical except that they read `nlp_features_llm.parquet` and write to the `llm/` subfolders.

| Script | Purpose |
|---|---|
| `08_splits.py` | Calendar-ordered patient-exclusive temporal split (cutoff 28 June 2025) and random patient-level split |
| `08b_fix_admission_dates.py` | Harmonises two- and four-digit-year admission and discharge dates, restricts the admission-to-testing interval, ward, unit and department to the linked admission, removes discharge summaries written at a later admission from the note features, and recomputes the note-timing summary (updates the feature tables in place; originals kept as `*_pre_datefix.parquet`) |
| `09_modeling.py`, `09_modeling_llm.py` | Fits Model A, Model B and baselines for 3 targets × 2 splits × 3 model families; patient-cluster bootstrap CIs (seed 42, 1000 resamples); inner-validation threshold and headline-family selection |
| `10_delta_ci.py`, `10_delta_ci_llm.py` | Paired Model B minus Model A differences (seed 123, 1000 resamples) |
| `11_tables_figures.py`, `11_tables_figures_llm.py` | Summary tables and draft figures |
| `12_shap_analysis.py`, `12_shap_analysis_llm.py` | SHAP values for headline models, aggregated over one-hot levels |
| `13_multiple_testing_correction.py`, `13_multiple_testing_correction_llm.py` | Bootstrap P values (5000 resamples), Bonferroni and Benjamini-Hochberg over the 36 comparisons |
| `14_shap_cre_all_models.py`, `14_shap_cre_all_models_llm.py` | CRE SHAP across model families |
| `15_recalibration.py`, `15_recalibration_llm.py`, `15c_recalibration_predictions_llm.py` | Isotonic recalibration (five-fold cross-validation within training data) and saved predictions for calibration plots |
| `16_subgroup_fairness.py`, `16_subgroup_fairness_llm.py` | Subgroup performance by gender and age group |
| `17_sensitivity_analyses.py`, `17_sensitivity_analyses_llm.py` | Note-present subset, direct-AES ESBL labels, unweighted models, baseline comparisons |

### 5. Additional analyses, full-precision values and figures
| Script | Purpose |
|---|---|
| `29_decomposition_sensitivity.py` | Decomposition of Model B (Model A + text-derived covariates, + LLM flags, + history or hospital-course flags) and sensitivity analyses: notes discharged before testing only, without the admission-to-testing interval, without species-level identification, all future isolates including returning patients, unweighted models, alternative CRE definitions, 30-day episodes and first isolates, admitted isolates only, species rule from training data, isolates with and without a note, matched operating point. First verifies that it reproduces the primary predictions exactly. Run `python scripts/29_decomposition_sensitivity.py` (all scenarios) or with scenario names, e.g. `S0 S4` (Table 5; Additional file 1 Tables S16 and S22–S24) |
| `30_data_quality_checks.py` | Unlabelled isolates, reconstruction of LLM call composition, repeat agreement within and between calls and test of batch dependence, species rule from training data, development sample size (Riley et al.), SHA-256 checksums (Additional file 1 Note S7 and Tables S26 and S27; Additional file 3) |
| `20b_decomposition_figures.py` | Supplementary Figs. S1 and S2 (decomposition and timing analyses) from the output of script 29 |
| `18_llm_ablation.py` | Earlier exploratory ablation of note feature groups; superseded by the decomposition in script 29 and not reported |
| `22_precise_core.py` | Re-derives all discrimination metrics and paired differences at full precision with the same seeds, used for half-up rounding in the manuscript; checks agreement with the 4-decimal tables |
| `22b_precise_subgroups_llm.py`, `22c_precise_sensitivity_llm.py`, `22d_precise_sensitivity_rulebased.py`, `22e_precise_ablation_llm.py` | Full-precision versions of scripts 16_llm, 17_llm, 17 (note-present analysis) and 18 (the last is not reported) |
| `23_supplement_extras.py` | Agreement between extraction methods across all note instances, repeatability of LLM extraction for documents linked to several isolates, validation sample composition, resistance by organism, missing-data facts |
| `24_split_characteristics.py` | Characteristics of the temporal training and test sets |
| `24b_table1_characteristics.py` | Table 1 (cohort characteristics, including outpatient status at testing), with checks that every section sums to the cohort |
| `25_observed_point_estimates.py` | Observed test-set differences used as point estimates for all model comparisons (CIs and P values unchanged from scripts 10, 13, 17, 18 and 22) |
| `28_decision_curve.py` | Decision curve analysis (net benefit) for the headline CRE and ESBL models using recalibrated probabilities (Figure 6) |
| `26_calibration_metrics.py` | Calibration intercept, slope and observed/expected ratio with patient-cluster bootstrap CIs, before and after isotonic recalibration |
| `20_paper_figures.py` | Figures 2, 3, 4, 5 and 7 and Supplementary Figs. S3–S5 (run after 19c) |
| `20a_figure1_study_design.py` | Figure 1 (study design: cohort assembly, note extraction and validation, temporal evaluation), with counts read from the pipeline outputs |

## Notes

- Point estimates for differences between models are observed test-set differences; confidence intervals are percentiles of the paired patient-cluster bootstrap. Scripts 10, 13, 17 and 18 also store the bootstrap mean difference, which is not used for reporting.
- Random seeds are fixed in every script (model fitting 42; bootstrap CIs 42; paired differences 123; validation bootstrap 2026; inter-rater bootstrap 2027).
- Scripts 19c, 20b, 29 and 30 were added after the primary analysis; scripts 18 and 22e implement an earlier exploratory ablation that script 29 supersedes and are kept for transparency.
- `02_deidentify.py` generates a random salt on first run and stores it with the identifier mapping in `data/deid/`, which must never be shared.
- All reported results come from a single execution of this pipeline (with `08b_fix_admission_dates.py` applied) from the archived de-identified data.
- Comments in some scripts record development history (for example, fixed bugs); they do not affect the analyses.
