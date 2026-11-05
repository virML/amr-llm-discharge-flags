"""Creates every output folder used by the analysis scripts.

Run once from the project root before the other scripts:  python scripts/00_create_output_folders.py
Some scripts write into sub-folders (for example tables/llm_precise) that would otherwise not yet exist
when the scripts are run in the documented order on a fresh copy of the project.
"""
from pathlib import Path

FOLDERS = [
    'data/processed', 'data/processed/fitted_models/llm', 'data/processed/model_outputs/llm',
    'data/llm', 'data/deid', 'paper',
    'tables', 'tables/llm', 'tables/llm_precise', 'tables/rb_precise', 'tables/obs',
    'tables/supp',
    'figures', 'figures/llm', 'figures/paper',
]
for f in FOLDERS:
    Path(f).mkdir(parents=True, exist_ok=True)
print('Created or confirmed', len(FOLDERS), 'output folders')
