# AGENTS.md

## Cursor Cloud specific instructions

This is **FRoGS** (Functional Representation of Gene Signatures) — a Python-based bioinformatics/ML research project. It has no web services, Docker, databases, or build system. All functionality is pure Python scripts.

### Running scripts

- Use `python3` (not `python`) to run scripts; the system does not alias `python`.
- Set `MPLBACKEND=Agg` when running any script that imports matplotlib (demo scripts) to avoid display errors.
- Scripts in `demo/` must be run from the `demo/` directory; scripts in `src/` must be run from the `src/` directory (they use relative paths like `../data/`).

### Lint

- Pylint is the CI lint tool: `pylint $(git ls-files '*.py')`. CI tests on Python 3.8–3.10.
- `pylint` is installed in `~/.local/bin`; ensure `PATH` includes it: `export PATH="$HOME/.local/bin:$PATH"`.

### Key demo commands (hello-world tests)

- `cd demo && python3 classifier.py` — classifies tissue-specific genes using FRoGS embeddings (~80% accuracy) vs one-hot (~30%).
- `cd demo && python3 gene_sig_representation.py` — classifies tissue-specific gene signatures (~100% vs ~86%).
- `cd src && python3 signature_embedding.py` — computes gene signature embeddings from pretrained FRoGS vectors.

### Known issues

- `LR_model/run_LR_models.py` has an unused `import util` referencing a missing module; the script will error on import. The `util` symbol is not used in the code.
- `data/L1000_PhaseI_and_II.csv` is referenced by `src/l1000_model.py` and `src/l1000_inference.py` but only exists as `.csv.gz`; decompress it first if running those scripts.
