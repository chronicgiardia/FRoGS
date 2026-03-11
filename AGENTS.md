# AGENTS.md

## Cursor Cloud specific instructions

### Overview

FRoGS (Functional Representation of Gene Signatures) is a Python bioinformatics research tool for encoding gene functions as vector embeddings. It uses a "word2vec"-style deep learning approach with TensorFlow. See `README.md` for full documentation.

### Running scripts

- **Demo scripts** run from the `demo/` directory: `python classifier.py`, `python gene_sig_representation.py`
- **Core scripts** run from the `src/` directory: `python signature_embedding.py`, `python gene_vec_model.py -h`, `python l1000_model.py -h`
- Scripts use relative paths (e.g., `../data/`) — always `cd` into the correct directory before running.

### Lint

CI runs `python3 -m pylint $(git ls-files '*.py')` from the repo root. Pylint only checks style — no project dependencies are installed in CI. Exit code is non-zero due to style warnings in the research code; this is expected.

### Key caveats

- `l1000_model.py` and `l1000_inference.py` require a large data file (`data/L1000_PhaseI_and_II.csv`) that is **not included** in the repository. Only the demo scripts and `signature_embedding.py` can run without it.
- `gene_vec_model.py --datatype go` requires the `goatools` package and `data/go.obo`; `--datatype archs4` works with the included data files.
- The `LR_model/` directory is incomplete — it references `model/*.model.pickle` and a `util` module not present in the repo.
- TensorFlow prints CUDA/oneDNN warnings on CPU-only machines; these are informational and harmless.
