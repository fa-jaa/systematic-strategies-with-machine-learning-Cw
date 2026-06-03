# experiments/

This folder contains optimised/experimental versions of notebooks.
Nothing here is part of the main pipeline — it can be deleted without breaking anything.

---

## cpcv_fast.py

An optimised version of `03_model_development/cpcv_energy_modelling.ipynb`.
Produces identical results but runs much faster.

### Why it's faster

The original notebook runs 15,840 candidates and recomputes triple-barrier labels
for every single one. That means labels get computed 15,840 times even though there
are only 240 unique ticker+barrier combinations.

This script fixes that with two things:

**1. Disk label cache**
- Triple-barrier labels are computed once per ticker+barrier combination (240 total)
- Saved to `data/labels/triple_barrier/{ticker}/{barrier_name}.csv`
- Every subsequent run loads from disk — no recomputation
- Second run onwards: 0 label computations

**2. Parallelisation**
- All 15,840 model candidates run in parallel across all CPU cores
- Uses threading backend (safe on Windows)

### Speed comparison

| | Original notebook | cpcv_fast.py |
|---|---|---|
| Label computations | 15,840 | 240 (first run) / 0 (after) |
| Parallel workers | 1 | All CPU cores |
| Estimated time (full grid) | ~10 hours | ~30-45 mins |

### How to run

From the project root (`systematic-strategies-with-machine-learning-Cw/`):

```bash
# Smoke mode — quick test (1 ticker, 1 barrier, 4 FS, 2 models = 8 candidates)
python experiments/cpcv_fast.py

# Full grid — all 15,840 candidates
python experiments/cpcv_fast.py --full

# Full grid + save results to data/models/cpcv_energy_fast/
python experiments/cpcv_fast.py --full --save

# Control number of parallel jobs (default = all cores)
python experiments/cpcv_fast.py --full --jobs 4
```

### Output files (only created with --save)

```
data/labels/triple_barrier/          ← label cache (created automatically)
    cl1s/
        ewma_d10_tp2_sl2.csv
        ewma_d20_tp2_sl2.csv
        ...
    ho1s/
        ...

data/models/cpcv_energy_fast/        ← model results (only with --save)
    path_level_results.csv           ← one row per CPCV fold per candidate
    candidate_summary.csv            ← one row per candidate, ranked by AUC
```

### What it does NOT change

- The original `03_model_development/cpcv_energy_modelling.ipynb` is untouched
- Same configs from `model_configs.py`
- Same CPCV logic, same feature selection, same models
- Same results — just faster
