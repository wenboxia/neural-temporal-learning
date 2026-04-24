# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-timescale temporal learning system for tabular data with concept drift, built on top of TabPFN (a pre-trained tabular foundation model). The system uses three hierarchical levels inspired by the prefrontal cortex (PFC): Slow Prior → Regime Module → Fast Corrector, which combine in logit space to adapt to distributional shift over time.

## Commands

```bash
# Install (with dev dependencies)
pip install -e ".[dev]"

# Phase 1: TabPFN baseline on synthetic data (~30-40 min for 5000 steps on CPU)
python scripts/run_baselines.py --dataset regime_switching
python scripts/run_baselines.py --dataset rotating_boundary   # auto-sets n_features=2
python scripts/run_baselines.py --dataset combined_drift --n_samples 5000

# Phase 2: TabPFN + Fast Corrector comparison
python scripts/run_phase2.py --dataset regime_switching \
    --n_samples 3000 --regime_length 500 --context_size 200

# Quick debug run (100 steps, ~1 min)
python scripts/run_baselines.py --dataset regime_switching --max_eval_steps 100
python scripts/run_phase2.py --dataset regime_switching --max_eval_steps 100 --n_samples 3000

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_synthetic_data.py
pytest tests/test_fast_corrector.py
```

All scripts must be run from the project root (`neural_1/`). Results (`.png` + `.npz`) are saved to `results/` (gitignored).

## Architecture

### Three-Level System

```
Input stream (X_t, y_t) ──▶ TemporalWindowLoader (sliding context window)
                                        │
                           ┌────────────▼────────────┐
                           │  Level 1: SlowPrior      │  Frozen TabPFN (in-context learning)
                           │  src/models/slow_prior.py│  Lazy-loads weights on first call
                           └────────────┬────────────┘
                                        │ proba, embeddings
                           ┌────────────▼────────────┐
                           │  Level 2: RegimeModule   │  NOT YET IMPLEMENTED
                           │  src/models/regime_module│  Sliding-window k-means on embeddings
                           └────────────┬────────────┘
                                        │ regime logit correction
                           ┌────────────▼────────────┐
                           │  Level 3: FastCorrector  │  FIFO buffer + KNN or EMA lookup
                           │  src/models/fast_corrector│ Zero learnable parameters
                           └────────────┬────────────┘
                                        │
                           y_final = sigmoid(logit_slow + logit_inter + logit_fast)
```

### Key Design Constraints

- **TabPFN is never fine-tuned** in Levels 1–3 (weights frozen). "Slow" means the context window changes, not the weights. Fine-tuning TabPFN (inter→slow consolidation) is reserved for Phase 4 and requires a GPU.
- **Context size ≤ 3000** for CPU-friendly inference. Each `SlowPrior.predict()` call re-runs `tabpfn.fit()` + `predict_proba()` — this is TabPFN's in-context learning mechanism, not training.
- **Logit-space addition**: TabPFN outputs probabilities → convert to logits → add corrections from Level 2 and Level 3 → sigmoid for final probability.
- **Prequential evaluation**: predict first, observe true label, then update the fast corrector. Never peek at future labels.

### Data Flow

`SyntheticDataset` → `TemporalWindowLoader` yields `WindowBatch(X_ctx, y_ctx, X_query, y_query)` at each time step. `X_ctx`/`y_ctx` is the sliding context window fed to TabPFN; `X_query` is the single sample to predict.

### Implemented Modules

| File | Status | Description |
|------|--------|-------------|
| `src/data/synthetic.py` | ✅ | 3 drift generators: `rotating_boundary`, `regime_switching`, `combined_drift` |
| `src/data/temporal_loader.py` | ✅ | Sliding window loader returning `WindowBatch` |
| `src/models/slow_prior.py` | ✅ | Frozen TabPFN wrapper with single-class fallback |
| `src/memory/buffer.py` | ✅ | FIFO buffer storing `(x, error, embedding)` with KNN/EMA query |
| `src/models/fast_corrector.py` | ✅ | Level 3 residual correction via KNN or EMA |
| `src/utils/metrics.py` | ✅ | `summarize_results()`, `window_accuracy()`, adaptation speed |
| `src/models/regime_module.py` | ❌ | Level 2 — not yet implemented |
| `src/consolidation/` | ❌ | Fast→inter and inter→slow consolidation — not yet implemented |
| `src/data/real_world.py` | ❌ | Real dataset loader — not yet implemented |

### Experiment Progress

- **Phase 1** (TabPFN baseline): Complete. 82.96% prequential accuracy on `regime_switching`. Drift causes ~12–13pp accuracy drop, recovering in ~61 steps.
- **Phase 2** (+ FastCorrector): Complete. KNN and EMA correctors slightly underperform baseline because stale buffer entries from the old regime corrupt corrections for ~50 steps post-drift.
- **Phase 3** (+ RegimeModule): Not started. Requires implementing `src/models/regime_module.py`.
- **Phase 4** (+ Consolidation + Real datasets): Not started. TabPFN fine-tuning needs a GPU (Colab recommended).

### Synthetic Datasets

All three generators produce `SyntheticDataset(X, y, regime_labels, drift_points, name)`:

- **`rotating_boundary`**: Gradual relationship drift. 2D Gaussian features, linear decision boundary rotates at `drift_speed=0.003` rad/step. Auto-uses `n_features=2` in scripts.
- **`regime_switching`**: Abrupt drift (used in all experiments). Cycles between `n_regimes=3` regimes with independent feature means and decision weights, each lasting `regime_length` steps.
- **`combined_drift`**: Mixed drift. First 5 features drift linearly by 2σ over full sequence; decision boundary changes abruptly every 3000 steps.

### Config System

`configs/default.yaml` sets defaults; experiment YAMLs in `configs/experiment/` override specific fields. Currently the scripts use `argparse` directly rather than loading YAML configs at runtime.
