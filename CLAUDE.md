# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A multi-timescale temporal learning system for tabular data with concept drift, built on top of TabPFN (a pre-trained tabular foundation model). The system uses three hierarchical levels inspired by the prefrontal cortex (PFC): Slow Prior (TabPFN) → Inter (gate-fused adapter) → Fast Corrector (KNN/EMA on residual buffer). Combined via residual-additive fusion. Currently in Phase 4 Day 1.5 (architectural redesign — Design A: per-regime adapter library).

## Commands

```bash
# Install (with dev dependencies)
pip install -e ".[dev]"

# Phase 1: TabPFN baseline on synthetic data (~25-30 min for 3000 steps on CPU)
python scripts/run_baselines.py --dataset regime_switching --n_samples 3000 --context_size 200
python scripts/run_baselines.py --dataset rotating_boundary --n_samples 3000 --context_size 200  # auto-sets n_features=2
python scripts/run_baselines.py --dataset combined_drift --n_samples 5000 --context_size 200

# Phase 1 with Oracle context-reset (Phase 4 Day 0.5 experiment)
python scripts/run_baselines.py --dataset regime_switching --oracle_context_reset --reset_size 50

# Phase 2: TabPFN + Fast Corrector comparison (runs 3 correctors per script)
python scripts/run_phase2.py --dataset regime_switching --n_samples 3000 --context_size 200

# Phase 3: Three-level system with GatedEnsemble + FastToInter consolidation
python scripts/run_phase3.py --dataset regime_switching --n_samples 3000 --context_size 200

# Phase 4 multi-seed automation (45 runs: 3 phase × 3 dataset × 5 seeds)
python scripts/run_multiseed.py --n_parallel 3

# Quick debug runs (~1-2 min each)
python scripts/run_baselines.py --dataset regime_switching --max_eval_steps 100
python scripts/run_phase3.py --dataset regime_switching --max_eval_steps 100 --n_samples 3000

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_synthetic_data.py
pytest tests/test_fast_corrector.py
pytest tests/test_gated_ensemble.py
pytest tests/test_fast_to_inter.py
```

All scripts must be run from the project root (`neural_1/`). Result PNGs are tracked in git; `.npz` files are gitignored.

## Architecture

### Three-Level System (current — V2 + Phase 3 v2+B+F)

```
Input stream (X_t, y_t) ──▶ TemporalWindowLoader (sliding context window)
                                        │
                           ┌────────────▼────────────┐
                           │  Level 1: SlowPrior      │  Frozen TabPFN (in-context learning)
                           │  src/models/slow_prior.py│  Lazy-loads weights on first call
                           └────────────┬────────────┘
                                        │ y_slow (probability)
                           ┌────────────▼────────────┐
                           │  Level 3: FastCorrector  │  FIFO buffer + KNN/EMA lookup
                           │  src/models/fast_corrector│  zero learnable params
                           └────────────┬────────────┘
                                        │ correction (residual)
                           ┌────────────▼────────────┐
                           │  Level 2: GatedEnsemble  │  gate (MLP→softmax 3-way)
                           │  src/models/gated_ensemble│ + adapter (MLP residual)
                           └────────────┬────────────┘
                                        │ (α, β, γ) weights, y_inter
                                        │
              y_final = clip(y_slow + β·y_inter + γ·correction, 0, 1)
                  (per-step MSE trains gate; consolidation trains adapter)
```

Orchestrator: `src/models/multi_timescale.py` `MultiTimescaleModel.step(X_ctx, y_ctx, x_t, y_t, t)`.

### Key Design Constraints

- **TabPFN is never fine-tuned** in any Level (weights frozen always). "Slow" means the context window changes, not the weights. The original V1 plan's "inter→slow consolidation via TabPFN fine-tuning" was cut in V2 due to GPU + catastrophic-forgetting risk.
- **Context size ≤ 3000** for CPU-friendly inference. Each `SlowPrior.predict()` call re-runs `tabpfn.fit()` + `predict_proba()` — TabPFN's in-context learning, not training.
- **Residual-additive fusion** (post v2): y_inter and correction are added as residuals on top of y_slow; gate weights (α, β, γ) sum to 1 via softmax but α is unused (kept for backward-compat / visualization). y_final clamped to [0, 1] post-MSE-loss for valid probability.
- **Prequential evaluation**: predict first → observe true label → update FastCorrector buffer → check consolidation trigger. Never peek at future labels.
- **Per-step MSE training**: gate optimizer steps each prequential step on `MSE(y_final, y_t)`. Adapter receives gradient ONLY during consolidation (Phase 3 F change).
- **Consolidation trigger** (Phase 3 B+F): `|mean(buffer.errors[-window:])| > threshold` (std-based AND condition removed) AND `t - last_consolidation_t >= cooldown` (prevents thrashing).

### Data Flow

`SyntheticDataset` → `TemporalWindowLoader` yields `WindowBatch(X_ctx, y_ctx, X_query, y_query, t)` at each time step. `X_ctx`/`y_ctx` is the sliding context window fed to TabPFN; `X_query` is the single sample to predict.

### Implemented Modules

| File | Status | Description |
|------|--------|-------------|
| `src/data/synthetic.py` | ✅ | 3 drift generators: `rotating_boundary`, `regime_switching`, `combined_drift`. Class-balanced via centered decision boundary (Phase 2.5 fix). |
| `src/data/temporal_loader.py` | ✅ | `TemporalWindowLoader` (sliding) + `CompositeWindowLoader` (fixed pool + sliding window) |
| `src/models/slow_prior.py` | ✅ | Frozen TabPFN wrapper with single-class fallback |
| `src/memory/buffer.py` | ✅ | FIFO buffer storing `(x, error, embedding)` with `recent_errors(n)` / `recent_features(n)` / KNN / EMA queries |
| `src/models/fast_corrector.py` | ✅ | Level 3 residual correction via KNN or EMA. `should_consolidate(window, threshold)` (Phase 3 B: simplified to mean-only condition). |
| `src/models/gated_ensemble.py` | ✅ | Level 2 GatedEnsemble (Phase 3A v2): `forward(x, y_slow, correction) → (y_final_raw, weights)`. Residual-additive fusion. |
| `src/consolidation/fast_to_inter.py` | ✅ | `FastToInterConsolidation.consolidate(...)`: MSE-distill buffer errors → adapter; clears buffer after. |
| `src/models/multi_timescale.py` | ✅ | `MultiTimescaleModel` orchestrator (Phase 3C). v2+B+F: separated gate/adapter optimizers + consolidation cooldown. |
| `src/utils/metrics.py` | ✅ | `summarize_results()`, `window_accuracy()`, adaptation speed; balanced acc + AUC-ROC (Phase 2.5). |
| `src/drift/error_detector.py` | 🚧 Phase 4 Day 1.5 | ADWIN / Page-Hinkley change-point detection on 1D error stream. |
| `src/regime/adapter_library.py` | 🚧 Phase 4 Day 1.5 | Per-regime AdapterLibrary (dict[int, MLP]) with hard routing. |
| `src/data/real_world.py` | ❌ | Real dataset loader — Phase 5+. |

### Experiment Progress

- **Phase 1** (TabPFN baseline): ✅ 82.96% prequential accuracy on `regime_switching` (single-seed); multi-seed mean 79.89% ± 0.99%. Drift causes ~12–13pp accuracy drop, recovering in ~61 steps.
- **Phase 2** (+ FastCorrector): ✅ KNN/EMA correctors. Multi-seed: NS difference vs Phase 1 across all 3 datasets. Single-seed initial reports of "82.5% best" were within noise.
- **Phase 2.5** (mentor feedback): ✅ Class-balance fix for synthetic data + multi-dataset Phase 2 + `CompositeWindowLoader` (fixed pool + sliding). Composite window: adaptation speed 126→53 steps but overall acc drops 11pp at fr=0.93.
- **Phase 3** (+ GatedEnsemble + Consolidation): ✅ Three-level system implemented (3A GatedEnsemble / 3B FastToInter / 3C MultiTimescaleModel) + 4-way ablation + 3-round architectural iteration (v2 residual-additive fusion / B loose trigger / F split optimizer + cooldown).
  - Multi-seed verdict (n=5, paired t-test):
    - `rotating_boundary`: **+1.00 pp sig (t=+4.07)** — only confirmed gain
    - `regime_switching`: −0.18 pp NS (t=−1.27) — initial single-seed −4.5pp was sampling artifact
    - `combined_drift`: **−0.47 pp sig negative (t=−5.62)** — reproducible regression
  - Diagnostic finding: gate's β ≈ 0 across all variants is RATIONAL — shared adapter has no transferable mid-timescale pattern across regimes. Fix requires per-regime isolation (Phase 4 Design A).
- **Phase 4 Day 0.5** (Cheap diagnostic): ✅ Oracle context-reset (drift-aware) gives **+0.51pp sig** on regime_switching (paired t=+3.25), confirms context pollution accounts for ~1/3 of post-drift gap. Decision: Design A (per-regime adapter library) over Design E (context-reset only) because E would forfeit the rotating_boundary gain.
- **Phase 4 Day 1.5** (architectural redesign): 🚧 In progress. Implementing `src/drift/error_detector.py` (ADWIN on 1D error stream) + `src/regime/adapter_library.py` (per-regime adapter dict + hard routing). Target: preserve rotating_boundary +1pp, flip combined_drift from −0.47pp to non-negative.
- **Phase 5** (paper writing): ❌ Pending Day 1.5 results. Framing: "Per-regime adapter library on frozen tabular foundation models for concept drift", with Phase 3 negative findings as motivation for per-regime isolation.

### Synthetic Datasets

All three generators produce `SyntheticDataset(X, y, regime_labels, drift_points, name)`:

- **`rotating_boundary`**: Gradual relationship drift. 2D Gaussian features, linear decision boundary rotates at `drift_speed=0.003` rad/step. Auto-uses `n_features=2` in scripts.
- **`regime_switching`**: Abrupt drift. Cycles between `n_regimes=3` regimes with **independent feature means and decision weights** (no transferable pattern across regimes by design), each lasting `regime_length` steps (default 500).
- **`combined_drift`**: Mixed drift. First 5 features drift linearly by 2σ over full sequence; decision boundary changes abruptly every 3000 steps.

### Config System

`configs/default.yaml` sets defaults; experiment YAMLs in `configs/experiment/` override specific fields. Currently the scripts use `argparse` directly rather than loading YAML configs at runtime.

## Key Documents

- `progress_report.md` — full experimental narrative (Phase 1 → Phase 4 Day 0.5)
- `phase4_plan.md` — Phase 4 plan (Cheap Diagnostic + Decision branch + Design A spec)
- `implementation_plan_v2.md` — V2 design spec (current code follows this)
- `idea_difference.md` — V1 (PDF) vs V2 design comparison
- `claude-code-workflow-setup.md` — `.claude/` config blueprint
- `results/oracle_summary.md` / `results/multiseed_summary.md` / `results/day05_decision.md` — Phase 4 Day 0.5 outputs

## Project-Level Subagents

Defined in `.claude/agents/`:
- `fullstack-engineer` — implements Python/ML code in `src/` and `scripts/`
- `tester` — writes pytest tests, runs them, reports failures (no implementation changes)
- `experimenter` — runs scripts, analyzes results, plots, writes result paragraphs (no `src/` changes)

Spawn via the built-in Agent tool ("用 fullstack-engineer 实现 ..."). Subagents respect the boundaries declared in their `.md` files.
