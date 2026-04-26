# Phase 4 Day 0.5 — 实验 0a Oracle Context-Reset

**日期**：2026-04-25
**配置**：regime_switching, n_samples=3000, context_size=200, 5 seeds [42, 123, 456, 789, 1024]
**对照**：baseline（无 reset） vs oracle（drift 后持续 soft reset，reset_size=50）
**实现**：drift 命中后，每步 X_ctx 截到最近 `reset_size + (t - last_drift_t)` 个样本，从 50 平滑增长回 200。每个 oracle run 累计 750 次截断。

---

## Per-seed overall accuracy

| seed | baseline | oracle (reset_size=50) | Δ |
|------|---------:|-----------------------:|------:|
| 42   | 0.7846   | 0.7921                 | +0.0075 |
| 123  | 0.7950   | 0.7975                 | +0.0025 |
| 456  | 0.8093   | 0.8125                 | +0.0032 |
| 789  | 0.8071   | 0.8093                 | +0.0022 |
| 1024 | 0.7982   | 0.8082                 | +0.0100 |

每个 seed 上 oracle 都 ≥ baseline，方向一致。

## Aggregate (mean ± std, ddof=1)

| 指标 | Baseline | Oracle reset_size=50 | Δ (oracle − baseline) | Paired t (n=5) |
|---|---|---|---|---|
| **总体 acc** | 0.7989 ± 0.0099 | 0.8039 ± 0.0087 | **+0.0051** (+0.51 pp) | **+3.25** ✓ |
| 漂移前 acc | 0.8236 ± 0.0141 | 0.8236 ± 0.0141 | 0 (无 drift 触发前) | — |
| 漂移后 acc | 0.6632 ± 0.0323 | 0.6868 ± 0.0231 | **+0.0236** (+2.36 pp) | **+2.45** (≈ 显著) |
| 适应速度 (步) | 68.4 ± 10.8 | 59.3 ± 8.6 | **−9.08 步** | **−4.72** ✓✓ |

显著性参考（df=4, two-sided）：critical |t| ≈ 2.78 for p<0.05。

- **总体 acc Δ=+0.51pp paired t=+3.25** > critical，**显著**。
- **post-drift Δ=+2.36pp paired t=+2.45**：略低于临界，p≈0.07，**边缘显著**（方向稳定，5/5 seeds 都正）。
- **适应速度快 9.08 步 paired t=−4.72**：**强显著**。

---

## 一句结论

> Oracle context-reset 在 regime_switching 上**确实**提升总体准确率 **+0.51pp**（80.39% vs 79.89%，paired t=+3.25，p<0.05，n=5），post-drift 准确率 +2.36pp、漂移恢复速度从 68 步缩短到 59 步（paired t=-4.72，强显著）。**Context 污染是 regime_switching 漂移退步的真实杠杆，但提升幅度远小于 -12-13pp 的漂移退步——杠杆只占 ~1/3，剩余 ~2/3 仍是 in-context learning 本身在新 regime 上的样本不足。**

---

## 决策表对位

按 `phase4_plan.md` 的决策点表：

| Oracle 总体准确率 mean | 落在哪个区间 |
|---|---|
| **80.39%** | **80-82% 中段** |

→ 决策**依赖** 0b multi-seed 的 rotating_boundary 上 Phase 3 vs Phase 1 是否显著：
- 显著（+1.18pp 在 std 之外）→ Design A
- 不显著（在 std 之内）→ Design E

故 0a 单独**不能定** Design E/A，需 0b 配合。

---

## 数据与产物

- `results/oracle_baseline_seed{42,123,456,789,1024}.{npz,png}` × 5
- `results/oracle_reset_seed{42,123,456,789,1024}.{npz,png}` × 5
- 每个 npz 含 `n_truncations`（oracle=750）、`oracle_context_reset` flag、`reset_size`
- log：`logs/oracle_runs.log`（4h29m wall）
