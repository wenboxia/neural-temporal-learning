# Phase 4 Day 0.5 — 实验 0b Multi-Seed 重跑 Phase 1 / 2 / 3

**日期**：2026-04-26 → 2026-04-27（45 runs，wall ~22h，n_parallel=3）
**配置**：3 phase × 3 dataset × 5 seed [42, 123, 456, 789, 1024]
**目的**：补 Phase 3 现有数字的方差估计；判断 Phase 3 vs Phase 1 在 rotating_boundary 上 +1.18pp 是否显著。
**显著性判据**：paired t-test, df=4, two-sided, critical |t| ≈ 2.78 → 报告中 **sig** = paired t 超过临界，**NS** = 噪声范围内。

---

## 主表：Overall accuracy (mean ± std, ddof=1, n=5)

| Dataset | Phase 1 (TabPFN) | Phase 2 (KNN) | Phase 3 (v2+B+F) |
|---|---|---|---|
| regime_switching | **0.7989 ± 0.0099** | 0.7993 ± 0.0095 | 0.7971 ± 0.0072 |
| rotating_boundary | 0.8207 ± 0.0056 | 0.8219 ± 0.0090 | **0.8306 ± 0.0076** |
| combined_drift | **0.8224 ± 0.0040** | 0.8210 ± 0.0039 | 0.8177 ± 0.0056 |

粗体 = 该数据集上的最佳。

## Phase 3 vs Phase 1（paired, n=5）

| Dataset | Δ (overall) | paired std | paired t | 结论 |
|---|---:|---:|---:|---|
| regime_switching  | −0.18 pp | 0.0032 | −1.27 | **NS**（噪声内）|
| **rotating_boundary** | **+1.00 pp** | **0.0055** | **+4.07** | **sig**（std 之外，方向正）|
| combined_drift   | **−0.47 pp** | 0.0019 | **−5.62** | **sig 负向**（Phase 3 反而退步）|

## Phase 2 (KNN) vs Phase 1（paired, n=5）

| Dataset | Δ (overall) | paired t | 结论 |
|---|---:|---:|---|
| regime_switching  | +0.04 pp | +0.36 | **NS** |
| rotating_boundary | +0.12 pp | +0.44 | **NS** |
| combined_drift   | −0.14 pp | −1.66 | **NS** |

→ KNN corrector 在三数据集上**全部非显著**，方向也不一致。

## Phase 3 post-drift / 适应速度（paired, vs Phase 1）

| Dataset | post-drift Δ | post t | speed Δ (步) | speed t |
|---|---:|---:|---:|---:|
| regime_switching | +0.52 pp | +1.54 [NS] | −3.88 | −1.95 [NS] |
| rotating_boundary | +0.73 pp | +1.21 [NS] | −3.08 | −2.53 [NS] |
| combined_drift | −0.20 pp | −0.53 [NS] | −9.00 | −1.06 [NS] |

post-drift 和适应速度都没到显著，但 rotating_boundary 上 Phase 3 把 adaptation_speed 压到 0（每个 seed 都是 0）。

---

## Per-seed 原始数据（overall_acc）

### regime_switching
| seed | Phase 1 | Phase 2 (KNN) | Phase 3 |
|---|---:|---:|---:|
| 42 | 0.7846 | 0.7886 | 0.7875 |
| 123 | 0.7950 | 0.7943 | 0.7925 |
| 456 | 0.8093 | 0.8114 | 0.8036 |
| 789 | 0.8071 | 0.8068 | 0.8043 |
| 1024 | 0.7982 | 0.7954 | 0.7975 |

### rotating_boundary
| seed | Phase 1 | Phase 2 (KNN) | Phase 3 |
|---|---:|---:|---:|
| 42 | 0.8302 | 0.8307 | 0.8379 |
| 123 | 0.8192 | 0.8200 | 0.8286 |
| 456 | 0.8183 | 0.8232 | 0.8361 |
| 789 | 0.8156 | 0.8075 | 0.8186 |
| 1024 | 0.8200 | 0.8279 | 0.8321 |

→ Phase 3 在所有 5 seeds 上都 ≥ Phase 1（5/5 同向），最差 +0.30pp（seed789），最好 +1.78pp（seed456）。

### combined_drift
| seed | Phase 1 | Phase 2 (KNN) | Phase 3 |
|---|---:|---:|---:|
| 42 | 0.8206 | 0.8213 | 0.8150 |
| 123 | 0.8187 | 0.8183 | 0.8113 |
| 456 | 0.8196 | 0.8171 | 0.8154 |
| 789 | 0.8252 | 0.8210 | 0.8219 |
| 1024 | 0.8279 | 0.8273 | 0.8250 |

→ Phase 3 在所有 5 seeds 上都 < Phase 1（5/5 反向退步）。

---

## 关键结论

1. **Phase 3 v2+B+F 在 rotating_boundary 上 +1.00pp 是真信号**，不是 single-seed 噪声。Phase 1 vs Phase 3 paired t=+4.07 显著超过 critical 2.78（p≈0.015 双侧），且 5/5 seeds 同向。多 seed 估值 +1.00pp 比 single-seed 旧值 +1.18pp 略低，在估计精度内一致。

2. **Phase 3 在 combined_drift 上 -0.47pp 退步显著**（paired t=-5.62）。Phase 3 不是普适改进，而是数据集相关的 trade-off。

3. **Phase 3 在 regime_switching 上 -0.18pp 处于噪声内**。先前 Phase 3 进度报告里的 "regime_switching -4.5pp 退步" 应该是 single-seed 不利点 + std≈1pp 的累积，多 seed 拉平后不成立。**重要修正：Phase 3 在 regime_switching 上并未真正退步，只是没有改进。**

4. **Phase 2 KNN 在三数据集上全部 NS**：FastCorrector KNN 不是有效杠杆（与之前 single-seed 直觉一致）。

---

## 决策点对位（按 phase4_plan.md 决策表）

| 输入 | 取值 |
|---|---|
| Oracle 总体准确率 (mean) | **80.39%**（来自 0a） |
| Oracle 落区间 | 80-82% |
| Phase 3 vs Phase 1 on rotating_boundary | **+1.00 pp, paired t=+4.07, sig**（差异在 std 之外） |

→ 决策表："80-82% × 显著" → **Design A**（per-regime adapter library，adapter 仍有救）。

---

## 数据与产物

- `results/multiseed_{phase1,phase2,phase3}_{regime_switching,rotating_boundary,combined_drift}_seed{42,123,456,789,1024}.{npz,png}` × 45
- `results/multiseed_extracted.json`：脚本提取的全部数字
- `results/multiseed_runlog.json`：driver 写入的执行记录（status / elapsed / log path）
- `logs/multiseed/multiseed_*.log` × 45：每 run 的 stdout/stderr
- 总 wall ~22h（n_parallel=3 on 10 cores），phase2 单 run 占 ~4h（内含 baseline+KNN+EMA 三 corrector 串行）
