# Phase 5 Stage B (re-aligned, B1+) — Insects Real-World Summary

**完成日期**：2026-05-29
**协议**：B1+ drift-aligned 4 segments × 5 seeds × 3 phases = **60 runs**，全部 `[ok]`
**数据集**：USP DS Insects abrupt_balanced (52,848 samples / 33 features) via Google Drive
**Binarization**：sex-pair `{2,4,11}→0` vs `{3,5,12}→1`（限制：exact species mapping not retrievable in experimental window）
**Wall time**：3606 min ≈ 60.1h CPU (n_parallel=2)
**版本说明**：本文档覆盖 2026-05-03 旧 A+ Stage B 版本；旧 45 runs 数据归档于 `results/archive_misaligned_stage_b/`（保留 transparency / methodology narrative arc）

---

## 0. Stage B 协议演化

| 协议 | 总 runs | F3 detector 触发 | 备注 |
|---|---|---|---|
| 原 A+ (start/middle/end)        | 45 | 0/15 | 14/15 segments 不含任何 documented drift |
| **B1+ aligned (early/mid/late_pre/late_post)** | **60** | **0/20** | 覆盖全 5/5 drift events，每 drift 距 segment 边界 ≥ 200 samples |

→ A+ → B1+ 协议升级后 detector 仍然 0/20 触发，**直接排除了 confound #1（A+ subsampling misalignment）作为 dominant 原因**。confound #2（binarization 稀释 abrupt P(y) shift）被证实为 dominant 机制——详细诊断见 [phase5_confound2_diagnostic.md](phase5_confound2_diagnostic.md)。

## 1. Acc by (config, segment) mean ± std

| Config | early | mid | late_pre | late_post |
|---|---|---|---|---|
| phase1   | 97.292 ± 0.000 | 96.021 ± 0.000 | 98.208 ± 0.000 | 96.729 ± 0.000 |
| phase3   | 97.133 ± 0.170 | 95.921 ± 0.074 | 98.162 ± 0.097 | 96.629 ± 0.168 |
| phase4a  | 97.075 ± 0.148 | 95.842 ± 0.141 | 98.100 ± 0.162 | 96.546 ± 0.140 |

Phase1 std=0 是 TabPFN 在固定 context + 固定数据下确定性产出的预期。

**Segment 难度**：late_pre (98.2%) > early (97.3%) ≈ late_post (96.7%) > mid (96.0%)。

## 2. Paired t-test n=20 (4 seg × 5 seed)

| Comparison | Δ (pp) | std (pp) | t | p | sig? |
|---|---|---|---|---|---|
| phase3 vs phase1    | −0.101 | 0.130 | −3.48 | 0.0025 | **sig 负** |
| **phase4a vs phase1**   | **−0.172** | **0.142** | **−5.42** | **<0.0001** | **sig 负** |
| phase4a vs phase3   | −0.071 | 0.160 | −1.98 | 0.0626 | NS (边缘) |

→ phase4a 比 phase1 显著差（−0.172pp p<0.0001），power n=20 揭示了原 A+ Stage B n=15 上 phase4a 也 sig 负但量级 (−0.075pp) 较小的真实方向。

### 2.1 Per-segment paired (n=5)

| Comparison | early | mid | late_pre | late_post |
|---|---|---|---|---|
| phase4a vs phase1 | **−0.217 sig** (p=0.031) | **−0.179 sig** (p=0.047) | −0.108 NS (p=0.208) | **−0.183 sig** (p=0.043) |
| phase4a vs phase3 | −0.058 NS | −0.079 NS | −0.062 NS | −0.083 NS |

**3/4 段** phase4a vs phase1 sig 负；late_pre 唯一 NS（drift @ local 4228 距末尾仅 772 samples，drift-post 窗口被 ADWIN cooldown 切短，但实际所有段 detector 都 0/5）。phase4a vs phase3 per-segment 全 NS——phase4a 的额外退化主要 vs phase1，与 phase3 几乎持平。

## 3. F3 detector triggers (Phase 4a, 20 runs)

| Segment | per-run events | triggered/5 | total |
|---|---|---|---|
| early     | [0, 0, 0, 0, 0] | 0/5 | 0 |
| mid       | [0, 0, 0, 0, 0] | 0/5 | 0 |
| late_pre  | [0, 0, 0, 0, 0] | 0/5 | 0 |
| late_post | [0, 0, 0, 0, 0] | 0/5 | 0 |
| **TOTAL** | | **0/20** | **0** |

**0/20 跨所有 segment + 所有 seed**。每个 segment 内的 documented drift（local t 1952 / 2672+4256 / 4228 / 4160）距 segment 边界都 ≥ 200，ADWIN min_subwindow + cooldown 缓冲足。protocol-side blame 完全排除。

## 4. F4 reuse 状态

```
Route events:           0
Route actions:          {} (空)
Consolidation events:   0
n_adapters_final:       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
```

**F4 在 Insects 上 vacuous**：detector 0 触发 → 0 routing → reuse 既无激活也无失活机会。F4 reuse 失活假说不能在 Insects 上被检验（即"无效证伪"——不是反例，是无信息）。

## 5. End / segment class imbalance

| Segment | y bincount | min/max | drift_points (local) |
|---|---|---|---|
| early     | [1679, 3121] | 0.538 | [2672, 4256] |
| mid       | [2122, 2678] | 0.792 | [1952] |
| late_pre  | [2957, 1843] | 0.623 | [4228] |
| late_post | [1324, 3476] | 0.381 | [4160] |

late_post 段最不平衡 (28%/72%) — drift @ 4160 后剩余 840 样本 y=1 主导，但 P1 仍达 96.73%（TabPFN 适应快）。无 metric NaN，class imbalance 未让评估失效。

## 6. 与合成 + Stage A 对比矩阵

| 数据集 | F3 触发率 | F4 reuse | Δ phase4a vs phase1 |
|---|---|---|---|
| Synthetic regime_switching (abrupt P(y)) | 12/15 (80%) | 0/25 routes (0%) | −0.29 NS |
| Synthetic rotating_boundary (gradual)     | 0/15 (0%) | — | **+1.52 sig** |
| Synthetic combined_drift (mixed)          | 9/15 (60%) | 0/14 routes (0%) | −0.47 NS |
| **Electricity (gradual, Stage A)**            | **1/15 (7%)** | 0/1 routes (0%) | −0.064 NS |
| **Insects abrupt (Stage B, A+ archived)**      | 0/15 (0%) | vacuous | −0.075 sig 负 |
| **Insects abrupt (Stage B1+ aligned)**       | **0/20 (0%)** | **vacuous** | **−0.172 sig 负** |

**Insects 真实 abrupt drift 上 F3 触发率从合成 regime 80% 退化到 0%**——这是 Phase 5 最强 negative finding，揭示了"indicator detector + binary 系统 + frozen TabPFN"组合在真实多类 abrupt drift 上的 systemic failure mode。

## 7. Stage B B1+ verdict

1. **F3 完全沉默 (0/20)** 在 drift-aligned + buffer-保证 协议下是 robust negative finding，不可归咎于 subsampling
2. **F4 vacuous**——detector 不触发就不能测 reuse；F4 在 Insects 上的 reuse 失活假说**无效证伪**
3. **phase4a 显著退化于 phase1 (−0.172pp p<0.0001)** 跨 n=20，3/4 segments per-segment sig 负——adapter library 引入的 input_dim=33 MLP cold-start cost 在 detector 不激活时是 net loss
4. **机制诊断**见 [phase5_confound2_diagnostic.md](phase5_confound2_diagnostic.md)：indicator mean shift |Δ| ≤ 0.019（全 segment 全 drift），比 ADWIN 阈值小 10×

---

## 文件清单

- `results/multiseed_phase{1,3,4a}_real_insects_{early,mid,late_pre,late_post}_seed{S}.npz` × 60
- `results/multiseed_phase{1,3,4a}_real_insects_{early,mid,late_pre,late_post}_seed{S}.png` × 60
- `results/multiseed_phase5_insects_aligned.partial.md` (live)
- `results/phase5_confound2_diag_insects_{seg}_{indicator,pred1,soft_err}.png` × 12 (γ 诊断)
- `results/archive_misaligned_stage_b/` (旧 A+ Stage B 45 npz + partial md，论文 methodology narrative)
- `logs/stage_b_insects_aligned_driver.log`
- `logs/multiseed/multiseed_phase{1,3,4a}_real_insects_{seg}_seed{S}.log` × 60
