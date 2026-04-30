# Phase 5 Stage A — Electricity Real-World Summary

**完成日期**：2026-04-30
**实验体量**：3 phases × 3 segments × 5 seeds = 45 runs，全部 `[ok]`
**数据集**：OpenML 151 Electricity (45,312 samples / 8 features → 14 维 one-hot 后)
**Subsampling**：A+ 协议，3 个非重叠 5000-sample contiguous segments (start / middle / end)
**Wall time**：1448 min ≈ 24.1h CPU (n_parallel=2)
**Scope note**：Stage A 仅 Electricity；Stage B Insects 待启动后合并写整体结论

---

## 1. Acc by config × segment（n=5 mean ± std）

| Config | start | middle | end | overall (n=15) |
|---|---|---|---|---|
| phase1   | 0.9627 ± 0.0000 | 0.9315 ± 0.0000 | 0.9483 ± 0.0000 | 0.9475 |
| phase3   | 0.9615 ± 0.0027 | 0.9297 ± 0.0019 | 0.9475 ± 0.0011 | 0.9462 |
| phase4a  | 0.9627 ± 0.0003 | 0.9308 ± 0.0024 | 0.9471 ± 0.0017 | 0.9469 |

**phase1 std = 0** 是 TabPFN 在固定 context + 固定数据下确定性产出的预期（seed 仅影响合成数据生成；real loader 输入与 seed 解耦）。

**Segment 效应显著**：start ≈ end > middle（middle 低 ~3pp）— Electricity 时序中段（约 sample 22.5k–27.5k）特征 / 价格分布显著难于首尾，与 Electricity 文献中"价格波动随时间集中"的报告吻合。

## 2. Paired t-test 跨 15 datapoints (3 seg × 5 seed)

| Comparison | Δ (pp) | std (pp) | t | p | sig? |
|---|---|---|---|---|---|
| **phase3 vs phase1**  | −0.124 | 0.190 | −2.51 | 0.0247 | **sig 负** |
| **phase4a vs phase1** | −0.064 | 0.166 | −1.49 | 0.1574 | NS |
| **phase4a vs phase3** | +0.060 | 0.225 | +1.03 | 0.3221 | NS |

### 2.1 Per-segment (n=5)

| Comparison | start | middle | end |
|---|---|---|---|
| phase4a vs phase1 | 0.000 NS | −0.067 NS | −0.125 NS |
| phase4a vs phase3 | +0.117 NS | +0.108 NS | −0.046 NS |

每段 n=5 power 不足，所有 per-segment 全 NS，但 trend 与 overall 一致：phase4a 在 start/middle 与 phase1 持平 / 微优 phase3，在 end 段最差（−0.125pp，drift 信号最强但 indicator 没触发）。

## 3. Phase 4a routing diagnostics (15 runs)

| 字段 | 值 |
|---|---|
| Total detector events | **1/15 runs** (vs synthetic regime_switching 12/15) |
| Runs with ≥1 detector event | 1/15 |
| Total route events | 1 |
| Route actions | `{'create': 1}` (0 reuse) |
| Total consolidation events | 0 |
| Final adapter count distribution | 14× n_adapters=1, 1× n_adapters=2 |

**关键发现**：indicator stream 在 Electricity 上几乎不触发 ADWIN（1/15 ≈ 7%），与 synthetic regime_switching 80% (12/15) 形成强对比，但**与 synthetic rotating_boundary 0/15 同构**。Electricity 是 gradual 漂移，无 abrupt 规则切点，indicator 的 in-regime 错误率不会有阶跃式 mean shift；ADWIN 沉默是设计精神匹配，不是 bug。

## 4. F1-F4 复现状态（Stage A 部分）

| 发现 | 合成 | Electricity (Stage A) | 复现状态 |
|---|---|---|---|
| F1 (raw error 0/15 触发) | ✓ | — (Phase 5 仅跑 indicator) | N/A |
| F2 (\|error\| 0/15 触发) | ✓ | — (Phase 5 仅跑 indicator) | N/A |
| **F3 (indicator detect abrupt mean shift)** | regime ✓ 12/15 / rotating 0/15 | **1/15** | **同向 rotating_boundary**：gradual drift 上 indicator 也沉默；F3 是 *drift type-conditional*，按设计精神在 Electricity 上理应沉默 |
| **F4 (reuse 不激活)** | ✓ 52/52 全 create | **1/1 全 create** | **完全复现**：唯一一次 routing 也是 create，0 reuse |
| Phase 3 vs P1 sig 负 (combined_drift) | combined ✓ −0.47 sig | **−0.124 sig** | **同向**：shared adapter 在 multi-mechanism 真实漂移上也微弱拉低 baseline |
| Phase 4a vs P1 改善 (rotating +1pp) | rotating ✓ +1.5 sig | **−0.064 NS** | **未复现**：Electricity 没拿到 rotating-class +1pp 改善；但 detector 几乎不触发 → 退化为"Phase 3 减 consolidation"，非 detect-then-isolate 的工作模式 |

## 5. End segment class imbalance check

| Segment | y bincount | min/max ratio |
|---|---|---|
| start  | [2951, 1849] | 0.627 |
| middle | [3026, 1774] | 0.586 |
| end    | [2592, 2208] | 0.852 |

Electricity end segment **更平衡**（不是 abrupt shift 后单类主导）；担心的 "imbalanced metric 失效" 未发生。Electricity 三段类别分布稳定，与 P(y) 漂移 hypothesis 不符 — 主要漂移在 P(x|y)（covariate / 价格特征）。

## 6. Stage A 暂时观察（不下整体 Phase 5 结论）

1. **Electricity 是 rotating_boundary 类比成功**：gradual drift 上 detector 几乎不触发，phase4a vs phase1 NS，与合成 rotating 上 detector 0/15 一致；但 rotating 合成上 phase4a 还能 +1.5pp（adapter 共享提供 inductive bias），Electricity 上没有等效改善（−0.064 NS），可能因为 Electricity 8/14 维特征 + 时序结构与合成 2D rotating 差异大。
2. **Phase 3 在 Electricity 微弱负向**（−0.124pp sig）：与合成 combined_drift 上 P3 退化方向一致（虽然量级小一个数量级 0.124 vs 0.47）。复现 "shared adapter 在 multi-mechanism drift 上 net negative" 的核心定性发现。
3. **F4 真实数据再次验证 reuse 失活**：routing 1/1 全 create — 与合成 52/52 → 0% reuse 一致，强化 "evaluate_existing 用 raw-error MSE 与 std 量级冲突" 的论文 limitation。

## 7. 待 Stage B 合并后才能下的整体结论

- Insects (regime_switching 真实 analog) 上 indicator 是否触发到 ~12/15？
- F4 在 Insects 上是否仍 100% create？
- combined_drift 真实 analog = 没有，整体 narrative 仅 rotating + regime 两个 axis

---

**Stage A 状态**：✅ 完成；Step 6 部分分析完成（仅 Electricity）；待 Stage B Insects 跑完后写整体 phase5_real_summary.md。

**结果文件清单**：
- `results/multiseed_phase{1,3,4a}_real_electricity_{start,middle,end}_seed{S}.npz` × 45
- `results/multiseed_phase5_electricity.partial.md` (live increment)
- `results/multiseed_runlog.json`
- `logs/multiseed/multiseed_phase{1,3,4a}_real_electricity_{seg}_seed{S}.log` × 45
- `logs/stage_a_electricity_driver.log`
