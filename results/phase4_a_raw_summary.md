# Phase 4 A — Per-Regime Adapter Library 多 seed 实验汇总

**实验完成日期**：2026-04-27
**Driver 总耗时**：331.9 min ≈ 5.5 h（n_parallel=2）
**配置**：3 数据集 × 5 seeds = 15 runs，全部 status=ok

详见 [phase4_plan.md](../phase4_plan.md) 的 "Design A" 段。

---

## TL;DR

> Phase 4 A 在所有 15 个 run 中 **ADWIN detector 0 次触发** → AdapterLibrary 全程只用 adapter 0 → 系统行为退化为"砍掉 consolidation 的 Phase 3 v2"。
>
> 三数据集 paired t-test vs Phase 3 全部 NS（统计上等价）。vs Phase 1 baseline：rotating_boundary +1.35pp sig（保住主要赢点）/ regime_switching −0.43pp sig（边缘退化但仍在 −1pp 验收线内）/ combined_drift −0.34pp sig 负向（**核心问题未解决**）。
>
> Day 1.5 当前实现下 Design A 的 routing 路径**未真正生效**。combined_drift 翻转为非负这个核心成功条件未达成。需 follow-up 调整 detector 输入信号（用 |error| 或 0/1 indicator 流而非 raw `y_t - y_slow`）。

---

## 1. 三数据集 overall_acc 多 seed 对比

n=5, mean ± std（ddof=1），seeds = [42, 123, 456, 789, 1024]

| 数据集 | Phase 1 baseline | Phase 3 v2+B+F | **Phase 4 A** |
|---|---|---|---|
| `regime_switching` | 79.89 ± 0.99 % | 79.71 ± 0.72 % | **79.46 ± 1.01 %** |
| `rotating_boundary` | 82.07 ± 0.56 % | 83.06 ± 0.76 % | **83.42 ± 0.77 %** |
| `combined_drift`   | 82.24 ± 0.40 % | 81.77 ± 0.56 % | **81.90 ± 0.47 %** |

## 2. Paired t-test (n=5, df=4, |t| ≥ 2.78 → α=0.05 two-sided)

### Phase 4 A vs Phase 1 baseline

| 数据集 | Δ (pp) | paired t | p (two-sided) | sign agreement | 显著性 | 验收线 | 验收 |
|---|---|---|---|---|---|---|---|
| `regime_switching`  | **−0.43** | −6.82 | 0.0024 | 5/5 | **✓ sig 负向** | ≥ −1pp | ✓ 通过（在 −1pp 内）|
| `rotating_boundary` | **+1.35** | +7.76 | 0.0015 | 5/5 | **✓ sig 正向** | ≥ +0.5pp | ✓ 超额（+0.85pp 余量）|
| `combined_drift`    | **−0.34** | −4.85 | 0.0083 | 5/5 | **✓ sig 负向** | ≥ −0.1pp（理想 ≥ +0.5pp）| **✗ 未达**（仍是 sig 负向）|

### Phase 4 A vs Phase 3 v2+B+F

| 数据集 | Δ (pp) | paired t | p (two-sided) | 显著性 |
|---|---|---|---|---|
| `regime_switching`  | −0.25 | −1.59 | 0.1863 | NS |
| `rotating_boundary` | +0.36 | +2.09 | 0.1043 | NS（边缘）|
| `combined_drift`    | +0.13 | +1.10 | 0.3338 | NS |

→ **Phase 4 A 在三数据集上与 Phase 3 v2+B+F 在统计上完全等价**。这不是巧合，是诊断字段直接证明的（见 §4）。

## 3. Post-drift accuracy（漂移后 100 步窗口准确率均值）

| 数据集 | Phase 1 | Phase 3 v2+B+F | Phase 4 A | Δ (4A vs P1) | t | sig? |
|---|---|---|---|---|---|---|
| `regime_switching`  | 66.32 ± 3.23 % | 66.84 ± 3.18 % | 66.80 ± 3.13 % | +0.48 | +1.47 | NS |
| `rotating_boundary` | 82.97 ± 0.84 % | 83.70 ± 1.92 % | 83.90 ± 1.63 % | +0.93 | +2.06 | NS（边缘） |
| `combined_drift`    | 60.00 ± 3.46 % | 59.80 ± 4.21 % | 60.40 ± 3.91 % | +0.40 | +1.63 | NS |

→ Post-drift 上各组均无显著差异，与 overall 一致。

## 4. 诊断字段：detector 与 routing 行为

| 数据集 | seed | detector_events | route_events | n_adapters_final | consolidation_events |
|---|---|---|---|---|---|
| regime_switching   | 42   | 0 | 0 | 1 | 0 |
| regime_switching   | 123  | 0 | 0 | 1 | 0 |
| regime_switching   | 456  | 0 | 0 | 1 | 0 |
| regime_switching   | 789  | 0 | 0 | 1 | 0 |
| regime_switching   | 1024 | 0 | 0 | 1 | 0 |
| rotating_boundary  | 42   | 0 | 0 | 1 | 0 |
| rotating_boundary  | 123  | 0 | 0 | 1 | 0 |
| rotating_boundary  | 456  | 0 | 0 | 1 | 0 |
| rotating_boundary  | 789  | 0 | 0 | 1 | 0 |
| rotating_boundary  | 1024 | 0 | 0 | 1 | 0 |
| combined_drift     | 42   | 0 | 0 | 1 | 0 |
| combined_drift     | 123  | 0 | 0 | 1 | 0 |
| combined_drift     | 456  | 0 | 0 | 1 | 0 |
| combined_drift     | 789  | 0 | 0 | 1 | 0 |
| combined_drift     | 1024 | 0 | 0 | 1 | 0 |

**汇总**：15/15 runs detector 0 触发；adapter library 全程只 active adapter 0；0 routing；0 consolidation。

## 5. 为什么 ADWIN 沉默？

**根因诊断**：

ADWIN 看的是 raw error 流 `error = y_t - y_slow`（其中 `y_slow ∈ [0,1]` 是 TabPFN 概率，`y_t ∈ {0,1}`）。在每个 regime 内：
- 正样本被正确分类时 error ≈ +0.2 ~ +0.4
- 正样本被误分类时 error ≈ +0.6 ~ +0.9
- 负样本被正确分类时 error ≈ −0.2 ~ −0.4
- 负样本被误分类时 error ≈ −0.6 ~ −0.9

只要正负样本大致均衡（合成数据集已通过 Phase 2.5 的 class-balance fix 保证），**raw error 在每个 regime 内的 mean ≈ 0**（正反向误差抵消）。新 regime 进来时错误率变大，但 |error| 的分布拉宽，**mean shift 仍接近 0**。

ADWIN 的 Hoeffding 切点检测的是 |mean(W0) − mean(W1)|，对均值近 0 的两段子窗它**结构性地看不到信号**。

对照实验 0a 的 Oracle context-reset：那里看的是真实 drift_points，不依赖检测。Oracle 在 regime_switching 上能挽回 ~1/3 损失（+0.51pp）。Phase 4 A 当前实现无法获得这个增量，因为 detector 不触发。

**Sanity check 反差**：在 Step 1 单测中，detector 用合成 mean-shift 信号（μ:0→0.5）能正确触发；用 Phase 1 npz 的 0/1 indicator 流（错误率 ∈ [0,1]）也能在已知漂移点附近触发。但用真实管线的 raw error ∈ [−1,1]（mean 近 0）就静默 —— 信号形态不匹配。

## 6. 验收结论

phase4_plan §"Design A 验收"原文：

| 数据集 | 验收线 | Phase 4 A 实际 | 验收 |
|---|---|---|---|
| `regime_switching` | ≥ −1 pp（不打破 NS） | −0.43 pp **sig** | △ 在 −1pp 内但**从 NS 变 sig 负向**（边缘退化）|
| `rotating_boundary` | ≥ +0.5 pp（保住主要赢点） | +1.35 pp sig | ✓ **超额** |
| `combined_drift` | ≥ −0.1 pp（理想 ≥ +0.5 pp） | −0.34 pp sig 负向 | ✗ **未达**（仍是 sig 负向，未翻转）|

**phase4_plan 核心成功条件**："rotating 不丢 + combined 翻成 non-negative" → **未达成**（rotating 保住，combined 未翻）。

**phase4_plan 失败条件**："rotating_boundary 失去 +1pp 赢点 → 触发 Day 1.5 内部回退讨论" → **未触发**（rotating 反而 +1.35pp）。

**整体定性**：Mixed bag, mostly null result。Design A 的"per-regime adapter library + routing"机制在当前实现下未真正激活；当前 5.5h 实验本质上是"无 consolidation 的 Phase 3 v2"对 Phase 1 的复测，与 Phase 3 在统计上等价。

## 7. Follow-up 选项

不做架构改动，仅修 detector 输入信号：

**选项 A**：把 detector 输入从 raw error 改为 `|error|` （∈ [0,1]，value_range=1.0）
- 单步绝对误差对错误率敏感，regime 切换时 mean 会从 ~0.25 跳到 ~0.5 以上
- 改一行：`self.detector.update(abs(error))` + detector 实例化时 `value_range=1.0`

**选项 B**：把 detector 输入改为 0/1 indicator（pred 是否等于 label）
- 等价于错误率信号；sanity check 已证能触发
- 需要在 step() 拿到 hard pred（已有 `int(pred ≥ 0.5)`）

**选项 C**：保留 raw error，放宽 ADWIN：value_range=2.0 → 1.0、delta 0.002 → 0.05
- 风险：增加 false positive，可能在稳态期乱触发 routing

**推荐**：选项 A（最低改动，保留连续信号信息）。预计单参数改动后重跑 5.5h，验证 detector 实际触发能否带动 combined_drift 翻转。

## 8. 数据落盘

- 15 个 npz：`results/multiseed_phase4a_{dataset}_seed{S}.npz`
- 增量 partial：`results/multiseed_phase4a.partial.md`（每个 seed 跑完时 append）
- 每 seed 主图：`results/multiseed_phase4a_{dataset}_seed{S}.png`（含 active adapter id 轨迹）
- 单 run 日志：`logs/multiseed/multiseed_phase4a_{dataset}_seed{S}.log`
- Driver 日志：`logs/multiseed_phase4a/driver.log`
- Run records：`results/multiseed_runlog.json`
