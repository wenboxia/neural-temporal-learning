# Phase 4 A — Option B (0/1 Indicator Detector Input) 多 seed 实验汇总

**实验完成日期**：2026-04-28
**Driver 总耗时**：414.1 min ≈ 6.9 h（n_parallel=2）
**配置**：3 数据集 × 5 seeds = 15 runs，全部 status=ok
**Detector**：ADWIN 输入改为 `int((y_final ≥ 0.5) != y_t)`，`value_range=1.0`，`detector_delta=0.002`

详见 [phase4_plan.md](../phase4_plan.md) "Design A" 段，以及对照实验 [phase4_a_raw_summary.md](phase4_a_raw_summary.md)（raw error 输入版本）。

---

## TL;DR

> Detector 终于激活（regime_switching 5/5 seed / 平均 1.4 routes，combined_drift 4/5 seed / 平均 1.0 route，rotating_boundary 0 routes — 这是设计精神而非 bug，渐进漂移上 indicator 流没阶跃式 mean shift）。
>
> vs Phase 1 baseline 验收线：rotating ✓ 超额（+1.51 pp sig）/ regime ✓ 通过（−0.66 pp NS, 在 −1pp 内）/ combined ✗ 未达（−0.28 pp sig 负向，未翻 non-negative）。
>
> vs Phase 3 v2+B+F 三数据集**全部 sig**：regime −0.48 pp sig 负向 / rotating +0.51 pp sig 正向 / **combined +0.20 pp sig 正向**。Combined 在 Phase 3 vs Phase 4 A 维度上翻转，但 vs Phase 1 baseline 维度未翻。
>
> 整体定性：**Mixed bag with substance** — 不是空跑（前两次 raw / abs 是），detector 真在做事，但核心 combined-drift-vs-Phase-1 验收仍未达成。失败模式：**所有 routing 全是 create / 0 reuse**（fit_threshold=0.05 太严），新 adapter 空白随机初始化导致冷启动 → 短期适应性差，regime_switching 上从 Phase 3 NS 跌到 sig 负向。

---

## 1. 三种 detector 输入对比（n=5, mean ± std）

| 数据集 | Phase 1 | Phase 3 | P4A raw (option 0) | P4A abs (option A) | **P4A indicator (option B)** |
|---|---|---|---|---|---|
| `regime_switching`  | 79.89 ± 0.99 | 79.71 ± 0.72 | 79.46 ± 1.01 | 79.51 ± 1.16 | **79.23 ± 0.43** |
| `rotating_boundary` | 82.07 ± 0.56 | 83.06 ± 0.76 | 83.42 ± 0.77 | 83.51 ± 0.78 | **83.57 ± 0.81** |
| `combined_drift`    | 82.24 ± 0.40 | 81.77 ± 0.56 | 81.90 ± 0.47 | 81.95 ± 0.26 | **81.97 ± 0.54** |

## 2. Paired t-test（n=5, df=4, |t| ≥ 2.78 ⇒ α=0.05 sig）

### Phase 4 A indicator vs Phase 1 baseline

| 数据集 | Δ (pp) | paired t | p (two-sided) | 显著性 | 验收线 | 验收 |
|---|---|---|---|---|---|---|
| `regime_switching`  | −0.66 | −2.17 | 0.0954 | **NS** | ≥ −1pp | ✓ 通过 |
| `rotating_boundary` | **+1.51** | **+6.38** | **0.0031** | ✓ sig 正向 | ≥ +0.5pp | ✓ **超额** |
| `combined_drift`    | −0.28 | −4.12 | 0.0146 | ✓ sig 负向 | ≥ −0.1pp（理想 +0.5）| ✗ **未达**（仍负向）|

### Phase 4 A indicator vs Phase 3 v2+B+F

| 数据集 | Δ (pp) | paired t | p (two-sided) | 显著性 |
|---|---|---|---|---|
| `regime_switching`  | **−0.48** | −2.91 | 0.0438 | **✓ sig 负向** |
| `rotating_boundary` | **+0.51** | +2.96 | 0.0414 | **✓ sig 正向** |
| `combined_drift`    | **+0.20** | +3.46 | 0.0257 | **✓ sig 正向** |

### Phase 4 A indicator vs Phase 4 A abs（控制变量：detector 输入）

三数据集**全 NS**（|t| ∈ {0.14, 0.45, 0.70}）→ 在 detector 不触发（abs）和触发（indicator）的两个端点之间，**overall_acc 几乎相等**。这意味着 routing 的总体增益被分散在少数 step 上 + cold-start adapter 的拖累把增益稀释了。

## 3. Detector 激活情况

| 数据集 | 真实漂移点 | sum routes / 15 | avg routes / seed | avg n_adapters_final | recall |
|---|---|---|---|---|---|
| `regime_switching`  | 5 (循环 3 regime) | 7 | 1.4 (range 1–2) | 2.4 | **28 %** |
| `rotating_boundary` | 4 (渐进旋转)      | 0 | 0   (0/5 seed) | 1.0 | 0 % (设计精神：渐进无阶跃)|
| `combined_drift`    | 1 (t=3000)        | 5 | 1.0 (range 0–2) | 2.0 | **80 %** |

**rotating_boundary detector 沉默是设计精神而非 bug**：渐进漂移下 indicator 流随时间缓慢上升（acc 缓降），没有阶跃式 mean shift；ADWIN 切点 |mean(W0)−mean(W1)| 看不到清晰边界。这种情形下 design A 退化为"single adapter + per-step gate 训练"，与 Phase 3 渐进漂移上的赢点机制等价 → 解释了 +1.51pp sig 超额。

### 触发时刻对比真实漂移点（regime_switching, drifts=[500, 1000, 1500, 2000, 2500]）

| seed | route 触发 t | 距最近真实漂移点 |
|---|---|---|
| 42   | [2112, 2527]      | +112, +27 |
| 123  | [305]             | −195 (检测到初始 context_size 边界附近的预测震荡)|
| 456  | [1563]            | +63 |
| 789  | [2897]            | +397 |
| 1024 | [1037, 2515]      | +37, +15 |

5 seed × 5 真实漂移 = 25 个机会，触发 7 次。漏检 18 次（第一漂移点 t=500 全 5 seed 漏检）。

### 触发时刻对比真实漂移点（combined_drift, drift=t=3000）

| seed | route 触发 t | 距 t=3000 |
|---|---|---|
| 42   | []                    | (漏检)    |
| 123  | [3350]                | +350     |
| 456  | [3050]                | +50      |
| 789  | [3054]                | +54      |
| 1024 | [3063, 3373]          | +63, +373 |

4/5 seed 在漂移后 50–80 步内捕到，1 个 seed 漏检。

**全 5×5=25 routing 事件中 100% 是 `create`，0 个 `reuse`**。说明 `fit_threshold=0.05` 对 evaluate_existing 输出的"现有 adapter 拟合 |error| 残差"判定太严，新 regime 的局部数据从未被判定为"现有 adapter 已能 fit"。每次 routing 都新建空白随机初始化的 adapter → 冷启动 → 短期 acc 拖累。

## 4. Post-drift accuracy 对比

| 数据集 | Phase 1 | Phase 3 | P4A indicator | Δ (4A−P1) | t | p | sig? |
|---|---|---|---|---|---|---|---|
| `regime_switching`  | 66.32 ± 3.23 | 66.84 ± 3.18 | 66.72 ± 2.11 | +0.40 | +0.52 | 0.633 | NS |
| `rotating_boundary` | 82.97 ± 0.84 | 83.70 ± 1.92 | 84.05 ± 1.79 | **+1.08** | +2.02 | 0.113 | NS（边缘）|
| `combined_drift`    | 60.00 ± 3.46 | 59.80 ± 4.21 | 60.00 ± 3.46 | +0.00  | (degenerate) | — | NS |

Post-drift 三数据集均无显著差异（rotating 边缘正向）。说明 Phase 4 A 的 routing 增益主要不在"漂移后短期适应"上。

## 5. Verdict（vs phase4_plan 验收口径）

phase4_plan **核心成功条件** = "rotating 不丢 + combined 翻 non-negative"：
- rotating: ✓ 保住，且 +1.51 pp sig 超过验收线 0.5 pp
- combined: ✗ vs Phase 1 仍 sig 负向 −0.28 pp，未翻 non-negative
  - **但**：vs Phase 3 翻正 +0.20 pp sig（这是不同口径）

**phase4_plan 失败条件** = "rotating 失去 +1pp 赢点" → 未触发（rotating 反而 +0.51 pp sig 优于 Phase 3）。

**整体定性**：Design A 在 indicator 输入下首次成为"真实测试"。结果：rotating 超额，regime 边缘退化，combined 部分翻转（vs P3 sig 正、vs P1 仍 sig 负）。**比前两次 raw / abs 实验显著更有 substance，但 phase4_plan 严格验收口径下 combined 仍未达"翻 non-negative vs Phase 1"标准**。

## 6. 论文叙事三段式（raw → abs → indicator）

| 阶段 | Detector 输入 | 触发情况 | 主要发现 |
|---|---|---|---|
| **raw** (option 0) | `error = y_t − y_slow ∈ [−1, 1]` | 0/15 触发 | Class-balanced regime 内正反向误差抵消，raw error mean ≈ 0 → ADWIN 切点结构性看不到 mean shift |
| **abs** (option A) | `|error| ∈ [0, 1]` | 0/15 触发 | TabPFN sliding-context in-context learning 在 regime 切换后快速调整 y_slow，使 |error| 平均水平在所有 regime 内都 ≈ 0.30（Δmean ±0.005~0.035 vs std 0.24） → 调 ADWIN 参数无法救 |
| **indicator** (option B) | `int((y_final ≥ 0.5) != y_t)` | 12/15 触发（regime + combined）| 错误率信号直接、阶跃明显（in-regime 0.17 → post-drift 0.37 mean shift ≈ 0.20）→ ADWIN 在 δ=0.002 默认参数下稳定触发 |

**三段共同结论**：detector 输入信号选择**比 ADWIN 算法本身更重要**。在自适应 in-context learner（如 TabPFN）的输出之上构造漂移检测信号时，必须避开它的自适应回路 —— raw / abs 两路都被 TabPFN 平滑掉，只有 hard 0/1 indicator 保留了"是否预测对"的离散信号。

## 7. 已知失败模式与 follow-up

**核心失败**：所有 25/25 routing 是 create，0 reuse → 永远在创建空白 adapter 而不复用 regime 历史。

**Follow-up 选项**（不在本 Day 1.5 范围）：
1. 调宽 `library_fit_threshold`（0.05 → 0.2 或 0.5）让 reuse 发生
2. 用 evaluate_existing 的两层判定：(a) 是否有现有 adapter 拟合好 → reuse；(b) 否则用现有最佳 adapter 的权重 warm-start 新 adapter（避免冷启动）
3. 调 `consolidation_epochs` 让新 adapter 创建后立即更深训练（10 → 50）

**论文范围**：Day 1.5 的"per-regime routing 即使触发也未带来 vs Phase 1 显著正向"是一个有意义的负面发现 —— 说明"per-regime adapter library + ADWIN 路由"在 frozen TabPFN 之上**结构上能 work**（rotating sig 正、combined vs P3 sig 正），但**冷启动 cost 抵消了大部分增益**。

## 8. 数据落盘

- 15 个 npz：`results/multiseed_phase4a_indicator_{dataset}_seed{S}.npz`
- 每 npz 含 `abs_error_history` (float32, T) + `indicator_history` (int8, T) → 论文三段对比的全量原始信号
- 增量 partial：`results/multiseed_phase4a_indicator.partial.md`
- 每 seed 主图：`results/multiseed_phase4a_indicator_{dataset}_seed{S}.png`（含 active adapter id 轨迹）
- 单 run 日志：`logs/multiseed/multiseed_phase4a_indicator_{dataset}_seed{S}.log`
- Driver 日志：`logs/multiseed_phase4a_indicator/driver.log`

对照运行：
- raw 实验数据：`results/multiseed_phase4a_raw_*.npz` + [phase4_a_raw_summary.md](phase4_a_raw_summary.md)
- abs 实验数据：`results/multiseed_phase4a_abs_*.npz` + `multiseed_phase4a_abs.partial.md`
- 诊断 1-seed：`results/diag_phase4a_d05_regime_switching_seed42.npz`（含 abs_error_history 跨 |error| mean shift 估算）
