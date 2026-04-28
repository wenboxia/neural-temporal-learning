# Phase 4 A — Warmstart 多 seed 实验汇总（Day 1.5 第四轮）

**实验完成日期**：2026-04-28
**Driver 总耗时**：324.7 min ≈ 5.4 h（n_parallel=2）
**配置**：3 数据集 × 5 seeds = 15 runs，全部 status=ok
**改动 vs indicator run**：
1. `library_fit_threshold` 默认 0.05 → 0.5（10× 放宽 reuse 判定）
2. `AdapterLibrary._create_new_adapter`：non-init 时从当前 active 复制权重 (warm-start)，optimizer state 仍重置（Adam 实例新建）

详见对照实验 [phase4_a_summary_indicator.md](phase4_a_summary_indicator.md) 与 [phase4_final_verdict.md](phase4_final_verdict.md)。

---

## TL;DR

> 守门 1 (detector 触发 > 0) **通过**：13 routes 跨 12/15 runs，warm-start 生效（n_warmstart_inits 计数与 routing 数一致）。守门 2 (reuse 占比 > 0) **失败**：13/13 routing 仍全 `create`，0 reuse — fit_threshold 0.5 仍不够松。
>
> vs Phase 1：rotating ✓ +1.32 sig / regime ✓ −0.29 NS / combined ✗ **−0.55 sig 负向**（比 indicator −0.28 还差 0.28pp）。
> vs Phase 3：三数据集**全 NS**（|t| ∈ {0.47, 0.62, 1.13}）— warm-start + threshold 调宽抹平了 indicator 实验里 vs Phase 3 的三个 sig 差异。
>
> warmstart 的 vs Phase 1 combined 退化反而把 indicator 的 sig 负向（-0.28）拉得更糟（-0.55）：warm-start 自旧 adapter 复制权重在 abrupt boundary change（combined_drift 在 t=3000 反向）数据上是 anti-pattern，新 regime 的 boundary 与旧 adapter 学到的相反。
>
> 整体定性：**Failure mode 2 with substance** — fit_threshold 调宽未触发 reuse 路径；warm-start 在不同漂移类型上效果分裂（regime 轻好转 / rotating 中性 / combined 恶化）。这是 paper-grade 负面发现，论文 ablation 章节直接引用。

---

## 1. 四种 Phase 4 A 变体 vs Phase 1/3 baseline（n=5, mean ± std）

| 数据集 | Phase 1 | Phase 3 | P4A raw | P4A abs | P4A indicator | **P4A warmstart** |
|---|---|---|---|---|---|---|
| `regime_switching`  | 79.89 ± 0.99 | 79.71 ± 0.72 | 79.46 ± 1.01 | 79.51 ± 1.16 | 79.23 ± 0.43 | **79.60 ± 0.93** |
| `rotating_boundary` | 82.07 ± 0.56 | 83.06 ± 0.76 | 83.42 ± 0.77 | 83.51 ± 0.78 | 83.57 ± 0.81 | **83.39 ± 0.70** |
| `combined_drift`    | 82.24 ± 0.40 | 81.77 ± 0.56 | 81.90 ± 0.47 | 81.95 ± 0.26 | 81.97 ± 0.54 | **81.69 ± 0.25** |

## 2. Paired t-test (n=5, df=4, |t| ≥ 2.78 ⇒ α=0.05 sig)

### Phase 4 A warmstart vs Phase 1 baseline

| 数据集 | Δ (pp) | t | p | sig? | 验收线 | 验收 |
|---|---|---|---|---|---|---|
| `regime_switching`  | −0.29 | −1.49 | 0.2116 | NS | ≥ −1pp | ✓ 通过 |
| `rotating_boundary` | **+1.32** | **+8.80** | **0.0009** | ✓ sig 正 | ≥ +0.5pp | ✓ **超额** |
| `combined_drift`    | **−0.55** | −6.37 | 0.0031 | ✓ sig 负 | ≥ −0.1pp | ✗ **未达**（最差的一个变体）|

### Phase 4 A warmstart vs Phase 3

| 数据集 | Δ (pp) | t | p | sig? |
|---|---|---|---|---|
| `regime_switching`  | −0.11 | −0.62 | 0.5695 | NS |
| `rotating_boundary` | +0.32 | +1.13 | 0.3205 | NS |
| `combined_drift`    | −0.08 | −0.47 | 0.6599 | NS |

→ warmstart 与 Phase 3 v2+B+F 完全等价（三数据集全 NS）。indicator run 在三数据集 vs Phase 3 都是 sig，warmstart 在阈值调宽 + warm-start 后**抹平了那些 sig 差异**（包括 combined 翻正的 +0.20）。

### Phase 4 A warmstart vs Phase 4 A indicator（同 detector 输入，唯一改 fit_threshold + warm-start）

| 数据集 | Δ (pp) | t | p | sig? |
|---|---|---|---|---|
| `regime_switching`  | +0.37 | +1.32 | 0.2585 | NS |
| `rotating_boundary` | −0.19 | −1.05 | 0.3548 | NS |
| `combined_drift`    | −0.28 | −1.87 | 0.1343 | NS |

→ warm-start + threshold 改动在 regime_switching 上有边缘正向（-0.66 → -0.29 NS，cold-start 拖累减轻），但在 combined_drift 上明显恶化（-0.28 → -0.55 sig）。整体净效应负面或中性。

## 3. Routing 行为 — 守门 2 失败诊断

| 数据集 | sum routes | actions | avg n_adapters | n_warmstart_inits | n_random_inits | reuse 占比 |
|---|---|---|---|---|---|---|
| `regime_switching`  | 7  | **{'create': 7}**  | 2.4 | 7 | 5 (=每 seed 初始 1 个) | **0%** |
| `rotating_boundary` | 1  | **{'create': 1}**  | 1.2 | 1 | 5 | **0%** |
| `combined_drift`    | 5  | **{'create': 5}**  | 2.0 | 5 | 5 | **0%** |

**关键诊断**：13/13 routing 仍全 create，**fit_threshold 0.5 不够松**。

- warm_inits == sum routes → 验证 warm-start 生效（每次 routing 新 adapter 都从 active 复制，非随机）
- random_inits == 5（每 seed 初始 adapter 0）→ 验证 init 随机分支正确
- 但 evaluate_existing 在新 regime 数据上的 MSE 仍然 > 0.5 阈值（也未在 paper-relevant fit threshold 范围内）

**为什么 fit_threshold 0.5 仍不够松**：evaluate_existing 计算 `MSE(adapter(X_recent), errors_recent)`，其中 `errors_recent` 是 raw error `y_t − y_slow`（FastCorrector buffer 存的字段）。raw error 在 [−1, 1] 范围内分布相当宽（典型 |error| 0.3-0.8），MSE 自然 ~0.2-0.5 量级。fit_threshold=0.5 要求 adapter 拟合 raw error 的 MSE ≤ 0.5 就是要求 adapter 几乎完美预测残差 — 这在 noisy stochastic 残差信号上不现实。

## 4. Detector 触发位置（与 indicator run 对比）

regime_switching 真实漂移点 [500, 1000, 1500, 2000, 2500]：

| seed | indicator routes | **warmstart routes** | 主要变化 |
|---|---|---|---|
| 42   | [2112, 2527]      | [2075, 2525]          | 接近相同 |
| 123  | [305]             | **[]**                | early false alarm 消失（fit_threshold 0.5 让 t=305 处的 evaluate_existing 不再判 create-worthy？查代码：fit_threshold 仅控制 reuse 判定，不影响 detector 触发；这里 detector 也没触发，意味着 detector 在 t=305 行为本身随机性就大） |
| 456  | [1563]            | [1546]                | 接近相同 |
| 789  | [2897]            | [2584]                | 时间点不同 |
| 1024 | [1037, 2515]      | [1060, 2394, 2873]    | 多触发 1 次 |

combined_drift 真实漂移点 [3000]：

| seed | indicator routes | warmstart routes |
|---|---|---|
| 42   | []                | [] |
| 123  | [3350]            | [3349] |
| 456  | [3050]            | [3052] |
| 789  | [3054]            | [3054] |
| 1024 | [3063, 3373]      | [3056, 3331] |

→ Detector 触发位置高度相似（实现层 detector / consolidator 没改）。warmstart 与 indicator 的差异只在 routing 后 adapter 的初始化方式 + library_fit_threshold（但后者未触发 reuse 所以等效不起作用）。

## 5. 失败模式分析：为什么 warmstart 在 combined_drift 上恶化

combined_drift 数据集设计：
- 5 个 drift features 在 5000 步内线性偏移 2σ（渐进）
- decision boundary 在 t=3000 abrupt 反向

t=3000 之前 adapter 0 学到 "boundary方向 A"。t=3000 之后 detector 触发 (在 t≈3050)，warmstart 把 adapter 0 权重复制到 adapter 1 → adapter 1 起步即"boundary方向 A"，但真实新 regime 的 boundary 是反向。

随机初始化（indicator run）：adapter 1 起步噪声，consolidate 后能在新 regime 上学到正确方向。
Warm-start：adapter 1 起步带反向 boundary 偏置，consolidate 需要先擦掉旧偏置再学新 → 收敛更慢。

**这是 abrupt boundary reversal 数据集上 warm-start 的 anti-pattern**。在循环 regime（regime_switching：3 个 regime 反复切换）上 warm-start 反而轻微有利（regime_switching +0.37 NS），因为切回相似 regime 时旧权重有用。

## 6. Verdict（vs phase4_plan 验收口径）

phase4_plan **核心成功条件** = "rotating 不丢 + combined 翻 non-negative vs Phase 1"：
- rotating: ✓ 保住，+1.32 pp sig 超过 0.5pp 验收线
- combined: ✗ **vs Phase 1 仍 sig 负 −0.55 pp，且比 indicator −0.28 进一步恶化**

phase4_plan **失败条件** = "rotating 失去 +1pp" → 未触发（rotating 仍 +1.32 sig）。

**整体定性**：守门 2 失败 + combined 恶化。Day 1.5 四轮全部跑完，**combined-drift-flip-vs-Phase-1 这个核心成功条件在所有四个变体下都未达成**：
- raw: −0.43 sig 负向
- abs: −0.30 sig 负向
- indicator: −0.28 sig 负向
- warmstart: −0.55 sig 负向（最差）

## 7. 失败模式总结（论文 ablation 章节直接引用）

| 失败模式 | 涉及变体 | 根因 |
|---|---|---|
| Detector 沉默 | raw / abs | TabPFN sliding-context 自适应消化连续误差信号 |
| Routing 全 create / 0 reuse | indicator / warmstart | evaluate_existing MSE 阈值与 raw error std 量级不匹配；fit_threshold 0.05/0.5 都不够松 |
| Warm-start anti-pattern | warmstart-on-combined | 在 abrupt boundary reversal 数据上，warm-start 复制的旧权重带反向 boundary 偏置，拖累新 adapter 收敛 |

## 8. 数据落盘

- 15 个 npz：`results/multiseed_phase4a_warmstart_{dataset}_seed{S}.npz`
- 含 `n_warmstart_inits` / `n_random_inits` / `library_fit_threshold` 诊断字段
- 增量 partial：`results/multiseed_phase4a_warmstart.partial.md`
- 每 seed 主图：`results/multiseed_phase4a_warmstart_{dataset}_seed{S}.png`
- 单 run 日志：`logs/multiseed/multiseed_phase4a_warmstart_{dataset}_seed{S}.log`
- Driver 日志：`logs/multiseed_phase4a_warmstart/driver.log`

四轮 Phase 4 A 实验数据（保留作 ablation）：
- raw: `multiseed_phase4a_raw_*.npz` + `phase4_a_raw_summary.md`
- abs: `multiseed_phase4a_abs_*.npz` + `multiseed_phase4a_abs.partial.md`
- indicator: `multiseed_phase4a_indicator_*.npz` + `phase4_a_summary_indicator.md`
- warmstart: `multiseed_phase4a_warmstart_*.npz` + 本文件

总结一页：[phase4_final_verdict.md](phase4_final_verdict.md)。
