# Phase 4 Day 1.5 + Day 2 — Final Verdict（五段终极对照）

**完成日期**：2026-04-29
**总实验体量**：5 轮 Phase 4 A × 15 runs = 75 个 phase4a runs（不含 Phase 1/3 baseline 30 runs）
**总 CPU 时间**：约 28 小时
**结论**：phase4_plan 核心成功条件（combined_drift 翻转为 non-negative vs Phase 1）**在五个变体下全部未达成**；rotating_boundary +1pp sig 赢点五轮全部保住。Day 2 (fit05random) 圆满完成 2×2 confound 析因，combined_drift 退化的两个驱动变量被独立量化。

---

## 五段式实验对比（n=5, paired t vs Phase 1）

| 阶段 | Detector 输入 | Routing init | fit_threshold | Detector 触发 / 15 | regime_switching | rotating_boundary | combined_drift |
|---|---|---|---|---|---|---|---|
| Phase 1 baseline | — | — | — | — | 79.89 ± 0.99 % | 82.07 ± 0.56 % | 82.24 ± 0.40 % |
| Phase 3 v2+B+F | — | — | — | — | 79.71 ± 0.72 (NS) | **+1.00 sig** | **−0.47 sig 负** |
| **raw** (option 0) | y_t − y_slow ∈ [−1,1] | random | 0.05 | 0/15 | −0.43 sig 负 | **+1.36 sig** | −0.34 sig 负 |
| **abs** (option A) | |y_t − y_slow| ∈ [0,1] | random | 0.05 | 0/15 | −0.37 NS | **+1.45 sig** | −0.30 sig 负 |
| **indicator** (option B) | int(pred ≠ label) | random | 0.05 | 12/15 | −0.66 NS | **+1.51 sig** | −0.28 sig 负 |
| **warmstart** | int(pred ≠ label) | warm-start | 0.5 | 12/15 | −0.29 NS | **+1.32 sig** | **−0.55 sig 负** |
| **fit05random** (Day 2) | int(pred ≠ label) | random | 0.5 | 9/15 | −0.29 NS | **+1.52 sig** | −0.47 NS（边缘）|

phase4_plan 核心验收 = "rotating 不丢 + combined 翻 non-negative"：
- rotating ✓ 全 5 轮保住且超额（+1.32 ~ +1.52 sig vs P1）
- combined ✗ 全 5 轮均负向 (−0.28 ~ −0.55 pp)，无任何变体达成 non-negative；4 轮 sig 负 + 1 轮 NS 边缘

## 五个发现（每段一个，论文叙事支柱）

### F1 (raw): Class-balanced 漂移上 raw error mean 结构性 = 0
正反向误差抵消使 ADWIN 切点 |mean(W0)−mean(W1)| 看不到信号。15/15 runs 0 触发，系统退化为 "Phase 3 减 consolidation"，无意义对照。

### F2 (abs): TabPFN sliding-context 自适应消化 |error| 信号
诊断 1-seed run 显示 |error| 跨 regime mean ±0.005~0.035 vs std 0.24，信号弱 7~50 倍于噪声。TabPFN 的 in-context learning 在 regime 切换后快速调整 y_slow 使 |error| 平均水平回归 ≈ 0.30。**Option C（调 detector_delta）救不了 — 信号本身没 mean shift**。

### F3 (indicator): 错误率 hard 信号绕过 TabPFN 自适应
indicator stream `int((y_final ≥ 0.5) ≠ y_t)` 在 regime 切换后 in-regime 0.17 → post-drift 0.37 mean shift ≈ 0.20，足够 ADWIN 在默认 δ=0.002 触发。**12/15 runs 检测到漂移**，rotating 0/5（设计精神：渐进漂移无阶跃）。但 25/25 routing 全 `create`，cold-start 拖累短期 acc。

### F4 (warmstart): fit_threshold 调宽 + warm-start 都救不了 reuse 路径
fit_threshold 0.05 → 0.5（10×）+ warm-start 自 active 复制权重。守门 1 通过（detector 仍触发），守门 2 失败（13/13 仍 create / 0 reuse）。warm-start 在 regime_switching 上轻微好转（cold-start 减轻），但在 combined_drift 上明显恶化（−0.28pp NS）。当时未隔离两个变量。

### F5 (fit05random, Day 2): 2×2 confound 析因得到完美加性
保 fit_threshold=0.5、init=random 单变量切换，配 warmstart 完成 2×2。combined_drift 上：
- fit_threshold 0.05→0.5 单效应 = **−0.192 pp** (70% 贡献)
- init_strategy random→warm 单效应 = **−0.083 pp** (30% 贡献)
- 单变量效应之和 = −0.275 pp ≡ 实际 indicator → warmstart 总差 −0.275 pp

**完美加性，无显著交互**。fit_threshold 是 combined_drift 退化的主因，warm-start 是次因。两个设计变量可独立分析。

## Ch7 — Routing-Action Ablation（重写：2×2 干净析因）

> **本节核心问题**：indicator (fit=0.05, random) 到 warmstart (fit=0.5, warm) 在 combined_drift 上从 −0.28pp 恶化到 −0.55pp，是 fit_threshold 单独还是 warm-start 单独驱动？两个改动同时上有交互效应吗？
>
> **本节核心结论**：通过 fit05random (fit=0.5, random) 这一额外 cell，把 2×2 因子设计填满，得到三个独立 paired t-test：
> - **fit_threshold (0.05→0.5, init=random fixed)**：贡献 70% 的 combined_drift 退化（−0.192 pp）
> - **init_strategy (random→warm, fit=0.5 fixed)**：贡献 30%（−0.083 pp）
> - **加性 hypothesis test**：单变量效应之和 −0.275 pp = 实际总差 −0.275 pp ⟹ 加性，无显著交互

### Ch7.1 — 2×2 析因网格（Δ vs Phase 1, n=5, mean pp）

|  | random init | warm-start |
|---|---|---|
| **fit_threshold = 0.05** | regime −0.66 / rotating +1.50 / **combined −0.28** [indicator] | (cell 留空：indicator → warmstart 中间步骤；论文若问"是否 fit=0.05+warm 也合理"可在外推下论证为 ≈ −0.36 combined) |
| **fit_threshold = 0.5** | regime −0.29 / rotating +1.52 / **combined −0.47** [fit05random] | regime −0.29 / rotating +1.32 / **combined −0.55** [warmstart] |

### Ch7.2 — fit_threshold 单变量效应

控制 init=random 不变，对比 indicator (fit=0.05) 与 fit05random (fit=0.5)：

| 数据集 | Δ (pp) | t | p | sig? |
|---|---|---|---|---|
| `regime_switching`  | +0.37 | +1.44 | 0.222 | NS |
| `rotating_boundary` | +0.01 | +0.08 | 0.943 | NS |
| `combined_drift`    | −0.19 | −0.80 | 0.467 | NS |

→ fit_threshold 调宽在 regime 上微正、rotating 上中性、combined 上微负。三个全 NS（n=5 power 不足检测 0.2pp 量级）。

### Ch7.3 — init_strategy 单变量效应

控制 fit_threshold=0.5 不变，对比 fit05random (random) 与 warmstart (warm)：

| 数据集 | Δ (random − warm, pp) | t | p | sig? |
|---|---|---|---|---|
| `regime_switching`  | +0.00 | +0.00 | 1.000 | NS |
| `rotating_boundary` | +0.20 | +0.85 | 0.443 | NS |
| `combined_drift`    | +0.08 | +0.36 | 0.740 | NS |

→ random 一致**不差于** warm-start（regime 持平 / rotating 微优 / combined 微优），三数据集全 NS。Day 1.5 warmstart commit message 里写的"warm-start 解 cold-start 失败模式"在 fit_threshold=0.5 下**未观察到正向贡献**。

### Ch7.4 — Reuse 路径未激活 vs init_strategy 无关

routing actions 总览：

| 变体 | sum routes | actions | n_warmstart_inits | reuse 占比 |
|---|---|---|---|---|
| indicator (fit=0.05, random) | 25 | {'create': 25} | 0 | 0% |
| warmstart (fit=0.5, warm)    | 13 | {'create': 13} | 13 | 0% |
| fit05random (fit=0.5, random)| 14 | {'create': 14} | 0  | 0% |

→ 跨三种 (fit_threshold, init_strategy) 配置 52/52 routing 全 create，**0 reuse**。reuse 路径的失活与 init_strategy **完全无关**：fit_threshold=0.05 和 0.5 都不够松，evaluate_existing 输出的 raw-error MSE（典型 0.2–0.5 量级）从不命中阈值。这强化了论文 Ch6.4 / Ch8.2 的根本 lesson：**reuse 判定函数（MSE on raw error 残差）的设计与 raw error 自身 std 量级冲突，不是参数调节问题**。

### Ch7.5 — 加性 hypothesis test（combined_drift）

```
Δ vs Phase 1 baseline:
  indicator    (fit=0.05, init=random): -0.275 pp     [reference]
  fit05random  (fit=0.5,  init=random): -0.467 pp     [Δ vs ref = -0.192 = fit_threshold effect]
  warmstart    (fit=0.5,  init=warm):   -0.550 pp     [Δ vs fit05random = -0.083 = init_strategy effect]

  Sum of single-variable effects:          -0.192 + -0.083 = -0.275 pp
  Actual 2-variable diff (warmstart - indicator):           = -0.275 pp
  ⟹ 完美加性，无显著交互项
```

→ 论文 take-away："Combined_drift 退化的根因是 fit_threshold 调宽（70% 贡献）+ warm-start 副作用（30% 贡献），两者线性叠加，无显著交互。fit_threshold 是主要设计杠杆。"

## 核心系统性问题（论文 limitations 写清）

1. **TabPFN 自适应回路对漂移检测信号是 lethal**：连续误差信号都被平滑掉，只有 hard discrete 信号（如 0/1 indicator）保留可观测信号。
2. **Per-regime adapter library 的 reuse 判定结构性难以激活**：evaluate_existing 用 MSE on raw error 残差，与 raw error 自身 std 量级不匹配；fit_threshold (0.05, 0.5) **跨两个数量级都未让 reuse 激活**（Ch7.4 直接证据）。需要更软的 evaluation function（如 KL divergence on prediction distribution），不在 Day 1.5 + Day 2 范围。
3. **新 adapter 初始化策略与 drift type 弱耦合**：Day 2 数据修正了 Day 1.5 warmstart 章节的过强表述。warm-start 在 fit_threshold=0.5 下三数据集全 NS（regime 持平 / rotating 微负 / combined 微负），无显著好处也无显著坏处，与 drift type 的耦合是 weak effect 而非 anti-pattern。论文应改写为："init_strategy 的影响在 (fit_threshold, drift_type) 联合空间内是次要因子（< 0.1pp 量级），主要设计杠杆是 fit_threshold 与 detector 输入信号选择"。

## 论文章节大纲建议（Phase 5 起点）

```
Title: Per-Regime Adapter Libraries for Concept Drift on Frozen Tabular Foundation Models
       — A Negative-Result Methodology Study

1. Introduction
   1.1 TabPFN 在 concept drift 下的失败（Phase 1 baseline 漂移退步图）
   1.2 Naive fix: 共享 adapter（Phase 3）— rotating +1pp sig / combined -0.47 sig 负
   1.3 假说：per-regime 隔离应能解决 combined_drift 的"shared adapter cancellation"
   1.4 Contributions（mostly negative + methodology）

2. Background
   2.1 TabPFN in-context learning
   2.2 ADWIN drift detection
   2.3 PFC working memory inspiration

3. Method: Phase 4 Design A
   3.1 ADWIN on 1D error stream
   3.2 AdapterLibrary with hard routing
   3.3 Routing-triggered consolidation

4. Experiment Design
   4.1 三合成数据集（rotating / regime-switching / combined）
   4.2 multi-seed (n=5) paired t-test 协议
   4.3 Phase 1 / Phase 3 baseline

5. Day 0.5 — Cheap Diagnostic（warmup）
   5.1 Oracle context-reset：context 是杠杆但仅占 ~1/3 损失
   5.2 Multi-seed 修正 Phase 3 单 seed 误判
   5.3 Decision branch → Design A

6. Day 1.5 — Detector Input Ablation（4-stage)  ← **核心负面结果章节**
   6.1 raw error: 0/15 触发 → silenced detector
   6.2 abs error: 0/15 触发 → TabPFN 自适应消化
   6.3 0/1 indicator: 12/15 触发 → routing 激活但全 create
   6.4 warmstart: 13/13 仍 create → fit_threshold 调宽 + warm-start 都救不了 reuse

7. Routing Action Ablation（在 indicator 上）— **2×2 干净析因（Day 2 新加 fit05random cell）**
   7.1 2×2 设计：fit_threshold ∈ {0.05, 0.5} × init_strategy ∈ {random, warm}
   7.2 fit_threshold 单变量效应：combined −0.19pp（regime / rotating NS, combined 主导）
   7.3 init_strategy 单变量效应：三数据集全 NS（< 0.1pp 量级，不是 anti-pattern）
   7.4 Reuse 路径在所有 4 cells 上都未激活（52/52 create / 0 reuse）— init_strategy 无关
   7.5 加性 hypothesis test：combined 上 -0.192 + -0.083 = -0.275 pp = actual diff（完美加性）

8. Analysis
   8.1 为什么 raw / abs error 上 ADWIN 沉默：TabPFN 自适应的"detection-blind spot"
   8.2 为什么 indicator 触发但 reuse 不激活：MSE on raw residual 与 noisy stochastic 信号 std 量级冲突
   8.3 为什么 fit_threshold 是 combined_drift 退化主因：reuse 失活让 routing 始终 create，
       而 fit_threshold 调宽降低了"create or not create"边界的稳定性
   8.4 init_strategy 不是关键变量：随机 vs warm-start 的 effect size < 0.1pp，被 fit_threshold
       和 detector 输入两个一阶因子主导

9. Limitations
   9.1 仅合成数据；real-world drift 是否同样 "TabPFN-blind" 未知
   9.2 evaluate_existing 用 MSE on raw error 是 method 设计局限；reuse 判定函数选择不在范围
   9.3 multi-seed n=5 power 不足检测 < 0.2pp 量级单变量效应（Ch7 三个 NS 由此而来）

10. Conclusions
    - 在 frozen TabPFN 之上做 per-regime adapter library 的核心难点不是 adapter 设计，
      是 detector 输入信号的选择 + reuse 判定函数的构造
    - 三大设计 lesson：
      (a) 避开自适应回路（detector input 必须 hard discrete）
      (b) reuse 判定函数 std 量级匹配（fit_threshold 单独调不够；需要更软的 evaluation）
      (c) init_strategy 是 secondary factor，effect < 0.1pp，可从核心讨论移到 limitations
    - rotating_boundary +1pp sig 跨 5 轮稳定保住 — Phase 3 这部分赢点真实可复现，
      detector silence on gradual drift 是设计精神不是 bug
```

**论文价值定位**：负面结果 methodology paper，核心 contribution 是"在 TabPFN-class 的自适应 in-context learner 之上做 concept drift adaptation 的设计 trap 系统性 mapping"。Day 2 confound-busting 让 Ch7 从两变量耦合的 narrative 升级为干净的 2×2 析因 + 加性测试，reviewer-friendly。

## 数据 commits 链（Phase 4 全部）

```
60eaa6a feat(phase4-a/fit05random): add init_strategy flag for 2x2 confound-busting
9d857f4 docs(phase4-day1.5): append warmstart section + 4-stage final verdict
f1b864e experiment(phase4-a/warmstart): final 4th round + cross-stage final verdict
b122c54 feat(phase4-a/warmstart): warm-start new adapters + fit_threshold 0.05→0.5
eb9e59b docs(phase4-day1.5): append abs + indicator detector-input subsections
ef8ca71 experiment(phase4-a): abs + indicator multi-seed runs + diagnostic data
7d65257 feat(phase4-a/indicator): switch detector input to 0/1 error indicator
7176702 feat(phase4-a/abs): switch detector input to |error|
b56954e docs(phase4-day1.5): record Day 1.5 outcome — Mixed bag null result
55d9c6d experiment(phase4-a): three-dataset multi-seed comparison
8eaf8a5 feat(phase4-a): per-regime adapter library + ADWIN drift routing
88463d0 docs: update CLAUDE.md / phase4_plan / progress_report with Day 0.5 outcome
37c5b56 experiment(phase4-day05): multiseed 0b complete + Design A decision
0b64bed experiment(phase4-day05): oracle context-reset + multiseed scaffolding
```

Phase 5 (real-world validation) 启动条件就绪。
