# Phase 4 Day 1.5 — Final Verdict（一页纸总结）

**完成日期**：2026-04-28
**总实验体量**：4 轮 × 15 runs = 60 个 phase4a runs（不含 Phase 1/3 baseline 30 runs）
**总 CPU 时间**：约 22 小时
**结论**：phase4_plan 核心成功条件（combined_drift 翻转为 non-negative vs Phase 1）**在四个变体下全部未达成**；rotating_boundary +1pp sig 赢点四轮全部保住。

---

## 四段式实验对比（n=5, paired t vs Phase 1）

| 阶段 | Detector 输入 | Routing init | fit_threshold | Detector 触发 / 15 | regime_switching | rotating_boundary | combined_drift |
|---|---|---|---|---|---|---|---|
| Phase 1 baseline | — | — | — | — | 79.89 ± 0.99 % | 82.07 ± 0.56 % | 82.24 ± 0.40 % |
| Phase 3 v2+B+F | — | — | — | — | 79.71 ± 0.72 (NS) | **+1.00 sig** | **−0.47 sig 负** |
| **raw** (option 0) | y_t − y_slow ∈ [−1,1] | random | 0.05 | 0/15 | −0.43 sig 负 | **+1.36 sig** | −0.34 sig 负 |
| **abs** (option A) | |y_t − y_slow| ∈ [0,1] | random | 0.05 | 0/15 | −0.37 NS | **+1.45 sig** | −0.30 sig 负 |
| **indicator** (option B) | int(pred ≠ label) | random | 0.05 | 12/15 | −0.66 NS | **+1.51 sig** | −0.28 sig 负 |
| **warmstart** | int(pred ≠ label) | warm-start | 0.5 | 12/15 | −0.29 NS | **+1.32 sig** | **−0.55 sig 负**（最差）|

phase4_plan 核心验收 = "rotating 不丢 + combined 翻 non-negative"：
- rotating ✓ 全 4 轮保住且超额（+1.32 ~ +1.51 sig vs P1）
- combined ✗ 全 4 轮均 sig 负向 (−0.28 ~ −0.55 pp)，无任何变体达成 non-negative

## 四个发现（每段一个，论文叙事支柱）

### F1 (raw): Class-balanced 漂移上 raw error mean 结构性 = 0
正反向误差抵消使 ADWIN 切点 |mean(W0)−mean(W1)| 看不到信号。15/15 runs 0 触发，系统退化为 "Phase 3 减 consolidation"，无意义对照。

### F2 (abs): TabPFN sliding-context 自适应消化 |error| 信号
诊断 1-seed run 显示 |error| 跨 regime mean ±0.005~0.035 vs std 0.24，信号弱 7~50 倍于噪声。TabPFN 的 in-context learning 在 regime 切换后快速调整 y_slow 使 |error| 平均水平回归 ≈ 0.30。**Option C（调 detector_delta）救不了 — 信号本身没 mean shift**。

### F3 (indicator): 错误率 hard 信号绕过 TabPFN 自适应
indicator stream `int((y_final ≥ 0.5) ≠ y_t)` 在 regime 切换后 in-regime 0.17 → post-drift 0.37 mean shift ≈ 0.20，足够 ADWIN 在默认 δ=0.002 触发。**12/15 runs 检测到漂移**，rotating 0/5（设计精神：渐进漂移无阶跃）。但 25/25 routing 全 `create`，cold-start 拖累短期 acc。

### F4 (warmstart): fit_threshold 调宽 + warm-start 都救不了 reuse 路径
fit_threshold 0.05 → 0.5（10×）+ warm-start 自 active 复制权重。守门 1 通过（detector 仍触发），守门 2 失败（13/13 仍 create / 0 reuse）。warm-start 在 regime_switching 上轻微好转（cold-start 减轻），但在 combined_drift 上明显恶化（-0.28pp NS）— 因为 abrupt boundary reversal 数据上复制旧权重是 anti-pattern。

## 核心系统性问题（论文 limitations 必须写清）

1. **TabPFN 自适应回路对漂移检测信号是 lethal**：在自适应 in-context learner 输出之上做 drift detection，必须避开它的自适应回路。连续误差信号都被平滑掉，只有 hard discrete 信号（如 0/1 indicator）保留可观测信号。
2. **Per-regime adapter library 的 reuse 判定结构性难以激活**：evaluate_existing 用 MSE on raw error 残差，与 raw error 自身 std 量级不匹配；任何"经验上合理"的 fit_threshold（0.05, 0.5）都不够松。需要更软的 evaluation function（如 KL divergence on prediction distribution），不在本 Day 1.5 范围。
3. **新 adapter 初始化策略与漂移类型耦合**：循环 regime 上 warm-start 微利，abrupt boundary reversal 上 warm-start 是 anti-pattern。论文需指出：optimal init 取决于 drift type，single global strategy 不存在。

## 论文章节大纲建议

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

7. Routing Action Ablation（在 indicator 上）
   7.1 fit_threshold 0.05 vs 0.5：reuse 路径全程未激活
   7.2 random init vs warm-start：drift-type-dependent
   7.3 Cold-start cost 在 regime_switching 与 combined_drift 上分裂

8. Analysis
   8.1 为什么 raw / abs error 上 ADWIN 沉默：TabPFN 自适应的"detection-blind spot"
   8.2 为什么 indicator 触发但 reuse 不激活：MSE on raw residual 与 noisy stochastic 信号 std 量级冲突
   8.3 为什么 warm-start 在 combined_drift 上恶化：boundary reversal anti-pattern

9. Limitations
   9.1 仅合成数据；real-world drift 是否同样 "TabPFN-blind" 未知
   9.2 evaluate_existing 用 MSE on raw error 是 method 设计局限
   9.3 没有探索 KL divergence-based reuse 判定

10. Conclusions
    - 在 frozen TabPFN 之上做 per-regime adapter library 的核心难点不是 adapter 设计，是 detector 输入信号的选择 + reuse 判定函数的构造
    - 三大设计 lesson：(a) 避开自适应回路 (b) 信号 std 量级匹配 fit threshold (c) init 策略与 drift type 耦合
    - rotating_boundary +1pp sig 跨 4 轮稳定保住 — Phase 3 这部分赢点是真实可复现，且 detector silence on gradual drift 是设计精神不是 bug
```

**论文价值定位**：负面结果 methodology paper，核心 contribution 是"在 TabPFN-class 的自适应 in-context learner 之上做 concept drift adaptation 的设计 trap 系统性 mapping"，而非"per-regime adapter 是有效 method"的正面结论。Reviewer-friendly 框架：诚实展示四轮 ablation 旅程 + 三个根本 design lesson。

## 数据 commits 链（Phase 4 全部）

```
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

Phase 5 论文写作下一会话起点。
