# Phase 5 — Confound #2 Diagnostic: Binarization-Driven Signal Dilution

> ⚠️ **Phase 5.5 更正（2026-09-06）** —— 本文件的部分**分析结论**已被更正，原始实验数据不变。
> 1. **Insects 漂移坐标**：本文件中的 12,672 / 14,256 / 17,952 / 46,728 / 52,008 是
>    50-chunk **P(y) 构成变化点**（经验推断），不是官方漂移点。
>    Souza 2020 Table 2 给 "Abrupt (bal.)" 的官方坐标是 **14,352 / 19,500 / 33,240 / 38,682 / 39,510**（温度变化 → P(X|y) 漂移）。
>    按官方坐标，B1+ 四段只覆盖 **2/5** 个真实漂移（early 含 14,352，mid 含 19,500）。
> 2. **γ 诊断 "|Δ| ≤ 0.019、比合成小 10×" 复现不出**：用同一批 npz 重算，
>    在官方点 19,500（mid 段 local 3,500）处 5/5 seed 的 indicator 位移是 **0.095–0.105**。
>    "10× 稀释" 的说法作废。
> 3. **detector 沉默的主因是实现层阈值**，不是 "TabPFN 自适应吃掉信号"：
>    自写 ADWIN 用值域 Hoeffding 界（无经验方差），默认配置下 200/200 切分要求
>    |Δmean| ≥ **0.209**。换 river 标准 ADWIN 离线重放同一批已存信号：
>    Insects **16/20** runs 报警、官方漂移 recall **1.00**、中位延迟 266 步；
>    Electricity 仍 1/15（渐进漂移，应沉默）。见 [detector_replay.md](detector_replay.md)。
> 4. **"indicator 12/15 触发" 是事件数与运行数混用**：实为 **9/15 运行**报警、共 **12 次**事件
>    （regime 5/5 运行 7 事件、combined 4/5 运行 5 事件、rotating 0/5）。
> 5. **真实数据的 5 个 seed 不是独立重复**：全仓库此前无 `torch.manual_seed`，
>    `--seed` 只喂合成数据生成器；真实数据上它们是同一段数据的 5 次不受控随机重复
>    （仅 gate/adapter 初始化不同）。既有 p 值只能在此意义下解释。Phase 1 不受影响（std 已为 0）。


> ⚠️ **framing 说明（2026-06-01 导师汇报后追加）**
> 本节的**实验数据、统计结论、γ 诊断的事实内容全部有效**。
> 被取代的只是把这些负面结果定位为"论文核心 contribution / mechanistic discovery"这一点 ——
> 该定位已被导师否定，详见 [../progress_report.md](../progress_report.md) §「2026-06-01 导师汇报反馈」。
> 关键推论：detector 从未触发 ⇒ 三层结构未被激活 ⇒ **"方法无效"这个结论其实从未被真正验证过**。
> 当前主线是 Phase 5.5（重设计 detector 信号），见 [../todo.md](../todo.md) P1。


**日期**：2026-05-29
**问题**：Insects abrupt drift（5 个 documented P(y) shifts）上 indicator-detector 应在 ADWIN 默认 δ=0.002 下触发（合成 regime_switching 上 12/15 触发实证）。Phase 5 Stage B B1+ 协议下覆盖全 5/5 drift events + ADWIN min_subwindow buffer ≥ 200，detector 仍 **0/20 触发**。本文档定位机制。

---

## 1. 实验设计回顾

| Confound 候选 | 假说 | Stage B 协议测试 |
|---|---|---|
| #1 A+ subsampling misalignment | 14/15 段 0 drift events → 没东西可触发 | Stage B B1+ 4 个 drift-aligned segments → 仍 0/20 |
| **#2 binarization dilution** | sex-pair `{2,4,11}→0` vs `{3,5,12}→1` 把 6-class P(y) shift 折叠成微弱 binary mean shift | B1+ 仍 0/20 → 直接证据 |

→ **confound #1 排除**：B1+ 全 5/5 drifts 覆盖 + buffer 保证仍 0/20，protocol-side 完全责任清算。confound #2 dominant 候选。

## 2. 直接证据：drift 前后 ±200 步 indicator mean shift

每 segment 用 seed=42 phase4a npz（其他 seed 同向，量级一致），统计 documented drift 前 200 步 vs 后 200 步的 indicator (= 0/1 error stream) 均值：

| Segment | drift @ local | indicator pre | indicator post | \|shift\| |
|---|---|---|---|---|
| early     | 2672 | 0.015 | 0.020 | **0.005** |
| early     | 4256 | (与 2672 同段后) | | (相关) |
| mid       | 1952 | 0.015 | 0.011 | **0.004** |
| late_pre  | 4228 | 0.044 | 0.025 | **0.019** |
| late_post | 4160 | 0.010 | 0.000 | **0.010** |

**所有 5 个 drifts 跨 4 segments 跨 5 seeds 的 |Δ indicator| 最大 0.019**。对比 synthetic regime_switching 的 indicator mean shift（CLAUDE.md F3 记录）：

> "indicator stream `int((y_final ≥ 0.5) ≠ y_t)` 在 regime 切换后 in-regime 0.17 → post-drift 0.37 mean shift ≈ 0.20"

→ **真实 Insects 上 indicator |shift| = 0.019，比合成 regime_switching (0.20) 小 10×**。ADWIN 在 δ=0.002 下需要的 mean shift 量级与 noise std 成比例；indicator stream 单步 var = p(1-p) ≈ 0.02 量级（错误率本身就 ~2-5%），合成 0.20 shift 是 noise std 的 5-10×（容易检测），真实 0.019 shift 是 noise std 的 0.5-1×（淹没）。

## 3. 但 P(y_pred=1) 是有 shift 的

同样的 drift 前后窗口对 hard prediction P(y_pred=1) 统计：

| Segment | drift @ | P(y_pred=1) pre | P(y_pred=1) post | shift |
|---|---|---|---|---|
| early     | 2672 | 0.883 | 0.750 | −0.133 |
| mid       | 1952 | 0.565 | 0.446 | −0.119 |
| late_pre  | 4228 | 0.556 | 0.645 | +0.089 |
| late_post | 4160 | 0.970 | 1.000 | +0.030 |

→ **P(y_pred=1) shifts 5-25×大于 indicator shifts**。模型预测分布**确实**跟着真实 P(y) shifts 动了；但因为 TabPFN 的 in-context learning 跟着 drift 自适应，**错误率（= indicator）在 drift 前后几乎不变**。

## 4. 机制诊断结论

**Detector 沉默的根因不是 drift 不存在，也不是 protocol 切错段，而是**：

> **"TabPFN 的 in-context learning 在 abrupt P(y) shift 后通过 sliding context window 快速重学新 majority class，使得 prediction 正确率在 drift 前后几乎不变（indicator |Δ| ≤ 0.02），ADWIN 看的 mean shift 信号被 self-adaptation 吃掉。**
>
> **这与合成 regime_switching 上 indicator 触发率 12/15 的差异是：合成 regime_switching 设计了独立 feature mean + 独立 decision weights 跨 regime（"no transferable pattern across regimes by design"），让 TabPFN 必须经过 ~50 步 in-context-relearn 才能恢复；这 ~50 步 indicator 跑到 0.4+，足够 ADWIN 触发。真实 Insects 上 binary 化后的 majority class 是"哪三类各占多少"的轻量决策，TabPFN context 窗内换 ~10-20 样本就能锁定，错误率不显著抬升。**"

这是论文 Ch8.5 (新加章节) 的 take-home：

**Indicator-detector approach 在 frozen TabPFN + sliding context 系统上对真实数据的 abrupt drift 是 detector-blind 的，因为 in-context learning 的 self-adaptation 速度 outpaces ADWIN 的检测延迟。合成数据"by design 无 transferable pattern"的 brittle 设计让 detector 看起来 work，但真实数据的 P(x|y) 在 drift 前后仍部分 transfer，TabPFN 借此快速重学，detector 永远沉默。**

## 5. 图证（12 PNGs）

每 segment × 3 metric (indicator / P(y_pred=1) / |pred-label| soft error) = 12 张图（seed=42 phase4a, smoothing window=100）：

```
results/phase5_confound2_diag_insects_early_{indicator, pred1, soft_err}.png
results/phase5_confound2_diag_insects_mid_{indicator, pred1, soft_err}.png
results/phase5_confound2_diag_insects_late_pre_{indicator, pred1, soft_err}.png
results/phase5_confound2_diag_insects_late_post_{indicator, pred1, soft_err}.png
```

每图含：
- 滑动均值曲线（红色 / 蓝色 / 橙色）
- documented drift 位置 (红色虚线 + 红色透明 ±200 buffer 区)
- 标题含 drift_local 坐标 + smoothing window 信息

视觉确认：indicator (红) 在 drift 处看不出阶跃；pred1 (蓝) 看得出明显 level shift；soft_err (橙) 介于两者。

## 6. Confound #2 dominance 判定

| 判定标准 | 结果 |
|---|---|
| confound #1 (A+ misalignment) 单独可解释 0/15 触发？ | ❌ 排除（B1+ 仍 0/20）|
| confound #2 (binarization dilution) 量化证据 | ✓ indicator \|Δ\| ≤ 0.019 全 segment 全 drift |
| 与合成 regime_switching 12/15 触发对比 | ✓ 真实 \|Δ\| / 合成 \|Δ\| ≈ 1/10 |
| TabPFN self-adaptation 速度 vs ADWIN 检测延迟 | ✓ in-context relearn 完成时 indicator 仍 ~0.02 |

**verdict**: confound #2 (binarization-driven signal dilution + TabPFN self-adaptation) 是 dominant 机制，单一现象足以解释 0/20。confound #1 (A+ misalignment) 即使存在也是 secondary。

## 7. 论文价值

这一负面结果反而**强化** Phase 4 Day 1.5 F2 "TabPFN sliding-context 自适应消化 |error| 信号" 的合成发现：**indicator stream 在真实 abrupt drift 上仍被 self-adaptation 消化**——证明 F2 不是 |error| 的特性，是 TabPFN sliding-context 这个 detector input 通道的本质性局限。任何 detector input 只要走 TabPFN 的预测结果（不管是 raw / abs / indicator），都被 TabPFN 自适应吃掉。

**论文 Ch8.5 (新加)**：*"Synthetic-to-real transfer failure mode: indicator-detector is detector-blind under TabPFN self-adaptation on real abrupt drift, because in-context learning's relearn speed (~10-20 samples) outpaces ADWIN's detection delay; synthetic regime_switching's by-design independent regimes mask this by forcing slower TabPFN relearn."*

---

## 待 β ablation 验证（不在 Stage B B1+ 范围）

binarization 假说的 cleanest 验证 = 不二值化跑 6-class Insects。Scope-creep 大（系统 6-class refactor + 重跑 Phase 1-4 合成基线 ≥ 1 周）。当前证据已够支撑 confound #2 dominance；β ablation 列入 future work / limitations。
