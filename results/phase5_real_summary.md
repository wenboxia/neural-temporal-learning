# Phase 5 — Real-World Validation Combined Verdict

> ⚠️ **framing 说明（2026-06-01 导师汇报后追加）**
> 本节的**实验数据、统计结论、γ 诊断的事实内容全部有效**。
> 被取代的只是把这些负面结果定位为"论文核心 contribution / mechanistic discovery"这一点 ——
> 该定位已被导师否定，详见 [../progress_report.md](../progress_report.md) §「2026-06-01 导师汇报反馈」。
> 关键推论：detector 从未触发 ⇒ 三层结构未被激活 ⇒ **"方法无效"这个结论其实从未被真正验证过**。
> 当前主线是 Phase 5.5（重设计 detector 信号），见 [../todo.md](../todo.md) P1。


**完成日期**：2026-05-29
**实验体量**：Electricity (Stage A, A+ 协议) 45 runs + Insects (Stage B1+ aligned 协议) 60 runs = **105 real-world runs** + 45 archived 原 A+ Insects runs（methodology narrative 保留）
**Wall time**：Stage A 24.1h + Stage B 48.3h (archived) + Stage B1+ 60.1h = **~132h CPU**
**详细数字**：[phase5_real_summary_electricity.md](phase5_real_summary_electricity.md) / [phase5_real_summary_insects.md](phase5_real_summary_insects.md) / [phase5_confound2_diagnostic.md](phase5_confound2_diagnostic.md)

---

## 1. F1-F4 + 改善信号 复现矩阵（合成 vs 真实）

| 发现 | Synthetic regime_switching | Synthetic rotating_boundary | Synthetic combined_drift | Electricity (gradual) | **Insects abrupt (B1+)** | 复现状态 |
|---|---|---|---|---|---|---|
| F1 raw error detector 沉默  | ✓ 0/15 | ✓ 0/15 | ✓ 0/15 | N/A (仅跑 indicator) | N/A | N/A |
| F2 \|error\| detector 沉默  | ✓ 0/15 | ✓ 0/15 | ✓ 0/15 | N/A | N/A | N/A |
| **F3 indicator detector 触发率** | 12/15 (80%) | 0/15 (0%) | 9/15 (60%) | **1/15 (7%)** | **0/20 (0%)** | **gradual 同向复现 (rotating/Electricity ≈ 0)**；**abrupt 在真实数据反向**（regime_switching 80% → Insects 0%）|
| **F4 reuse 不激活** | ✓ 0/25 routes (0%) | — (0 detector) | ✓ 0/14 routes (0%) | ✓ 1/1 route create | **vacuous** (0 routes total) | Electricity 上完全复现；Insects 上**无效证伪**（detector 0 → 无 routing → reuse 假说不能检验）|
| Δ phase3 vs phase1 sig 负 (combined_drift) | NS | NS | **−0.47 sig** | **−0.124 sig** | **−0.101 sig** | **真实数据同向复现**（量级小一个数量级但显著性强）|
| Δ phase4a vs phase1 改善 (rotating_boundary) | NS | **+1.52 sig** | NS | −0.064 NS | **−0.172 sig 负** | **未复现**（rotating-class +1pp 改善是合成 artifact）|

## 2. 五个论文级 verdict（Phase 5 主结论）

### V1. F3 在真实 abrupt drift 上完全失效（**最强 negative finding**）
合成 regime_switching 的 12/15 indicator 触发率，在 protocol 充分对齐 + ADWIN buffer 保证 + drift 全覆盖下，在真实 Insects abrupt drift 上退化为 **0/20**。机制诊断（[phase5_confound2_diagnostic.md](phase5_confound2_diagnostic.md)）证明这不是 protocol 失败，而是 **TabPFN sliding-context self-adaptation 速度 outpaces ADWIN detection delay** —— in-context relearn ~10-20 步内 indicator |Δ| 仅 0.02，比合成的 0.20 信号小 10×。

**这一发现把 Phase 4 Day 1.5 F2 ("TabPFN sliding-context 自适应消化 |error| 信号") 扩展到 indicator stream**：任何走 TabPFN prediction 通道的 detector input 在真实 P(x|y) 部分 transfer 的 abrupt drift 上都被 self-adaptation 吃掉。

### V2. F4 reuse 失活在 Electricity 上完全复现，在 Insects 上无效证伪
- Electricity: 1/1 routes = create（100%），与合成 52/52 一致 ✓
- Insects: 0 routes total（detector 不触发）→ reuse 假说 vacuous，**不是反例，是无信息**

### V3. Phase 3 在真实数据上 sig 负向同向复现
- Synthetic combined_drift: −0.47 sig
- Electricity: **−0.124 sig** (p=0.025)
- Insects B1+: **−0.101 sig** (p=0.0025)

虽然量级小一个数量级（synthetic 0.47 vs real 0.10-0.12），但定性同向 + 真实数据 power 强（n=15/20 paired t）。**Shared adapter 在真实 multi-mechanism drift 上 net negative 是 paper-grade 可复现** finding。

### V4. rotating_boundary 的 +1pp 改善是合成 artifact
- Synthetic rotating_boundary: **+1.52 sig** （5 轮稳定保住）
- Electricity (gradual real-world analog): **−0.064 NS**
- Insects: 不适用（abrupt）

合成 rotating_boundary 的 2D Gaussian + 简单线性边界旋转 + n_features=2 是 brittle 设定；真实 gradual drift 上 phase4a 没拿到合成-观察到的改善。**论文 framing 必须诚实承认**：rotating +1pp 不是 Phase 4 设计的本质优势，是合成数据 inductive bias 的 artifact。

### V5. phase4a 在真实数据上整体 net negative
- Electricity 跨 n=15: −0.064 NS（边缘负向）
- Insects B1+ 跨 n=20: **−0.172 sig 负 (p<0.0001)**
- 3/4 Insects segments per-segment sig 负

Detector 沉默时 adapter library 的 input_dim≥14 (Electricity) / =33 (Insects) MLP cold-start cost 没人付 → 净 net loss。**Phase 4 Design A 真实数据上不工作**。

## 3. Stage B 协议演化的 methodology narrative

旧 A+ 协议 (start/middle/end) 上 Insects detector 0/15，**post-hoc empirical analysis** (50-chunk P(y) L1 shift 诊断) 识别出 uniform partitioning 与 Insects 5 个 documented drift positions (~12.7k / 14.3k / 17.9k / 46.7k / 52.0k) 不对齐——14/15 segments 包含 0 drift events。重新设计 4 个 drift-aligned 非重叠 5000-sample segments 覆盖全 5/5 drifts，保持 non-overlap 和 ≥200-sample ADWIN buffer。原 misaligned 数据归档于 `results/archive_misaligned_stage_b/`（保留 transparency）。

B1+ 协议下仍 0/20 → **A+ misalignment 单独不足以解释 detector 沉默** → 触发 confound #2 (binarization dilution) 诊断 → 找到 dominant 机制（indicator |Δ| 比合成小 10×）。**协议演化本身成为 paper-level methodology contribution**：负面结果数据集上的 systematic root-cause isolation。

## 4. 论文（Phase 6）章节影响

### Ch9 Real-World Validation 大纲（基于 Phase 5 结果）
- 9.1 Datasets & A+ subsampling protocol（rationale + ETA）
- 9.2 Stage A Electricity: F4 复现 / F3 在 gradual drift 上沉默（matches synthetic rotating）
- 9.3 Stage B A+ misalignment → B1+ aligned protocol revision (methodology narrative)
- 9.4 Stage B B1+ results: F3 0/20 仍沉默 → confound #2 诊断
- 9.5 V1-V5 combined verdict + 与合成对比矩阵

### Ch8 新加 §8.5 — "Indicator-detector 在真实 abrupt drift 上的 systemic failure mode"
扩展 Phase 4 Day 1.5 F2 ("TabPFN sliding-context 自适应消化") 到 indicator stream，定位 TabPFN self-adaptation 速度（~10-20 步 in-context relearn）与 ADWIN 检测延迟的对抗机制。

### Ch10 Limitations 加 binarization confound 一段
6-class → 2-class sex-pair 折叠稀释 abrupt P(y) shift 是 dominant 机制；6-class native ablation 在 Phase 5 scope 外，列入 future work / limitations（系统 6-class refactor + 重跑全部 Phase 1-4 合成基线 ≥ 1 周）。

### Ch7 / Day 2 2×2 confound 析因仍然成立
真实数据负面结果不改变合成 fit_threshold × init 析因；那是 design ablation，与真实-vs-合成 transfer 正交。

## 5. Phase 5 总 verdict

**Phase 5 完成，105 datapoints 实证，5 个 paper-level verdicts (V1-V5) 全部明确**：

- F3 真实 abrupt 失效是论文最强 negative contribution
- F4 在 Electricity 复现，在 Insects 无效证伪
- Phase 3 sig 负向真实数据同向复现
- rotating +1pp 改善是合成 artifact
- Phase 4 Design A 真实数据 net negative

**Phase 5 不要求 "赢" 的设计精神实现**：findings 复现（无论正反），methodology narrative 自洽，paper 起点就绪。**论文核心 contribution 升级**：从"在 TabPFN-class 自适应 in-context learner 之上做 concept drift adaptation 的设计 trap 系统性 mapping"扩展到"该 mapping 在真实 abrupt drift 上的 systemic failure mode 量化定位"。

---

**下一步**：Phase 6 论文撰写。建议优先：
1. Ch1-3 (intro / background / method) — 2-3 天
2. Ch4-7 (experiments + 合成结果，已 90% 数据齐) — 3-4 天
3. Ch8-10 (analysis + Phase 5 + limitations) — 2-3 天
4. Ch11 conclusions + abstract — 1 天

毕业版 1-2 周可成稿。
