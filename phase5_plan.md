# Phase 5 — Real-World Validation Plan

**日期**：2026-04-28（plan 起草）/ 2026-05-29（Phase 5 完成）
**前置依赖**：Phase 4 Day 1.5 已完成四轮 detector input ablation（raw / abs / indicator / warmstart）。Phase 4 Day 2 收尾任务（option B confound-busting）见 §0。
**后续**：Phase 6 = 毕业论文撰写。

## ✅ Phase 5 完成状态汇总（2026-05-29 收尾）

- §0 Phase 4 Day 2 (option B fit05random) ✅ 完成（commit `5d47ce3`）
- §1-5 Stage A Electricity (45 runs, A+ 3-segment) ✅ 完成（commit `cb13e08`）
- §1-5 Stage B Insects (原 A+ 协议 45 runs) ⚠️ **方法学失败**：14/15 segments 不含 documented drift → 归档至 `results/archive_misaligned_stage_b/`
- §1-5 Stage B1+ Insects (drift-aligned 4-segment, 60 runs) ✅ 完成（commit `e368ca6`）
- γ Confound #2 Diagnostic ✅ 完成 — 机制定位"TabPFN absorption outpaces ADWIN delay"

**Phase 5 核心 5 verdicts**（详见 `results/phase5_real_summary.md`）：
- **V1**: F3 indicator detector 在真实 abrupt drift 上**完全失效** (0/20) — 扩展 F2 到 indicator stream
- **V2**: F4 reuse 失活在 Electricity 完全复现 (1/1 create)，Insects vacuous
- **V3**: Phase 3 sig 负向真实数据同向复现（Electricity −0.124 / Insects −0.101）
- **V4**: rotating_boundary +1pp 改善是合成 artifact (Electricity gradual −0.064 NS)
- **V5**: phase4a 真实数据 net negative (Insects **−0.172 sig p<0.0001**)

**关键 framing 升级**：原计划"Negative-Result Methodology Paper"→ 升级为"**Mechanistic Discovery + Methodology Contribution**"。γ 诊断量化了 indicator |Δ| ≤ 0.019 (vs 合成 0.20，10× 稀释)，将 F3 失败从"vague 现象"升级为"机制定位的 fundamental property"。

**β binarization ablation 决定不做**：γ 已机制定位（TabPFN absorption 是主因，不是 binarization），β 5.4 天会得到冗余实证。

**Phase 6 路径**：CPU only 不租 GPU，不动算法，重组叙事即可。预计 drafting 2-3 周。

---

## 0. Phase 4 Day 2 收尾（先做，5.5h）— option B confound-busting

### 0.1 背景

Phase 4 Day 1.5 warmstart 实验同时改了两个变量：
- `library_fit_threshold` 0.05 → 0.5
- AdapterLibrary `_create_new_adapter` 从 random init 改为 warm-start（active 复制权重）

→ 当前 2×2 析因网格缺一格 (fit_threshold=0.5, random init)。reviewer 必问"combined_drift 上从 −0.28 (indicator) 恶化到 −0.55 (warmstart)，是 fit_threshold 还是 warm-start 的锅？"

### 0.2 实验配置

| 参数 | 值 |
|---|---|
| Detector 输入 | `int((y_final ≥ 0.5) ≠ y_t)` (indicator) |
| ADWIN delta | 0.002 |
| ADWIN value_range | 1.0 |
| `library_fit_threshold` | **0.5** |
| Adapter init | **random**（不 warm-start） |
| Datasets | regime_switching / rotating_boundary / combined_drift |
| Seeds | [42, 123, 456, 789, 1024] |
| n_parallel | 2 |
| 命名 | `multiseed_phase4a_fit05random_*` |

### 0.3 守门检查（前 2 seed 完成时）

- 守门 1：`n_routes > 0`（detector 仍触发）
- 守门 2：`route_action` 含 reuse 或全 create 但 ≠ warmstart 的退步幅度 → 都 OK，不 abort

### 0.4 完成后输出

- `results/multiseed_phase4a_fit05random_*.npz` × 15
- `results/phase4_a_summary_fit05random.md`
- 更新 [results/phase4_final_verdict.md](results/phase4_final_verdict.md)：四轮 → 五段表格，Ch7 重写为 2×2 干净析因
- 更新 [progress_report.md](progress_report.md) Phase 4 Day 2 节
- commit

### 0.5 ETA

~5.5h（同 Day 1.5 单轮）。完成后 Phase 5 启动。

---

## 1. Phase 5 总体目标

**导师要求**：算法开发可用合成数据，但论文最终发现必须用真实数据验证。

**Phase 5 目标**：在 canonical drift benchmark 上验证 Phase 4 Day 1.5 的四个发现是否复现：

| 发现 | 内容 | 预期验证方式 |
|---|---|---|
| F1 | Class-balanced raw error mean ≈ 0 → ADWIN 盲 | 真实数据上 raw error mean 是否同样接近 0？ |
| F2 | TabPFN sliding-context 消化 |error| 信号 | 真实数据上 |error| 跨 regime 是否同样被平滑？|
| F3 | Indicator stream 绕过 TabPFN 自适应 → 触发 | 真实数据上 indicator 是否同样 12+/15 触发？ |
| F4 | fit_threshold + warm-start 救不了 reuse 路径 | 真实数据 routing actions 是否仍以 create 为主？ |

**Phase 5 不要求"赢"**：findings 复现（即使是负面方向）即论文有效；如果 findings 在真实数据上**不复现**，写"synthetic-trained intuition 真实数据上不通用"+ mechanistic explanation，仍是 paper-grade 结果。

---

## 2. 数据集决策（已锁定）

### 2.1 选定的两个

| 数据集 | OpenML id | 规模 | 特征 | drift 类型 | 类比合成 | 验证目标 |
|---|---|---|---|---|---|---|
| **Electricity** | 151 | 45,312 | 8 | 季节性 + 价格 covariate | rotating_boundary | F-rotating（+1pp sig 真实复现）|
| **Insects (incremental)** | TBD（OpenML 检索 "insects-incremental"，备选 MOA ARFF） | ~57k | ~33 | 6 个 batch regime | regime_switching | F-routing（per-regime hypothesis 真实测试）|

### 2.2 砍掉的候选

- **Airlines (OpenML 1169)**：drift 性质 ≈ Electricity，冗余
- **Gas Sensor Array Drift (UCI)**：regime drift 但文献引用 < Insects
- **OpenML-CC18 全套**：静态 benchmark，不测 drift，跟 paper 主题无关

### 2.3 combined_drift 真实类比 = 没有

合成 `combined_drift`（covariate gradual + boundary abrupt 同时存在）**没有 canonical 真实数据集**对应。论文里诚实承认：

> Combined_drift represents an adversarial stress test combining gradual covariate and abrupt boundary mechanisms; while not directly mapped to a canonical real-world benchmark, it tests robustness to multi-mechanism drift coexistence.

`combined_drift` 仅作为合成-only stress test，Phase 5 不做真实数据对应。

---

## 3. Subsampling 协议（已锁定 A+ 跨两数据集）

### 3.1 为什么是 A+

- Electricity 45k / Insects ~57k 全跑 CPU 不现实（estimate 56–600h）
- 单一 contiguous 5000 段（A）样本选择敏感：可能被质疑"为什么是这一段"
- 文献已知 drift 段（C）在 Electricity 上文献证据弱（"整段都漂移"，少 crisp 时刻指认），Insects 上 regime 标签本身就是 drift，C 不适用
- A+ 三段统一协议跨数据集，paired t-test 协议干净

### 3.2 具体切法

每数据集取**三个非重叠 contiguous 5000-sample 段**：
- start: index [0, 5000)
- middle: index [N//2 - 2500, N//2 + 2500)
- end: index [N - 5000, N)

**Insects 例外**：如果 Insects 数据集组织方式是 6 个 contiguous batch，"middle" 段需对齐到 batch 边界确保跨 regime（实施时验证）。

### 3.3 论文 protocol 一句话

> "Each real-world dataset is partitioned into three non-overlapping 5000-sample contiguous segments (start / middle / end of stream). Each segment is evaluated with 5 random seeds across Phase 1 (TabPFN baseline) / Phase 3 (shared adapter) / Phase 4 (indicator) configurations, yielding 15 datapoints per phase per dataset for paired t-test."

### 3.4 已知 limitation（论文写明）

> "Due to CPU compute budget, full streaming evaluation on the 45k+ samples is infeasible. We evaluate on 5000-sample segments per dataset; future GPU-accelerated work should validate findings on full streams."

---

## 4. 实验矩阵

```
2 datasets × 3 segments × 5 seeds × 3 phases (1/3/4 indicator) = 90 runs
ETA: ~45h CPU，分 2-3 天跑完
```

不跑 Phase 4 raw/abs/warmstart 在真实数据（Day 1.5 已确证 indicator 是唯一让 detector 真激活的 input）。

---

## 5. Step 计划

### Step 1 — Real-world data loader

**新文件**：`src/data/real_world.py`

- `load_electricity()`: OpenML id=151 → `pd.DataFrame` → `(X, y)` numpy array
- `load_insects(variant='incremental')`: OpenML 或 ARFF → `(X, y, batch_id)`
- `take_segment(X, y, segment_id ∈ {'start', 'middle', 'end'}, size=5000)`: 时序切片
- `load_real_world(name, segment_id, seed)`: 统一入口，返回 `(X, y, drift_points, name)` 与现有 `SyntheticDataset` 接口对齐
- 处理：缺失值、类别特征 one-hot、归一化（per-segment 训练时不允许 leak future stats）

**测试**：`tests/test_real_world.py`
- shape / dtype / drift_points 非空 / 不同 segment_id 不重叠

### Step 2 — Plumb 真实数据进现有 scripts

修改：
- `scripts/run_baselines.py`
- `scripts/run_phase3.py`
- `scripts/run_phase4_a.py`
- `scripts/run_multiseed.py`

新增 flags：
- `--dataset_source` ∈ {`synthetic`, `real`}（默认 synthetic 保持向后兼容）
- `--segment_id` ∈ {`start`, `middle`, `end`}（仅 real 时生效）

模型代码（`src/models/`）**完全不动**——adapter library / detector / gated_ensemble 都是数据无关。

### Step 3 — Smoke tests（30 min）

每数据集 × 1 segment × 1 seed × 1 phase × `max_eval_steps=100`：
- 6 个 smoke runs：(electricity / insects) × (P1 / P3 / P4_indicator)
- 验证不崩 / npz 字段齐 / 数值合理
- 检查 indicator stream 上 ADWIN 是否触发（前 100 步可能太短，主要验流程）

### Step 4 — Multi-seed Electricity（~22h）

`run_multiseed.py --configs phase1,phase3,phase4a_indicator --datasets electricity --segments start,middle,end --seeds 42,123,456,789,1024 --n_parallel 2`

45 runs 串行 / 部分并行。增量 `results/multiseed_real_electricity.partial.md`。

### Step 5 — Multi-seed Insects（~22h）

同 Step 4，dataset=insects。

### Step 6 — Final analysis

每数据集每 phase 跨 15 datapoints (3 segments × 5 seeds) paired t-test：
- Phase 4 indicator vs Phase 1
- Phase 4 indicator vs Phase 3

输出：
- `results/phase5_real_summary.md` — 数字 + 复现状态表
- 更新 `progress_report.md` Phase 5 节
- 更新 [results/phase4_final_verdict.md](results/phase4_final_verdict.md) 加真实数据列

### Step 7 — Cross-validation table（论文核心）

| 发现 | 合成 | Electricity | Insects | 复现？|
|---|---|---|---|---|
| F1 (raw 0 触发) | ✓ | ? | ? | TBD |
| F2 (abs 0 触发) | ✓ | ? | ? | TBD |
| F3 (indicator 12/15 触发) | ✓ | ? | ? | TBD |
| F4 (reuse 不激活) | ✓ | ? | ? | TBD |
| rotating +1pp / Electricity 同向 | ✓ | ? | — | TBD |
| regime per-regime / Insects 同向 | ✓ | — | ? | TBD |

---

## 6. 验收标准

**Phase 5 不要求 Phase 4 indicator 在真实数据上"赢"**。三种结果都 paper-grade：

| 结果 | 论文 framing |
|---|---|
| **同向复现**（Electricity 同 rotating +1pp、Insects 同 regime per-regime hypothesis 同 fail） | "Synthetic findings replicate on canonical drift benchmarks → 论文 contribution 验证" |
| **部分复现**（一个数据集复现，另一个不） | "Findings are dataset-specific; 我们 dissect why" |
| **完全不复现** | "Synthetic-trained intuition 真实数据上不通用 + mechanistic explanation" |

唯一 hard fail：**实验跑不出来**（数据接入坏 / TabPFN 在真实数据上 OOM / 时序协议 leak）。这种情况要 root-cause fix。

---

## 7. Phase 6 — 论文撰写（Phase 5 完成后启动）

> ⚠️ **本节（§7 论文大纲 + §8 时间预估）已过时，不要直接使用。**
> - §7.1 的标题 *"A Negative-Result Methodology Study"* 与章节列表是 Phase 5 之前的版本，
>   其"负面结果 methodology paper"定位已于 2026-06-01 被导师否定
> - §8 的时间预估已失效（Phase 5 实际耗时 ~132h CPU，远超表中的 45h）
> - 目标已从"期刊投稿"降级为 **KTH 硕士答辩 pass**
>
> **当前有效的规划见 [todo.md](todo.md)**；方向纠正的完整记录见
> [progress_report.md](progress_report.md) §「2026-06-01 导师汇报反馈」。

### 7.1 论文结构（10 章）

详细大纲见 [results/phase4_final_verdict.md](results/phase4_final_verdict.md) §"论文章节大纲建议"，本计划列**调整后**版本：

```
Title: Multi-Timescale Adapter Libraries for Concept Drift on Frozen Tabular
       Foundation Models — A Negative-Result Methodology Study

(标题去 PFC，用 ML 术语)

1. Introduction
2. Background
   2.1 TabPFN in-context learning
   2.2 ADWIN drift detection
   2.3 Multi-timescale ensemble — biological motivation note
       (PFC working memory 灵感来源 + 明确写"frozen TabPFN 限制无 weight-level
        consolidation，故本文 contribution 不声称生物建模")
3. Method: Phase 4 Design A
4. Experiment Design
5. Day 0.5 — Cheap Diagnostic（Oracle context-reset + multi-seed 修正）
6. Day 1.5 — Detector Input Ablation（raw / abs / indicator 三段 + 在合成数据）
   (warmstart 不在 Ch6，移到 Ch7)
7. Routing Ablation — 2×2 析因（fit_threshold × init，含 0.5/random + 0.5/warm）
   (Day 2 option B 跑完后 2×2 完整)
   7.x cold-start cost by drift type → 移到 Ch8 Analysis
8. Analysis（mechanistic decomposition of F1-F4）
9. Real-World Validation — Phase 5（Electricity + Insects）
10. Limitations
11. Conclusions

Appendix A: Phase 2 / 2.5 探索（KNN/EMA FastCorrector + CompositeWindowLoader），
            未进主线但保留实验诚实性
Appendix B: combined_drift 无真实 analog 的 framing 段
Appendix C: 详细 npz 字段说明 + 复现指引
```

### 7.2 关键叙事拆分

- **毕业论文版**：完整含 Phase 2/2.5 appendix，Phase 5 即使发现"不复现"也大段讨论；中文 + 英文摘要
- **简化扩展版**（如需要）：删 appendix A/B，主章节砍到 8 章左右；英文

### 7.3 anticipate reviewer attack（每条至少 limitations 一句）

1. ADWIN 选型 → "ADWIN 是无参数 Hoeffding 界检测器，CPU 友好；BOCPD 在 binary stream 上参数更多；future work 对比"
2. fit_threshold 不试 ≥1.0 → "≥1.0 进入 false-reuse 区，与 detect-then-isolate 设计精神冲突"
3. Real data subsample → §3.4 已 cover
4. Non-foundation baseline (LightGBM + online retrain) → "future work: cross-architecture 对比"
5. Real-world combined drift → §2.3 已 cover

---

## 8. 时间预估

| 阶段 | 时间 |
|---|---|
| Phase 4 Day 2 (option B) | 5.5h |
| Phase 5 Step 1-2 (loader + plumb) | 半天 |
| Phase 5 Step 3 (smoke) | 30 min |
| Phase 5 Step 4-5 (Electricity + Insects multi-seed) | ~45h CPU，分 2-3 日 |
| Phase 5 Step 6-7 (analysis + verdict update) | 半天 |
| Phase 6 论文撰写（毕业版） | 2-4 周（看进度） |
| Phase 6 简化版（可选）| 1-2 周（毕业版完成后） |

总：Phase 5 约 1 周（含等实验），Phase 6 数周。

---

## 9. 决策快查（本计划锁定项）

- [x] combined_drift **无真实 analog**（不补第三数据集）
- [x] Subsampling 用 **A+**（3 contiguous segments × 5 seeds = 15 datapoints/phase/dataset）
- [x] 真实数据集组合 = **Electricity + Insects（incremental）**
- [x] **不跑 OpenML-CC18**（静态 benchmark，与 drift 论文无关）
- [x] 真实数据**只跑 Phase 4 indicator**，不跑 raw/abs/warmstart（Day 1.5 已证 indicator 是唯一激活 detector 的 input）
- [x] Phase 4 Day 2 (option B) 跑 **fit_threshold=0.5 + random init**，补齐 2×2 析因
- [x] 论文标题**去 PFC**，Ch2.3 半页保留生物灵感动机
- [x] Phase 2/2.5 进 **Appendix A**，毕业论文留、简化版可砍
