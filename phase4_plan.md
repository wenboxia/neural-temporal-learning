# Phase 4 Plan — TabPFN 漂移适应：Cheap Diagnostic → Decision → Build

**初始日期**：2026-04-25
**Day 0.5 完成日期**：2026-04-27（决策已落定 → Design A）
**目的**：基于 4 家 LLM（DeepSeek / Gemini / Qwen / Claude）的独立 cross-review 共识，在投入主要工程时间之前先做 0.5 天 cheap diagnostic 收集决策数据，再分支到 Design E（Dynamic Context Reset）或 Design A（Regime Detection + Adapter Library）二选一。

> **当前状态（2026-04-27）**：Day 0.5 全部完成 ✅，Design A 已选定。Day 1.5 实施待启动。详见下方 [Day 0.5 实际结果与决策](#day-05-实际结果与决策已定) 段。

---

## Context

Phase 3（v2 + B + F）单 seed 跑出 regime_switching 上 -4.5pp 退步（已 commit 至 6928142），诊断旅程指向 "adapter 跨 regime 学不到 pattern"。

但 4 家 LLM cross-review 一致指出我们漏掉了 **TabPFN 的 sliding context window 本身可能是更大的杠杆** —— 漂移后 context window 里 80% 是旧 regime 数据，污染 in-context learning，可能占 Phase 1 漂移退步 -12-13pp 中的大头。

如果这个假说成立，最便宜的修法不是搞复杂的 adapter library，而是检测到漂移就清空 / 截断 TabPFN context。Phase 4 设计先用 Oracle 实验验证假说，再决定走哪条路。

同时 Phase 3 所有数字都是单 seed，没有方差估计 —— 这是方法学硬伤，Phase 4 同步补 multi-seed。

**关于 adapter 是否保留**：决策延后到 Day 0.5 末，由 Oracle 结果决定。详见决策点表。

> ⚠️ **Day 0.5 后的关键修正**：multi-seed 显示 Phase 3 在 regime_switching 上的 -4.5pp 是**单 seed 抽样噪声**，实际 mean -0.18pp NS（n=5, t=-1.27）。"Phase 3 在 regime_switching 上灾难性失败" 这个原 motivation 不成立。Design A 的 motivation 转为：保 rotating_boundary 的 +1.00pp 真实赢 + 救 combined_drift 的 -0.47pp 真实退步。详见下方决策段。

---

## Phase 4 路线图

```
Day 0.5: Cheap Diagnostic（并行做）
  ├─ 实验 0a: Oracle context-reset on regime_switching
  └─ 实验 0b: Multi-seed (5) 重跑现有 Phase 1 / 2 / 3 v2+B+F

Day 0.5 末: 决策点 → 选 Design E 或 Design A

Day 1.5-3: 实施所选 Design + 三数据集 × 5 seeds 完整对比 + commit

Day 3-5: Phase 5（论文撰写）准备
```

---

## Day 0.5: Cheap Diagnostic（详细 spec）

### 实验 0a: Oracle Context-Reset

**目的**：验证 "context 污染是 regime_switching 退步根因" 假说。

**实现**（修改 scripts/run_baselines.py，加一个 flag，不写新文件）：

```python
parser.add_argument("--oracle_context_reset", action="store_true",
                    help="在已知 drift_points 处强制截断 context 到 reset_size")
parser.add_argument("--reset_size", type=int, default=50,
                    help="oracle reset 后 context 截留多少步")
```

主循环里：
```python
if args.oracle_context_reset and t in drift_points_set:
    # 把 X_ctx 截到最近 reset_size 步
    X_ctx = X_ctx[-args.reset_size:]
    y_ctx = y_ctx[-args.reset_size:]
```

**测试集**：regime_switching, n_samples=3000, context_size=200, **5 seeds**

**对照组**：
- baseline (不 reset)：5 seeds × 1 配置 = 5 runs
- oracle (reset on drift, reset_size=50)：5 seeds × 1 配置 = 5 runs

**输出**：
- `results/oracle_baseline_seed{S}.npz` × 5
- `results/oracle_reset_seed{S}.npz` × 5
- `results/oracle_summary.md`：两组 mean ± std

### 实验 0b: Multi-seed 重跑现有 Phase 1/2/3

**目的**：补 Phase 3 现有数字的方差估计 —— 特别要验证 rotating_boundary +1.18pp 是否在 std 之外。

**实现**：新建 `scripts/run_multiseed.py`，自动化跑：

| 配置 | 数据集 | seeds | 命令模板 |
|---|---|---|---|
| Phase 1 | regime_switching | 5 | `run_baselines.py --dataset regime_switching --seed S` |
| Phase 1 | rotating_boundary | 5 | `run_baselines.py --dataset rotating_boundary --seed S` |
| Phase 1 | combined_drift | 5 | `run_baselines.py --dataset combined_drift --n_samples 5000 --seed S` |
| Phase 2 (KNN) | 同三数据集 | 5 each | `run_phase2.py --fast_method knn --seed S` |
| Phase 3 v2+B+F | 同三数据集 | 5 each | `run_phase3.py --seed S` |

总计：**3 配置 × 3 数据集 × 5 seeds = 45 runs**

**CPU 时间**：每个 25-30 min，串行跑 ~20 小时（不可行）。优化：
- 三数据集**串行**（避免 OOM 和 cache 互相影响）
- 同数据集内 5 seeds **后台并行 2-3 个**（CPU 4 核足够）
- 实测时长：~6-8 小时，挂着跑一晚上

**Seeds**：[42, 123, 456, 789, 1024] 固定五个。

**输出**：
- 每个 run 一个 .npz：`results/multiseed_{config}_{dataset}_seed{S}.npz`
- 汇总报告 `results/multiseed_summary.md`：每格 `mean ± std`，标注哪些差异在 std 之外（显著）哪些在 std 之内（噪声）

### Day 0.5 决策点（规则）

汇总两个实验，按规则分支：

| Oracle 总体准确率 (mean) | Phase 3 vs Phase 1 on rotating_boundary（multi-seed） | → 走哪个 Design |
|---|---|---|
| **≥ 82%** | 任意 | **Design E**（context 是杠杆，简洁优先，砍 adapter） |
| **80-82%** | 差异在 std 之外（显著） | **Design A**（adapter 仍有救，per-regime 隔离） |
| **80-82%** | 差异在 std 之内（噪声） | **Design E**（adapter 也没显著贡献，simpler 优先） |
| **< 80%** | 任意 | **Design A**（context 不是杠杆，回到原 D 路线） |

---

## Day 0.5 实际结果与决策（已定）

**Day 0.5 完成于 2026-04-27**。两个实验产出 + 决策：

### Oracle (实验 0a) 结果（5 seeds, regime_switching）

| 指标 | Baseline | Oracle reset_size=50 | Δ | Paired t |
|---|---|---|---|---|
| 总体 acc | 79.89 ± 0.99% | **80.39 ± 0.87%** | **+0.51 pp** | **+3.25 ✓ p<0.05** |
| 漂移后 acc | 66.32 ± 3.23% | 68.68 ± 2.31% | +2.36 pp | +2.45 (≈ p=0.07) |
| 适应速度 | 68.4 ± 10.8 步 | 59.3 ± 8.6 步 | **−9.08 步** | **−4.72 ✓✓ 强显著** |

→ Oracle 落 **80-82% 中段**。

### Multi-seed (实验 0b) 结果（5 seeds × 3 配置 × 3 数据集）

Phase 3 v2+B+F vs Phase 1 baseline（paired t, df=4, critical |t|≈2.78）：

| 数据集 | Δ (Phase 3 − Phase 1) | Paired t | 显著性 |
|---|---|---|---|
| **rotating_boundary** | **+1.00 pp** | **+4.07** | **✓ sig**（5/5 同向）|
| `regime_switching` | −0.18 pp | −1.27 | NS（不显著）|
| `combined_drift` | **−0.47 pp** | **−5.62** | **✓ sig 负向**（5/5 同向退步）|

Phase 2 KNN vs Phase 1 baseline：三数据集**全部 NS**。

### 决策

落第 2 行（Oracle 80-82% × rotating_boundary 显著）→ **Design A**

但**理由变了**（multi-seed 修正了 Phase 3 灾难叙事）：
- 旧理由："救 regime_switching 的 -4.5pp 灾难" ❌ 灾难根本不存在
- **新理由**：
  1. **保 rotating_boundary +1.00pp 显著赢** —— Design E 在渐进漂移上 ADWIN 不会触发，会丢掉这 +1pp 退回 Phase 1
  2. **救 combined_drift -0.47pp 显著退步** —— 共享 adapter 在混合漂移的两个 regime 间互相冲销，per-regime 隔离正面应对
  3. regime_switching 顺其自然（已 NS，per-regime 期望仍 neutral）

### Day 0.5 关键发现（影响 Phase 5 论文叙事）

1. **Phase 3 -4.5pp 是单 seed 噪声**（multi-seed n=5 后归零至 NS）。整个 Phase 3 v2/B/F 诊断旅程是"修一个不存在的灾难"，但发现的 β=0 现象本身仍是有效观察。
2. **Context 污染是 lever 但不是 THE lever**：Oracle 在 regime_switching 上只挽回 post-drift 损失的 ~1/3。剩余 ~2/3 是 TabPFN 在新 regime 上需要积累足够新样本（in-context learning 本身）。Design E ceiling ≈ Oracle ≈ 80.4%（仍 < Phase 1 baseline 82.96%）。
3. **Phase 3 真正的失败模式不是 regime_switching，是 combined_drift**（-0.47pp sig 负向），共享 adapter 在混合漂移上有结构性问题。这是 Day 1.5 必须解决的真实问题。

### 数据 commits

- `0b64bed`：oracle context-reset + multiseed scaffolding
- `37c5b56`：multiseed 0b complete + Design A decision

详见 `results/oracle_summary.md` / `results/multiseed_summary.md` / `results/day05_decision.md`。

---

## Design E: Dynamic Context Reset（**未选定**，仅作 Phase 5 论文 ablation 引用）

**核心**：用 ADWIN 监控 1D 误差流，检测到 drift 就截断 TabPFN context window。**砍掉 adapter / gate / consolidation 全套**，只保留 TabPFN + FastCorrector。

**架构对比**：
```
Phase 3 v2+B+F:  TabPFN → FastCorrector → GatedEnsemble[gate, adapter] → y_final
Design E:        TabPFN(dynamic_context) → FastCorrector → y_final
                       ↑ ADWIN 监控
```

### 待新建文件
- `src/drift/__init__.py`
- `src/drift/error_detector.py`：ADWIN 实现（自己写或 import river；如用 river 需先 pip install → 受 deny 限制需要 user 手动 install）
- `src/data/temporal_loader.py`：扩展 `TemporalWindowLoader`，加 `truncate_to(n)` 方法可外部调用
- `scripts/run_phase4_e.py`：Phase 4 E 入口脚本

### 单测
- `tests/test_error_detector.py`：ADWIN 在已知漂移信号上能 detect；3-4 条
- `tests/test_temporal_loader.py`：truncate_to 后 X_ctx/y_ctx 形状正确；2 条

### 实验
- **三数据集 × 5 seeds × Design E** = 15 runs
- 对比：Phase 1 baseline / Phase 3 v2+B+F / **Design E**
- 输出：`results/phase4_e_{dataset}_seed{S}.npz`
- 汇总：`results/phase4_e_summary.md`，含 mean ± std + 相对各 baseline 的 pp 差异

### 验收
- regime_switching: Design E ≥ Phase 1 + 2pp（mean，且 std 内显著）
- rotating_boundary: Design E ≥ Phase 1（不退步）
- combined_drift: Design E ≥ Phase 1 - 1pp

### Commit
- 实施完毕：`feat(phase4-e): dynamic context reset with ADWIN drift detection`
- 数据：`experiment(phase4-e): three-dataset comparison vs phase 1/3 baselines`

---

## Design A: Regime Detection + Adapter Library（**当前实施路径** — Day 1.5）

**核心**：在 v2 代码基础上，加 ADWIN 检测，硬路由到 K 个 adapter 字典。每个 adapter 只在其 active regime 时接收训练梯度。

**架构对比**：
```
Phase 3 v2+B+F: 单一 adapter，全周期 per-step + consolidation 双训练
Design A:       adapters[k]，仅 active regime k 的 adapter 接收 consolidation 梯度
                ADWIN 报警 → routing index 切换 → 当前 adapter 冻结，新 adapter 激活
```

### 待新建文件
- `src/drift/error_detector.py`：同 Design E
- `src/regime/__init__.py`
- `src/regime/adapter_library.py`：`AdapterLibrary` 类，维护 `dict[int, MLP]` + 路由
- `scripts/run_phase4_a.py`：Phase 4 A 入口脚本

### 修改现有文件
- `src/models/multi_timescale.py`：加 detector + adapter_library；
  - per-step backward 不更新 adapter（仅 active regime 的 adapter 在 consolidation 时更新）
  - consolidation 触发条件改成 "detector 报警 → 巩固当前 active adapter"

### 单测
- `tests/test_error_detector.py`：同 E
- `tests/test_adapter_library.py`：路由正确，新 regime 创建新 adapter，frozen 状态正确；4-5 条

### 实验
- 三数据集 × 5 seeds × Design A = 15 runs
- 对比：Phase 1 / Phase 3 v2+B+F / **Design A**
- 输出：`results/phase4_a_{dataset}_seed{S}.npz`
- 汇总：`results/phase4_a_summary.md`，含 mean ± std + 相对各 baseline 的 pp 差异 + paired t-test

### 验收（基于 Day 0.5 multi-seed 修正后的真实 Phase 3 数字）

| 数据集 | Phase 3 v2+B+F (n=5) | Design A 验收线 | 含义 |
|---|---|---|---|
| `regime_switching` | -0.18 pp NS | ≥ -1 pp（mean，不显著退步） | 不打破 NS |
| `rotating_boundary` | **+1.00 pp sig** | ≥ +0.5 pp（mean，sig 或边缘 sig） | 保住主要赢点 |
| `combined_drift` | **-0.47 pp sig 负向** | ≥ -0.1 pp（理想 ≥ +0.5 pp） | 翻转或至少消除显著退步 |

**核心成功条件**：在 rotating_boundary 不丢的前提下，combined_drift 翻成 non-negative。其它任意结果（包括 regime_switching 微正）都是 bonus。

**失败条件**：rotating_boundary 失去 +1pp 赢点（说明 per-regime 隔离破坏了渐进漂移上 adapter 的 transferable pattern）→ 触发 Day 1.5 内部回退讨论。

### Commit
- 实施完毕：`feat(phase4-a): per-regime adapter library with ADWIN drift routing`
- 数据：`experiment(phase4-a): three-dataset comparison vs phase 1/3 baselines`

---

## 不动的文件

Phase 3 v2+B+F 当前代码全部**保留**：
- `src/models/gated_ensemble.py`
- `src/models/multi_timescale.py`
- `src/consolidation/fast_to_inter.py`
- `src/models/fast_corrector.py`
- `scripts/run_phase3.py`

它们是 Phase 5 论文 "Phase 3 负面结果章节" 的引用代码。Phase 4 只新增不删除。

---

## 时间总预估

| 阶段 | 工作量 | 累计 |
|---|---|---|
| 实验 0a 实施 + 跑 5 seeds | 0.3 天 | 0.3 |
| 实验 0b 自动化 + 后台跑 45 runs | 挂机 6-8h，活跃工作 0.3 天 | 0.6 |
| 决策点 + 写汇总报告 | 0.2 天 | 0.8 |
| Design E **或** A 实施（含单测） | 1.5-2 天 | 2.3-2.8 |
| 三数据集 × 5 seeds × 选定 Design | 后台 6-8h，活跃 0.5 天 | 2.8-3.3 |
| commit + progress_report 更新 | 0.5 天 | 3.3-3.8 |

**3.3-3.8 天，符合 2-5 天预算**。

---

## 验证（end-to-end）

Phase 4 完成判定（5 项必须都满足）：

1. Oracle 实验 + multi-seed 数据已 commit；oracle_summary.md / multiseed_summary.md 在项目里
2. Design E 或 A 在三数据集上跑完，每个数字带 mean ± std
3. 论文级数据汇总：Phase 1 / Phase 2 / Phase 3 v2+B+F / Phase 4 选定 Design 四列对比
4. progress_report.md 更新含 Phase 4 全段（Oracle / 决策 / Design 实施 / 结果）
5. 所有改动 git 本地 commit，至少 5+ 个新 commit 落地

进入 Phase 5（论文）的前置条件：上述 1-5 全部完成。

---

## 论文 framing（Phase 5 用，已选定 A 路径）

**选定路径**：Design A（per-regime adapter library）。

- **Title 候选**：*Per-Regime Adapter Libraries for Concept Drift on Frozen Tabular Foundation Models*
- **核心 contribution**：用 ADWIN 触发 + adapter 字典实现 "PFC working memory" 隐喻；证明对**混合漂移**和**渐进漂移**都有效，对**纯突变漂移**保持中性
- **Phase 3 章节定位**：作为"为什么共享 adapter 在混合漂移上结构性失败、为什么需要 per-regime 隔离"的设计动机
- **Day 0.5 章节**（重要）：multi-seed 验证 + Oracle ablation。multi-seed 部分负责"修正 Phase 3 单 seed 误判"的方法学诚实性（reviewer 友好）；Oracle ablation 部分证明 context 污染只解释 ~1/3 损失，剩余必须靠 per-regime adapter
- **Phase 3 v2/B/F 三轮 iteration**：作为 ablation 章节的 supporting material（而非主结果），展示"为什么不同的修法都救不了共享 adapter，per-regime 隔离才是结构性解"

**论文叙事弧**：
1. Motivation：TabPFN 在概念漂移下的失败（Phase 1 baseline 漂移退步图）
2. Strawman attempt：共享 adapter（Phase 3）—— 在渐进漂移上 +1pp，在混合漂移上 -0.47pp，揭示 root cause（β=0 是 RATIONAL，跨 regime 共享是诅咒）
3. Method：per-regime adapter library + ADWIN 在 1D 误差流的硬路由
4. Results：rotating ≥ Phase 3 / combined 翻转 / regime 中性
5. Analysis：context 杠杆有限（Oracle）+ adapter 隔离才是关键

把 Phase 3 的负面发现当作 paper 的 strength（诚实展示设计旅程 + 提供 informative ablations），符合 reviewer 对深度分析的偏好。
