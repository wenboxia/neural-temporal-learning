# TODO — Phase 5.5 重启计划（活文档）

> 入口文档见 [`start_prompt.md`](start_prompt.md)（**先读那个**）。
> **本文件是 Phase 5.5 的唯一活计划**：每完成一步、每次改计划，都直接改这里，
> 并在文末「进度日志」记一行。`phase4_plan.md` / `phase5_plan.md` 是已完成阶段的历史记录，不再更新。
>
> **计划创建**：2026-09-06 ｜ **最后更新**：2026-09-06
> **当前目标**：完成 KTH 硕士论文答辩（pass 即可，时间不紧张）

---

## 0. Context

冻结 TabPFN 之上的多时间尺度概念漂移适应。导师 2026-06-01 的诊断：
检测器在真实数据上不触发 ⇒ 三层结构从未被激活 ⇒ "方法无效" 这个结论从未被真正验证过。
要求重设计检测信号，争取正面结果；负面结果保留但降级，不作卖点。

**2026-09-06 重启，三方（Claude / Astra / Opus）交叉核实后确认的事实**：

1. **Insects 官方变点**（Souza 2020 Table 2, "Abrupt (bal.)", 52,848 instances）：
   **14,352 / 19,500 / 33,240 / 38,682 / 39,510**（温度变化 30°C → 20°C → ~35°C → …，即 P(X\|y) 漂移）。
   仓库原用的 12,672 / 14,256 / 17,952 / 46,728 / 52,008 是 50-chunk **P(y) 构成变化点**。
   实测类条件特征偏移：经验点 0.05–0.09（全流中位数 0.07）vs 官方点 0.25–0.49；
   且 12,672 与 14,256 都在同一段连续 class 5（[12,598, 14,352) 共 1,754 条）内部。
   **B1+ 四段按官方坐标只覆盖 2/5 个真实漂移**（early 含 14,352，mid 含 19,500）。
2. **检测器沉默的主因是实现层阈值**（✅ 已验证并修复，见 Step 1）。自写 ADWIN 用值域 Hoeffding 界、
   无经验方差，默认配置 200/200 切分要求 \|Δmean\| ≥ **0.209**，而真实 indicator 位移 ≈ 0.10。
3. **每个温度段末尾是一长串 class 5**（1754 / 74 / 97 / 537 / 822 条）；14,352 与 38,682 前 200 条 100% class 5。
   ⇒ TabPFN 探针的掉幅混有"类别消失-重现"效应，**不能用掉幅大小选任务**。
4. **class ID → 物种/性别无官方映射**（Souza 2020 与 USP 仓库均未提供）。
   任何二值化只能称 **ID 分组**，不能称 sex / species 分类。论文 Limitations 必写。
5. **文档已订正**（Step 2 完成）：γ 数字、9/15 计数、2×2 缺格、seed 未绑定 torch。
6. **检测延迟 145–447 步 ≈ TabPFN 滑窗自恢复时间（约 100–200 步）**
   ⇒ "报警后再校正" 的抢救窗口可能很小 ⇒ **必须先做判别性对照**（Step 5），
   不能直接上规模。检测器的评价指标是**延迟**，不只是"触不触发"。
7. **算力**：MacBook M2 Pro 只能 CPU（MPS 慢 5.6×，1.2 s/步）；ROG 幻 14（NVIDIA）2026-09-13 左右可用，
   预计 4–6×。**所有需互相比较的运行都在同一台机器上跑**（GPU/CPU 浮点结果有细微差别）。
8. **6 类改造成本**（7-agent 只读勘察 + 2 个 adversarial reviewer）：33 处 / 19 文件，含数学改动
   （概率向量重归一化、MSE 均值缩减致梯度缩 ~1/K、阈值 / lr / 检测器全部重标定、测试重写、
   与既有二值数字不可比），**45–60 工时 + ROG 重跑 12–15 h**。

---

## 1. 已定决策

| 决策 | 内容 |
|---|---|
| **三条创新点都做** | ① 对比信号检测器（导师路径 A）② 适应-遗忘双评价（导师维度 C，与路径 B 合并成前沿图）③ 诚实消融（四分支 oracle 对照） |
| **标签方案 = C（先二值后 6 类）** | 本周与第一批 ROG 实验用 pair 二值 `pair_A_vs_B`（{2,3} vs {4,5}）跑通检测-动作链；现有 {2,4,11}/{3,5,12} 更名 `pair_parity`。若检测-动作链有增益，再投 6 类原生作论文主数字 |
| **Git** | 每完成一项本地 commit 并推 main，显式列文件，绝不带私人转写 |
| **导师沟通** | 第一批结果出来后统一同步（段切错 + γ 更正 + river 结果一起讲） |
| **本周边界** | MacBook 只做编码与零成本诊断；长实验一律留给 ROG |

---

## 2. 本周任务（MacBook）

每项完成后：`pytest tests/` 全绿 → commit → push → 在进度日志记一行。

- [x] **Step 1 — 检测器**：`src/drift/error_detector.py` 加 `RiverADWINDetector` 包装 + `make_detector` 工厂，
      `--detector_impl {own,river}`（own 保留为消融）；`scripts/diag_detector_replay.py` 对 35 条已存
      indicator 流做 δ 扫描，出延迟 / 误报表 → [`results/detector_replay.md`](results/detector_replay.md)。
      **结果见 §5。**
- [x] **Step 2 — 校准硬伤**：`src/utils/seeding.py` + 接进四个脚本；`real_world.py` 坐标常量改名并注明官方 vs 经验；
      订正 `progress_report.md` / `results/phase5_*.md` / `results/phase4_final_verdict.md`。
- [x] **Step 5 — ActionPolicy + oracle 触发**（顺序已调整到 Step 3 之前，理由见 Context 6）
      `src/models/multi_timescale.py` + `scripts/run_phase4_a.py`：
      动作 `{route_adapter, context_reset, buffer_clear, none}` × 触发源 `{detector, oracle}`。
      **设计守卫（来自 adversarial 勘察，13 条中的关键项）**：
      - 报警在**下一步预测前**生效，与 `run_baselines.py --oracle_context_reset` 同一时刻表（避免差一步）
      - `detector.clear()` 改为可选，默认行为不变（否则既有 npz 不可比）
      - `use_adapter_library=False`（Phase 3 路径，detector/library 为 None）不能被弄坏
      - oracle 触发**不清空**影子检测器，否则延迟测不准
      - `drift_points` 为空（Electricity）或超出 `--max_eval_steps` 时**报错**，不能静默变成"永不适应"
      - context 截断后有**最小长度 + 类别覆盖**守卫（否则单类 context 触发 SlowPrior fallback → 误报循环）
      - 输出 stem 带 action / trigger 标签，避免不同分支互相覆盖 npz
      - 报警后**只用切点之后的样本**训练新 adapter（否则在用旧概念数据训练）
- [x] **Step 3 — 标签与切段**：`real_world.py` 加 `label_scheme ∈ {pair_parity, pair_A_vs_B}`，
      过滤行后把 `drift_points` 重映射到 `kept_ids` 坐标；新增官方变点居中的 2500–3000 段（`aligned_v2`）；
      `--label_scheme` 穿过三个脚本 + `run_multiseed.py` 的 out_tag / npz_path / build_cmd（约 8 处调用点）。
- [x] **Step 4 — A0 对比信号诊断**（结果见 §5.1）：在已有 mid 段（含官方点 19,500），滑窗路直接用 npz 已存预测，
      stale 路一次**批量** TabPFN 调用（已实测逐样本独立、批量快 7.4×，成本≈0）；
      比较 contrast / indicator / P(y) 三种信号在 river ADWIN 下的**延迟与误报** → `scripts/diag_contrast_signal.py`。
- [x] **Step 6 — 遗忘回测** `src/utils/forgetting.py`：段首留出集不进 context / buffer / 训练；
      回测前后**模型状态哈希不变**的单测。
- [x] **Step 7 — 双记忆 loader + fixed_ratio**：`CompositeWindowLoader` 接进 `run_phase4_a.py`
      （照 `run_baselines.py:122-140`）；`DualMemoryLoader`（短 FIFO + 类均衡长期库，**加年龄上限**，
      yield-then-push）；`run_multiseed.py` 的 out_tag 带 fr / dual，避免 skip-existing 误跳。
      ⚠️ 勘察警告：在类别均衡的流上朴素双记忆会退化成纯滑窗，长期库必须有年龄上限才有意义。
- [x] **Step 8 — metrics**：共用恢复目标 A\*（同段同 seed 的 Phase 1）+ "固定窗口内少犯错误数"
      （从变点起算 / 从报警起算）；保留旧字段以便与既有结果对比。
      ⚠️ 现有 `adaptation_speed` 用各方法**自己**的漂移前均值 ×0.95 作门槛，准确率低的方法反而更容易"恢复"。
- [x] **Step 9 — ROG 运行清单**：[`ROG_RUNBOOK.md`](ROG_RUNBOOK.md)（含 WSL2 安装、GPU 校准、五个批次、中止判据）。
- [ ] 回头更新 `start_prompt.md` 的架构章节（当前只加了更正 banner）

---

## 3. 留给 ROG 的实验

每批 ≤ 3 天 CPU 等价，做完停下汇报。

1. **新基线**：新段 × `pair_A_vs_B` 的 Phase 1（1 run，确定性）+ 双记忆基线。
   **中止判据**：acc 要有 headroom（不能像 pair_parity 那样 96–98%）且 ≥2 段漂移后跌 ≥5pp。不达标就停下汇报。
2. **判别性对照**（最关键）：oracle 即时触发 vs river 报警 × 四分支动作，
   固定数据 / context / 初始化，比较触发后**同一批样本**的累计错误。三种可能结论：
   - oracle 即时触发也无增益 → 问题在动作本身，优化信号救不了
   - oracle 有效、实际报警无效 → 瓶颈是检测时机，路径 A 才有意义
   - 实际报警也有效 → 直接扩大规模
3. 检测器驱动的三层激活 vs Phase 1 vs 双记忆（3 seeds）。
4. 前沿图：fixed_ratio × 有无校正 × 双记忆，adaptation-vs-forgetting。
5. 若 2–3 有增益 → 6 类改造 + 重跑；胜出配置补 5 seeds，paired t-test。

---

## 4. 验证方式

- `pytest tests/` 全绿（**当前 193 passed**）+ 每步新增单测。
- Step 1 的 river 重放数字可复现（mid 段官方点 local 3,500，延迟 145–172 步）。
- Step 4 出 A0 图与延迟表；**contrast 不优于 indicator 也如实记录**。
- Step 5–8 只跑 `--max_eval_steps 100` smoke，不在 MacBook 上跑长实验。

---

## 5.1 Step 4 结果：对比信号 vs 错误指示器（2026-09-06）

导师路径 A：把 **stale（context 固定在段首、不适应）vs sliding（适应）两路预测的差值**
喂给检测器，替代"适应之后的误差"。在官方变点居中的新段上（`pair_A_vs_B`，river ADWIN δ=0.002，
命中窗口 = 变点前 100 ~ 变点后 600 步）：

| 检测输入 | 检出 | 误报 | d2 延迟 | d3 延迟 | d2 位移 | d3 位移 |
|---|---|---|---|---|---|---|
| `contrast_prob`（路径 A）| **2/2** | 0 | 31 | −45 | **0.491** | 0.028 |
| `contrast_hard`（路径 A）| **2/2** | 0 | 37 | −22 | **0.500** | 0.085 |
| `pred1`（模型类先验）| **2/2** | 0 | 37 | −31 | 0.500 | **0.340** |
| `indicator`（现状）| **1/2** | 0 | **18** | 未检出 | 0.240 | 0.165 |

**结论：路径 A 的前提成立，但优势在召回而不在延迟。**

- **召回**：对比信号 2/2，现状 indicator **只有 1/2**（d3_33240 完全漏检）。这是路径 A 的实质收益。
- **延迟**：d2 上 indicator 反而最快（18 步 vs 31–37）。**对比信号没有压短延迟**。
- **误报**：无漂移的 `d0_control` 段上四种信号**全部零报警**，说明检出不是靠放宽阈值换来的。
- **成本前提被推翻**：stale 路一次批量算完只要 **2–5 s**，sliding 路逐步要 **304–801 s**。
  路径 A 额外成本 ≈ 0，todo 原先"实验成本翻倍"的估计作废。
- **意外发现**：`pred1`（模型输出的类先验，比对比信号更简单、无需第二路预测）
  同样 2/2 且在 d3 上位移最大（0.340）。**值得在批次二里一并作为检测输入对照。**

⚠️ d3 的三次检出都落在标注点**之前** 22–45 步。Souza 的变点标的是温度**设定**切换时刻，
传感器读数会提前变，所以判据允许早报（`--pre_tolerance`，默认 100）。
按旧判据（只认变点之后）这三次会被误记为误报，从而系统性低估检测器。

信号已缓存在 `results/contrast_signal_cache.npz`，改判据可零成本重算。

---

## 5. 关键数字速查（含 Phase 5.5 更正）

| 指标 | 数值 |
|---|---|
| Phase 1 baseline（合成） | 79.89 ± 0.99% (regime_switching, n=5) |
| Phase 3 vs P1（合成） | rotating +1.00 sig / regime NS / combined −0.47 sig 负 |
| Oracle context-reset（合成） | +0.51 pp sig（context 污染只占漂移损失的 ~1/3） |
| Phase 4 detector 触发（合成，**计数已订正**） | raw 0/15、abs 0/15、indicator **9/15 运行 12 事件**、warmstart 同、fit05random 9/15 |
| Phase 5 Electricity phase4a vs P1 | −0.064 NS，detector 1/15 |
| Phase 5 Insects B1+ phase4a vs P1 | −0.172 sig p<0.0001，detector 0/20 |
| **自写 ADWIN 所需 \|Δmean\|**（默认配置 200/200 切分） | **0.209** ← 真实信号只有 0.10，结构性触发不了 |
| **river ADWIN 重放（δ=0.002）** | Insects **16/20 runs 报警、官方漂移 recall 1.00、中位延迟 266 步**；Electricity 1/15（渐进，应沉默） |
| **官方点 19,500 处 indicator 位移** | **0.095–0.105**（5/5 seed）← 旧文档的 "≤0.019、10× 稀释" 已作废 |
| **Insects 官方 drift 位置** | **14,352 / 19,500 / 33,240 / 38,682 / 39,510**（Souza 2020 Table 2） |
| Insects 经验 P(y) 变化点（旧用，非官方） | 12,672 / 14,256 / 17,952 / 46,728 / 52,008 |
| Insects B1+ 四段（旧切法，只覆盖 2/5 官方点） | early [10000,15000) / mid [16000,21000) / late_pre [42500,47500) / late_post [47848,52848) |
| TabPFN 探针在官方点 33,240 的掉幅 | pair_parity 100→83% ／ pair_A_vs_B 99→62% ／ 6 类 95→42%（**含类别重现效应，勿单独用于选任务**）|

---

## 6. 论文写作（阻塞于 §2–§3）

- [ ] 有结论后，和用户 + 导师一起定论文骨架
- [ ] 起草（中文正文 + 英文摘要）
- [ ] 答辩 PPT 草稿 → 导师说要先看一版再正式答辩

**已知写作约束**：目标是 pass，不是期刊 novelty；标题去 PFC 用 Multi-Timescale / Hierarchical；
Ch2.3 保留半页 PFC 灵感但明确不声称生物建模；Phase 2 / 2.5 进附录；
Phase 3 / 4 / 5 的负面结果保留但定位为方法演进的中间步骤；
Limitations 必写：① 合成 rotating +1pp 未迁移到真实数据 ② Insects 标签只是 ID 分组、无官方物种/性别映射
③ Phase 5 的段切法只覆盖 2/5 官方漂移 ④ 真实数据的多 seed 不是独立数据流。

> ⚠️ [`phase5_plan.md`](phase5_plan.md) §7 的 10 章大纲已过时，不要直接用。

---

## 7. 明确不做的事

| 不做 | 原因 |
|---|---|
| 加第三个真实数据集 | 导师认可现有两个（一渐进一突发）已够 |
| 现在就做 6 类改造 | 决策 C：先用二值验证检测-动作链有没有增益，有了再投 45–60 工时 |
| fit_threshold / init_strategy 等细粒度 ablation | 导师："治根不治本……应该是最后补充的实验，不应该是核心实验" |
| 投稿版论文 / 期刊投稿 | 目标已降级为硕士答辩 pass |
| 重跑 Phase 1–4 合成实验 | 数据已齐全且已 commit |
| 新 detector 设计先在合成数据上验证 | 导师：直接在真实数据上验证 |
| 改整体算法架构 | 导师：先确认方法可不可行，"如果完全不行，我觉得可以再看" |
| 在 MacBook 上跑长实验 | MPS 比 CPU 还慢 5.6×；等 ROG |
| `git add -A` / `git add .` | 会把导师私人录音转写推到**公开**仓库 |

---

## 8. 仓库卫生（延续项）

- [x] 导师录音转写加进 `.gitignore`（2026-09-05）
- [x] `.gitignore` 改动已提交（含 `advisor_demo_*.md`）
- [x] `results/smoke/` 与 `results/smoke_oracle.png` 已 gitignore
- [x] README 里 `run_phase4_a.py` 的 flag 修正 `--datasets` → `--dataset`
- [ ] `phase5_plan.md` §7.1 / §8 已过时（已加 superseded 标注，确定新骨架后可整段重写）

---

## 9. 进度日志

| 日期 | 步骤 | 结果 | commit |
|---|---|---|---|
| 2026-09-06 | Step 1 检测器 | river ADWIN 包装 + 零成本离线重放。**自写版 Insects 0/20 → river 16/20 运行报警、官方漂移 recall 1.00、中位延迟 266 步**；Electricity 仍 1/15。证实导师判断：是检测器把方法拦下来了。7 个新单测 | `bd07253` |
| 2026-09-06 | Step 2 校准 | 加 `set_global_seed` 到四个脚本；Insects 坐标常量拆成官方 / 经验两组并注明；四份文档加更正 banner（γ 数字、9/15、2×2 缺格、seed 语义）。115 passed | `2ff0ad7` |
| 2026-09-06 | 计划归位 | Phase 5.5 计划从对话搬进本文件，成为唯一活文档；Step 5 与 Step 3 顺序对调 | — |
| 2026-09-06 | Step 3 标签与切段 | `label_scheme {pair_parity, pair_A_vs_B}` + `aligned_v2` 官方变点居中 5 段（含无漂移对照段 d0_control）。**关键正确性点**：丢行后漂移坐标按保留行累计数重映射（d2 的 1500→767、d4 的 1082/1910→880/1432），否则 oracle 会触发在错样本上。驱动加 `variant_tag` 防 skip-existing 误跳、`--extra_args` 只传给认识的脚本。15 个新单测 | `ae62fc2` |
| 2026-09-06 | Step 6+4 遗忘/对比信号 | `src/utils/forgetting.py`（留出集剔除 + state_hash 零污染断言 + best-minus-final 遗忘定义）；`scripts/diag_contrast_signal.py`。17 个新单测 | `3327095` |
| 2026-09-06 | Step 7 双记忆 | `DualMemoryLoader`（类均衡淘汰 + 年龄上限）作 KDD 2026 文献基线；**实测确认退化性质**：类别均衡时与纯滑窗 context 完全相同，只在单类长段才分歧（单类段深处保有 25 条少数类样本，滑窗为 0）。14 个新单测 | `19d9316` |
| 2026-09-06 | Step 8 metrics | 共用恢复目标 A\* + errors_avoided（可从变点或**报警时刻**起算）；撤掉"各方法自己的门槛"。14 个新单测，193 passed | `18ee2d9` |
| 2026-09-06 | Step 4 重评分 | 允许早报后结论变了：对比信号 **2/2 召回**、indicator 只有 **1/2**（漏 d3）；但延迟没压短（d2: 31–37 vs 18）。`d0_control` 零误报。`pred1` 同样 2/2 且 d3 位移最大，值得进批次二 | `(见下)` |
| 2026-09-06 | Step 9 ROG 手册 | [`ROG_RUNBOOK.md`](ROG_RUNBOOK.md)：MPS 实测比 CPU 慢 5.6× 的依据、WSL2 安装、GPU 校准基准（41s/0.9471）、五个批次含中止判据 | `25baf51` |
| 2026-09-06 | Step 5 ActionPolicy | `--action_on_alarm {route_adapter,context_reset,buffer_clear,none}` × `--trigger_source {detector,oracle}` + `--oracle_lag`。守卫全部落地：context_reset 在**预测前**生效（与 run_baselines oracle 同一时刻表）、最小长度 + 类别覆盖、空 oracle 报错、oracle 下 detector 转影子模式不被 clear、默认动作 auto-resolve 保 Phase 3 路径、`--consolidate_on_post_alarm_data` 可把巩固推迟到报警之后、输出 stem 带分支名。18 个新单测，133 passed | `a7dfadb` |
