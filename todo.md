# TODO — 待办清单

> 入口文档见 [`start_prompt.md`](start_prompt.md)（**先读那个**）。
> 本文件是可执行的待办清单，含实验技术规格。
>
> **最后更新**：2026-09-05
> **当前目标**：完成 KTH 硕士毕业论文答辩（pass 即可，时间不紧张）

---

## 状态速览

| 优先级 | 分类 | 状态 |
|---|---|---|
| P0 | 认知对齐 | 待做 |
| P1 | 导师指定核心实验（Phase 5.5） | 待做 ← **主线** |
| P2 | 论文写作 | 阻塞于 P1 |
| P3 | 仓库卫生 | 部分待做 |
| — | 明确不做的事 | 见文末 |

> **用户希望接手的 AI 自己判断优先级**。下面的 P1 三项（路径 A / 路径 B / 维度 C）导师说是"平行"的，
> 但成本与产出概率不同，建议读完后向用户提建议再动手。

---

## P0 — 认知对齐（先做，~30 min）

- [ ] 读 [`start_prompt.md`](start_prompt.md) 全文，重点 §5「方向说明」
- [ ] 读 `6_1和导师汇报的录音_包含两份转写.txt` 前半部分（GPT-6 Astra 版），重点 00:02–13:00 技术讨论段
      - 讯飞听见那份（文件后半）专业名词转写不准，仅作交叉参考
- [ ] 确认理解这个转变：**从"把负面结果包装成发现"→"重新设计 detector 争取正面结果"**

**为什么重要**：项目文档里 Phase 5 的结论曾被定位为论文核心 contribution，这个定位已被导师否定。
不先对齐会沿着错误方向继续做。

---

## P1 — 导师指定核心实验（Phase 5.5）← 主线

**背景**：真实数据上 detector 触发率 Electricity 1/15、Insects 0/20 → 三层结构根本没被激活
→ "方法无效"这个结论其实**从未被真正验证过**。导师认为方法本身可能可行，是 detector 拦住了。

导师原话：
> "如果你现在检测器都不触发的话，你的三层的结构其实就完全体现不了作用，你现在这个方法其实就无法去验证。"
> "所以我们就要把这个场景去设计到，就是说能够更容易的让它触发，然后看看我这算法的效果。"

**共同前提**：新实验**直接在真实数据上跑，不必先过合成**（导师："合成数据有点太理想了，只适合做可行性分析"）。

---

### P1-A — 路径 A：对比信号 detector ⭐ 最贴合导师核心诊断

**思路**（导师原话）：
> "你可以比较它的适应前后的差距，适应前后的差距作为 detect 的信号。因为你现在如果用它适应以后的误差的话，那个信号就很弱了。"
> "一个是滑窗 0 就不适应，一个是滑窗 300，两种对比，那个差作为信号去作为检测器的信号。这个刻画的就是它的 drift。"
> "基于检测信号可以判定它 drift 的强弱，如果你的差值很大说明 drift 比较明显，对吧？很直观。"

即：**不再把"适应之后的误差"喂给 ADWIN，而是把"适应 vs 不适应两路预测的差值"喂给 ADWIN。**

#### A0 — 便宜的信号质量诊断（**先做这个，不要跳过**）

**为什么**：路径 A 会让 TabPFN 调用翻倍（单次真实数据 run 从 ~1.5–2.25h 涨到 ~3–4.5h）。
上规模前先花 2–3h 确认信号真的更强。这套做法在 Phase 4 Day 0.5 的 Oracle 诊断上验证过是有效的。

**怎么做**：
1. 选 Insects `late_post` 段（含 drift @ local t=4160）+ `early` 段（含 drift @ 2672 / 4256），seed=42
2. 每个 prequential 步跑**两次** `SlowPrior.predict()`：
   - **stale 路**：context 固定为该段最早的 200 个样本（永不更新）→ 代表"不做 in-context 适应"
   - **sliding 路**：当前滑动 context（现有行为）→ 代表"做 in-context 适应"
3. 记录三条序列：两路的 `predict_proba` 差值 `|p_stale - p_sliding|`、两路硬预测是否不一致 `int(pred_stale != pred_sliding)`、以及现有的 indicator 流
4. 对每个 documented drift 点，算前后 ±200 步的均值差 `|Δ|`，与现有 indicator 的基线对比

**判据（关键）**：现有 indicator 流在真实 Insects 上 `|Δ| ≤ 0.019`，比合成的 0.20 小 10×，这就是 ADWIN 沉默的原因
（见 [`results/phase5_confound2_diagnostic.md`](results/phase5_confound2_diagnostic.md)）。
**对比信号的 `|Δ|` 要显著大于 0.019 才值得上规模**；理想是接近或超过 0.1。

**成本**：~2–3h。不需要改模型代码，可以写成独立诊断脚本（例如 `scripts/diag_contrast_signal.py`）。
**产出**：诊断图 + 一段结论。若 `|Δ|` 没有明显提升，**立刻停下来汇报**，不要硬上规模。

#### A1 — 实现（A0 通过后再做）

- [ ] `src/models/multi_timescale.py` 加 `detector_input` 参数：`{"indicator"（默认，向后兼容）, "contrast"}`
      - 当前 detector 输入硬编码在 [multi_timescale.py:304-306](src/models/multi_timescale.py#L304-L306)
- [ ] `contrast` 模式下 `step()` 需要拿到 stale context —— 建议由调用方（脚本层）传入，避免模型层持有数据
- [ ] `scripts/run_phase4_a.py` 加 `--detector_input` flag（默认 `indicator`）
- [ ] 补单元测试到 `tests/test_multi_timescale_phase4a.py`
- [ ] 用 `--max_eval_steps 100` smoke 一遍再上规模

**成本**：半天到一天编码 + 测试。

#### A2 — 真实数据验证

- [ ] 先跑**小规模**：Insects 2 个含 drift 的段 × 2 seeds × {phase1, phase4a}，看 detector 触发率与准确率
- [ ] 通过后再考虑扩到完整 multiseed

**成本**：小规模 ~8 组 × 3–4.5h ÷ n_parallel=2 ≈ 15–18h。完整 sweep 会到 100h+，**按预算裁剪，不要默认全跑**。

---

### P1-B — 路径 B：fixed_ratio（提高固定池比例）

**思路**（导师原话）：
> "你固定更多，它的 error 本来就更大了……两种路径都是为了提升它的检测器的信号强度，然后去验证你的校正器。"

即：让模型少适应一点，误差信号就不会被 in-context learning 吃掉，detector 就能看到东西。

**现状**：`CompositeWindowLoader`（[src/data/temporal_loader.py:109](src/data/temporal_loader.py#L109)）已经实现且支持 `fixed_ratio`，
但**只接进了 `run_baselines.py`**；`run_phase3.py` / `run_phase4_a.py` 仍然只用 `TemporalWindowLoader`。

- [ ] 把 `--fixed_ratio` + `CompositeWindowLoader` plumb 进 `run_phase4_a.py`
      （照搬 `run_baselines.py` 里现成的分支写法即可，约 10 行）
- [ ] 在真实数据上扫几档 `fixed_ratio ∈ {0.0, 0.33, 0.67}`，1–2 个含 drift 的段 × 1–2 seeds
- [ ] 记录每档的：detector 触发次数、overall acc、drift 后恢复速度

**关键权衡**：Phase 2.5 已知 `fixed_ratio` 提高会让适应速度变快但**总体准确率下降**（fr=0.67 时掉 2.5pp，fr=0.93 时掉 11pp）。
所以这条路径要找的是**触发率提升足以补偿准确率损失**的甜点，不是无脑调高。

**成本**：plumb 半小时；实验 ~6 组 × 1.5–2.25h ÷ 2 ≈ 5–7h。
**这条比路径 A 便宜得多，可以先做**（不需要双倍 TabPFN 调用）。

---

### P1-C — 维度 C：遗忘 / 适应 trade-off ⭐ 导师点名最可能出正面结果

**思路**（导师原话）：
> "拿之前那批数据也作为一个指标，看一下之前上面的效果，做一个适应跟遗忘之间的平衡，这个可以作为一个点。"
> "遗忘程度跟你的适应能力一定是 trade-off 的关系，不可能既要都要。这个就是看你的方法能不能取得一个平衡。"
> "你的适应能力差不多的情况下，你的前面的遗忘能不能减少。如果这个是可以的话，那它也可以作为你一个正面结果的一个点。"

即：引入**第二个评估维度**。现在只看"适应新 regime 有多快/多准"，再加上"适应之后还记不记得旧 regime"。
如果我们的多时间尺度系统在**适应能力持平**的前提下**遗忘更少**，这就是一个可以写进论文的正面结果。

**为什么这条有吸引力**：
- 不依赖 detector 是否触发 → 不被 P1-A/B 阻塞，可以并行
- 概念上直接对应系统设计动机（slow 层保长期、fast 层保近期），叙事自洽
- 是 continual learning 领域的标准评估维度（stability-plasticity trade-off / backward transfer），有现成文献可引

**现状**：**代码里完全不存在**遗忘/回测相关实现，需要新写。

- [ ] 新建回测评估 harness（建议 `src/utils/forgetting.py` 或扩展 `src/utils/metrics.py`）
      - 在流的若干检查点，用当前模型回测**更早 regime 的留出样本**，记录准确率
      - 注意：回测必须用留出样本，不能用已经进过 context/buffer 的样本，否则不是真的在测遗忘
      - 注意：回测**不能污染** prequential 主循环的状态（buffer / gate 参数都不能因回测而更新）
- [ ] 在 Insects（突发漂移，regime 边界清楚）上对比 phase1 / phase3 / phase4a 的遗忘曲线
- [ ] 产出 adaptation-vs-forgetting 二维图（横轴适应能力、纵轴遗忘程度，每个方法一个点/一条曲线）

**成本**：编码 1 天（含测试）；实验可复用已有 run 或小规模新跑 ~10h。

---

### P1 优先级建议（供和用户讨论）

导师说三项"平行"，但从成本/产出看：

| 项 | 成本 | 出正面结果的概率 | 依赖 |
|---|---|---|---|
| **P1-B** fixed_ratio | 最低（~6h，代码几乎现成） | 中（有准确率损失的权衡） | 无 |
| **P1-C** 遗忘 trade-off | 中（1 天编码 + ~10h 实验） | **较高**（导师点名，且不依赖 detector 触发） | 无 |
| **P1-A** 对比信号 | 最高（编码 1 天 + 实验成本翻倍） | 中高（最贴合导师核心诊断） | A0 诊断先过 |

一个可能的顺序：**P1-B（最便宜，快速看到 detector 能不能被推动）→ P1-A0（便宜诊断）→ P1-C（并行开工）→ P1-A1/A2**。
但这只是建议，**请和用户确认后再动手**。

---

## P2 — 论文写作（KTH 硕士毕业论文）

**阻塞于 P1** —— 论文主线取决于 Phase 5.5 能不能拿到正面结果。

- [ ] P1 有结论后，和用户+导师一起定论文骨架
- [ ] 起草（中文正文 + 英文摘要）
- [ ] 答辩 PPT 草稿 → 导师说要先看一版再正式答辩

**已知的写作约束**：
- 目标是 **pass**，不是期刊 novelty。不需要预演 reviewer 攻击、不做投稿版
- 标题去 PFC，用 Multi-Timescale / Hierarchical 等 ML 术语
- Ch2.3 保留半页 PFC 灵感动机，但明确写"因 frozen TabPFN 无 weight-level consolidation，不声称生物建模"
- Phase 2 / 2.5 进附录（保留实验诚实性）
- Phase 3 / 4 / 5 的负面结果**保留**，定位为方法演进的中间步骤，不作核心卖点
- Limitations 必写：合成 rotating +1pp 没迁移到真实数据（mechanism-specific，不普适）

> ⚠️ [`phase5_plan.md`](phase5_plan.md) §7 里那份 10 章大纲和标题是 Phase 5 之前的旧版，**已过时，不要直接用**。

**导师要求的流程**（录音 14:52–15:14、16:50–17:14）：
- 进度随时更新到 GitHub 仓库，有需要讨论的点随时联系
- **开始写论文时要告诉导师**，导师要一起看
- 答辩 PPT 先出草稿一起过，正式答辩前再对一次

---

## P3 — 仓库卫生

- [x] **把导师录音转写加进 `.gitignore`**（隐私 P0）— 已完成 2026-09-05
      仓库是公开的（github.com/wenboxia/neural-temporal-learning），转写此前未被忽略也未追踪，
      一次 `git add -A` 就会泄露。已加 `*录音* / *转写* / *.txt` 规则并验证生效；历史提交中无 .txt 被追踪
- [x] 提交 pending 的 `.gitignore` 改动（含 `advisor_demo_*.md`）— 已完成 2026-09-05
- [x] 处理 `results/smoke/` 与 `results/smoke_oracle.png`（Phase 5 smoke 残留）— 已加 gitignore 2026-09-05
      （本地保留，不进仓库；无论文价值）
- [x] 修 [`README.md`](README.md) 里 `run_phase4_a.py` 的 flag：`--datasets` → `--dataset`（单数）— 已完成 2026-09-05
- [ ] [`phase5_plan.md`](phase5_plan.md) §7.1 旧标题/章节列表、§8 ETA 表已过时 → 已加 superseded 标注，
      后续若确定新论文骨架可整段重写

---

## 明确不做的事

| 不做 | 原因 |
|---|---|
| 加第三个真实数据集 | 导师认可现有两个（一渐进一突发）已够，"这几个数据应该已经算比较干净了" |
| β binarization ablation（6-class native Insects） | 成本 ≥ 1 周（系统重构 + 重跑全部基线），不在主线 |
| fit_threshold / init_strategy 等细粒度 ablation | 导师："治根不治本……应该是最后补充的一些实验，它不应该是核心实验" |
| 投稿版论文 / 期刊投稿 | 目标已降级为硕士答辩 pass |
| 重跑 Phase 1–4 合成实验 | 数据已齐全且已 commit |
| 新 detector 设计先在合成数据上验证 | 导师：直接在真实数据上验证 |
| 改整体算法架构 | 导师：先确认方法可不可行，"如果完全不行，我觉得可以再看" —— 排在 P1 之后 |

---

## 附：关键数字速查

| 指标 | 数值 |
|---|---|
| Phase 1 baseline | 79.89 ± 0.99% (regime_switching, n=5) |
| Phase 3 vs P1 | rotating +1.00 sig / regime NS / combined −0.47 sig 负 |
| Oracle context-reset | +0.51 pp sig（context 只占损失的 ~1/3） |
| Phase 4 detector 触发（合成） | raw 0/15、abs 0/15、indicator 12/15、warmstart 12/15、fit05random 9/15 |
| Phase 5 Electricity phase4a vs P1 | −0.064 NS，detector 1/15 |
| Phase 5 Insects B1+ phase4a vs P1 | −0.172 sig p<0.0001，detector **0/20** |
| γ 信号稀释 | 真实 indicator \|Δ\| ≤ 0.019 vs 合成 0.20（**10× 稀释**）← 路径 A 要打败的基线 |
| Insects drift 位置（绝对坐标） | 12,672 / 14,256 / 17,952 / 46,728 / 52,008 |
| Insects B1+ 四段 | early [10000,15000) / mid [16000,21000) / late_pre [42500,47500) / late_post [47848,52848) |
