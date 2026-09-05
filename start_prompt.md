# START HERE — 项目交接说明

> **给接手这个项目的任何 AI**：这是唯一的入口文档。读完这一份，你就知道项目是什么、进行到哪、下一步该做什么、以及哪些方向已经被否决。
> 具体待办清单在 [`todo.md`](todo.md)。
>
> **最后更新**：2026-09-05

---

## 1. 项目一句话定位

在**冻结的 TabPFN**（预训练表格基础模型）之上，构建一个多时间尺度（slow / inter / fast）的适配器系统，用来应对表格数据流中的**概念漂移**（concept drift）。

灵感来自前额叶皮层（PFC）的多时间尺度记忆结构，但因为 TabPFN 权重不做微调，**不声称做生物建模**（见 §6 偏差说明）。

## 2. 目标（2026-09-05 更新，已降级）

**完成 KTH 硕士毕业论文答辩，只追求 pass，做得好一点更好。**

- ❌ 不再以发表期刊论文为目标（这是早期目标，已放弃）
- ✅ 交付物 = 一篇能通过答辩的硕士毕业论文（中文正文 + 英文摘要）
- 答辩形式：线上；时间不紧张，无需赶工

这条降级直接影响写作标准：不需要论证 novelty 的强度，不需要预演 reviewer 攻击，不需要投稿版。重点是**工作量完整、方法讲清楚、实验诚实、结论能自圆其说**。

## 3. 研究问题与系统架构

**问题**：TabPFN 在概念漂移下会掉点（漂移后准确率降 12–13pp，需 ~61 步恢复）。能否在不动 TabPFN 权重的前提下，用多时间尺度结构补上这个损失？

```
输入流 (X_t, y_t) ──▶ TemporalWindowLoader (滑动 context 窗口)
                                    │
                       ┌────────────▼────────────┐
                       │  Level 1: SlowPrior      │  冻结 TabPFN（in-context learning）
                       │  src/models/slow_prior.py│
                       └────────────┬────────────┘
                                    │ y_slow
                       ┌────────────▼────────────┐
                       │  Level 3: FastCorrector  │  FIFO buffer + KNN/EMA 残差校正
                       │  src/models/fast_corrector.py│  零可学习参数
                       └────────────┬────────────┘
                                    │ correction
                       ┌────────────▼────────────┐
                       │  Level 2: GatedEnsemble  │  gate (MLP→softmax 三路)
                       │  + AdapterLibrary (P4 A) │  + ADWIN drift detector
                       │                          │  + per-regime adapter 硬路由
                       └────────────┬────────────┘
                                    │ (α, β, γ) + active adapter id
                                    │
          y_final = clip(y_slow + β·y_inter + γ·correction, 0, 1)
```

编排器：`MultiTimescaleModel.step(X_ctx, y_ctx, x_t, y_t, t)`（[src/models/multi_timescale.py](src/models/multi_timescale.py)）

### 硬约束（不可违反）

| 约束 | 说明 |
|---|---|
| **TabPFN 权重绝不微调** | "slow" 指 context 变、权重不变。V1 计划里的 Inter→Slow fine-tuning consolidation 已在 V2 砍掉（GPU 不足 + catastrophic forgetting 风险） |
| **CPU only** | MacBook 本地跑，`context_size ≤ 3000` |
| **Prequential 协议** | 先预测 → 观测真实标签 → 更新。**绝不能偷看未来标签** |
| **归一化不能泄漏** | 真实数据的 StandardScaler 只 fit 前 200 个样本，再 transform 全段 |

### 融合与训练细节

- **residual-additive 融合**：`y_inter` 和 `correction` 是加在 `y_slow` 上的残差；gate 三路权重经 softmax 和为 1，但 `α` 实际未被使用（保留作向后兼容/可视化）。`y_final` 最后 clip 到 [0,1]
- **per-step 训练**：每个 prequential 步用 `MSE(y_final, y_t)` 更新 **gate**；**adapter 只在 consolidation 时**才拿到梯度（Phase 3 F 改动，gate/adapter 优化器分离）
- **consolidation 触发**：Phase 3 用 `|mean(buffer.errors[-window:])| > threshold` + cooldown；Phase 4 A 改为纯 **ADWIN detector 驱动**（routing → consolidate active adapter）

### 模块地图

| 文件 | 职责 | 关键 API |
|---|---|---|
| `src/data/synthetic.py` | 3 个合成漂移生成器 | `make_dataset(name, **kw) → SyntheticDataset` |
| `src/data/real_world.py` | Electricity / Insects 加载 + 二值化 + 无泄漏归一化 | `load_real_world(name, segment_id, insects_aligned=False)` |
| `src/data/temporal_loader.py` | 滑动窗口 / 组合窗口（固定池+滑窗） | `TemporalWindowLoader`, `CompositeWindowLoader(fixed_ratio=)` |
| `src/models/slow_prior.py` | **Level 1**：冻结 TabPFN 包装器，懒加载权重 | `predict(X_ctx, y_ctx, X_query)` — **无状态，context 显式传入** |
| `src/memory/buffer.py` | FIFO 工作记忆，存 `(x, error, embedding)` | `recent_errors(n)` / KNN / EMA 查询 |
| `src/models/fast_corrector.py` | **Level 3**：KNN/EMA 残差校正，零可学习参数 | `correct(x)`, `should_consolidate(window, threshold)` |
| `src/models/gated_ensemble.py` | **Level 2**：gate(MLP→softmax 三路) + adapter(MLP 残差) | `forward(x, y_slow, correction) → (y_final_raw, weights)` |
| `src/consolidation/fast_to_inter.py` | 快→中巩固：把 buffer 误差 MSE 蒸馏进 adapter | `consolidate(...)`，结束后清空 buffer |
| `src/drift/error_detector.py` | ADWIN 变点检测（1D 误差流） | `update(value) → bool`, `clear()` |
| `src/regime/adapter_library.py` | Per-regime adapter 库（`nn.ModuleDict`）+ 硬路由 | `route(...)`, `active_optimizer()` |
| `src/models/multi_timescale.py` | **编排器**，三层串起来 | `step(X_ctx, y_ctx, x_t, y_t, t)` — **detector 输入硬编码在 ~L304** |
| `src/utils/metrics.py` | 评估指标 | `summarize_results()`, `window_accuracy()`, balanced acc, AUC |

脚本：`run_baselines.py`(P1) / `run_phase2.py` / `run_phase3.py` / `run_phase4_a.py` / `run_multiseed.py`(批量驱动)。
测试：`tests/` 9 个文件，**105 passed**。

### 数据集清单

**合成**（`src/data/synthetic.py`，全部二分类、类别平衡）：

| 数据集 | 漂移类型 | 机制 |
|---|---|---|
| `rotating_boundary` | 渐进 | 2D 高斯特征，线性决策边界以 `drift_speed=0.003` rad/步旋转 |
| `regime_switching` | 突发 | 循环 `n_regimes=3` 个体制，**每个体制特征均值和决策权重互相独立**（by design 无可迁移模式），每段 `regime_length=500` |
| `combined_drift` | 混合 | 前 5 维特征全程线性漂移 2σ + 决策边界每 3000 步突变 |

**真实**（`src/data/real_world.py`）：

| 数据集 | 来源 | 规模 | 漂移 | 注意 |
|---|---|---|---|---|
| Electricity | OpenML id=151 | 45,312 × 8 特征（one-hot 后 14 维） | 渐进 / 季节性，无 crisp 切点 | 时序按 (date, period) 已排序；target UP/DOWN → 1/0 |
| Insects | USP DS `abrupt_balanced`（Google Drive，river master 的直链；river 0.23 内置 URL 已 404） | 52,848 × 33 特征 | **突发**，5 个 documented drift @ 12,672 / 14,256 / 17,952 / 46,728 / 52,008 | 原始 **6 类**，按 sex-pair 二值化 `{2,4,11}→0` vs `{3,5,12}→1`（物种映射查不到，Limitations 需写明） |

**Insects B1+ drift-aligned 四段**（当前使用的切法，覆盖全 5/5 drift）：
`early [10000,15000)` / `mid [16000,21000)` / `late_pre [42500,47500)` / `late_post [47848,52848)`，互不重叠，每个 drift 距段边界 ≥ 744 样本（满足 ADWIN 缓冲）。

## 4. 项目履历：从头到尾做了什么

> 一行一个阶段太薄了，这里按「做了什么 / 结论 / 踩过的坑」三列写。
> **完整叙事（含全部数字和实验图）见 [`progress_report.md`](progress_report.md)（1083 行）。**

| 阶段 | 做了什么 | 结论 | 踩过的坑 / 教训 |
|---|---|---|---|
| **Phase 1** | 冻结 TabPFN 滑动窗口 baseline，三数据集 multi-seed | 79.89 ± 0.99% (regime_switching)；漂移后掉 12–13pp，需 ~61 步恢复 | 确立了"TabPFN 对漂移脆弱"这个动机 |
| **Phase 2** | + Level 3 FastCorrector（KNN / EMA 残差校正） | **三数据集全 NS**，无显著改善 | 单 seed 曾报 "82.5% best"，其实在噪声内。→ 进论文 Appendix A |
| **Phase 2.5** | 导师 2026-03-11 反馈四条：①类别平衡修复（体制内曾 86:14）②多数据集 Phase 2 ③`CompositeWindowLoader`（固定池+滑窗）④补 balanced acc / AUC | 组合窗口把适应速度 126→62 步，**但总体准确率掉 2.5pp**（fr=0.67）；fr=0.93 时掉 11pp | **适应速度与总体准确率无法两全** —— 这个权衡在 Phase 5.5 路径 B 会再次出现 |
| **Phase 3** | + Level 2 GatedEnsemble（三路 softmax 门控）+ Fast→Inter 巩固；做了 4-way ablation + **三轮架构迭代**（v2 残差融合 / B 松触发 / F 分离优化器+cooldown） | rotating **+1.00 sig** / regime −0.18 **NS** / combined **−0.47 sig 负** | ⚠️ **单 seed 误判教训**：初期单 seed 报 regime_switching "−4.5pp 灾难"，multi-seed (n=5) 重跑归零为 −0.18 NS，纯抽样噪声。**此后所有结论一律 n=5 paired t-test**<br>🔑 **核心发现**：gate 的 **β ≈ 0** 跨全部数据集 —— 共享 adapter 跨 regime 无可迁移模式，gate 理性地忽略它。这是 Phase 4 走 per-regime 隔离的动机 |
| **外部 cross-review** | 4 家 LLM 独立评审当时的结论 | 共识指出 multi-seed 是必须补的方法学要求 | 若不补，后续论文会基于错误数字做错误叙事 |
| **Phase 4 Day 0.5** | Oracle context-reset：漂移后强制把 context 截断到最近 50 样本（模拟"完美 detector"） | **+0.51pp sig** (paired t=+3.25) | **context 污染只占损失的 ~1/3**，剩下 2/3 是新体制样本不足。→ 决定走 Design A 而非只做 context-reset |
| **Phase 4 Day 1.5** | 实施 Design A（ADWIN + per-regime AdapterLibrary + 硬路由），做 **4 轮 detector 输入 ablation** | raw error **0/15 触发**、\|error\| **0/15**、**0/1 indicator 12/15**、warmstart 12/15 | 🔑 raw error 在类别平衡数据上 mean ≈ 0，ADWIN 结构性看不到；\|error\| 被 TabPFN 自适应消化。**只有硬离散信号能绕过自适应**<br>⚠️ 但 25/25 routing 全是 `create`，**reuse 路径从未激活** |
| **Phase 4 Day 2** | 补 2×2 析因缺失格（fit_threshold=0.5 × random init） | fit_threshold 主因 **70%**、init_strategy 次因 **30%**、**完美加性无交互** | ⚠️ **overclaim 修正教训**：Day 1.5 写的"warm-start 是 anti-pattern"是过强表述，Day 2 数据显示它只是 secondary factor (<0.1pp)。诚实修正写进方法学 |
| **Phase 5 Stage A** | Electricity 45 runs（A+ 协议 start/middle/end × 5 seeds × 3 phases），~24h | phase4a vs P1 **−0.064 NS**；detector **1/15**；F4 reuse 复现（1/1 全 create） | 渐进漂移上 detector 沉默符合设计精神（同合成 rotating 0/15）；但合成 rotating 的 +1pp 增益**没迁移过来** |
| **Phase 5 Stage B**（已归档） | Insects 同样用 A+ 三段协议，45 runs，~48h | detector **0/15** | ⚠️⚠️ **最大的坑**：事后诊断发现 A+ 的三段里 **14/15 段根本不含任何 documented drift**（5 个 drift 全在切片窗口外）。这个 0/15 是 trivially correct，**结论无效，45 个 run 白跑**。数据归档在 `results/archive_misaligned_stage_b/`<br>👉 **教训：切片前必须先定位 drift 位置** |
| **Phase 5 Stage B1+** | 重设计为 4 个 drift-aligned 段覆盖全 5/5 drift，60 runs，~60h | detector **仍 0/20**（此时是 valid 数据点）；phase4a vs P1 **−0.172 sig p<0.0001**；3/4 段单独也 sig 负 | 协议层面的解释被排除 → 逼出下面的 γ 诊断 |
| **Phase 5 γ 诊断** | 对每个 drift 前后 ±200 步分析三种信号（12 张图） | indicator \|Δ\| **≤ 0.019**（合成是 0.20，**10× 稀释**）；但 P(y_pred=1) shift 有 0.03–0.13 | 🔑 模型**确实**跟着漂移动了，只是 TabPFN 在 ~10–20 步内就把 P(y) shift 吸收掉，错误率几乎不抬升，ADWIN 看不到切点 |
| **2026-06-01 导师汇报** | 汇报全部进展 + 4 个 framing 偏差 | **否定了把上述负面结果当论文卖点**；指出 detector 不触发是设计问题，给了三条补救路径 | 见 §5 —— **这是当前方向的依据** |
| **Phase 5.5** | 🚧 待启动：路径 A 对比信号 / 路径 B fixed_ratio / 维度 C 遗忘 trade-off | — | 规格见 [`todo.md`](todo.md) P1 |

---

## 5. ⚠️ 方向说明 —— 读到这里请务必看完

### 5.1 已被否定的方向

Phase 5 结束后，项目文档一度把这个结论当作论文核心卖点：

> "Foundation model 的 self-adaptation 速度 outpaces change-point detector 的检测延迟 —— 这是一个 fundamental property discovery，不只是方法失败。"
> 候选标题：*"When Foundation Models Outrun Drift Detectors"*

**2026-06-01 导师汇报后，这个 framing 被否定了。** 导师原话（录音转写）：

> "这个答辩的时候比较容易被质疑，我觉得这个不太好，还是不太稳。还是得有一些正面的，我们还是按照正面来推进。"
> "如果都是负面结果的话，那这个论文其实也不太好写。我们得尽快定出来一版比较正面的结果，那个是论文能不能立足的一个关键。"

**用户本人的判断与导师一致**：负面结果不作为项目的核心展示内容；硬把负面结果说成"发现了什么"站不住脚。

### 5.2 导师的诊断：不是根本性质，是检测器没做好

导师认为 detector 不触发是**设计问题**，方法本身可能是可行的：

> "现在你之所以没有提升，只是因为你的检测器其实还没起作用……我觉得算法本身应该是可行的，只不过你的检测器把它拦截下来了。"
> "如果你现在检测器都不触发的话，你的三层的结构其实就完全体现不了作用，你现在这个方法其实就无法去验证。"

也就是说：**detector 0/20 不触发 ⇒ 三层结构根本没被激活 ⇒ 现有的"方法无效"结论其实没有被真正验证过。**

### 5.3 现在的主线

```
重新设计 detector 信号（导师给了两条平行路径）
        ↓
  让 detector 真正触发起来
        ↓
  才能验证校正器/适配器到底有没有效果
        ↓
  争取拿到正面结果 → 论文主线
```

导师给的三个具体方向（技术规格见 [`todo.md`](todo.md) P1）：

- **路径 A — 对比信号 detector**：用"滑窗=0（不做 in-context）"与"滑窗=300（做 in-context）"两路预测的**差值**作为 detector 输入。"差值差的就是 drift"，差值大 = drift 明显
- **路径 B — fixed_ratio**："你固定更多，它的 error 本来就更大了" —— 提高固定池比例，让误差信号别被自适应吃掉
- **维度 C — 遗忘/适应 trade-off**：拿早期 regime 的数据回测，看遗忘程度。导师点名这是**可能产出正面结果的点**

### 5.4 负面结果怎么处理

**保留，但降级。**

- ❌ 不删除 —— 答辩时被问"你试过哪些方案"必须答得出来；Phase 3/4 的 ablation 本身就是工作量证明
- ❌ 不作为论文的核心 contribution 或卖点
- ✅ 定位为**方法演进过程中的中间步骤**，放正文的方法探索章节或附录
- ✅ 所有实验数字、统计结果、γ 诊断的**事实内容全部有效**，被取代的只是"这是个重大发现"这个定位

---

## 6. 与最初设想的四个偏差（论文里要讲清楚）

1. **PFC framing 弱化**：标题去掉 PFC，用 "Multi-Timescale / Hierarchical" 等 ML 术语。因为 V1 计划里对应 PFC 长期记忆巩固的 Inter→Slow fine-tuning 被砍了，最关键的生物对应链断了，硬声称生物建模会被攻击。保留半页灵感动机即可。
2. **Inter→Slow consolidation 被砍**：只保留 Fast→Inter（蒸馏 buffer 误差到 adapter），TabPFN 全程冻结。原因：GPU 不足 + catastrophic forgetting 风险 + CPU only 不可行。
3. **合成数据上的 rotating_boundary +1pp 增益没有迁移到真实数据**：合成的"决策边界旋转"机制与 Electricity 的"covariate seasonal shift"机制不同，属于 mechanism-specific，不普适。这条要诚实写进 Limitations。
4. **论文定位**：不是"我们的方法赢了 X pp"式的 method paper。最终定位取决于 Phase 5.5（导师三方向）的结果 —— 拿到正面结果就写正面，拿不到再和导师讨论退路。

> 上面 4 条在 2026-06-01 汇报中向导师讲过。导师对偏差 1、3 没有明确反对；对"把负面结果当发现"（原偏差 2 的表述）明确反对，见 §5.1。

---

## 7. 文件地图

### 先读这些
| 文件 | 内容 |
|---|---|
| **`start_prompt.md`**（本文件） | 项目入口 |
| **[`todo.md`](todo.md)** | 待办清单 + 实验技术规格 |
| [`progress_report.md`](progress_report.md) | 完整实验叙事 Phase 1→5，含全部结果图 |
| [`CLAUDE.md`](CLAUDE.md) | 代码结构、命令清单、模块表、设计约束 |

### 实验结果细节
| 文件 | 内容 |
|---|---|
| [`results/phase4_final_verdict.md`](results/phase4_final_verdict.md) | Phase 4 五段对照大表 |
| [`results/phase5_real_summary.md`](results/phase5_real_summary.md) | Phase 5 合并结论 + F1-F4 复现矩阵 |
| [`results/phase5_real_summary_electricity.md`](results/phase5_real_summary_electricity.md) | Electricity 详细数字 |
| [`results/phase5_real_summary_insects.md`](results/phase5_real_summary_insects.md) | Insects 详细数字 |
| [`results/phase5_confound2_diagnostic.md`](results/phase5_confound2_diagnostic.md) | γ 诊断（信号稀释量化） |
| [`results/archive_misaligned_stage_b/`](results/archive_misaligned_stage_b/) | 旧 A+ 协议归档（协议演进的诚实记录） |

> ⚠️ Phase 5 系列文档顶部都有 framing 说明：**数据有效，"核心 contribution" 的定位已被导师反馈取代**。

### 计划文档
| 文件 | 状态 |
|---|---|
| [`phase4_plan.md`](phase4_plan.md) | 历史记录，Phase 4 已完成 |
| [`phase5_plan.md`](phase5_plan.md) | Phase 5 部分已完成；**§7 论文大纲已过时**，以 `todo.md` 为准 |
| [`implementation_plan_v2.md`](implementation_plan_v2.md) | V2 设计规格，当前代码遵循 |

### 私人文件（**已 gitignore，绝不可提交**）
| 文件 | 内容 |
|---|---|
| `6_1和导师汇报的录音_包含两份转写.txt` | 导师汇报录音转写。**两份转写，以 GPT-6 Astra 那份（文件前半部分）为准**，讯飞听见那份专业名词准确度差 |
| `advisor_demo_2026-06-01.md` | 汇报演示脚本 |

## 8. 环境与常用命令

```bash
# 安装
pip install -e ".[dev]"
pip install openml river          # Phase 5 真实数据需要

# 测试（应 105 passed）
pytest tests/

# Phase 1 基线（合成）
python scripts/run_baselines.py --dataset regime_switching --n_samples 3000 --context_size 200

# Phase 4 A（per-regime adapter library + ADWIN routing，合成）
python scripts/run_phase4_a.py --dataset regime_switching --n_samples 3000 --context_size 200

# 真实数据单次运行（注意是 --dataset 单数）
python scripts/run_phase4_a.py --dataset electricity --dataset_source real \
  --segment_id start --context_size 200

# 真实数据 Insects drift-aligned 四段
python scripts/run_phase4_a.py --dataset insects --dataset_source real \
  --insects_aligned --segment_id early --context_size 200

# multiseed 批量（真实数据）
python scripts/run_multiseed.py --dataset_source real --datasets insects \
  --insects_aligned --configs phase1,phase3,phase4a \
  --seeds 42,123,456,789,1024 \
  --segments early,mid,late_pre,late_post --n_parallel 2 --partial_tag _insects

# 快速调试（1-2 分钟）
python scripts/run_phase4_a.py --dataset regime_switching --max_eval_steps 100
```

**所有脚本必须从项目根目录 `neural_1/` 运行。** 结果 PNG 进 git，`.npz` 被 gitignore。

### 运行时间参考（CPU，实测）
| 场景 | 耗时 |
|---|---|
| 合成 3000 步单次 | ~25–30 min |
| 真实数据 5000 步单次 | ~1.5–2.25 h |
| Electricity 45 runs (n_parallel=2) | ~24 h |
| Insects 60 runs (n_parallel=2) | ~60 h |

跑长实验请用后台任务，并增量写 partial md。

## 9. 不要做的事

| ❌ 不要 | 原因 |
|---|---|
| 重跑 Phase 1–4 的合成实验 | 数据已齐全且已 commit，重跑纯浪费（合成三数据集跑一轮要三四天） |
| 再做 fit_threshold / init_strategy 之类细粒度 ablation | 导师明确说"治根不治本……应该是最后补充的一些实验，它不应该是核心实验" |
| 把 "mechanistic discovery / foundation model outruns detector" 当论文卖点 | 已被导师否定，见 §5.1 |
| 新实验先在合成数据上验证 | 导师："合成数据有点太理想了，只适合做可行性分析，实际效果还得看真实数据上" —— **新的 detector 设计直接在真实数据上验证** |
| 加第三个真实数据集 | 导师认可现有两个（一渐进一突发）已经够，且"这几个数据应该已经算比较干净了" |
| 做 6-class native Insects 的 β binarization ablation | 成本高（系统重构 + 重跑全部基线 ≥ 1 周），不在主线上 |
| 删除既有的负面结果文档 | 学术诚信 + 答辩要答得出"试过什么" |
| `git add -A` / `git add .` | 会把导师私人录音转写和演示脚本提交到**公开** GitHub 仓库。已加 gitignore 兜底，但仍请显式 `git add <具体文件>` |
| 一上来就启动 60 runs 的大规模 sweep | 先做便宜的信号质量诊断验证方向，再上规模（见 `todo.md` P1-A0） |

## 10. 立即行动

### 必读（按顺序，不要跳）

| # | 文件 | 为什么必读 |
|---|---|---|
| 1 | 本文件 | 方向、禁区、架构、履历 |
| 2 | [`todo.md`](todo.md) | 待办 + Phase 5.5 实验技术规格 |
| 3 | `6_1和导师汇报的录音_包含两份转写.txt` **前半部分**（GPT-6 Astra 版，重点 00:02–13:00 技术讨论段） | 导师给的方向原文，是当前主线的依据。后半是讯飞听见转写，专业名词不准，仅作交叉参考 |
| 4 | [`progress_report.md`](progress_report.md)（1083 行） | **项目主记录**。§4 履历表只是它的摘要；完整数字、实验图、每轮迭代的推理过程都在这里。写论文时这是主要素材来源 |

### 按需读（§7 文件地图有全部清单）

`CLAUDE.md`（代码结构与命令，Claude Code 会自动加载）、
`results/phase4_final_verdict.md`（Phase 4 五段对照）、
`results/phase5_real_summary.md`（Phase 5 合并结论）、
`results/phase5_confound2_diagnostic.md`（γ 诊断）。

> ⚠️ `results/` 下 phase5 系列文档顶部都有 framing 说明，标明哪些结论的**定位**已被导师否定（数据本身有效）。读的时候留意。

### 读完后先回报，不要直接写代码

向用户说明三件事，等确认后再动手：

1. **你对项目的理解**：解决什么问题、用什么方法、走到哪、为什么方向在 2026-06 发生转变（3-5 句）
2. **哪些结论已被真正验证、哪些没有** —— 这是本项目最关键的一点，答不出说明没读透
3. **你建议从 [`todo.md`](todo.md) 的哪一项开始，理由是什么**

用户明确希望接手的 AI 自己判断优先级，而不是被预设顺序绑住。

> 有一件事值得先想清楚再问用户：导师给的三个方向里，路径 A（对比信号）技术上最直接、最贴合导师的核心诊断，但 TabPFN 调用翻倍会让实验成本翻倍；维度 C（遗忘 trade-off）是导师点名"最可能产出正面结果"的点，且不依赖 detector 是否触发。这两者哪个先做，值得和用户对齐。
