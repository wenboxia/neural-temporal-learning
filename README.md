# Neural Temporal Learning — 基于 TabPFN 的多时间尺度概念漂移适应系统

在**冻结的 TabPFN**（pre-trained tabular foundation model）之上构建多时间尺度（slow / inter / fast）adapter library 系统，应对表格数据的概念漂移问题。受前额叶皮层（PFC）多时间尺度结构启发，但因 TabPFN 权重不可微调（CPU only + 防 catastrophic forgetting），不声称生物建模。

**项目状态**：Phase 1-5 完成 → **Phase 5.5（导师指定的 detector 重设计）进行中** → Phase 6 毕业论文
**目标**：完成 KTH 硕士毕业论文答辩
**最后更新**：2026-09-05

> 🔎 **接手这个项目？先读 [`start_prompt.md`](start_prompt.md)**，待办见 [`todo.md`](todo.md)。

---

## 当前阶段进展

| Phase | 状态 | 内容 | 关键产出 |
|---|---|---|---|
| Phase 1 | ✅ | TabPFN baseline 三数据集 multi-seed | 79.89 ± 0.99% on regime_switching |
| Phase 2 | ✅ | + Fast Corrector (KNN/EMA) 残差校正 | 三数据集全 NS（进 Appendix A）|
| Phase 2.5 | ✅ | 类别平衡修复 + Composite Window 协议 | 适应速度 126→62 步 |
| Phase 3 | ✅ | + GatedEnsemble (Level 2) + Fast→Inter 巩固 | rotating +1.00 sig / combined -0.47 sig 负 |
| Phase 4 Day 0.5 | ✅ | Oracle context-reset 诊断 | +0.51 pp sig（context 是 1/3 lever）|
| Phase 4 Day 1.5 | ✅ | 4-stage detector input ablation | raw/abs/indicator/warmstart |
| Phase 4 Day 2 | ✅ | 2×2 confound 解耦 (fit_threshold × init) | fit_threshold 主因 70% |
| **Phase 5** | ✅ | **真实数据验证** (Electricity + Insects abrupt) | **5 个 paper-grade verdicts** |
| **Phase 5.5** | 🚧 **进行中** | **导师指定的 detector 重设计**（对比信号 / fixed_ratio / 遗忘 trade-off） | 待产出 |
| Phase 6 | ⏸ 阻塞于 5.5 | 毕业论文撰写（KTH 硕士，目标 pass） | 论文主线取决于 5.5 结果 |

---

## Phase 5 核心 5 个 Verdicts

完整数据 + paired t-test 见 [`results/phase5_real_summary.md`](results/phase5_real_summary.md)。

| # | Verdict | 量化证据 |
|---|---|---|
| **V1** | F3 indicator detector 在真实 abrupt drift 上**完全失效** | 0/20 触发跨全 4 segments 全 5 seeds |
| **V2** | F4 reuse 路径失活在 Electricity 复现，在 Insects vacuous | Electricity 1/1 全 create，Insects 0 routes |
| **V3** | Phase 3 sig 负向真实数据**同向复现** | Electricity −0.124 sig / Insects −0.101 sig |
| **V4** | rotating_boundary +1pp 是**合成 artifact** | Electricity gradual −0.064 NS（机制不同 → 不复现）|
| **V5** | Phase 4a 真实数据 **net negative** | Insects −0.172 sig p<0.0001 |

---

## 关键机制定位（论文核心 contribution）

**γ Confound #2 Diagnostic** ([`results/phase5_confound2_diagnostic.md`](results/phase5_confound2_diagnostic.md)) 量化了为什么 detector 在真实 abrupt drift 上失效：

```
indicator 信号 |Δ| ≤ 0.019（真实 Insects abrupt drift 前后）
vs 合成 regime_switching 0.20
→ 10× 信号稀释
```

**机制**：TabPFN 的 sliding context window 在 ~10-20 步内通过 in-context learning 吸收掉 P(y) shift，预测准确率几乎不变 → indicator stream 没有明显切点 → ADWIN change-point detector 无法触发。

> ⚠️ **framing 已修正（2026-06-01 导师汇报后）**
>
> 上面这条机制定位**曾被当作论文核心 contribution**（"foundation models outrun drift detectors"），该定位**已被导师否定**：
>
> > "这个答辩的时候比较容易被质疑，我觉得这个不太好，还是不太稳。还是得有一些正面的，我们还是按照正面来推进。"
>
> 导师的诊断是：detector 不触发是**设计问题**而非根本性质 —— "算法本身应该是可行的，只不过你的检测器把它拦截下来了"。
> 由于 detector 从未触发，三层结构根本没被激活，**"方法无效" 这个结论其实从未被真正验证过**。
>
> **γ 诊断的实验数据与统计结论全部有效**，被取代的只是"这是论文核心发现"这个定位。
> 当前主线是 Phase 5.5：重新设计 detector 信号让它触发起来，再验证方法的真实效果。详见 [`todo.md`](todo.md) P1。

---

## 文件指引

### 入口
- [**`start_prompt.md`**](start_prompt.md) — **项目交接入口**：定位 / 目标 / 方向说明 / 不要做的事
- [**`todo.md`**](todo.md) — 待办清单 + Phase 5.5 实验技术规格

### 核心叙事
- [`progress_report.md`](progress_report.md) — **完整实验 narrative**（Phase 1 → Phase 5 Combined Verdict + Phase 6 stub），含所有实验结果图
- [`CLAUDE.md`](CLAUDE.md) — 项目说明 + 命令清单 + 模块表

### 计划与决策记录
- [`phase4_plan.md`](phase4_plan.md) — Phase 4 plan (Cheap Diagnostic + Design A spec)
- [`phase5_plan.md`](phase5_plan.md) — **Phase 5 plan + Phase 6 outline**（active 计划）
- [`implementation_plan_v2.md`](implementation_plan_v2.md) — V2 设计 spec（当前代码遵循）

### Phase 4 实验输出
- [`results/phase4_final_verdict.md`](results/phase4_final_verdict.md) — **Day 1.5 五段终极对照表 + 论文 10 章大纲**
- Day 0.5 oracle / multi-seed / Design A 决策细节：见 [progress_report.md](progress_report.md) §Phase 4 Day 0.5
- Day 1.5 / Day 2 各轮详细 summary：见 [progress_report.md](progress_report.md) §Phase 4 Day 1.5 + §Phase 4 Day 2

### Phase 5 实验输出
- [`results/phase5_real_summary.md`](results/phase5_real_summary.md) — **Stage A + B1+ combined verdict + F1-F4 矩阵**
- [`results/phase5_real_summary_electricity.md`](results/phase5_real_summary_electricity.md) — Stage A 完整数字
- [`results/phase5_real_summary_insects.md`](results/phase5_real_summary_insects.md) — Stage B1+ 完整数字
- [`results/phase5_confound2_diagnostic.md`](results/phase5_confound2_diagnostic.md) — **γ 诊断 + TabPFN absorption 机制定位**
- [`results/archive_misaligned_stage_b/`](results/archive_misaligned_stage_b/) — 旧 A+ misaligned Stage B 数据归档（保留 methodology narrative）

---

## 系统架构（当前 — V2 + Phase 3 v2+B+F + Phase 4 Design A）

```
Input stream (X_t, y_t) ──▶ TemporalWindowLoader (sliding context window)
                                        │
                           ┌────────────▼────────────┐
                           │  Level 1: SlowPrior      │  Frozen TabPFN (in-context learning)
                           └────────────┬────────────┘
                                        │ y_slow
                           ┌────────────▼────────────┐
                           │  Level 3: FastCorrector  │  FIFO buffer + KNN/EMA 残差
                           └────────────┬────────────┘
                                        │ correction
                           ┌────────────▼────────────┐
                           │  Level 2: GatedEnsemble  │  gate (MLP→softmax 3-way)
                           │  + AdapterLibrary (P4 A) │  + ADWIN drift detector
                           │                          │  + per-regime adapter 硬路由
                           └────────────┬────────────┘
                                        │ (α, β, γ) + active adapter id
                                        │
              y_final = clip(y_slow + β·y_inter + γ·correction, 0, 1)
```

**关键约束**：
- TabPFN 权重**绝不微调**（"slow" 指 context 不变 weights）
- Context size ≤ 3000（CPU friendly）
- Prequential evaluation（先预测 → 观测 → 更新）

---

## 快速开始

```bash
# 安装
pip install -e ".[dev]"
pip install openml river  # Phase 5 真实数据用

# 跑测试
pytest tests/

# Phase 1 基线
python scripts/run_baselines.py --dataset regime_switching --n_samples 3000 --context_size 200

# Phase 4 A（per-regime adapter library + ADWIN routing）
python scripts/run_phase4_a.py --dataset regime_switching --n_samples 3000

# Phase 5 真实数据（Electricity）—— 注意单次运行脚本用 --dataset（单数）
python scripts/run_phase4_a.py --dataset electricity --dataset_source real --segment_id start

# 完整 multi-seed 批量
python scripts/run_multiseed.py --dataset_source real --datasets insects \
  --insects_aligned --configs phase1,phase3,phase4a \
  --seeds 42,123,456,789,1024 \
  --segments early,mid,late_pre,late_post --n_parallel 2
```

---

## 数据集 references

- **Electricity** (OpenML 151)：Harries 1999 / Gama 2004 — concept drift 标准 benchmark
- **Insects (abrupt_balanced)**：Souza et al. (2020) "Challenges in benchmarking stream learning algorithms with real-world data" arXiv 2005.00113

合成数据集：rotating_boundary / regime_switching / combined_drift 见 [`src/data/synthetic.py`](src/data/synthetic.py)。

---

## License & Contact

- 学术项目，毕业论文相关
- Author: Wenbo Xia (wenboxia)
- Repo: https://github.com/wenboxia/neural-temporal-learning
