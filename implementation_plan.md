# 多时间尺度时序学习系统 — 实现计划

## Context

导师提出了一个受大脑前额叶皮层（PFC）启发的时序表格数据学习框架。核心问题是：传统时间序列模型无法显式处理**特征语义漂移**（同一特征值在不同时间含义不同）和**关系漂移**（特征间关系随时间变化）。本项目在 TabPFN（预训练表格数据基础模型）之上，构建一个三层多时间尺度适应系统，通过层间巩固机制实现持续有效的学习和预测。

项目从零开始，当前目录仅有一份 PDF 文档。

---

## 项目目录结构

```
neural_1/
├── pyproject.toml
├── README.md
├── Makefile
├── .gitignore
├── configs/
│   ├── default.yaml
│   ├── experiment/
│   │   ├── synthetic_abrupt.yaml
│   │   ├── synthetic_gradual.yaml
│   │   ├── electricity.yaml
│   │   └── weather.yaml
│   └── sweep/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── slow_prior.py             # Level 1: TabPFN 包装器（冻结/LoRA）
│   │   ├── regime_module.py          # Level 2: 体制检测 + 适配器
│   │   ├── fast_corrector.py         # Level 3: 工作记忆 + 校正
│   │   ├── multi_timescale.py        # 总调度器，组合三层
│   │   └── baselines.py             # XGBoost, CatBoost, vanilla TabPFN 等基线
│   ├── regime/
│   │   ├── __init__.py
│   │   ├── detector.py               # 体制检测（聚类/变点检测）
│   │   ├── embeddings.py             # 体制嵌入库
│   │   └── adapter.py                # 残差 MLP 适配器
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── buffer.py                 # FIFO 工作记忆缓冲区
│   │   └── attention_lookup.py       # 基于注意力的误差检索
│   ├── consolidation/
│   │   ├── __init__.py
│   │   ├── fast_to_inter.py          # 快→中蒸馏
│   │   └── inter_to_slow.py          # 中→慢微调
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py              # 合成漂移数据生成器
│   │   ├── real_world.py             # 真实数据集加载器
│   │   ├── temporal_loader.py        # 时序窗口化 DataLoader
│   │   └── drift_types.py            # 漂移注入工具函数
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # 主训练循环（prequential）
│   │   ├── evaluator.py              # 评估指标
│   │   └── consolidation_scheduler.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py          # W&B 集成
│       └── metrics.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_baselines.py
│   └── visualize.py
├── tests/
│   ├── test_slow_prior.py
│   ├── test_regime_module.py
│   ├── test_fast_corrector.py
│   ├── test_consolidation.py
│   ├── test_synthetic_data.py
│   └── test_integration.py
├── notebooks/
│   ├── 01_tabpfn_exploration.ipynb
│   ├── 02_synthetic_data_viz.ipynb
│   ├── 03_regime_detection_dev.ipynb
│   ├── 04_results_analysis.ipynb
│   └── 05_ablation_study.ipynb
└── results/                          # gitignored
```

---

## 模块实现计划

### Module A: 数据管线 (`src/data/`) — 1.5~2周

**`drift_types.py`** — 漂移注入原语：
- `abrupt_drift`: 在某时刻突然切换特征分布
- `gradual_drift`: 在时间区间内线性插值两种分布
- `recurring_drift`: 周期性在多种体制间切换
- `feature_drift`: 改变单个特征的语义（如均值漂移）
- `relationship_drift`: 改变 X→y 的映射关系

**`synthetic.py`** — 合成数据集：
| 数据集 | 漂移类型 | 描述 |
|--------|---------|------|
| `rotating_boundary` | 渐进关系漂移 | 2D高斯特征，线性决策边界随时间旋转 |
| `shifting_means` | 特征语义漂移 | 特征均值单向漂移 |
| `regime_switching` | 突变+循环 | 在2-3个分布间交替切换 |
| `combined_drift` | 两种兼有 | 特征分布 + 决策边界同时变化 |

每个生成器输出 `(X_t, y_t, regime_label_t, t)` 元组，5000-50000个样本。

**`real_world.py`** — 真实数据集：
- Electricity (OpenML #151): 45,312样本，二分类
- Gas Sensor Array Drift (UCI): 13,910样本，36个月显式漂移
- NOAA Weather: ~18,000样本，季节/趋势漂移

**`temporal_loader.py`** — 时序窗口化加载器：
- 输出 `(context_window, query_batch)` 对
- 支持滑动窗口和扩展窗口模式
- 参数：`window_size`, `step_size`, `context_size`

### Module B: Level 1 — 慢速先验 (`src/models/slow_prior.py`) — 1周

```python
class SlowPrior(nn.Module):
    # 包装 TabPFNClassifier，提取中间嵌入
    def forward(self, X_ctx, y_ctx, X_query) -> (predictions, embeddings)
    def fine_tune(self, accumulated_data, lr, epochs)  # 用于中→慢巩固
```

- 默认冻结模式：调用 TabPFN 的 `fit` + `predict_proba`
- 使用 `tabpfn-extensions` 的 `TabPFNEmbedding` 提取内部表征给 Level 2 使用
- 微调路径：参考 `finetune_tabpfn_v2` 方法，全量微调 + 小学习率 + 早停

**关键点：** TabPFN 通过 in-context learning 工作——整个训练集作为上下文一次前向传播。"慢"指的是模型权重冻结，只有上下文数据变化。

### Module C: Level 2 — 体制模块 (`src/regime/` + `src/models/regime_module.py`) — 2~3周

**`detector.py`** — 体制检测：
- 主方案：在 TabPFN 嵌入上做**滑动窗口 k-means 聚类**
- 输出当前体制索引 k_t
- 备选：BOCPD（贝叶斯在线变点检测）

**`embeddings.py`** — 体制嵌入库：
- 维护可学习嵌入字典 `{c^(k)}`，维度 d_regime=64~128
- 新体制出现时，用该窗口 TabPFN 嵌入均值初始化

**`adapter.py`** — 残差 MLP 适配器（非完整超网络）：
- 输入：`concat(tabpfn_embedding, regime_embedding)`
- 架构：`Linear(d_emb+d_regime, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, n_outputs)`
- 输出：对 TabPFN 预测的残差校正

### Module D: Level 3 — 快速校正器 (`src/memory/` + `src/models/fast_corrector.py`) — 1周

**`buffer.py`** — FIFO 工作记忆缓冲区：
- 大小 B=50~200，存储 `(x_t, error_t, embedding_t)`
- `error_t = y_t - (ŷ_slow + ŷ_inter)`
- 支持 push, query, reset, decay 操作

**`attention_lookup.py`** — 注意力检索：
- `correction = softmax(q @ K^T / sqrt(d)) @ V`
- 单层注意力，无可训练参数（key=存储嵌入, value=存储误差）

### Module E: 总调度器 (`src/models/multi_timescale.py`) — 1周

```python
class MultiTimescaleModel(nn.Module):
    def predict(self, X_ctx, y_ctx, X_query):
        y_slow, embeddings = self.slow_prior(X_ctx, y_ctx, X_query)
        y_inter, regime_id = self.regime_module(embeddings, X_query)
        y_fast = self.fast_corrector(embeddings)
        return y_slow + y_inter + y_fast  # logit空间相加

    def step(self, X_ctx, y_ctx, X_t, y_t):
        # predict -> evaluate -> update fast corrector
```

**关键设计：** 对于分类任务，三层在 **logit 空间**相加：TabPFN 输出概率 → 转logit → 加上中间层和快速层的logit校正 → sigmoid得到最终概率。

### Module F: 巩固机制 (`src/consolidation/`) — 1.5~2周

**`fast_to_inter.py`** — 快→中蒸馏：
- 触发条件：缓冲区校正在某特征空间区域持续同方向偏移
- 损失：`L = E[|adapter_new(emb, c_k) - (adapter_old(emb, c_k) + stored_error)|²]`
- 蒸馏后重置快速校正器缓冲区

**`inter_to_slow.py`** — 中→慢微调：
- 触发条件：积累大量数据后（如跨越10次体制转换）
- 全量微调 TabPFN，学习率 1e-5~1e-6，激进早停

### Module G: 训练与评估 (`src/training/`) — 1.5周

- **Prequential 评估**：先预测后更新
- 指标：prequential accuracy/RMSE、窗口指标、体制转换处准确率、适应速度、ECE
- 实验追踪：Weights & Biases

---

## 实验计划

### Stage 1: 基线建立
运行 vanilla TabPFN、Drift-Resilient TabPFN、XGBoost（逐窗口重训）、CatBoost

### Stage 2: 逐组件开发与消融
- Level 1 单独
- Level 1 + Level 3（慢先验 + 快速校正）
- Level 1 + Level 2（慢先验 + 体制模块）
- 三层全系统（无巩固）

### Stage 3: 完整系统
- 加入快→中巩固
- 加入中→慢巩固

### Stage 4: 消融实验
- 缓冲区大小敏感性
- 体制数量敏感性
- 巩固频率敏感性
- 上下文窗口大小敏感性

---

## 实现顺序（约12周）

| 阶段 | 周次 | 任务 |
|------|------|------|
| **基础搭建** | 1-3 | 项目脚手架、合成数据、时序加载器、TabPFN包装器、探索性notebook |
| **快速校正器** | 3-4 | FIFO缓冲区、注意力检索、Level 1+3 集成、合成数据验证 |
| **体制模块** | 4-7 | 体制检测、嵌入库、MLP适配器、三层完整集成 |
| **巩固机制** | 7-9 | 快→中蒸馏、中→慢微调、调度器、端到端集成测试 |
| **评估与写作** | 9-12 | 真实数据集、全基线对比、消融实验、可视化、论文写作 |

---

## 关键技术决策

1. **TabPFN 嵌入提取**：使用 `tabpfn-extensions` 的 `TabPFNEmbedding`
2. **体制检测**：滑动窗口 k-means（简单可调试），备选 BOCPD
3. **适配器架构**：残差 MLP（非完整超网络，TabPFN参数量太大）
4. **微调策略**：全量微调（非LoRA，TabPFN的批量推理引擎与LoRA冲突）
5. **任务类型**：先做二分类，后扩展回归
6. **预测融合**：logit空间加法
7. **MAML备选**：作为基线对比实现在 `baselines.py`

---

## 硬件约束：CPU/MacBook 环境

由于没有本地 GPU，需要注意：
- **TabPFN 支持 CPU 推理**，但速度较慢。上下文窗口建议控制在 1000~3000 样本内
- **中→慢巩固（微调 TabPFN）** 是最耗 GPU 的部分。建议：
  - 开发阶段在 CPU 上用小数据集验证逻辑正确性
  - 正式实验用 **Google Colab Pro**（约 ¥70/月）或学校服务器跑微调
  - 如果实在无 GPU，可将中→慢巩固简化为训练 LoRA adapter 或直接跳过，专注快→中巩固
- **合成数据实验** 完全可以在 CPU 上完成（数据量小）
- **真实数据集大规模实验** 建议用 Colab 或找同学借服务器账号
- 建议安装 `torch` 的 CPU 版本以节省磁盘空间

## 依赖

```
torch>=2.1, tabpfn>=2.0, tabpfn-extensions, scikit-learn>=1.3,
numpy>=1.24, pandas>=2.0, openml, wandb, hydra-core>=1.3,
omegaconf, matplotlib, seaborn, pytest, tqdm
```

---

## 主要参考

- **Drift-Resilient TabPFN** (NeurIPS 2024) — 最接近的已有工作，主要对比基线
- **finetune_tabpfn_v2** (LennartPurucker) — 中→慢微调的参考实现
- **TabPFN** (PriorLabs) — 基础模型

---

## 验证方案

1. **单元测试**：每个模块独立测试（输出形状、基本逻辑）
2. **集成测试**：100样本小合成数据上端到端运行
3. **合成数据验证**：在已知漂移点的数据上，验证体制检测准确率 >80%
4. **消融验证**：每加一个组件，性能应优于或等于之前的子系统
5. **真实数据对比**：在 Electricity 等数据集上超越 vanilla TabPFN 和 Drift-Resilient TabPFN

---

## 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| TabPFN 嵌入提取太慢 | 按窗口缓存嵌入；减小上下文窗口 |
| 体制检测噪声大 | 设最小体制持续时间；固定k=3起步 |
| 加法校正结构受限 | 备选门控机制：`α·y_slow + β·y_inter + γ·y_fast` |
| 巩固导致性能退化 | 巩固前存检查点；保守学习率；验证后才接受更新 |
