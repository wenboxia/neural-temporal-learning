# 多时间尺度时序学习系统 — 实现计划 V2（稳妥路线）

## Context

本计划基于 V1 的反思，砍掉了三个最大风险点：
1. **砍掉 TabPFN 全量微调**（中→慢巩固）—— OOM 风险 + 灾难性遗忘风险太高
2. **砍掉高维 K-means 体制检测** —— 改用软门控机制，避免噪声导致体制频繁误切换
3. **砍掉裸 logit 相加** —— 改用学习权重的门控加权融合，避免数值饱和

核心策略：**先跑通 MVP 拿到基线成绩，再逐步叠加复杂度。**

---

## 项目目录结构

```
neural_1/
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── default.yaml
│   └── experiment/
│       ├── synthetic_rotate.yaml
│       ├── synthetic_regime.yaml
│       └── electricity.yaml
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── slow_prior.py             # Level 1: TabPFN 冻结包装器（绝不微调）
│   │   ├── fast_corrector.py         # Level 3: KNN/EMA 快速校正器
│   │   ├── gated_ensemble.py         # Level 2: 软门控融合（替代硬体制检测）
│   │   ├── multi_timescale.py        # 总调度器
│   │   └── baselines.py             # vanilla TabPFN, XGBoost, CatBoost
│   ├── memory/
│   │   ├── __init__.py
│   │   └── buffer.py                 # FIFO 工作记忆缓冲区
│   ├── consolidation/
│   │   ├── __init__.py
│   │   └── fast_to_inter.py          # 快→中蒸馏（唯一的巩固机制）
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py              # 合成漂移数据生成器
│   │   ├── real_world.py             # 真实数据集加载器
│   │   └── temporal_loader.py        # 时序窗口化 DataLoader
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # 主训练循环（prequential）
│   │   └── evaluator.py              # 评估指标
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py
│       └── metrics.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_baselines.py
│   └── visualize.py
├── tests/
│   ├── test_slow_prior.py
│   ├── test_fast_corrector.py
│   ├── test_gated_ensemble.py
│   ├── test_consolidation.py
│   └── test_integration.py
├── notebooks/
│   ├── 01_tabpfn_exploration.ipynb
│   ├── 02_synthetic_data_viz.ipynb
│   ├── 03_error_analysis.ipynb
│   └── 04_results_analysis.ipynb
└── results/                          # gitignored
```

与 V1 对比变化：
- 删除 `src/regime/` 整个目录（不再做硬体制检测）
- 删除 `inter_to_slow.py`（不再微调 TabPFN）
- 新增 `gated_ensemble.py`（软门控融合，替代硬体制 + logit 裸加）
- 整体文件数减少约 30%，降低工程复杂度

---

## 四阶段实现路线

### Phase 1: 绝对冻结与观察（第 1-2 周）—— 保底基线

**目标：** 证明"TabPFN 确实存在应对概念漂移不足的问题"（论文 Motivation）

**实现内容：**

1. **项目脚手架**：`pyproject.toml`, `.gitignore`, 目录结构, 配置系统
2. **合成数据生成器** (`src/data/synthetic.py`)：
   - `rotating_boundary`：2D 高斯特征，决策边界随时间旋转（关系漂移）
   - `regime_switching`：在 2-3 个分布间突变切换（突变漂移）
   - 每个生成器输出 `(X_t, y_t, regime_label_t, t)`，规模 5000~10000 样本
3. **时序窗口加载器** (`src/data/temporal_loader.py`)：
   - 滑动窗口模式，输出 `(X_ctx, y_ctx, X_query, y_query)`
   - 参数：`context_size=500`, `step_size=1`
4. **TabPFN 冻结包装器** (`src/models/slow_prior.py`)：

```python
class SlowPrior:
    """绝对冻结的 TabPFN，仅用作预测器 + 特征提取器"""
    def __init__(self, device="cpu"):
        self.model = TabPFNClassifier(device=device)

    def predict(self, X_ctx, y_ctx, X_query):
        self.model.fit(X_ctx, y_ctx)
        proba = self.model.predict_proba(X_query)
        return proba  # 概率空间，不转 logit

    # 没有 fine_tune 方法，永远冻结
```

5. **基线评估**：在合成数据上运行 vanilla TabPFN（滑动窗口），记录逐步预测误差

**里程碑：** 画出"误差 vs 时间"曲线图，在漂移点处误差骤增，证明 TabPFN 无法自适应漂移。

**关键文件：**
- `src/data/synthetic.py`
- `src/data/temporal_loader.py`
- `src/models/slow_prior.py`
- `notebooks/01_tabpfn_exploration.ipynb`

---

### Phase 2: 轻量级快速工作记忆（第 3-4 周）—— 核心创新点 1

**目标：** 用最简单的方式实现即时纠错，指标超越 vanilla TabPFN

**实现内容：**

1. **FIFO 缓冲区** (`src/memory/buffer.py`)：

```python
class WorkingMemoryBuffer:
    """固定容量的 FIFO 队列，存储最近的 (特征, 预测误差) 对"""
    def __init__(self, capacity=100):
        self.capacity = capacity
        self.features = []   # 存储 x_t
        self.errors = []     # 存储 error_t = y_t - ŷ_slow_t

    def push(self, feature, error):
        if len(self.features) >= self.capacity:
            self.features.pop(0)
            self.errors.pop(0)
        self.features.append(feature)
        self.errors.append(error)

    def knn_correction(self, query, k=5):
        """找到最相似的 K 个历史样本，返回其误差均值作为补偿"""
        if len(self.features) == 0:
            return 0.0
        distances = [np.linalg.norm(query - f) for f in self.features]
        topk_idx = np.argsort(distances)[:k]
        correction = np.mean([self.errors[i] for i in topk_idx])
        return correction

    def ema_correction(self, alpha=0.1):
        """指数移动平均：对最近误差做加权平均"""
        if len(self.errors) == 0:
            return 0.0
        weights = [alpha * (1 - alpha) ** i for i in range(len(self.errors) - 1, -1, -1)]
        return np.average(self.errors, weights=weights)

    def clear(self):
        self.features.clear()
        self.errors.clear()
```

2. **快速校正器** (`src/models/fast_corrector.py`)：

```python
class FastCorrector:
    """基于 KNN 或 EMA 的轻量级即时纠错"""
    def __init__(self, buffer_size=100, method="knn", k=5, alpha=0.1):
        self.buffer = WorkingMemoryBuffer(buffer_size)
        self.method = method
        self.k = k
        self.alpha = alpha

    def correct(self, x_query):
        if self.method == "knn":
            return self.buffer.knn_correction(x_query, self.k)
        else:
            return self.buffer.ema_correction(self.alpha)

    def update(self, x_t, error_t):
        self.buffer.push(x_t, error_t)

    def reset(self):
        self.buffer.clear()
```

3. **两层系统集成**（Level 1 + Level 3）：

```python
# 伪代码：prequential 评估循环
for t in range(len(data)):
    X_ctx, y_ctx = get_context_window(t)

    # Level 1: TabPFN 预测
    y_slow = slow_prior.predict(X_ctx, y_ctx, X_query)

    # Level 3: 快速校正
    correction = fast_corrector.correct(X_query)
    y_pred = y_slow + correction  # 概率空间的简单补偿

    # 观测真实值后更新缓冲区
    error = y_true - y_slow
    fast_corrector.update(X_query, error)
```

**注意：** Phase 2 阶段的融合是**概率空间的简单加法**（而非 logit），因为只有两层且 correction 量级很小（接近 0 附近的残差），数值饱和风险可控。到 Phase 3 引入门控后替换。

**里程碑：** 在合成数据上，Level 1+3 的 prequential accuracy 显著优于 Level 1 单独。**到这一步，中期答辩内容已足够。**

**关键文件：**
- `src/memory/buffer.py`
- `src/models/fast_corrector.py`
- `src/training/trainer.py`（prequential 循环）

---

### Phase 3: 软门控融合 + 快→中巩固（第 5-7 周）—— 核心创新点 2

**目标：** 引入可学习的中间层，实现"经验沉淀"（快→中巩固）

**实现内容：**

1. **门控融合网络** (`src/models/gated_ensemble.py`)：

```python
class GatedEnsemble(nn.Module):
    """
    软门控融合，替代硬体制检测 + logit 裸加。
    根据输入动态分配三层的权重，避免数值饱和。
    """
    def __init__(self, input_dim, hidden_dim=64, n_outputs=1):
        super().__init__()
        # 门控网络：输入特征 -> 3 个权重（softmax 归一化）
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3 = slow, inter, fast
        )
        # 中间层适配器：学习对 TabPFN 预测的残差校正
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, x, y_slow, y_fast):
        # 门控权重
        weights = F.softmax(self.gate(x), dim=-1)  # [batch, 3]
        alpha, beta, gamma = weights[:, 0], weights[:, 1], weights[:, 2]

        # 中间层预测
        y_inter = self.adapter(x)

        # 加权融合（概率空间，经 softmax 归一化的权重保证不会数值爆炸）
        y_final = alpha * y_slow + beta * y_inter + gamma * y_fast
        return y_final, weights
```

**设计要点：**
- **不做硬性体制划分**：门控网络根据输入特征自动决定三层的贡献权重
- **权重经 softmax 归一化**：α + β + γ = 1，保证输出永远在合理范围内，彻底避免数值饱和
- **中间层 adapter 很轻量**：两层 MLP，参数量极小，CPU 上训练毫无压力
- 当数据稳定时，gate 会倾向于高 α（信任 TabPFN）；当快速漂移时，gate 会提高 γ（信任快速校正）

2. **快→中巩固** (`src/consolidation/fast_to_inter.py`)：

```python
class FastToInterConsolidation:
    """
    当快速校正器在一段时间内持续产生相似方向的校正时，
    将这些模式蒸馏到中间层 adapter 中，然后清空缓冲区。
    """
    def __init__(self, threshold=0.05, window=50):
        self.threshold = threshold  # 触发蒸馏的最小平均校正幅度
        self.window = window        # 观察窗口大小

    def should_consolidate(self, buffer):
        """当缓冲区中最近 window 个校正值方向一致且幅度足够时触发"""
        if len(buffer.errors) < self.window:
            return False
        recent = buffer.errors[-self.window:]
        mean_correction = np.mean(recent)
        std_correction = np.std(recent)
        # 均值明显偏离 0 且方差较小 → 系统性偏移 → 该巩固了
        return abs(mean_correction) > self.threshold and std_correction < abs(mean_correction)

    def consolidate(self, gated_ensemble, buffer, X_recent, y_slow_recent, optimizer, epochs=10):
        """让中间层 adapter 学习去拟合快速校正的模式"""
        corrections = torch.tensor(buffer.errors[-self.window:], dtype=torch.float32)
        X_tensor = torch.tensor(X_recent, dtype=torch.float32)
        y_slow_tensor = torch.tensor(y_slow_recent, dtype=torch.float32)

        for _ in range(epochs):
            y_inter = gated_ensemble.adapter(X_tensor)
            # 目标：让 adapter 输出接近快速校正的值
            loss = F.mse_loss(y_inter.squeeze(), corrections)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 巩固完成后，清空部分缓冲区
        buffer.clear()
```

3. **三层完整系统** (`src/models/multi_timescale.py`)：

```python
class MultiTimescaleModel:
    def __init__(self, input_dim, config):
        self.slow_prior = SlowPrior(device="cpu")
        self.fast_corrector = FastCorrector(
            buffer_size=config.buffer_size,
            method=config.fast_method,
        )
        self.gated_ensemble = GatedEnsemble(input_dim)
        self.consolidator = FastToInterConsolidation(
            threshold=config.consolidation_threshold,
            window=config.consolidation_window,
        )
        self.optimizer = torch.optim.Adam(
            self.gated_ensemble.parameters(), lr=1e-3
        )

    def step(self, X_ctx, y_ctx, x_t, y_t):
        """一步：预测 → 评估 → 更新"""
        # Level 1: TabPFN 冻结预测
        y_slow = self.slow_prior.predict(X_ctx, y_ctx, x_t)

        # Level 3: 快速校正
        y_fast = y_slow + self.fast_corrector.correct(x_t)

        # Level 2 + 门控融合
        x_tensor = torch.tensor(x_t, dtype=torch.float32).unsqueeze(0)
        y_slow_t = torch.tensor([y_slow], dtype=torch.float32)
        y_fast_t = torch.tensor([y_fast], dtype=torch.float32)
        y_final, weights = self.gated_ensemble(x_tensor, y_slow_t, y_fast_t)

        # 观测真实值后更新
        error = y_t - y_slow
        self.fast_corrector.update(x_t, error)

        # 检查是否需要巩固
        if self.consolidator.should_consolidate(self.fast_corrector.buffer):
            self.consolidator.consolidate(...)

        return y_final.item(), weights
```

**里程碑：** 证明三层系统优于两层（Level 1+3），且巩固机制使系统在长期漂移中表现更稳定。

**关键文件：**
- `src/models/gated_ensemble.py`
- `src/consolidation/fast_to_inter.py`
- `src/models/multi_timescale.py`

---

### Phase 4: 实验与论文（第 8-10 周）

**目标：** 完善实验对比，撰写论文

**实验矩阵：**

| 模型变体 | 描述 | 对应消融 |
|---------|------|---------|
| Baseline: TabPFN | 冻结 TabPFN + 滑动窗口 | Level 1 only |
| Baseline: XGBoost | 逐窗口重训的 XGBoost | 传统 ML 基线 |
| Baseline: CatBoost | 逐窗口重训的 CatBoost | 传统 ML 基线 |
| Ours-Fast | TabPFN + KNN 快速校正 | Level 1 + Level 3 |
| Ours-EMA | TabPFN + EMA 快速校正 | Level 1 + Level 3 (变体) |
| Ours-Gated | TabPFN + 门控融合（无巩固） | Level 1 + 2 + 3 |
| **Ours-Full** | **TabPFN + 门控融合 + 快→中巩固** | **完整系统** |

**数据集：**

| 数据集 | 类型 | 规模 | 环境 |
|--------|------|------|------|
| `rotating_boundary` | 合成（渐进关系漂移） | 10,000 | CPU |
| `regime_switching` | 合成（突变漂移） | 10,000 | CPU |
| `combined_drift` | 合成（混合漂移） | 10,000 | CPU |
| Electricity (OpenML) | 真实 | 45,312 | Colab |

**评估指标：**
- Prequential Accuracy（逐步预测准确率）
- Window Accuracy（按窗口的准确率，观察漂移恢复）
- Adaptation Speed（漂移后恢复到基线准确率所需步数）
- 门控权重可视化（展示系统如何动态分配信任）

**关键产出图表：**
1. 误差 vs 时间曲线（所有模型对比，漂移点标注）
2. 门控权重 vs 时间（展示 α, β, γ 如何随漂移变化）
3. 巩固前后对比（展示快→中蒸馏的效果）
4. 消融实验汇总表
5. 缓冲区大小敏感性分析

**关键文件：**
- `scripts/run_baselines.py`
- `scripts/evaluate.py`
- `scripts/visualize.py`
- `notebooks/04_results_analysis.ipynb`

---

## 时间线总览

| 周次 | 阶段 | 核心产出 | 保底目标 |
|------|------|---------|---------|
| 1-2 | Phase 1: 冻结与观察 | 数据管线 + TabPFN 基线 + 漂移可视化 | 论文 Motivation 图 |
| 3-4 | Phase 2: 快速校正 | KNN/EMA 校正器 + Level 1+3 集成 | **中期答辩可用** |
| 5-7 | Phase 3: 门控融合 + 巩固 | 完整三层系统 | 核心实验结果 |
| 8-10 | Phase 4: 实验与写作 | 消融实验 + 真实数据 + 论文 | **终稿提交** |

---

## 关键技术决策（V2 修订）

| 决策点 | V1 方案（已废弃） | V2 方案（采用） | 理由 |
|--------|-----------------|----------------|------|
| TabPFN 微调 | 全量微调 / LoRA | **绝不微调，永远冻结** | OOM + 灾难性遗忘风险 |
| 体制检测 | 高维 K-means 聚类 | **软门控网络（MoE Router 风格）** | K-means 在高维嵌入上噪声大 |
| 预测融合 | logit 空间裸加 | **softmax 归一化加权（α+β+γ=1）** | 避免数值饱和和梯度消失 |
| 快速校正 | Attention + 可学习参数 | **KNN / EMA（无参数）** | 极简、鲁棒、CPU 友好 |
| 巩固机制 | 快→中 + 中→慢 两级 | **仅快→中蒸馏** | 中→慢需要 GPU，风险太大 |

---

## 硬件约束适配

- **Phase 1-3 全部可在 MacBook CPU 上完成**（合成数据规模小，TabPFN CPU 推理足够）
- **Phase 4 真实数据集**：Electricity 数据集建议用 Google Colab（免费版即可，TabPFN 推理不需要 GPU，但 Colab 的 RAM 更大）
- **门控网络训练**：参数量 < 10K，CPU 上训练一个 epoch 耗时 < 1 秒
- **无任何模块需要 GPU 训练**（TabPFN 冻结 + 门控网络极轻量 + KNN 无参数）

---

## 依赖

```
torch>=2.1
tabpfn>=2.0
scikit-learn>=1.3
numpy>=1.24
pandas>=2.0
openml
matplotlib
seaborn
pyyaml
pytest
tqdm
```

与 V1 对比删除：`tabpfn-extensions`（不再需要嵌入提取）、`wandb`（轻量项目用 matplotlib 即可）、`hydra-core` + `omegaconf`（改用简单 YAML + dataclass）

---

## 验证方案

1. **Phase 1 验证**：TabPFN 在漂移点误差骤增的可视化图（论文 Figure 1）
2. **Phase 2 验证**：Level 1+3 的 prequential accuracy > Level 1 单独（至少提升 3-5%）
3. **Phase 3 验证**：
   - 完整系统 > Level 1+3（消融实验）
   - 门控权重可视化：漂移时 γ 升高，稳定时 α 升高
   - 巩固前后对比：巩固后 fast buffer 清空，中间层能独立预测漂移方向
4. **Phase 4 验证**：在 Electricity 真实数据集上优于 vanilla TabPFN 和逐窗口 XGBoost

---

## 风险与缓解（V2）

| 风险 | 缓解措施 |
|------|---------|
| TabPFN CPU 推理太慢 | 缩小 context_size 到 300-500；减少合成数据规模 |
| KNN 校正效果不够好 | 尝试 EMA 替代；调整 buffer_size 和 K |
| 门控网络过拟合 | 加 dropout；用小学习率；early stopping |
| 巩固触发条件不准 | 用简单的均值/方差阈值，从宽松开始逐步收紧 |
| 时间不够 | Phase 2 完成即可中期答辩；Phase 3 完成即可写出完整论文 |

---

## 论文故事线

1. **Motivation**：TabPFN 虽强，但面对概念漂移束手无策（Phase 1 的图）
2. **Core Idea**：受 PFC 启发的多时间尺度适应 —— 快速工作记忆 + 软门控中间层 + 经验巩固
3. **Method**：三层架构详细描述 + 巩固机制的数学形式化
4. **Experiments**：消融实验证明每一层都有贡献；真实数据集验证泛化能力
5. **Analysis**：门控权重可视化 —— 系统如何"自动"识别漂移并调整策略
