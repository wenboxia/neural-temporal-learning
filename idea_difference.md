# 原始 Idea（PDF）vs 当前实现计划（V2）对比

---

## 一、完全一致的部分

### 1. 核心问题定义
两者均以**特征语义漂移（Feature Semantics Drift）**和**关系漂移（Relationship Drift）**为核心问题，批判传统时序模型无法显式建模语义变化（如通货膨胀后"Revenue=1M"含义不同）。

### 2. PFC 神经科学隐喻
Brain ↔ ML 的映射完全保留：

| 神经概念（PFC） | ML 类比 |
|---------------|---------|
| 短期工作记忆 | 快速适应模块（Level 3）|
| 长期突触可塑性 | 慢速基础模型（Level 1）|
| 跨时间尺度交互 | 层间巩固机制 |
| 突触权重变化 | 参数适配 |

### 3. 三层架构的基本骨架
PDF 和 V2 均采用**三层预测叠加**的结构：

```
Level 1 (Slow)  ←→  TabPFN 冻结，长期先验
Level 2 (Inter) ←→  中间适配层，捕捉体制结构
Level 3 (Fast)  ←→  工作记忆，即时误差补偿
```

### 4. Level 1：TabPFN 永久冻结
PDF 明确："TabPFN = slow Bayesian prior"，"TabPFN and regime module are **fixed**"。V2 同样绝不微调 TabPFN，实现完全一致。

### 5. Level 3：FIFO 工作记忆缓冲区
PDF 图中 Level 3 = "Working Memory Buffer (FIFO, Recent Errors)"，V2 中 `buffer.py` 完全按此实现，capacity、push/clear 操作均对应。

### 6. 快→中巩固的触发逻辑
PDF 描述："If the fast system is consistently compensating in the **same direction** in a region of feature space → suggests true structural relationship drift → distil into regime module."
V2 中 `should_consolidate()` 判断条件：`abs(mean_err) > threshold AND std < abs(mean_err)`，直接数值化了上述文字逻辑。

### 7. 巩固后重置快速缓冲区
PDF："Slightly **reset or shrink** fast corrections in that area (like sleep-dependent consolidation)."
V2 巩固后调用 `buffer.clear()`，完全对应。

### 8. Prequential 评估协议
两者均隐含"先预测后更新"的在线评估方式，不做离线批量训练。

---

## 二、有差异的部分

### 差异 1：Level 3 快速校正的具体实现

| 维度 | PDF 原始 Idea | V2 实现计划 |
|------|-------------|------------|
| 机制 | **Attention/Lookup**（注意力检索）| **KNN + EMA**（无参数） |
| 公式 | `ŷ_fast = softmax(q·K^T/√d) · V` | KNN: 欧氏距离最近邻均值；EMA: 指数衰减加权均值 |
| 参数 | 有（注意力 Q/K/V 投影）| 无（零参数）|
| 理由差异 | PDF 认为注意力检索更精准 | V2 认为 KNN/EMA 更 CPU 友好、无过拟合风险 |

**评价**：V2 是合理的工程简化。KNN 在语义上等价于"找最相似历史样本的误差"，与注意力的软检索方向一致，但实现复杂度大幅降低。

---

### 差异 2：Level 2 的体制模块设计

| 维度 | PDF 原始 Idea | V2 实现计划 |
|------|-------------|------------|
| 体制检测 | **硬性**体制检测（Regime Detection 选出 c_t） | **无**硬性检测，改为软门控 |
| 体制表示 | 可学习体制嵌入字典 `{c^(k)}`，新体制时初始化新嵌入 | 无体制嵌入，门控网络输出即为软体制权重 |
| 适配器 | Hypernetwork / Head（以体制嵌入为条件） | 简单两层 MLP（直接以输入特征为条件） |
| 预测符号 | ŷ_inter（独立的中间层预测） | 门控网络同时输出 α/β/γ 和 y_inter |

**评价**：这是 V2 最大的架构简化。PDF 的 K-means + 嵌入字典方案在高维 TabPFN 嵌入上噪声大、难调试；V2 用软门控代替，更稳定但失去了"显式体制标识"的可解释性。

---

### 差异 3：预测融合方式

| 维度 | PDF 原始 Idea | V2 实现计划 |
|------|-------------|------------|
| 融合公式 | 直接相加（图中 Σ 符号，三路相加） | 软门控加权：α·y_slow + β·y_inter + γ·y_fast，α+β+γ=1 |
| 数值约束 | 无（可能数值饱和） | Softmax 归一化，输出永远在概率域 |
| 可学习性 | 无（固定相加） | 权重 α/β/γ 由门控网络动态学习 |

**评价**：V2 的加权融合比 PDF 的裸加更安全，同时门控权重提供了额外的可解释性（可视化哪层在何时被信任）。

---

### 差异 4：Timescale 3（慢→更慢，TabPFN 微调）

| 维度 | PDF 原始 Idea | V2 实现计划 |
|------|-------------|------------|
| 是否实现 | 是（Timescale 3：fine-tune TabPFN 或 LoRA 适配器） | **完全删除** |
| PDF 描述 | "Use large amounts of accumulated data to fine-tune TabPFN so that the prior itself better reflects recurring non-stationarities" | N/A |
| 删除理由 | N/A | OOM 风险 + 灾难性遗忘 + 无 GPU 硬件约束 |

**评价**：这是最显著的功能缺失。PDF 认为 TabPFN 本身也应随时间演化（类比皮层慢速突触重构），V2 为了工程可行性放弃了这一层。论文中需要明确说明并用消融实验验证其影响。

---

### 差异 5：FMT（Feature Modulation Transformer）

| 维度 | PDF 原始 Idea | V2 实现计划 |
|------|-------------|------------|
| 是否涉及 | 是（PDF 第一页：temporal embeddings ψ(t) + feature-wise transformations to normalize semantics） | **未实现** |
| 功能 | 生成特征级变换，将不同时期的特征语义对齐 | N/A |
| 删除理由 | N/A | 工程复杂度高；TabPFN 的 in-context learning 本身对特征分布有一定鲁棒性 |

**评价**：PDF 提出 FMT 是为了显式处理特征语义漂移（同一特征在不同时期含义不同），这是论文标题中"Feature semantics drift"的核心回应。V2 完全未实现此部分，是一个概念层面的缺口，后续论文写作时需要说明如何处理这一问题（或将其列为 Limitation）。

---

### 差异 6：MAML 风格元学习

| 维度 | PDF 原始 Idea | V2 实现计划 |
|------|-------------|------------|
| 是否涉及 | 是（Arch C：Meta-initialized Fast Learner，MAML-ish） | 仅作为对比基线提及，不作为主方案 |
| PDF 描述 | "Train a base model to be easy to fine-tune on new windows"，每次新体制从 θ₀ 出发做几步梯度更新 | N/A |

**评价**：PDF 将 MAML 列为可选架构之一（Arch C），V2 正确地将其降级为基线对比，因为 MAML 需要元训练阶段，工程成本高且不符合严格的 prequential 协议。

---

## 三、总结对照表

| 组件 | PDF 原始 | V2 实现 | 状态 |
|------|---------|---------|------|
| 问题定义（漂移类型） | 特征语义 + 关系漂移 | 同上 | ✅ 完全一致 |
| PFC 三时间尺度隐喻 | 快/中/慢 | 快/中（慢已删）| ⚠️ 部分保留 |
| Level 1: TabPFN 冻结 | 冻结先验 | 永久冻结 | ✅ 完全一致 |
| Level 3: 快速校正机制 | Attention/Lookup | KNN / EMA | ⚠️ 简化替换 |
| Level 3: FIFO 缓冲区 | Working Memory Buffer | `buffer.py` | ✅ 完全一致 |
| Level 2: 体制检测 | 硬性 K-means + 嵌入字典 | 软门控网络 | 🔄 方向一致，实现不同 |
| 预测融合 | 直接求和 Σ | 软加权 α/β/γ | 🔄 改进替换 |
| 快→中巩固 | 蒸馏，重置缓冲区 | 同，`fast_to_inter.py` | ✅ 完全一致 |
| 慢层微调（Timescale 3） | fine-tune TabPFN / LoRA | **删除** | ❌ 未实现 |
| FMT 特征语义对齐 | temporal embeddings ψ(t) | **未实现** | ❌ 未实现 |
| MAML 元学习 | Arch C 候选方案 | 仅作基线对比 | ⚠️ 降级处理 |
