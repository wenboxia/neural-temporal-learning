"""
Phase 3C：三时间尺度编排器（MultiTimescaleModel）

此模块属于 Phase 3C，将 SlowPrior / FastCorrector / GatedEnsemble /
FastToInterConsolidation 四个子模块串联成单一 prequential 接口，
每步调用 step() 完成：慢层预测 → 快层校正 → 门控融合 → 在线训练 → buffer 更新 → 按需巩固。

偏离 plan V2 的两处设计决策（在 MultiTimescaleModel 类 docstring 中说明）：
  [决策 1] 每步以 MSE 训练 gate + adapter
  [决策 2] y_final clamp 到 [0, 1] 后再作为预测输出

使用流程（prequential online 场景）：

    import numpy as np
    from src.models.multi_timescale import MultiTimescaleModel

    model = MultiTimescaleModel(input_dim=10)

    for batch in loader:
        x_t   = batch.X_query[0]           # (n_features,)  1D
        y_t   = float(batch.y_query[0])    # 标量 0/1
        X_ctx = batch.X_ctx                 # (context_size, n_features)
        y_ctx = batch.y_ctx                 # (context_size,)

        y_pred, weights = model.step(X_ctx, y_ctx, x_t, y_t, t=batch.t)
        # y_pred:  float ∈ [0, 1]，正类概率
        # weights: np.ndarray shape (3,)，[α_slow, β_inter, γ_fast]
"""

import numpy as np
import torch
import torch.nn as nn

from src.models.slow_prior import SlowPrior
from src.models.fast_corrector import FastCorrector
from src.models.gated_ensemble import GatedEnsemble
from src.consolidation.fast_to_inter import FastToInterConsolidation


class MultiTimescaleModel:
    """
    三时间尺度编排器（Phase 3C）。

    功能：
        将 SlowPrior（Level 1）、FastCorrector（Level 3）、
        GatedEnsemble（Phase 3A 门控融合）、FastToInterConsolidation（Phase 3B）
        组合为统一的 prequential 接口，每步 step() 按固定顺序执行六步流程。

    偏离 plan V2 的设计决策：
        [决策 1] 每步 MSE 训练：step() 在观测真实标签 y_t 后，
            对 y_final_raw（未 clamp）计算 MSE(y_final_raw, y_t)，
            执行 backward + optimizer.step()，在线更新 gate 和 adapter 参数。
            选用 MSE 而非 BCE：adapter 输出无界，y_final 偶尔超出 [0,1]，
            MSE 对值域宽容；BCE 在 log(0) 附近数值不稳定。
        [决策 2] y_final clamp 后作为预测输出：
            clamp 发生在 MSE 计算之后（loss 使用原始 y_final_raw），
            返回值使用 torch.clamp(y_final_raw, 0, 1).item()，
            确保外部调用方拿到合法概率。
    """

    def __init__(
        self,
        input_dim: int,
        buffer_size: int = 100,
        fast_method: str = "knn",
        knn_k: int = 5,
        ema_alpha: float = 0.15,
        consolidation_threshold: float = 0.05,
        consolidation_window: int = 50,
        consolidation_epochs: int = 10,
        gate_hidden_dim: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        n_estimators: int = 4,
    ):
        """
        Args:
            input_dim:               原始特征维度（必须与数据集一致）
            buffer_size:             FastCorrector 工作记忆容量
            fast_method:             "knn" 或 "ema"
            knn_k:                   KNN 近邻数
            ema_alpha:               EMA 平滑系数
            consolidation_threshold: 触发巩固的最小平均误差绝对值
            consolidation_window:    巩固观察窗口（同时是 FastToInterConsolidation.window）
            consolidation_epochs:    每次巩固的梯度更新步数
            gate_hidden_dim:         GatedEnsemble gate/adapter 隐藏层宽度
            lr:                      Adam 学习率（应用于 GatedEnsemble 全部参数）
            device:                  "cpu"（TabPFN 约束，不支持其他设备）
            n_estimators:            TabPFN 集成数量
        """
        assert input_dim > 0, f"input_dim 必须 > 0，收到: {input_dim}"
        assert device == "cpu", f"当前仅支持 CPU，收到: {device}"

        self.input_dim = input_dim
        self.consolidation_window = consolidation_window
        self.consolidation_threshold = consolidation_threshold
        self._step_count: int = 0
        self.consolidation_events: list = []

        # ── 子模块初始化 ──────────────────────────────────────────────
        self.slow_prior = SlowPrior(device=device, n_estimators=n_estimators)

        self.fast_corrector = FastCorrector(
            buffer_size=buffer_size,
            method=fast_method,
            k=knn_k,
            alpha=ema_alpha,
        )

        self.gated_ensemble = GatedEnsemble(
            input_dim=input_dim,
            hidden_dim=gate_hidden_dim,
            n_outputs=1,
        )

        self.consolidator = FastToInterConsolidation(
            threshold=consolidation_threshold,
            window=consolidation_window,
            epochs=consolidation_epochs,
        )

        # ── 优化器：仅优化 GatedEnsemble（TabPFN 权重绝不微调）─────────
        self.optimizer = torch.optim.Adam(
            self.gated_ensemble.parameters(), lr=lr
        )

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    def step(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        x_t: np.ndarray,
        y_t: float,
        t: int = -1,
    ):
        """
        执行单步 prequential 推理与更新。

        Args:
            X_ctx: (context_size, input_dim) 上下文特征，喂给 TabPFN
            y_ctx: (context_size,) 上下文标签（0/1）
            x_t:   (input_dim,) 当前时步查询特征（1D）
            y_t:   当前时步真实标签（0 或 1，标量）
            t:     全局时间步坐标，由调用方传入（对应数据集原始下标，从 context_size 起步）

        Returns:
            y_pred:  float ∈ [0, 1]，clamp 后的正类概率（可直接用于 ≥0.5 判断）
            weights: np.ndarray shape (3,)，门控权重 [α_slow, β_inter, γ_fast]

        consolidation_events 存的是全局时间步坐标，由调用方通过 t 参数提供；
        consolidation 触发由 fast_corrector.should_consolidate 单点判断；
        consolidate() 内部有 assert 兜底形状，外层不加额外保护（YAGNI）。
        """
        assert isinstance(t, int), f"t 必须为 int，收到: {type(t)}"
        assert X_ctx.ndim == 2, (
            f"X_ctx 应为 2D 数组 (context_size, input_dim)，收到 shape: {X_ctx.shape}"
        )
        assert X_ctx.shape[1] == self.input_dim, (
            f"X_ctx 特征维度应为 {self.input_dim}，收到: {X_ctx.shape[1]}"
        )
        x_t = np.asarray(x_t, dtype=np.float32).ravel()
        assert x_t.shape[0] == self.input_dim, (
            f"x_t 长度应为 {self.input_dim}，收到: {x_t.shape[0]}"
        )
        y_t = float(y_t)

        # ── Step 1：慢层预测（TabPFN in-context learning）────────────
        X_query_2d = x_t[np.newaxis, :]                   # (1, input_dim)
        proba = self.slow_prior.predict_proba(X_ctx, y_ctx, X_query_2d)
        y_slow: float = float(proba[0, 1])                # 正类概率，标量

        # ── Step 2：快层校正 ─────────────────────────────────────────
        correction: float = self.fast_corrector.correct(x_t)
        y_fast_prob: float = float(np.clip(y_slow + correction, 0.0, 1.0))

        # ── Step 3：门控融合 ─────────────────────────────────────────
        x_tensor = torch.tensor(x_t[np.newaxis, :], dtype=torch.float32)   # (1, D)
        y_slow_t  = torch.tensor([[y_slow]],      dtype=torch.float32)     # (1, 1)
        y_fast_t  = torch.tensor([[y_fast_prob]], dtype=torch.float32)     # (1, 1)

        self.gated_ensemble.train()
        y_final_raw, weights_tensor = self.gated_ensemble(
            x_tensor, y_slow_t, y_fast_t
        )                                                  # (1,1), (1,3)

        # ── Step 4：每步 MSE 训练（偏离 plan V2 决策 1）────────────────
        y_t_tensor = torch.tensor([[y_t]], dtype=torch.float32)            # (1, 1)
        loss = nn.functional.mse_loss(y_final_raw, y_t_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ── Step 5：观测后更新 buffer ────────────────────────────────
        error: float = y_t - y_slow
        self.fast_corrector.update(x_t, error)

        # ── Step 6：按需巩固 ─────────────────────────────────────────
        if self.fast_corrector.should_consolidate(
            window=self.consolidation_window,
            bias_threshold=self.consolidation_threshold,
        ):
            X_recent = self.fast_corrector.buffer.recent_features(
                self.consolidation_window
            )
            self.consolidator.consolidate(
                gated_ensemble=self.gated_ensemble,
                fast_corrector=self.fast_corrector,
                X_recent=X_recent,
                optimizer=self.optimizer,
            )
            self.consolidation_events.append(t)

        self._step_count += 1  # 保留用于 __repr__ 调试展示，不再用于事件记录

        # ── 返回：clamp 后概率（偏离 plan V2 决策 2）+ 门控权重 ─────
        # clamp 发生在 MSE loss 计算之后，loss 使用的是 y_final_raw（无 clamp）
        y_pred: float = torch.clamp(y_final_raw.detach(), 0.0, 1.0).item()
        weights_np: np.ndarray = (
            weights_tensor.detach().squeeze(0).numpy()     # (3,)
        )
        return y_pred, weights_np

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def reset_fast(self) -> None:
        """手动清空快速校正器缓冲区（如在已知漂移点处调用）。"""
        self.fast_corrector.reset()

    def __repr__(self) -> str:
        return (
            f"MultiTimescaleModel("
            f"input_dim={self.input_dim}, "
            f"step_count={self._step_count}, "
            f"consolidation_events={len(self.consolidation_events)}, "
            f"fast={self.fast_corrector.method})"
        )
