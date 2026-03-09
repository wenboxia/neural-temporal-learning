"""
时序窗口化加载器

将时序数据集拆分为一系列 (上下文窗口, 查询样本) 对，
供 TabPFN in-context learning 使用。

用法示例：
    dataset = make_rotating_boundary(n_samples=10000)
    loader = TemporalWindowLoader(dataset.X, dataset.y, context_size=500, step_size=1)
    for batch in loader:
        X_ctx, y_ctx, X_query, y_query, t = batch
        ...
"""

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np


@dataclass
class TemporalBatch:
    """单个时间步的数据批次。"""
    X_ctx: np.ndarray      # 上下文特征 (context_size, n_features)
    y_ctx: np.ndarray      # 上下文标签 (context_size,)
    X_query: np.ndarray    # 当前查询特征 (1, n_features)
    y_query: np.ndarray    # 当前真实标签 (1,)
    t: int                 # 时间步索引


class TemporalWindowLoader:
    """
    滑动窗口时序加载器。

    对于每个时间步 t（从 context_size 开始），
    返回：
      - 上下文窗口：[t - context_size, t) 的历史数据
      - 查询样本：时间步 t 的单个样本

    这是 prequential（先测试后训练）评估的标准设置：
    先用历史数据预测当前样本，再将当前样本的真实标签用于更新。
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        context_size: int = 500,
        step_size: int = 1,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):
        """
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签向量 (n_samples,)
            context_size: 上下文窗口大小（TabPFN 的"训练集"样本数）
            step_size: 每次滑动的步长（通常为 1）
            start: 起始时间步（默认 = context_size，确保上下文非空）
            end: 结束时间步（默认 = n_samples）
        """
        assert len(X) == len(y), "X 和 y 长度必须一致"
        assert context_size >= 1, "context_size 至少为 1"

        self.X = X
        self.y = y
        self.context_size = context_size
        self.step_size = step_size
        self.start = start if start is not None else context_size
        self.end = end if end is not None else len(X)

        assert self.start >= context_size, (
            f"start ({self.start}) 必须 >= context_size ({context_size})"
        )

    def __len__(self) -> int:
        """返回总时间步数。"""
        return max(0, (self.end - self.start + self.step_size - 1) // self.step_size)

    def __iter__(self) -> Generator[TemporalBatch, None, None]:
        """迭代所有时间步，每次 yield 一个 TemporalBatch。"""
        for t in range(self.start, self.end, self.step_size):
            X_ctx = self.X[t - self.context_size: t]
            y_ctx = self.y[t - self.context_size: t]
            X_query = self.X[t: t + 1]
            y_query = self.y[t: t + 1]
            yield TemporalBatch(
                X_ctx=X_ctx,
                y_ctx=y_ctx,
                X_query=X_query,
                y_query=y_query,
                t=t,
            )

    def get_batch(self, t: int) -> TemporalBatch:
        """随机访问：获取时间步 t 的批次。"""
        assert self.context_size <= t < self.end, (
            f"t={t} 超出范围 [{self.context_size}, {self.end})"
        )
        return TemporalBatch(
            X_ctx=self.X[t - self.context_size: t],
            y_ctx=self.y[t - self.context_size: t],
            X_query=self.X[t: t + 1],
            y_query=self.y[t: t + 1],
            t=t,
        )
