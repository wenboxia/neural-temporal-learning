"""
Level 1: 慢速先验 —— 冻结的 TabPFN 包装器

永远不微调 TabPFN 权重。
每个时间步接受一个上下文窗口（X_ctx, y_ctx）和查询样本 X_query，
通过 in-context learning 给出预测概率。

使用注意事项：
  - TabPFN 对 context_size 有上限（默认 10000），实际建议 ≤ 3000（CPU 友好）。
  - TabPFN 每次 fit 都会在内存中存储上下文，每步调用开销主要来自前向传播。
  - 本模块不依赖 GPU，可在 CPU 上运行。
"""

import warnings
from typing import Tuple

import numpy as np


class SlowPrior:
    """
    冻结的 TabPFN 包装器。

    对外接口：
        predict(X_ctx, y_ctx, X_query) -> (proba, pred_label)
    """

    def __init__(self, device: str = "cpu", n_estimators: int = 8):
        """
        Args:
            device: 'cpu' 或 'cuda'
            n_estimators: TabPFN 内部集成数量，越大越慢越准
                          CPU 上建议 4~8
        """
        self.device = device
        self.n_estimators = n_estimators
        self._model = None
        self._is_fitted = False

    def _get_model(self):
        """懒加载 TabPFN，避免 import 时就下载权重。"""
        if self._model is None:
            try:
                from tabpfn import TabPFNClassifier
            except ImportError as e:
                raise ImportError(
                    "请先安装 tabpfn: pip install tabpfn"
                ) from e
            self._model = TabPFNClassifier(
                device=self.device,
                n_estimators=self.n_estimators,
            )
        return self._model

    def predict(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        给定上下文和查询样本，返回预测概率和预测标签。

        Args:
            X_ctx:   (context_size, n_features) 上下文特征
            y_ctx:   (context_size,) 上下文标签（0/1 二分类）
            X_query: (batch_size, n_features) 查询特征

        Returns:
            proba:       (batch_size, n_classes) 每类的预测概率
            pred_labels: (batch_size,) 预测的类别（argmax）
        """
        model = self._get_model()

        # TabPFN 要求上下文至少包含两个类别
        unique_classes = np.unique(y_ctx)
        if len(unique_classes) < 2:
            # 极端情况：上下文只有一类，直接返回多数类
            majority = unique_classes[0]
            n_query = len(X_query)
            proba = np.zeros((n_query, 2))
            proba[:, int(majority)] = 1.0
            return proba, np.full(n_query, majority, dtype=int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_ctx, y_ctx)
            proba = model.predict_proba(X_query)

        pred_labels = np.argmax(proba, axis=1)
        return proba, pred_labels

    def predict_proba(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        """仅返回概率（convenience wrapper）。"""
        proba, _ = self.predict(X_ctx, y_ctx, X_query)
        return proba

    def predict_label(
        self,
        X_ctx: np.ndarray,
        y_ctx: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        """仅返回预测标签（convenience wrapper）。"""
        _, pred_labels = self.predict(X_ctx, y_ctx, X_query)
        return pred_labels
