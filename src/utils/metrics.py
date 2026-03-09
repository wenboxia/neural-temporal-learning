"""
评估指标

核心指标：
  - prequential_accuracy: 逐步预测准确率（先预测后更新）
  - window_accuracy: 按时间窗口计算的准确率序列
  - adaptation_speed: 漂移后恢复到基线准确率所需步数
"""

from typing import Dict, List, Optional

import numpy as np


def prequential_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    fading_factor: float = 1.0,
) -> float:
    """
    Prequential（先测试后训练）准确率。

    fading_factor < 1 时，对更近的预测赋予更高权重（衰减加权）。
    fading_factor = 1 等价于普通准确率。

    Args:
        predictions: (n,) 预测标签
        labels:      (n,) 真实标签
        fading_factor: 衰减因子，范围 (0, 1]

    Returns:
        加权准确率，范围 [0, 1]
    """
    n = len(predictions)
    assert len(labels) == n
    if n == 0:
        return 0.0

    correct = (predictions == labels).astype(float)
    if fading_factor == 1.0:
        return float(np.mean(correct))

    # 指数衰减权重：最近的样本权重最高
    weights = np.array([fading_factor ** (n - 1 - i) for i in range(n)])
    return float(np.average(correct, weights=weights))


def window_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    window_size: int = 200,
) -> np.ndarray:
    """
    按滑动窗口计算准确率序列。

    用于可视化模型性能随时间的变化，特别是在漂移点附近的跌落和恢复。

    Args:
        predictions: (n,) 预测标签
        labels:      (n,) 真实标签
        window_size: 窗口大小

    Returns:
        accs: (n - window_size + 1,) 每个窗口的准确率
    """
    n = len(predictions)
    assert len(labels) == n
    correct = (predictions == labels).astype(float)

    # 用累积和实现 O(n) 滑动窗口
    cumsum = np.concatenate([[0], np.cumsum(correct)])
    window_sums = cumsum[window_size:] - cumsum[:-window_size]
    return window_sums / window_size


def adaptation_speed(
    predictions: np.ndarray,
    labels: np.ndarray,
    drift_points: List[int],
    baseline_acc: float,
    window_size: int = 50,
    offset: int = 0,
) -> Dict[int, Optional[int]]:
    """
    计算每个漂移点后恢复到 baseline_acc 所需的步数。

    Args:
        predictions:  (n,) 预测标签
        labels:       (n,) 真实标签
        drift_points: 漂移发生的时间步列表（全局索引）
        baseline_acc: 参考准确率（通常是漂移前的平均值）
        window_size:  用于计算局部准确率的小窗口
        offset:       predictions 数组对应的起始全局时间步

    Returns:
        dict: {drift_point -> 恢复步数 (None 表示未恢复)}
    """
    result = {}
    n = len(predictions)
    correct = (predictions == labels).astype(float)

    for dp in drift_points:
        local_start = dp - offset
        if local_start < 0 or local_start >= n:
            result[dp] = None
            continue

        recovered_steps = None
        for i in range(local_start, min(n - window_size, n)):
            window_acc = np.mean(correct[i: i + window_size])
            if window_acc >= baseline_acc:
                recovered_steps = i - local_start
                break
        result[dp] = recovered_steps

    return result


def summarize_results(
    predictions: np.ndarray,
    labels: np.ndarray,
    drift_points: List[int],
    window_size: int = 200,
    offset: int = 0,
) -> Dict:
    """
    一次性计算并返回所有关键指标的汇总字典。

    Args:
        predictions:  (n,) 预测标签
        labels:       (n,) 真实标签
        drift_points: 漂移点列表（全局索引）
        window_size:  窗口大小
        offset:       predictions 起始的全局时间步

    Returns:
        包含 overall_acc, pre_drift_acc, post_drift_acc, window_accs,
        avg_adaptation_speed 的字典
    """
    overall = prequential_accuracy(predictions, labels)
    win_accs = window_accuracy(predictions, labels, window_size)

    # 漂移前后准确率（各取最近 window_size 步作为代表）
    pre_drift_accs, post_drift_accs = [], []
    for dp in drift_points:
        local = dp - offset
        if 0 < local < len(predictions):
            pre_start = max(0, local - window_size)
            pre_drift_accs.append(float(np.mean(predictions[pre_start:local] == labels[pre_start:local])))
            post_end = min(len(predictions), local + window_size)
            post_drift_accs.append(float(np.mean(predictions[local:post_end] == labels[local:post_end])))

    baseline = float(np.mean(pre_drift_accs)) if pre_drift_accs else overall
    speeds = adaptation_speed(predictions, labels, drift_points, baseline_acc=baseline * 0.95, offset=offset)
    valid_speeds = [v for v in speeds.values() if v is not None]

    return {
        "overall_acc": overall,
        "pre_drift_acc": float(np.mean(pre_drift_accs)) if pre_drift_accs else None,
        "post_drift_acc": float(np.mean(post_drift_accs)) if post_drift_accs else None,
        "window_accs": win_accs,
        "avg_adaptation_speed": float(np.mean(valid_speeds)) if valid_speeds else None,
    }
