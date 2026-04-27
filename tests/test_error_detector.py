"""
Phase 4 A 单元测试：ADWINErrorDetector

不依赖 TabPFN / torch，纯 numpy + 自实现 ADWIN，离线运行 < 1s。
覆盖：
  1. 稳定流不触发漂移
  2. 显著均值跳变能在合理延迟内被检测
  3. cooldown 抑制重复触发
  4. clear() 后窗口归零
  5. 太短窗口不会误触发
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.drift.error_detector import ADWINErrorDetector


class TestADWINErrorDetector:

    def test_stationary_stream_no_drift(self):
        """同分布 N(0, 0.05²) 噪声流不应触发漂移。"""
        rng = np.random.default_rng(0)
        det = ADWINErrorDetector(delta=0.002, min_subwindow=30, cooldown=20)
        n_drifts = 0
        for _ in range(1000):
            x = float(rng.normal(0.0, 0.05))
            if det.update(x):
                n_drifts += 1
        assert n_drifts == 0, f"稳定流不应触发漂移，实触发 {n_drifts} 次"

    def test_mean_shift_detected(self):
        """前 300 步均值 0，后 300 步均值 0.5 → 应在 shift 后短期内检测到。"""
        rng = np.random.default_rng(1)
        det = ADWINErrorDetector(delta=0.002, min_subwindow=30, cooldown=50)
        drift_steps = []
        for t in range(600):
            mu = 0.0 if t < 300 else 0.5
            x = float(rng.normal(mu, 0.05))
            if det.update(x):
                drift_steps.append(t + 1)  # detector 内部 t 从 1 起

        assert len(drift_steps) >= 1, (
            f"明显均值漂移应被至少检测 1 次，实际未触发"
        )
        first = drift_steps[0]
        # 漂移真实点是 t=300（第 301 个观测开始变化）
        # 允许检测延迟 ≤ 100 步
        assert 300 <= first <= 400, (
            f"首次漂移检测应在 [300, 400] 内，实际 t={first}"
        )

    def test_cooldown_suppresses_duplicate_alarms(self):
        """同一漂移不应在 cooldown 期内反复触发。"""
        rng = np.random.default_rng(2)
        det = ADWINErrorDetector(
            delta=0.002, min_subwindow=30, cooldown=100,
        )
        drift_steps = []
        for t in range(500):
            mu = 0.0 if t < 200 else 0.6
            x = float(rng.normal(mu, 0.05))
            if det.update(x):
                drift_steps.append(t + 1)

        # 至少检测 1 次；相邻检测点必须间隔 ≥ cooldown
        assert len(drift_steps) >= 1
        for a, b in zip(drift_steps[:-1], drift_steps[1:]):
            assert b - a >= det.cooldown, (
                f"相邻漂移间隔 {b-a} 小于 cooldown={det.cooldown}"
            )

    def test_clear_resets_window(self):
        """clear() 后窗口长度归零；后续短期不会立刻触发漂移。"""
        rng = np.random.default_rng(3)
        det = ADWINErrorDetector(delta=0.002, min_subwindow=30, cooldown=10)
        for _ in range(100):
            det.update(float(rng.normal(0.0, 0.05)))
        assert len(det) > 0

        det.clear()
        assert len(det) == 0, f"clear 后窗口应为空，实为 {len(det)}"

        # 注入 60 步极弱信号（仍不到 2·min_subwindow），不应触发
        for _ in range(50):
            triggered = det.update(float(rng.normal(0.0, 0.05)))
            assert not triggered, "clear 后短期内不应触发漂移"

    def test_short_window_no_false_positive(self):
        """样本数 < 2·min_subwindow 时永不触发漂移。"""
        rng = np.random.default_rng(4)
        det = ADWINErrorDetector(delta=0.5, min_subwindow=50, cooldown=0)
        # 即使注入剧烈跳变，前 99 步也不该触发
        for t in range(99):
            mu = -1.0 if t < 50 else 1.0
            x = float(np.clip(rng.normal(mu, 0.01), -1, 1))
            assert not det.update(x), (
                f"窗口未达 2·min_subwindow=100 时不应触发，t={t} 触发"
            )

    def test_n_drifts_counter_monotonic(self):
        """n_drifts 计数器随漂移触发单调递增。"""
        rng = np.random.default_rng(5)
        det = ADWINErrorDetector(delta=0.002, min_subwindow=30, cooldown=80)
        prev = 0
        for t in range(800):
            # 三段：[0, 0.5, -0.3]
            if t < 250:
                mu = 0.0
            elif t < 500:
                mu = 0.5
            else:
                mu = -0.3
            det.update(float(rng.normal(mu, 0.05)))
            assert det.n_drifts >= prev, "n_drifts 不应回退"
            prev = det.n_drifts

        assert det.n_drifts >= 2, (
            f"两次漂移应至少检测 2 次，实际 {det.n_drifts}"
        )
