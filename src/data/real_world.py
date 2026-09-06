"""
真实世界漂移数据集加载器 (Phase 5)

提供与 `SyntheticDataset` 接口对齐的 `RealWorldDataset`，覆盖：
  - Electricity (OpenML id=151)：~45k 样本，8 特征（含 1 类别），二分类，gradual / seasonal
  - Insects abrupt_balanced (USP DS via Google Drive)：52,848 样本，33 数值特征，
    原 6 类多分类按 sex-pair 二值化（Phase 5 决策）

Prequential 协议硬要求：normalization 仅 fit on 前 200 样本，再 transform 全 segment，
严禁全段 fit（会泄漏未来统计进入 baseline 评估）。
"""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class RealWorldDataset:
    X: np.ndarray  # (n_samples, n_features) float32
    y: np.ndarray  # (n_samples,) int (0/1)
    drift_points: List[int] = field(default_factory=list)
    name: str = ""


# ---------------------------------------------------------------------------
# Insects abrupt_balanced
# ---------------------------------------------------------------------------

# Google Drive 直链来自 river master (`river/datasets/insects.py`).
# river 0.23 内置 URL 已 404 (USP labic.icmc.usp.br 已下线 creme/ 路径)。
_INSECTS_GDRIVE = {
    "abrupt_balanced": (
        "https://drive.google.com/uc?export=download&"
        "id=1WQoIuuVgiuXfzv4kvao6XuLQG37V923O&confirm=t"
    ),
}

# 首次下载后通过 sha256(open(path,'rb').read()).hexdigest() 计算并写死，
# 防止 Google Drive 静默替换文件（as of 2026-04-29）。
_INSECTS_SHA256 = {
    # as of 2026-04-29，首次下载实测；如 Google Drive 静默替换会抛错
    "abrupt_balanced": "f368a6f4b7f28ce2e9aa0a9e542e1f6924999cae3e8607f0637549798cb7b94a",
}

_INSECTS_CACHE_DIR = os.path.expanduser("~/.cache/insects_drift")
_INSECTS_CSV_FEATURE_COLS = [f"f{i}" for i in range(1, 34)]

# Phase 5 决策：按相邻 ID 配对的奇偶二值化。class IDs [2,3,4,5,11,12] 被**推断**为
# 3 物种 × 2 性别（{2,3} / {4,5} / {11,12} 是相邻整数对），每对偶数 ID → 0、奇数 ID → 1。
# ⚠️ Phase 5.5 更正：Souza 2020 与 USP 仓库均**未提供** class ID → 物种/性别的对应表
# （原文只说 6 类来自 Aedes aegypti / Aedes albopictus / Culex quinquefasciatus 的雌雄）。
# 因此这只能称 "ID 分组 (pair parity)"，不能称 sex 分类；论文 Limitations 须写明。
_INSECTS_BINARIZE_MAP = {2: 0, 4: 0, 11: 0, 3: 1, 5: 1, 12: 1}

# ── 漂移坐标：官方 vs 经验推断（Phase 5.5 更正，2026-09-06）────────────────────
#
# 官方坐标（Souza et al. 2020, Table 2, "Abrupt (bal.)", 52,848 instances）。
# 漂移由捕虫器内温度变化引起（30°C → 20°C → ~35°C → ...），即 P(X|y) 变化。
_INSECTS_OFFICIAL_DRIFT_POINTS = [14_352, 19_500, 33_240, 38_682, 39_510]

# 经验推断坐标（2026-04-29 的 50-chunk P(y) 诊断产物）。
# ⚠️ Phase 5 文档一度把这组升格成 "documented drift"，这是错的：它们是**标签构成**
# 突变点，不是温度漂移点。实测（2026-09-06）：
#   - 这些点处的类条件特征偏移 |Δμ|/σ 仅 0.05–0.09（≈全流中位数 0.07），
#     而官方点处是 0.25–0.49；
#   - 12,672 与 14,256 都落在同一段连续 class 5（[12,598, 14,352) 共 1,754 条）内部，
#     不是那段单类区间的真实边界。
# 保留此常量仅为复现既有 B1+ 结果；新实验一律用官方坐标。
_INSECTS_EMPIRICAL_PY_SHIFT_POINTS = [12_672, 14_256, 17_952, 46_728, 52_008]

# 向后兼容别名（既有代码路径引用）；语义 = 经验 P(y) 变化点，非官方漂移点。
_INSECTS_DRIFT_POINTS_HINT = _INSECTS_EMPIRICAL_PY_SHIFT_POINTS

# Phase 5 Stage B re-aligned (B1+, 2026-05-01)：按**经验 P(y) 变化点**对齐的 4 段。
# ⚠️ Phase 5.5 更正：按官方坐标看，这 4 段只覆盖 2/5 个真实漂移
#   （early 含 14,352 → local 4,352；mid 含 19,500 → local 3,500；
#    late_pre / late_post 不含任何官方漂移）。
# 保留此切法仅为复现既有 60 runs；新实验用官方坐标居中的段（见 todo.md P1）。
_INSECTS_ALIGNED_BOUNDS = {
    "early":     (10_000, 15_000),  # 经验点 12,672 + 14,256；官方点 14,352（local 4,352）
    "mid":       (16_000, 21_000),  # 经验点 17,952；官方点 19,500（local 3,500）
    "late_pre":  (42_500, 47_500),  # 经验点 46,728；无官方漂移
    "late_post": (47_848, 52_848),  # 经验点 52,008；无官方漂移
}
_INSECTS_ALIGNED_SEGMENTS = list(_INSECTS_ALIGNED_BOUNDS.keys())


def _ensure_insects_csv(variant: str = "abrupt_balanced") -> str:
    if variant not in _INSECTS_GDRIVE:
        raise ValueError(
            f"Unsupported Insects variant: {variant!r}; "
            f"available: {list(_INSECTS_GDRIVE)}"
        )
    os.makedirs(_INSECTS_CACHE_DIR, exist_ok=True)
    target = os.path.join(_INSECTS_CACHE_DIR, f"{variant}.csv")

    if os.path.exists(target):
        # 缓存命中：校验 sha256（首轮 PENDING 时跳过校验，仅打印实际 hash）
        with open(target, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        expected = _INSECTS_SHA256.get(variant, "PENDING")
        if expected != "PENDING" and actual != expected:
            raise RuntimeError(
                f"Cached Insects CSV checksum mismatch for {variant!r}\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                f"  path:     {target}\n"
                f"Delete the file to re-download."
            )
        if expected == "PENDING":
            print(
                f"[insects] sha256({variant})={actual}  "
                f"(write into _INSECTS_SHA256 to enable verification)"
            )
        return target

    # Cache miss → download
    url = _INSECTS_GDRIVE[variant]
    print(f"[insects] downloading {variant} from Google Drive (~14 MB)...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        data = r.read()
    with open(target, "wb") as f:
        f.write(data)
    actual = hashlib.sha256(data).hexdigest()
    print(f"[insects] saved to {target} ({len(data)} bytes)")
    print(f"[insects] sha256={actual}")
    return target


def load_insects(
    segment_id: str = "start",
    size: int = 5000,
    variant: str = "abrupt_balanced",
    insects_aligned: bool = False,
) -> RealWorldDataset:
    """加载 Insects 二值化 segment。

    insects_aligned=False (default): segment_id ∈ {start, middle, end}, size 任意 ≤ N
    insects_aligned=True (Phase 5 B1+): segment_id ∈ {early, mid, late_pre, late_post}，
      bounds 由 `_INSECTS_ALIGNED_BOUNDS` 硬编码（覆盖全 5/5 drift），size 参数被忽略
      （4 段都固定 5000 samples）。
    """
    csv_path = _ensure_insects_csv(variant=variant)
    cols = _INSECTS_CSV_FEATURE_COLS + ["class"]
    df = pd.read_csv(csv_path, header=None, names=cols)

    # 二值化：class IDs → 0/1 sex-pair binarization
    unknown = set(df["class"].unique()) - set(_INSECTS_BINARIZE_MAP.keys())
    if unknown:
        raise RuntimeError(
            f"Unexpected Insects class IDs {unknown}; "
            f"binarization map covers only {sorted(_INSECTS_BINARIZE_MAP)}"
        )
    y_full = df["class"].map(_INSECTS_BINARIZE_MAP).to_numpy(dtype=np.int64)
    X_full = df[_INSECTS_CSV_FEATURE_COLS].to_numpy(dtype=np.float32)

    if insects_aligned:
        if segment_id not in _INSECTS_ALIGNED_BOUNDS:
            raise ValueError(
                f"insects_aligned=True requires segment_id ∈ "
                f"{_INSECTS_ALIGNED_SEGMENTS}, got {segment_id!r}"
            )
        seg_start, seg_end = _INSECTS_ALIGNED_BOUNDS[segment_id]
        X_seg = X_full[seg_start:seg_end].copy()
        y_seg = y_full[seg_start:seg_end].copy()
    else:
        if segment_id not in {"start", "middle", "end"}:
            raise ValueError(
                f"insects_aligned=False requires segment_id ∈ "
                f"{{start, middle, end}}, got {segment_id!r}"
            )
        X_seg, y_seg = take_segment(X_full, y_full, segment_id=segment_id, size=size)
        seg_start, seg_end = _segment_bounds(len(X_full), segment_id, size)

    X_seg = _prequential_normalize(X_seg, fit_size=200)

    local_drift = [
        int(d - seg_start)
        for d in _INSECTS_DRIFT_POINTS_HINT
        if seg_start < d < seg_end
    ]
    suffix = "aligned_" if insects_aligned else ""
    return RealWorldDataset(
        X=X_seg,
        y=y_seg,
        drift_points=local_drift,
        name=f"insects_{variant}_{suffix}{segment_id}",
    )


# ---------------------------------------------------------------------------
# Electricity (OpenML 151)
# ---------------------------------------------------------------------------


def load_electricity(segment_id: str = "start", size: int = 5000) -> RealWorldDataset:
    """加载 Electricity 二分类 segment。

    OpenML id=151，N=45,312，8 features (1 类别 day-of-week + 7 数值)。
    时序按 (date, period) 已排序，不重排。target = class ∈ {UP, DOWN} → {0, 1}.

    缺失值策略：drop rows（OpenML 151 的原版无缺失，dropna 是 defensive no-op）。
    """
    import openml  # 懒加载，避免合成实验路径强依赖

    print("[electricity] fetching OpenML 151 (cached at ~/.openml/ if previously downloaded)...")
    ds = openml.datasets.get_dataset(
        151, download_data=True, download_qualities=False, download_features_meta_data=False
    )
    X_df, y_series, _, _ = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )

    # drop missing rows (defensive)
    n_before = len(X_df)
    valid = X_df.notna().all(axis=1) & y_series.notna()
    X_df = X_df.loc[valid].reset_index(drop=True)
    y_series = y_series.loc[valid].reset_index(drop=True)
    if len(X_df) < n_before:
        print(f"[electricity] dropped {n_before - len(X_df)} rows with NaN")

    # 类别特征 one-hot；数值列直通
    X_encoded = pd.get_dummies(X_df, drop_first=False)
    X_full = X_encoded.to_numpy(dtype=np.float32)
    # target: class 是字符串 "UP"/"DOWN" → 0/1 (UP=1)
    y_full = (y_series.astype(str).str.upper() == "UP").to_numpy(dtype=np.int64)

    X_seg, y_seg = take_segment(X_full, y_full, segment_id=segment_id, size=size)
    X_seg = _prequential_normalize(X_seg, fit_size=200)

    # Electricity 漂移是 gradual/seasonal，无 crisp drift points
    return RealWorldDataset(
        X=X_seg, y=y_seg, drift_points=[], name=f"electricity_{segment_id}"
    )


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _segment_bounds(n: int, segment_id: str, size: int) -> tuple[int, int]:
    if size > n:
        raise ValueError(f"size {size} exceeds dataset length {n}")
    if segment_id == "start":
        return 0, size
    if segment_id == "end":
        return n - size, n
    if segment_id == "middle":
        mid = n // 2
        return mid - size // 2, mid - size // 2 + size
    raise ValueError(
        f"segment_id must be one of {{start, middle, end}}, got {segment_id!r}"
    )


def take_segment(
    X: np.ndarray, y: np.ndarray, segment_id: str = "start", size: int = 5000
) -> tuple[np.ndarray, np.ndarray]:
    """Contiguous 时序切片，保持原顺序，不 shuffle。"""
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
    lo, hi = _segment_bounds(len(X), segment_id, size)
    return X[lo:hi].copy(), y[lo:hi].copy()


def _prequential_normalize(X: np.ndarray, fit_size: int = 200) -> np.ndarray:
    """StandardScaler fit on 前 fit_size 样本，transform 全段。Prequential 协议硬要求。"""
    if len(X) < fit_size:
        raise ValueError(
            f"segment too short ({len(X)}) for fit_size={fit_size}; "
            f"need at least {fit_size} samples for the initial-context fit"
        )
    scaler = StandardScaler()
    scaler.fit(X[:fit_size])
    out = scaler.transform(X).astype(np.float32, copy=False)
    return out


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------


def load_real_world(
    name: str, segment_id: str = "start", size: int = 5000,
    insects_aligned: bool = False, **kwargs,
) -> RealWorldDataset:
    """
    name ∈ {"electricity", "insects"};
    segment_id ∈ {"start", "middle", "end"} 或 (insects_aligned=True 时)
                  {"early", "mid", "late_pre", "late_post"}.

    Insects 当前固定 variant="abrupt_balanced"（Phase 5 §决策），可由 kwargs 传入覆盖。
    insects_aligned 仅对 Insects 生效，Electricity 忽略。
    """
    if name == "electricity":
        return load_electricity(segment_id=segment_id, size=size)
    if name == "insects":
        variant = kwargs.pop("variant", "abrupt_balanced")
        return load_insects(
            segment_id=segment_id, size=size, variant=variant,
            insects_aligned=insects_aligned,
        )
    raise ValueError(f"unknown real-world dataset {name!r}")
