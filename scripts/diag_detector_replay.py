"""
Phase 5.5 Step 1 — 检测器离线重放诊断（零 CPU 成本，不跑 TabPFN）

背景：Phase 4/5 的 phase4a npz 里存了每步的 0/1 错误指示器 `indicator_history`。
检测器只消费这条 1D 流，所以换检测器实现 / 扫 δ **不需要重跑实验**，
把已存的 35 条流重放一遍即可。

对每条流报告：报警时刻、相对 documented 变点的检测延迟、变点外的误报数。
Insects 用 Souza 2020 Table 2 的官方变点（14352/19500/33240/38682/39510），
而不是仓库里此前用的 P(y) 构成变化点 —— 后者是标签构成突变，非温度漂移。

用法：
    python scripts/diag_detector_replay.py                       # 默认 δ 网格
    python scripts/diag_detector_replay.py --deltas 0.002,0.05   # 自定义
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.drift.error_detector import make_detector

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")

# Souza 2020 Table 2, INSECTS Abrupt (balanced), 52,848 instances.
OFFICIAL_INSECTS_DRIFTS = [14352, 19500, 33240, 38682, 39510]

# B1+ 段边界（绝对坐标），用于把官方变点映射到段内局部坐标
INSECTS_SEG_BOUNDS = {
    "early": (10_000, 15_000),
    "mid": (16_000, 21_000),
    "late_pre": (42_500, 47_500),
    "late_post": (47_848, 52_848),
}


def official_local_drifts(segment: str) -> list[int]:
    lo, hi = INSECTS_SEG_BOUNDS[segment]
    return [d - lo for d in OFFICIAL_INSECTS_DRIFTS if lo < d < hi]


def replay(indicator: np.ndarray, impl: str, delta: float, cooldown: int,
           context_size: int = 200) -> list[int]:
    """重放一条 indicator 流，返回报警的**段内局部**时刻（= 数组下标 + context_size）。"""
    det = make_detector(
        impl, delta=delta, min_subwindow=30, max_window=400,
        value_range=1.0, cooldown=cooldown,
    )
    return [i + context_size for i, v in enumerate(indicator) if det.update(float(v))]


def score(alarms: list[int], drifts: list[int], tolerance: int) -> dict:
    """检测延迟 / 命中 / 误报。每个变点只允许被最早的一次报警命中。"""
    hits, delays, used = 0, [], set()
    for d in drifts:
        cand = [a for a in alarms if 0 <= a - d <= tolerance and a not in used]
        if cand:
            hits += 1
            used.add(cand[0])
            delays.append(cand[0] - d)
    return {
        "n_alarms": len(alarms),
        "hits": hits,
        "n_drifts": len(drifts),
        "false_alarms": len(alarms) - hits,
        "delays": delays,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=str, default="0.002,0.01,0.05,0.2")
    ap.add_argument("--cooldown", type=int, default=80)
    ap.add_argument("--tolerance", type=int, default=600,
                    help="报警落在变点后多少步内算命中")
    ap.add_argument("--out", type=str,
                    default=os.path.join(RESULTS, "detector_replay.md"))
    args = ap.parse_args()
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]

    streams = []  # (dataset, segment, seed, indicator, official_local_drifts)
    for seg in INSECTS_SEG_BOUNDS:
        for f in sorted(glob.glob(
                os.path.join(RESULTS, f"multiseed_phase4a_real_insects_{seg}_seed*.npz"))):
            d = np.load(f, allow_pickle=True)
            streams.append(("insects", seg, int(d["seed"][0]),
                            d["indicator_history"], official_local_drifts(seg)))
    for seg in ("start", "middle", "end"):
        for f in sorted(glob.glob(
                os.path.join(RESULTS, f"multiseed_phase4a_real_electricity_{seg}_seed*.npz"))):
            d = np.load(f, allow_pickle=True)
            # Electricity 无 documented 变点 → 任何报警都记为误报（渐进漂移应沉默）
            streams.append(("electricity", seg, int(d["seed"][0]),
                            d["indicator_history"], []))

    print(f"loaded {len(streams)} saved indicator streams")

    rows = []
    for impl in ("own", "river"):
        for delta in deltas:
            agg = defaultdict(lambda: {"alarms": 0, "hits": 0, "drifts": 0,
                                       "fa": 0, "delays": [], "runs": 0,
                                       "runs_with_alarm": 0})
            for ds, seg, seed, ind, drifts in streams:
                al = replay(ind, impl, delta, args.cooldown)
                s = score(al, drifts, args.tolerance)
                a = agg[ds]
                a["alarms"] += s["n_alarms"]; a["hits"] += s["hits"]
                a["drifts"] += s["n_drifts"]; a["fa"] += s["false_alarms"]
                a["delays"] += s["delays"]; a["runs"] += 1
                a["runs_with_alarm"] += int(s["n_alarms"] > 0)
            for ds, a in agg.items():
                rows.append({
                    "impl": impl, "delta": delta, "dataset": ds,
                    "runs_with_alarm": f"{a['runs_with_alarm']}/{a['runs']}",
                    "alarms": a["alarms"],
                    "recall": (a["hits"] / a["drifts"]) if a["drifts"] else None,
                    "false_alarms": a["fa"],
                    "median_delay": (float(np.median(a["delays"])) if a["delays"] else None),
                })

    lines = [
        "# Detector replay — own vs river ADWIN on saved indicator streams",
        "",
        "**零 CPU 成本诊断**：检测器只消费 1D 的 0/1 错误指示器流，该流已存在 Phase 5 的 npz 里"
        "（`indicator_history`），因此换实现 / 扫 δ 无需重跑 TabPFN。",
        "",
        f"- 流数：{len(streams)}（Insects B1+ 20 + Electricity A+ 15）",
        f"- Insects 变点用 **Souza 2020 Table 2 官方坐标** {OFFICIAL_INSECTS_DRIFTS}，"
        "不是仓库此前使用的 P(y) 构成变化点",
        "- Electricity 无 documented 变点（渐进漂移），任何报警都计为误报",
        f"- 命中容差 = 变点后 {args.tolerance} 步；cooldown = {args.cooldown}",
        "",
        "| impl | δ | dataset | runs with ≥1 alarm | alarms | recall | false alarms | median delay |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rec = f"{r['recall']:.2f}" if r["recall"] is not None else "—"
        dly = f"{r['median_delay']:.0f}" if r["median_delay"] is not None else "—"
        lines.append(
            f"| {r['impl']} | {r['delta']} | {r['dataset']} | {r['runs_with_alarm']} | "
            f"{r['alarms']} | {rec} | {r['false_alarms']} | {dly} |"
        )

    lines += ["", "## Per-stream detail (river, δ=0.002)", "",
              "| dataset | segment | seed | err rate | official drifts (local) | alarms (local) | delays |",
              "|---|---|---|---|---|---|---|"]
    for ds, seg, seed, ind, drifts in streams:
        al = replay(ind, "river", 0.002, args.cooldown)
        s = score(al, drifts, args.tolerance)
        lines.append(
            f"| {ds} | {seg} | {seed} | {float(ind.mean()):.3f} | {drifts} | "
            f"{al} | {s['delays']} |"
        )

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
