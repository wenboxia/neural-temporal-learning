"""
Phase 4 Day 0.5 - 实验 0b: Multi-seed 重跑 Phase 1 / Phase 2 / Phase 3

3 配置 × 3 数据集 × 5 seeds = 45 runs

用法：
    python scripts/run_multiseed.py --dry_run         # 仅打印命令
    python scripts/run_multiseed.py                   # 串行执行
    python scripts/run_multiseed.py --n_parallel 3    # 同数据集内并行 3 个 seed

约束：
- 三数据集**串行**（每完成一个 dataset 再开下一个），避免跨 dataset OOM 互扰
- 同一数据集内可并行 N 个 seed（默认 1）
- 结果落盘：results/multiseed_{config}_{dataset}_seed{S}.npz
- 每个 run 完成后写 results/multiseed_summary.partial.md（增量 dump）
- 已存在的 npz 自动 skip（resumable）
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# 项目根（脚本在 scripts/ 下）
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

SEEDS = [42, 123, 456, 789, 1024]
DATASETS = ["regime_switching", "rotating_boundary", "combined_drift"]
CONFIGS = ["phase1", "phase2", "phase3", "phase4a"]

# 各 (config, dataset) 的 base 命令（不含 --seed / --out_tag）
def build_base_cmd(config: str, dataset: str) -> list[str]:
    """返回该 (config, dataset) 的命令模板，调用方再加 --seed 和 --out_tag"""
    if config == "phase1":
        script = "scripts/run_baselines.py"
    elif config == "phase2":
        script = "scripts/run_phase2.py"
    elif config == "phase3":
        script = "scripts/run_phase3.py"
    elif config == "phase4a":
        script = "scripts/run_phase4_a.py"
    else:
        raise ValueError(config)

    cmd = [sys.executable, script, "--dataset", dataset]

    if dataset == "rotating_boundary":
        # run_baselines.py 自动设 n_features=2，phase2/3 需显式指定
        if config != "phase1":
            cmd += ["--n_features", "2"]
    elif dataset == "combined_drift":
        cmd += ["--n_samples", "5000"]
    elif dataset == "regime_switching":
        # 默认 n_samples=5000 for phase1, 3000 for phase2/3 - 统一到 3000 与 plan 对齐
        if config == "phase1":
            cmd += ["--n_samples", "3000"]

    # context_size 统一为 200（与 0a / Phase 2 / Phase 3 现有实验对齐）
    cmd += ["--context_size", "200"]
    return cmd


def out_tag(config: str, dataset: str, seed: int) -> str:
    return f"multiseed_{config}_{dataset}_seed{seed}"


def npz_path(config: str, dataset: str, seed: int) -> Path:
    return RESULTS_DIR / f"{out_tag(config, dataset, seed)}.npz"


def build_full_cmd(config: str, dataset: str, seed: int) -> list[str]:
    cmd = build_base_cmd(config, dataset)
    cmd += ["--seed", str(seed), "--out_tag", out_tag(config, dataset, seed)]
    return cmd


def read_overall_acc(config: str, dataset: str, seed: int) -> float | None:
    """从 npz 中读取 overall_acc。Phase 2 取 KNN 列。"""
    path = npz_path(config, dataset, seed)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    if config == "phase2":
        # 多 corrector npz：取 KNN 那列
        for key in ["overall_acc_TabPFN_p_KNN", "overall_acc_TabPFN_p_KNN_"]:
            if key in data.files:
                return float(data[key][0])
        return None
    if "overall_acc" in data.files:
        return float(data["overall_acc"][0])
    return None


def read_phase4a_metrics(dataset: str, seed: int) -> dict | None:
    """读取 phase4a npz 中的关键诊断字段（用于 partial md）。"""
    path = npz_path("phase4a", dataset, seed)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    out = {
        "overall_acc": float(data["overall_acc"][0]) if "overall_acc" in data.files else None,
        "post_drift_acc": (
            float(data["post_drift_acc"][0])
            if "post_drift_acc" in data.files and not np.isnan(data["post_drift_acc"][0])
            else None
        ),
        "n_routes": int(data["route_t"].shape[0]) if "route_t" in data.files else 0,
        "n_adapters": int(data["n_adapters_final"][0]) if "n_adapters_final" in data.files else 1,
        "n_detector_events": int(data["detector_events"].shape[0]) if "detector_events" in data.files else 0,
    }
    return out


def append_phase4a_partial_row(rec: dict) -> None:
    """每个 phase4a 任务跑完，把这一行 append 到 partial md。"""
    out_path = RESULTS_DIR / "multiseed_phase4a.partial.md"
    header = "| seed | dataset | overall_acc | post_drift_acc | n_routes | n_adapters | wall_time |\n"
    sep = "|---|---|---|---|---|---|---|\n"
    init = not out_path.exists()
    metrics = read_phase4a_metrics(rec["dataset"], rec["seed"]) or {}
    with open(out_path, "a") as fh:
        if init:
            fh.write("# Phase 4 A multi-seed partial summary (live)\n\n")
            fh.write("Increment-appended after each finished seed. Re-run safe (no header dedup needed if file exists).\n\n")
            fh.write(header)
            fh.write(sep)
        oa = metrics.get("overall_acc")
        pd = metrics.get("post_drift_acc")
        oa_s = f"{oa:.4f}" if oa is not None else "—"
        pd_s = f"{pd:.4f}" if pd is not None else "—"
        fh.write(
            f"| {rec['seed']} | {rec['dataset']} | {oa_s} | {pd_s} | "
            f"{metrics.get('n_routes', 0)} | {metrics.get('n_adapters', 1)} | "
            f"{rec['elapsed_sec']:.0f}s |\n"
        )


def run_one(task: tuple[str, str, int], log_dir: Path) -> dict:
    """跑一个 (config, dataset, seed)，返回 dict 含 status/elapsed/overall_acc。"""
    config, dataset, seed = task
    log_path = log_dir / f"{out_tag(config, dataset, seed)}.log"
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_full_cmd(config, dataset, seed)
    t0 = time.time()
    try:
        with open(log_path, "w") as fh:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"fail({proc.returncode})"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"exc:{type(e).__name__}"

    acc = read_overall_acc(config, dataset, seed)
    return {
        "config": config, "dataset": dataset, "seed": seed,
        "status": status, "elapsed_sec": elapsed, "overall_acc": acc,
        "log": str(log_path.relative_to(ROOT)),
    }


def write_partial_summary():
    """根据 results/ 下现存 multiseed_*.npz 写 partial summary 表格。"""
    rows = []
    for config in CONFIGS:
        for dataset in DATASETS:
            accs = []
            for s in SEEDS:
                a = read_overall_acc(config, dataset, s)
                if a is not None:
                    accs.append(a)
            if accs:
                arr = np.array(accs)
                rows.append({
                    "config": config, "dataset": dataset,
                    "n": len(accs), "mean": arr.mean(), "std": arr.std(ddof=1) if len(accs) > 1 else 0.0,
                    "values": accs,
                })

    out_path = RESULTS_DIR / "multiseed_summary.partial.md"
    with open(out_path, "w") as fh:
        fh.write("# Multi-seed partial summary (live)\n\n")
        fh.write(f"_Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
        fh.write("| Config | Dataset | n_seeds | overall_acc mean ± std |\n")
        fh.write("|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['config']} | {r['dataset']} | {r['n']}/5 | "
                     f"{r['mean']:.4f} ± {r['std']:.4f} |\n")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 0b: multi-seed driver")
    p.add_argument("--dry_run", action="store_true",
                   help="仅打印命令，不执行")
    p.add_argument("--n_parallel", type=int, default=1,
                   help="同一数据集内并行 seed 数（默认 1=串行）")
    p.add_argument("--configs", type=str, default=",".join(CONFIGS),
                   help=f"逗号分隔 config 子集，可选 {CONFIGS}")
    p.add_argument("--datasets", type=str, default=",".join(DATASETS),
                   help=f"逗号分隔 dataset 子集，可选 {DATASETS}")
    p.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)),
                   help="逗号分隔 seed 列表")
    return p.parse_args()


def main():
    args = parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = LOGS_DIR / "multiseed"

    # 计划：外层 dataset 串行，每个 dataset 内 (config × seeds) 池
    total = len(configs) * len(datasets) * len(seeds)
    print(f"=== Multi-seed driver ===")
    print(f"configs={configs}  datasets={datasets}  seeds={seeds}")
    print(f"total runs: {total}  parallelism within dataset: {args.n_parallel}")
    print(f"dry_run: {args.dry_run}")

    if args.dry_run:
        print("\n--- planned commands (dry-run) ---")
        n_skip = 0
        n_plan = 0
        for dataset in datasets:
            print(f"\n## dataset = {dataset}")
            for config in configs:
                for seed in seeds:
                    if npz_path(config, dataset, seed).exists():
                        print(f"  [skip] {npz_path(config, dataset, seed).name}")
                        n_skip += 1
                        continue
                    cmd = build_full_cmd(config, dataset, seed)
                    print("  " + " ".join(cmd))
                    n_plan += 1
        print(f"\n--- summary: {n_plan} runs to execute, {n_skip} skipped (already present) ---")
        return

    overall_t0 = time.time()
    all_records = []

    for dataset in datasets:
        print(f"\n========== dataset: {dataset} ==========")
        # 该 dataset 下所有待执行任务（跳过已存在的）
        tasks = []
        for config in configs:
            for seed in seeds:
                if npz_path(config, dataset, seed).exists():
                    print(f"  [skip] {npz_path(config, dataset, seed).name}")
                    # 仍然写入 partial 让 summary 能反映
                    continue
                tasks.append((config, dataset, seed))

        if not tasks:
            continue

        n_par = max(1, args.n_parallel)
        if n_par == 1:
            # 串行
            for task in tasks:
                rec = run_one(task, log_dir)
                all_records.append(rec)
                print(f"  [{rec['status']}] {rec['config']} / {rec['dataset']} / seed{rec['seed']} "
                      f"acc={rec['overall_acc']} elapsed={rec['elapsed_sec']:.0f}s log={rec['log']}")
                write_partial_summary()
        else:
            # 同 dataset 内并行
            with ProcessPoolExecutor(max_workers=n_par) as ex:
                futs = {ex.submit(run_one, t, log_dir): t for t in tasks}
                for fut in as_completed(futs):
                    rec = fut.result()
                    all_records.append(rec)
                    print(f"  [{rec['status']}] {rec['config']} / {rec['dataset']} / seed{rec['seed']} "
                          f"acc={rec['overall_acc']} elapsed={rec['elapsed_sec']:.0f}s log={rec['log']}")
                    write_partial_summary()
                    if rec["config"] == "phase4a":
                        append_phase4a_partial_row(rec)

    total_elapsed = time.time() - overall_t0
    print(f"\n=== ALL DONE in {total_elapsed/60:.1f} min ===")
    write_partial_summary()

    # 落盘运行记录
    rec_path = RESULTS_DIR / "multiseed_runlog.json"
    with open(rec_path, "w") as fh:
        json.dump(all_records, fh, indent=2)
    print(f"runlog: {rec_path}")


if __name__ == "__main__":
    main()
