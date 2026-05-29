"""
Phase 4 A 评估脚本：MultiTimescaleModel + ADWIN + AdapterLibrary

在 Phase 3C 脚本基础上启用 use_adapter_library=True，
保留窗口准确率 / 门控权重轨迹绘图，新增 detector_events / route_events / 各
adapter 使用次数等诊断字段落盘。

用法：
    cd neural_1
    python scripts/run_phase4_a.py --dataset regime_switching \
        --n_samples 3000 --context_size 200

    # 快速 smoke (~1-2 分钟)
    python scripts/run_phase4_a.py --dataset regime_switching \
        --max_eval_steps 100 --n_samples 3000
"""

import argparse
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.real_world import load_real_world
from src.data.synthetic import make_dataset
from src.data.temporal_loader import TemporalWindowLoader
from src.models.multi_timescale import MultiTimescaleModel
from src.utils.metrics import summarize_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 A: MultiTimescaleModel + ADWIN + AdapterLibrary"
    )

    # ── 基础参数（对齐 run_phase3.py）────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="regime_switching",
        choices=["rotating_boundary", "regime_switching", "combined_drift",
                 "electricity", "insects"],
    )
    parser.add_argument(
        "--dataset_source", type=str, default="synthetic",
        choices=["synthetic", "real"],
        help="数据来源：synthetic（默认，向后兼容）或 real（Phase 5）",
    )
    parser.add_argument(
        "--segment_id", type=str, default="start",
        choices=["start", "middle", "end",
                 "early", "mid", "late_pre", "late_post"],
        help="real 数据集 segment 选择（A+: start/middle/end；B1+ aligned: early/mid/late_pre/late_post）",
    )
    parser.add_argument(
        "--segment_size", type=int, default=5000,
        help="real 数据集的 segment 大小（默认 5000，A+ 协议；aligned 模式下被忽略）",
    )
    parser.add_argument(
        "--insects_aligned", action="store_true",
        help="Insects 用 4 个 drift-aligned segments（Phase 5 Stage B1+）；其他 dataset 忽略",
    )
    parser.add_argument("--n_samples", type=int, default=3000)
    parser.add_argument("--n_features", type=int, default=10,
                        help="rotating_boundary 自动改为 2")
    parser.add_argument("--regime_length", type=int, default=500)
    parser.add_argument("--n_regimes", type=int, default=3)
    parser.add_argument("--drift_speed", type=float, default=0.003)
    parser.add_argument("--context_size", type=int, default=200)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--n_estimators", type=int, default=4)
    parser.add_argument("--max_eval_steps", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_tag", type=str, default=None,
                        help="输出文件 stem（默认 phase4_a_{dataset}）")
    parser.add_argument("--seed", type=int, default=42)

    # ── Phase 3 既有参数 ────────────────────────────────────────────────
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--fast_method", type=str, default="knn",
                        choices=["knn", "ema"])
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--ema_alpha", type=float, default=0.15)
    parser.add_argument("--consolidation_threshold", type=float, default=0.05)
    parser.add_argument("--consolidation_window", type=int, default=50)
    parser.add_argument("--consolidation_epochs", type=int, default=10)
    parser.add_argument("--consolidation_cooldown", type=int, default=100)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # ── Phase 4 A 新增参数 ──────────────────────────────────────────────
    parser.add_argument("--max_adapters", type=int, default=8,
                        help="AdapterLibrary 容量上限")
    parser.add_argument("--library_fit_threshold", type=float, default=0.5,
                        help="route 时复用现有 adapter 的 MSE 上限（0.5 是 warmstart 默认；"
                             "v1 indicator run 用 0.05 导致 25/25 全 create 0 reuse）")
    parser.add_argument("--library_init_strategy", type=str, default="warm",
                        choices=["warm", "random"],
                        help="新 adapter 初始化策略：warm=自 active 复制（Day 1.5 默认），"
                             "random=永远随机初始化（Day 2 confound-busting）")
    parser.add_argument("--detector_delta", type=float, default=0.002,
                        help="ADWIN 置信参数（小=保守）")
    parser.add_argument("--detector_min_subwindow", type=int, default=30,
                        help="ADWIN 切点两侧最小子窗")
    parser.add_argument("--detector_cooldown", type=int, default=80,
                        help="ADWIN 漂移声明后冷却步数")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # rotating_boundary 必须用 n_features=2
    if args.dataset_source == "synthetic" and args.dataset == "rotating_boundary" and args.n_features == 10:
        args.n_features = 2

    print(f"\n{'='*60}")
    print(f"Phase 4 A: MultiTimescaleModel + ADWIN + AdapterLibrary")
    print(f"数据集: {args.dataset} | 样本数: {args.n_samples} | seed: {args.seed}")
    print(f"context_size: {args.context_size} | window_size: {args.window_size}")
    print(f"buffer_size: {args.buffer_size} | fast_method: {args.fast_method}")
    print(f"max_adapters: {args.max_adapters} | "
          f"library_fit_threshold: {args.library_fit_threshold} | "
          f"init_strategy: {args.library_init_strategy}")
    print(f"detector_delta: {args.detector_delta} | "
          f"min_subwindow: {args.detector_min_subwindow} | "
          f"cooldown: {args.detector_cooldown}")
    print(f"{'='*60}\n")

    # ── 生成 / 加载数据 ──────────────────────────────────────────────────
    if args.dataset_source == "real":
        aligned_tag = " [aligned]" if args.insects_aligned else ""
        print(f"加载真实数据集 {args.dataset} segment={args.segment_id}{aligned_tag}...")
        dataset = load_real_world(
            args.dataset, segment_id=args.segment_id, size=args.segment_size,
            insects_aligned=args.insects_aligned,
        )
        args.n_features = dataset.X.shape[1]
    else:
        kwargs = {
            "n_samples": args.n_samples,
            "random_seed": args.seed,
            "n_features": args.n_features,
        }
        if args.dataset == "rotating_boundary":
            kwargs["drift_speed"] = args.drift_speed
        elif args.dataset == "regime_switching":
            kwargs["regime_length"] = args.regime_length
            kwargs["n_regimes"] = args.n_regimes
        dataset = make_dataset(args.dataset, **kwargs)
    print(f"漂移点 ({len(dataset.drift_points)} 个): {dataset.drift_points}")

    loader = TemporalWindowLoader(
        dataset.X, dataset.y,
        context_size=args.context_size,
        step_size=1,
    )
    total_steps = len(loader)
    if args.max_eval_steps is not None:
        total_steps = min(total_steps, args.max_eval_steps)

    # ── 初始化 MultiTimescaleModel（开 use_adapter_library）──────────────
    model = MultiTimescaleModel(
        input_dim=args.n_features,
        buffer_size=args.buffer_size,
        fast_method=args.fast_method,
        knn_k=args.knn_k,
        ema_alpha=args.ema_alpha,
        consolidation_threshold=args.consolidation_threshold,
        consolidation_window=args.consolidation_window,
        consolidation_epochs=args.consolidation_epochs,
        consolidation_cooldown=args.consolidation_cooldown,
        gate_hidden_dim=args.gate_hidden_dim,
        lr=args.lr,
        device="cpu",
        n_estimators=args.n_estimators,
        # ── Phase 4 A ──
        use_adapter_library=True,
        max_adapters=args.max_adapters,
        library_fit_threshold=args.library_fit_threshold,
        library_init_strategy=args.library_init_strategy,
        detector_delta=args.detector_delta,
        detector_min_subwindow=args.detector_min_subwindow,
        detector_cooldown=args.detector_cooldown,
    )

    # ── Prequential 主循环 ──────────────────────────────────────────────
    predictions: list = []
    labels: list = []
    gate_weights_trajectory: list = []
    active_id_trajectory: list = []

    print(f"开始评估（共 {total_steps} 步）...")
    t0 = time.time()

    for i, batch in enumerate(loader):
        if i >= total_steps:
            break

        x_t = batch.X_query[0]
        y_t = int(batch.y_query[0])

        pred, weights = model.step(
            batch.X_ctx, batch.y_ctx, x_t, float(y_t), t=batch.t
        )

        predictions.append(int(pred >= 0.5))
        labels.append(y_t)
        gate_weights_trajectory.append(weights)
        active_id_trajectory.append(model.adapter_library.active_id)

        if (i + 1) % 500 == 0:
            acc = float(np.mean(np.array(predictions) == np.array(labels)))
            print(f"  步 {i+1:5d}/{total_steps} | acc: {acc:.3f} | "
                  f"detect: {len(model.detector_events)} | "
                  f"routes: {len(model.route_events)} | "
                  f"adapters: {model.adapter_library.n_adapters()}")

    elapsed = time.time() - t0

    preds_arr = np.array(predictions)
    labels_arr = np.array(labels)
    gate_weights_arr = np.stack(gate_weights_trajectory, axis=0)
    active_id_arr = np.array(active_id_trajectory, dtype=np.int32)
    consolidation_events = model.consolidation_events
    detector_events = model.detector_events
    # route_events: list[(t, action_str, active_id)] — 拆 3 列存 npz
    if model.route_events:
        route_t = np.array([r[0] for r in model.route_events], dtype=np.int64)
        route_action = np.array([r[1] for r in model.route_events])
        route_active_id = np.array([r[2] for r in model.route_events], dtype=np.int32)
    else:
        route_t = np.array([], dtype=np.int64)
        route_action = np.array([], dtype="<U10")
        route_active_id = np.array([], dtype=np.int32)

    # ── 指标 ─────────────────────────────────────────────────────────────
    results = summarize_results(
        preds_arr, labels_arr,
        drift_points=dataset.drift_points,
        window_size=args.window_size,
        offset=args.context_size,
    )
    win_accs = results["window_accs"]

    speed_str = (f"{results['avg_adaptation_speed']:.1f}"
                 if results["avg_adaptation_speed"] is not None else "N/A")

    print(f"\n--- Phase 4 A 结果 ---")
    print(f"  总体准确率: {results['overall_acc']:.4f} | "
          f"漂移前: {results['pre_drift_acc'] or 0:.4f} | "
          f"漂移后: {results['post_drift_acc'] or 0:.4f} | "
          f"适应速度: {speed_str} | 耗时: {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print("Phase 4 A 汇总：")
    print(f"{'指标':<20} {'值':>12}")
    print("-" * 35)
    print(f"{'总体准确率':<20} {results['overall_acc']:>12.4f}")
    print(f"{'Balanced Accuracy':<20} {results['balanced_acc']:>12.4f}")
    auc_str = f"{results['auc_roc']:.4f}" if results["auc_roc"] is not None else "N/A"
    print(f"{'AUC-ROC':<20} {auc_str:>12}")
    print(f"{'漂移前准确率':<20} {(results['pre_drift_acc'] or 0):>12.4f}")
    print(f"{'漂移后准确率':<20} {(results['post_drift_acc'] or 0):>12.4f}")
    print(f"{'适应速度 (步)':<20} {speed_str:>12}")
    print(f"{'='*60}")
    print(f"detector 触发次数: {len(detector_events)}")
    print(f"route 事件数:      {len(model.route_events)}")
    print(f"consolidation 数:  {len(consolidation_events)}")
    print(f"最终 adapter 数:   {model.adapter_library.n_adapters()}")
    print(f"adapter usage:     {dict(model.adapter_library.usage)}")
    if model.route_events:
        actions = [r[1] for r in model.route_events]
        from collections import Counter
        print(f"route action 分布: {dict(Counter(actions))}")

    # ── 绘图（与 phase3 同结构 + 漂移检测/路由叠加）───────────────────────
    offset = args.context_size
    t_win = np.arange(len(win_accs)) + offset + args.window_size // 2
    t_gate = np.arange(len(gate_weights_arr)) + offset

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # 上图：窗口准确率 + 漂移点 + detector / route 标记
    axes[0].plot(t_win, win_accs, color="steelblue", linewidth=1.8,
                 label="Phase 4 A", alpha=0.9)
    for dp in dataset.drift_points:
        axes[0].axvline(dp, color="red", linestyle="--", alpha=0.4, linewidth=1)
    axes[0].axvline(-1, color="red", linestyle="--", alpha=0.4, linewidth=1,
                    label="True Drift")
    for de in detector_events:
        axes[0].axvline(de, color="purple", linestyle=":", alpha=0.5, linewidth=1)
    if detector_events:
        axes[0].axvline(-1, color="purple", linestyle=":", alpha=0.5, linewidth=1,
                        label="Detector")
    for rt, action, _ in model.route_events:
        color = {"create": "darkgreen", "switch": "orange", "reuse": "gray"}.get(
            action, "black"
        )
        axes[0].axvline(rt, color=color, linestyle="-", alpha=0.6, linewidth=1.2)
    axes[0].set_ylabel("Window Accuracy")
    axes[0].set_ylim(0.4, 1.05)
    axes[0].set_title(
        f"Phase 4 A on '{args.dataset}' (seed={args.seed})\n"
        f"context={args.context_size}, max_adapters={args.max_adapters}, "
        f"detector_δ={args.detector_delta}",
        fontsize=12,
    )
    axes[0].legend(fontsize=9, loc="lower left")
    axes[0].grid(True, alpha=0.3)

    # 中图：门控权重 α/β/γ
    axes[1].plot(t_gate, gate_weights_arr[:, 0], color="steelblue",
                 linewidth=1.4, label="α (slow)", alpha=0.9)
    axes[1].plot(t_gate, gate_weights_arr[:, 1], color="darkorange",
                 linewidth=1.4, label="β (inter)", alpha=0.9)
    axes[1].plot(t_gate, gate_weights_arr[:, 2], color="seagreen",
                 linewidth=1.4, label="γ (fast)", alpha=0.9)
    for dp in dataset.drift_points:
        axes[1].axvline(dp, color="red", linestyle="--", alpha=0.3, linewidth=1)
    axes[1].set_ylabel("Gate Weight")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # 下图：active adapter id 轨迹
    axes[2].step(t_gate, active_id_arr, where="post",
                 color="darkgreen", linewidth=1.6, label="active adapter id")
    for dp in dataset.drift_points:
        axes[2].axvline(dp, color="red", linestyle="--", alpha=0.3, linewidth=1)
    for de in detector_events:
        axes[2].axvline(de, color="purple", linestyle=":", alpha=0.4, linewidth=1)
    axes[2].set_ylabel("Active Adapter ID")
    axes[2].set_xlabel("Time Step")
    max_id = int(active_id_arr.max()) if active_id_arr.size else 0
    axes[2].set_yticks(list(range(max_id + 1)))
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    stem = args.out_tag if args.out_tag is not None else f"phase4_a_{args.dataset}"
    png_path = os.path.join(args.results_dir, f"{stem}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存至: {png_path}")

    # ── 保存数值结果 ─────────────────────────────────────────────────────
    npz_path = os.path.join(args.results_dir, f"{stem}.npz")
    np.savez(
        npz_path,
        predictions=preds_arr,
        labels=labels_arr,
        drift_points=np.array(dataset.drift_points),
        window_accs=win_accs,
        overall_acc=np.array([results["overall_acc"]]),
        balanced_acc=np.array([results["balanced_acc"]]),
        auc_roc=np.array([results["auc_roc"] if results["auc_roc"] is not None else np.nan]),
        pre_drift_acc=np.array([results["pre_drift_acc"] or np.nan]),
        post_drift_acc=np.array([results["post_drift_acc"] or np.nan]),
        avg_adaptation_speed=np.array(
            [results["avg_adaptation_speed"] if results["avg_adaptation_speed"] is not None else np.nan]
        ),
        gate_weights_trajectory=gate_weights_arr,
        active_id_trajectory=active_id_arr,
        consolidation_events=np.array(consolidation_events),
        detector_events=np.array(detector_events),
        route_t=route_t,
        route_action=route_action,
        route_active_id=route_active_id,
        n_adapters_final=np.array([model.adapter_library.n_adapters()]),
        seed=np.array([args.seed]),
        abs_error_history=np.array(model.abs_error_history, dtype=np.float32),
        indicator_history=np.array(model.indicator_history, dtype=np.int8),
        n_warmstart_inits=np.array([model.adapter_library.n_warmstart_inits]),
        n_random_inits=np.array([model.adapter_library.n_random_inits]),
        library_fit_threshold=np.array([args.library_fit_threshold]),
        library_init_strategy=np.array([args.library_init_strategy]),
    )
    print(f"数值结果已保存至: {npz_path}")


if __name__ == "__main__":
    main()
