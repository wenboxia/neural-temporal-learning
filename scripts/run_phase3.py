"""
Phase 3C 评估脚本：三时间尺度编排器（MultiTimescaleModel）

此脚本属于 Phase 3C，用单一 MultiTimescaleModel 接口完成 prequential 评估，
并可视化窗口准确率 + 门控权重轨迹，保存 .png 和 .npz 结果。

用法：
    cd neural_1
    python scripts/run_phase3.py --dataset regime_switching \
        --n_samples 3000 --context_size 200

    # 快速调试（~100 步）
    python scripts/run_phase3.py --dataset regime_switching \
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

from src.data.synthetic import make_dataset
from src.data.temporal_loader import TemporalWindowLoader
from src.models.multi_timescale import MultiTimescaleModel
from src.utils.metrics import summarize_results, window_accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3C: MultiTimescaleModel (Level1 + Level2 + Level3) Evaluation"
    )

    # ── 基础参数（对齐 run_phase2.py）────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default="regime_switching",
        choices=["rotating_boundary", "regime_switching", "combined_drift"],
        help="合成数据集类型",
    )
    parser.add_argument("--n_samples", type=int, default=3000,
                        help="生成序列总长度")
    parser.add_argument("--n_features", type=int, default=10,
                        help="特征维度（rotating_boundary 自动改为 2）")
    parser.add_argument("--regime_length", type=int, default=500,
                        help="regime_switching 每段体制的长度")
    parser.add_argument("--n_regimes", type=int, default=3,
                        help="regime_switching 体制数量")
    parser.add_argument("--drift_speed", type=float, default=0.003,
                        help="rotating_boundary 边界旋转速度（弧度/步）")
    parser.add_argument("--context_size", type=int, default=200,
                        help="喂给 TabPFN 的上下文窗口大小（≤3000）")
    parser.add_argument("--window_size", type=int, default=100,
                        help="计算窗口准确率的滑动窗口大小")
    parser.add_argument("--n_estimators", type=int, default=4,
                        help="TabPFN 集成数量")
    parser.add_argument("--max_eval_steps", type=int, default=None,
                        help="最多评估步数（None = 跑完全部）")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="结果输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # ── Phase 3 特定参数 ──────────────────────────────────────────────────
    parser.add_argument("--buffer_size", type=int, default=100,
                        help="FastCorrector 工作记忆容量")
    parser.add_argument("--fast_method", type=str, default="knn",
                        choices=["knn", "ema"],
                        help="快速校正方法（knn 或 ema）")
    parser.add_argument("--knn_k", type=int, default=5,
                        help="KNN 近邻数")
    parser.add_argument("--ema_alpha", type=float, default=0.15,
                        help="EMA 平滑系数")
    parser.add_argument("--consolidation_threshold", type=float, default=0.05,
                        help="触发巩固的最小平均误差绝对值")
    parser.add_argument("--consolidation_window", type=int, default=50,
                        help="巩固观察窗口大小（也用于 FastToInterConsolidation）")
    parser.add_argument("--consolidation_epochs", type=int, default=10,
                        help="每次巩固执行的梯度更新步数")
    parser.add_argument("--consolidation_cooldown", type=int, default=100,
                        help="两次巩固之间的最小间隔步数（防 thrashing）")
    parser.add_argument("--gate_hidden_dim", type=int, default=64,
                        help="GatedEnsemble gate/adapter 隐藏层宽度")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="GatedEnsemble Adam 学习率")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # rotating_boundary 必须用 n_features=2（与 run_phase2 保持一致）
    if args.dataset == "rotating_boundary" and args.n_features == 10:
        args.n_features = 2

    print(f"\n{'='*60}")
    print(f"Phase 3C: MultiTimescaleModel（三时间尺度编排器）")
    print(f"数据集: {args.dataset} | 样本数: {args.n_samples}")
    print(f"context_size: {args.context_size} | window_size: {args.window_size}")
    print(f"buffer_size: {args.buffer_size} | fast_method: {args.fast_method} | "
          f"knn_k: {args.knn_k} | ema_α: {args.ema_alpha}")
    print(f"gate_hidden_dim: {args.gate_hidden_dim} | lr: {args.lr}")
    print(f"consolidation_threshold: {args.consolidation_threshold} | "
          f"consolidation_window: {args.consolidation_window} | "
          f"consolidation_epochs: {args.consolidation_epochs} | "
          f"consolidation_cooldown: {args.consolidation_cooldown}")
    print(f"{'='*60}\n")

    # ── 生成数据 ──────────────────────────────────────────────────────────
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

    # ── 初始化 MultiTimescaleModel ────────────────────────────────────────
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
    )

    # ── Prequential 主循环 ────────────────────────────────────────────────
    predictions: list = []
    labels: list = []
    gate_weights_trajectory: list = []

    print(f"开始评估（共 {total_steps} 步）...")
    t0 = time.time()

    for i, batch in enumerate(loader):
        if i >= total_steps:
            break

        x_t = batch.X_query[0]           # (n_features,)
        y_t = int(batch.y_query[0])

        pred, weights = model.step(
            batch.X_ctx, batch.y_ctx, x_t, float(y_t), t=batch.t
        )
        # pred: float ∈ [0,1]；weights: np.ndarray (3,) = [α_slow, β_inter, γ_fast]

        predictions.append(int(pred >= 0.5))
        labels.append(y_t)
        gate_weights_trajectory.append(weights)

        if (i + 1) % 500 == 0:
            acc = float(np.mean(np.array(predictions) == np.array(labels)))
            print(f"  步 {i+1:5d}/{total_steps} | 准确率: {acc:.3f}")

    elapsed = time.time() - t0

    preds_arr = np.array(predictions)
    labels_arr = np.array(labels)
    gate_weights_arr = np.stack(gate_weights_trajectory, axis=0)  # (T, 3)
    consolidation_events: list = model.consolidation_events         # list[int]

    # ── 计算指标 ─────────────────────────────────────────────────────────
    results = summarize_results(
        preds_arr, labels_arr,
        drift_points=dataset.drift_points,
        window_size=args.window_size,
        offset=args.context_size,
    )
    win_accs = results["window_accs"]

    print(f"\n--- Phase 3C 结果 ---")
    speed_str = (f"{results['avg_adaptation_speed']:.1f}"
                 if results["avg_adaptation_speed"] is not None else "N/A")
    print(f"  总体准确率: {results['overall_acc']:.4f} | "
          f"漂移前: {results['pre_drift_acc'] or 0:.4f} | "
          f"漂移后: {results['post_drift_acc'] or 0:.4f} | "
          f"适应速度: {speed_str} | "
          f"耗时: {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print("Phase 3C 汇总：")
    print(f"{'指标':<20} {'值':>12}")
    print("-" * 35)
    print(f"{'总体准确率':<20} {results['overall_acc']:>12.4f}")
    print(f"{'Balanced Accuracy':<20} {results['balanced_acc']:>12.4f}")
    auc_str = f"{results['auc_roc']:.4f}" if results["auc_roc"] is not None else "N/A"
    print(f"{'AUC-ROC':<20} {auc_str:>12}")
    print(f"{'漂移前准确率':<20} {(results['pre_drift_acc'] or 0):>12.4f}")
    print(f"{'漂移后准确率':<20} {(results['post_drift_acc'] or 0):>12.4f}")
    print(f"{'适应速度 (步)':<20} {speed_str:>12}")
    print(f"{'='*60}\n")
    print(f"巩固事件数：{len(consolidation_events)}")

    # ── 绘图 ──────────────────────────────────────────────────────────────
    offset = args.context_size
    t_win = np.arange(len(win_accs)) + offset + args.window_size // 2
    t_gate = np.arange(len(gate_weights_arr)) + offset

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # 上图：窗口准确率 + 漂移点 + 巩固事件
    axes[0].plot(t_win, win_accs, color="steelblue", linewidth=1.8,
                 label="MultiTimescaleModel", alpha=0.9)

    for dp in dataset.drift_points:
        axes[0].axvline(dp, color="red", linestyle="--", alpha=0.4, linewidth=1)
    axes[0].axvline(-1, color="red", linestyle="--", alpha=0.4, linewidth=1,
                    label="Drift Point")

    for ce in consolidation_events:
        axes[0].axvline(ce, color="green", linestyle="-",
                        alpha=0.4, linewidth=1)
    if consolidation_events:
        # 为图例添加一条代表性巩固线
        axes[0].axvline(-1, color="green", linestyle="-", alpha=0.4,
                        linewidth=1, label="Consolidation Event")

    axes[0].set_ylabel("Window Accuracy", fontsize=12)
    axes[0].set_ylim(0.4, 1.05)
    axes[0].set_title(
        f"Phase 3C: MultiTimescaleModel on '{args.dataset}'\n"
        f"(context={args.context_size}, buffer={args.buffer_size}, "
        f"method={args.fast_method}, gate_hidden={args.gate_hidden_dim})",
        fontsize=13,
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 下图：三条门控权重曲线 α/β/γ
    axes[1].plot(t_gate, gate_weights_arr[:, 0], color="steelblue",
                 linewidth=1.5, label="α (slow)", alpha=0.9)
    axes[1].plot(t_gate, gate_weights_arr[:, 1], color="darkorange",
                 linewidth=1.5, label="β (inter)", alpha=0.9)
    axes[1].plot(t_gate, gate_weights_arr[:, 2], color="seagreen",
                 linewidth=1.5, label="γ (fast)", alpha=0.9)

    for dp in dataset.drift_points:
        axes[1].axvline(dp, color="red", linestyle="--", alpha=0.4, linewidth=1)
    for ce in consolidation_events:
        axes[1].axvline(ce, color="green", linestyle="-",
                        alpha=0.4, linewidth=1)

    axes[1].set_ylabel("Gate Weight", fontsize=12)
    axes[1].set_xlabel("Time Step", fontsize=12)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(args.results_dir, f"phase3_{args.dataset}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存至: {png_path}")

    # ── 保存数值结果 ──────────────────────────────────────────────────────
    npz_path = os.path.join(args.results_dir, f"phase3_{args.dataset}.npz")
    np.savez(
        npz_path,
        predictions=preds_arr,
        labels=labels_arr,
        drift_points=np.array(dataset.drift_points),
        window_accs=win_accs,
        overall_acc=np.array([results["overall_acc"]]),
        gate_weights_trajectory=gate_weights_arr,
        consolidation_events=np.array(consolidation_events),
    )
    print(f"数值结果已保存至: {npz_path}")


if __name__ == "__main__":
    main()
