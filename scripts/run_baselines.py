"""
Phase 1 基线评估脚本

在合成数据集上运行 vanilla TabPFN（滑动窗口），
记录逐步预测误差，生成论文 Motivation 图：
  "TabPFN 在漂移点处误差骤增"

用法：
    cd neural_1
    python scripts/run_baselines.py --dataset rotating_boundary
    python scripts/run_baselines.py --dataset regime_switching
    python scripts/run_baselines.py --dataset combined_drift --n_samples 5000
"""

import argparse
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 使环境不依赖 display（MacBook 本地运行无需此行，Colab/服务器需要）
matplotlib.use("Agg")

# 将项目根目录加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.synthetic import make_dataset
from src.data.temporal_loader import TemporalWindowLoader
from src.models.slow_prior import SlowPrior
from src.utils.metrics import summarize_results, window_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="regime_switching",
        choices=["rotating_boundary", "regime_switching", "combined_drift"],
        help="合成数据集名称",
    )
    parser.add_argument("--n_samples", type=int, default=5000, help="样本总数")
    parser.add_argument("--n_features", type=int, default=10, help="特征维度（rotating_boundary 建议用 2）")
    parser.add_argument("--regime_length", type=int, default=500, help="每个体制持续的样本数（regime_switching 专用）")
    parser.add_argument("--n_regimes", type=int, default=3, help="体制数量（regime_switching 专用）")
    parser.add_argument("--drift_speed", type=float, default=0.003, help="决策边界旋转速度（rotating_boundary 专用）")
    parser.add_argument("--context_size", type=int, default=300, help="TabPFN 上下文窗口大小")
    parser.add_argument("--window_size", type=int, default=100, help="评估窗口大小")
    parser.add_argument("--n_estimators", type=int, default=4, help="TabPFN 集成数（越小越快，CPU 建议 4）")
    parser.add_argument("--max_eval_steps", type=int, default=None, help="最多评估多少步（调试用，None = 全量）")
    parser.add_argument("--results_dir", type=str, default="results", help="输出目录")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_tabpfn_baseline(args):
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"数据集: {args.dataset}")
    print(f"样本数: {args.n_samples} | 上下文大小: {args.context_size}")
    print(f"TabPFN n_estimators: {args.n_estimators}")
    print(f"{'='*60}\n")

    # 1. 生成数据
    print("正在生成合成数据...")
    kwargs = {"n_samples": args.n_samples, "random_seed": args.seed, "n_features": args.n_features}
    if args.dataset == "rotating_boundary":
        kwargs["drift_speed"] = args.drift_speed
    elif args.dataset == "regime_switching":
        kwargs["regime_length"] = args.regime_length
        kwargs["n_regimes"] = args.n_regimes
    dataset = make_dataset(args.dataset, **kwargs)

    print(f"  总样本数: {len(dataset.X)}")
    print(f"  特征维度: {dataset.X.shape[1]}")
    print(f"  类别分布: {np.bincount(dataset.y)}")
    print(f"  漂移点 ({len(dataset.drift_points)} 个): {dataset.drift_points[:10]}")

    # 2. 初始化加载器和模型
    loader = TemporalWindowLoader(
        dataset.X,
        dataset.y,
        context_size=args.context_size,
        step_size=1,
    )
    model = SlowPrior(device="cpu", n_estimators=args.n_estimators)

    # 3. Prequential 评估循环
    all_preds = []
    all_labels = []
    step_times = []

    total_steps = len(loader)
    if args.max_eval_steps is not None:
        total_steps = min(total_steps, args.max_eval_steps)

    print(f"\n开始 Prequential 评估（共 {total_steps} 步）...")
    print("注意：首步需要下载/加载 TabPFN 权重，可能需要几秒钟。\n")

    start_total = time.time()
    for i, batch in enumerate(loader):
        if i >= total_steps:
            break

        t0 = time.time()
        _, pred_label = model.predict(batch.X_ctx, batch.y_ctx, batch.X_query)
        step_times.append(time.time() - t0)

        all_preds.append(pred_label[0])
        all_labels.append(batch.y_query[0])

        # 进度提示
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - start_total
            avg_step = np.mean(step_times[-100:])
            remaining = avg_step * (total_steps - i - 1)
            print(
                f"  步 {i+1:5d}/{total_steps} | "
                f"当前准确率: {np.mean(np.array(all_preds) == np.array(all_labels)):.3f} | "
                f"均步耗时: {avg_step:.3f}s | "
                f"预计剩余: {remaining/60:.1f}min"
            )

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total_time = time.time() - start_total

    # 4. 计算指标
    offset = args.context_size
    results = summarize_results(
        all_preds,
        all_labels,
        drift_points=dataset.drift_points,
        window_size=args.window_size,
        offset=offset,
    )

    print(f"\n{'='*60}")
    print("评估结果 (TabPFN Baseline):")
    print(f"  总体 Prequential 准确率: {results['overall_acc']:.4f}")
    if results["pre_drift_acc"] is not None:
        print(f"  漂移前平均准确率:         {results['pre_drift_acc']:.4f}")
    if results["post_drift_acc"] is not None:
        print(f"  漂移后平均准确率:         {results['post_drift_acc']:.4f}")
    if results["avg_adaptation_speed"] is not None:
        print(f"  平均适应速度 (步):        {results['avg_adaptation_speed']:.1f}")
    print(f"  总耗时:                   {total_time:.1f}s")
    print(f"  均步耗时:                 {np.mean(step_times):.3f}s")
    print(f"{'='*60}\n")

    # 5. 绘图：误差随时间变化（论文 Motivation 图）
    win_accs = results["window_accs"]
    t_axis = np.arange(len(win_accs)) + offset + args.window_size // 2

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # 上图：滑动窗口准确率
    axes[0].plot(t_axis, win_accs, color="steelblue", linewidth=1.2, label="TabPFN (baseline)")
    axes[0].set_ylabel("Window Accuracy", fontsize=12)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(
        f"TabPFN Baseline on '{args.dataset}' — Drift Detection\n"
        f"(context_size={args.context_size}, window_size={args.window_size})",
        fontsize=13,
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 标注漂移点
    for dp in dataset.drift_points:
        if offset <= dp < offset + len(win_accs) + args.window_size:
            axes[0].axvline(dp, color="red", linestyle="--", alpha=0.5, linewidth=1)
    axes[0].axvline(dataset.drift_points[0] if dataset.drift_points else -1,
                    color="red", linestyle="--", alpha=0.5, linewidth=1, label="Drift Point")
    axes[0].legend(fontsize=10)

    # 下图：逐步误差（1 = 错误，0 = 正确），用平滑曲线展示
    error_series = (all_preds != all_labels).astype(float)
    # 用卷积平滑
    smooth_kernel = np.ones(args.window_size) / args.window_size
    smoothed_error = np.convolve(error_series, smooth_kernel, mode="valid")
    t_error = np.arange(len(smoothed_error)) + offset + args.window_size // 2

    axes[1].fill_between(t_error, smoothed_error, alpha=0.4, color="tomato", label="Error Rate (smoothed)")
    axes[1].plot(t_error, smoothed_error, color="tomato", linewidth=1)
    axes[1].set_ylabel("Error Rate", fontsize=12)
    axes[1].set_xlabel("Time Step", fontsize=12)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    for dp in dataset.drift_points:
        axes[1].axvline(dp, color="red", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    out_path = os.path.join(args.results_dir, f"baseline_{args.dataset}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"图表已保存至: {out_path}")

    # 6. 保存数值结果
    np.savez(
        os.path.join(args.results_dir, f"baseline_{args.dataset}.npz"),
        predictions=all_preds,
        labels=all_labels,
        drift_points=np.array(dataset.drift_points),
        window_accs=win_accs,
        overall_acc=np.array([results["overall_acc"]]),
    )
    print(f"数值结果已保存至: results/baseline_{args.dataset}.npz")

    return results


if __name__ == "__main__":
    args = parse_args()

    # rotating_boundary 默认用 2D 特征（方便可视化决策边界旋转）
    if args.dataset == "rotating_boundary" and args.n_features == 10:
        print("提示: rotating_boundary 自动使用 n_features=2 以便可视化，如需其他维度请显式指定 --n_features")
        args.n_features = 2

    run_tabpfn_baseline(args)
