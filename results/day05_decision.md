# Phase 4 Day 0.5 — 决策记录

**日期**：2026-04-27
**输入**：`results/oracle_summary.md`（实验 0a） + `results/multiseed_summary.md`（实验 0b）
**决策**：**Design A — Regime Detection + Adapter Library**

---

## 决策表对位（来自 phase4_plan.md）

| Oracle 总体准确率 (mean) | Phase 3 vs Phase 1 on rotating_boundary | → 走哪个 Design |
|---|---|---|
| ≥ 82% | 任意 | E |
| **80-82%** | **差异在 std 之外（显著）** | **A** ✓ 本次落点 |
| 80-82% | 差异在 std 之内（噪声） | E |
| < 80% | 任意 | A |

### 本次取值

- **Oracle 总体准确率 mean = 80.39%**（5 seeds, regime_switching, reset_size=50）→ 落 80-82% 中段
- **Phase 3 vs Phase 1 on rotating_boundary = +1.00 pp**, paired std=0.55pp, **paired t=+4.07**, df=4 → 显著超过临界 2.78（p≈0.015 双侧），5/5 seeds 同向

→ 决策表行 "80-82% × 显著" → **Design A**

---

## 选定 Design A，因为：

### 1. Oracle 杠杆中等而非压倒性
Oracle context-reset 只把 regime_switching 总体准确率从 79.89% 推到 80.39%（+0.51 pp 显著但幅度小）。漂移点处的 -12-13 pp 退步只有 ~1/3 由 context 污染造成，剩下 ~2/3 来自新 regime 上 in-context learning 本身的样本不足。**Context 是杠杆，但不足以单独支撑全部改进**——这与决策表 "≥82% 走 Design E" 的临界不符。

### 2. Phase 3 v2+B+F 的 adapter 路线在 rotating_boundary 上有真信号
multi-seed +1.00 pp 显著（paired t=+4.07，5/5 同向），证伪了 "Phase 3 全军覆没、adapter 没用" 的悲观假设。adapter library 路线**有抢救价值**，per-regime 隔离设计目标对路。

### 3. Phase 3 在 combined_drift 上 -0.47 pp 显著退步揭示当前架构问题
共享单一 adapter 的 v2+B+F 设计无法处理"逐步偏移 + 周期 boundary 变化"这种混合漂移：adapter 一直被多 regime 数据训练造成模糊。**Design A 的 per-regime 字典正面回应这个问题**——每个 adapter 只对 active regime 学习，combined_drift 上路由会随 boundary 变化切换，避免参数模糊。

### 4. 与论文 framing 一致
按 phase4_plan.md 论文章节："Per-Regime Adapter Libraries for Concept Drift on Frozen Tabular Foundation Models"。Phase 3 在 rotating_boundary 上的 +1pp 显著正向 + combined_drift 上的负向退步，正好作为 "为什么共享 adapter 失败、为什么需要 per-regime 隔离" 的设计动机。

---

## 修正先前结论

multi-seed 重跑还修正了 Phase 3 进度报告里一处误判：

> Phase 3 进度报告说 "Phase 3 v2+B+F 在 regime_switching 上 -4.5pp 退步"

实际多 seed 取值：Phase 3 79.71 ± 0.72%，Phase 1 79.89 ± 0.99%，paired t=-1.27（NS）。**Phase 3 在 regime_switching 上不退步，只是没改进，差值在噪声内**。原 -4.5pp 是 single-seed 不利点 + std 累积造成的过判。

→ Phase 3 真正的 trade-off 模式：
- rotating_boundary：**显著 +1.00pp**（adapter 学到 boundary 旋转 pattern）
- regime_switching：噪声内（adapter 跨 regime 学不到稳定 pattern）
- combined_drift：**显著 -0.47pp**（多 regime 共享单 adapter 反作用）

---

## 下一步（Phase 4 Day 1.5+）

按 plan 走 Design A 实施路径：

| 步骤 | 待新建文件 | 重点 |
|---|---|---|
| 1 | `src/drift/error_detector.py` | ADWIN（自写或 import river） |
| 2 | `src/regime/adapter_library.py` | `dict[int, MLP]` + 路由 |
| 3 | `src/models/multi_timescale.py`（修改） | 替换共享 adapter 为 library，consolidation 仅训练 active adapter |
| 4 | `scripts/run_phase4_a.py` | 实验入口 |
| 5 | 三数据集 × 5 seeds × Design A | 15 runs |
| 6 | `results/phase4_a_summary.md` | mean ± std + paired t vs Phase 1 / Phase 3 |

验收（来自 plan）：
- regime_switching: Design A ≥ Phase 1 + 2pp（mean，且 std 内显著）
- rotating_boundary: Design A ≥ Phase 1（不退步），目标 ≥ Phase 3
- combined_drift: Design A ≥ Phase 1 - 1pp（即至少修复 v2+B+F 的退步）

**实施暂停**：Day 0.5 到此结束，等用户确认 Design A 决策后再启动 Day 1.5。
