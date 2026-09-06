# ROG 幻 14 运行手册（Phase 5.5）

> MacBook（M2 Pro）只能用 CPU：实测 TabPFN 在 Apple GPU (MPS) 上 **6.6 s/步**，
> 比 CPU 的 **1.2 s/步** 还慢 5.6×（部分算子无原生实现，要回退 CPU 再来回搬数据）。
> 92% 的耗时在 Transformer 前向，正是 NVIDIA 显卡擅长的部分，所以长实验搬到 ROG。
>
> **纪律：所有需要互相比较的运行必须在同一台机器上跑完。** GPU 与 CPU 的浮点结果
> 有细微差别，混跑会把机器差异算进方法差异里。既有的 Phase 1–5 结果是 MacBook CPU 跑的，
> 所以 Phase 5.5 的**全部**新运行（含新基线）都在 ROG 上重跑，不与旧数字直接相减。

---

## 0. 一次性环境搭建（Windows + WSL2）

建议用 WSL2 的 Ubuntu，而不是原生 Windows：省掉路径与编码的坑，NVIDIA 显卡在 WSL2 里可直接用。

```powershell
# PowerShell（管理员），装完会要求重启
wsl --install -d Ubuntu
```

重启后在 Ubuntu 里：

```bash
sudo apt update && sudo apt install -y python3.10 python3.10-venv python3-pip git
nvidia-smi        # 必须能看到显卡；看不到就先在 Windows 侧装/更新 NVIDIA 驱动
```

```bash
git clone https://github.com/wenboxia/neural-temporal-learning.git
cd neural-temporal-learning
python3.10 -m venv .venv && source .venv/bin/activate
```

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch
```

```bash
pip install -e ".[dev]" && pip install openml river
```

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
pytest tests/ -q        # 应 206 passed
```

**Insects 数据**：首次运行会从 Google Drive 下载约 14 MB 到 `~/.cache/insects_drift/`，
并校验 sha256。若下载失败，可从 MacBook 直接拷贝该目录。

**注意**：当前 `SlowPrior` 硬编码 `device="cpu"`，`MultiTimescaleModel` 也 assert 了这一点。
上 GPU 前需要先把 device 打通（`src/models/slow_prior.py:28` 与
`src/models/multi_timescale.py` 的 assert）。**这是 ROG 上的第一件事**，
改完先跑 `pytest tests/ -q` 再跑下面任何一批。

---

## 1. 先做一次 GPU 校准（15 分钟）

上规模之前先确认两件事：速度真的提升了，且结果与 CPU 一致到可接受范围。

```bash
python scripts/run_phase4_a.py --dataset insects --dataset_source real --segment_id d3_33240 --aligned_v2 --label_scheme pair_A_vs_B --context_size 200 --n_estimators 1 --max_eval_steps 1800 --detector_impl river --detector_input pred1 --action_on_alarm context_reset --trigger_source oracle --results_dir results/gpu_check
```

ROG 上的准确率应与 MacBook CPU 一致或极接近，耗时应明显更短。
**若准确率差异 > 0.5pp，先停下排查，不要开跑批量。**
（用 `d3_33240` 而不是 `d2_19500`：后者在 `pair_A_vs_B` 下漂移前是单类，加载器会拒绝，见 §6.5。）

---

## 2. 批次一：新基线（约 2–4 h）

目的：确认新的标签方案 + 官方变点居中的切段**有 headroom**，否则后面都白做。

```bash
python scripts/run_multiseed.py --dataset_source real --datasets insects --aligned_v2 --label_scheme pair_A_vs_B --segments d3_33240,d4_double,d0_control --configs phase1 --seeds 42 --n_parallel 2 --variant_tag v2AvsB_base --partial_tag _p55_base
```

```bash
python scripts/run_multiseed.py --dataset_source real --datasets insects --aligned_v2 --label_scheme pair_A_vs_B --segments d3_33240,d4_double,d0_control --configs phase1 --seeds 42 --n_parallel 2 --variant_tag v2AvsB_dual --partial_tag _p55_dual --extra_args "--context_loader dual --short_ratio 0.5 --long_max_age 2000"
```

**中止判据（不满足就停下汇报，不要往下跑）**：
- 总体准确率有 headroom（不能像旧的 `pair_parity` 那样 96–98%）；
- 至少 2 个含漂移的段，漂移后窗口准确率相对漂移前跌 **≥ 5pp**。

Phase 1 在真实数据上是确定性的（TabPFN 冻结、context 固定），所以 **1 个 seed 就够**，
不要跑 5 个 seed 浪费算力。

---

## 3. 批次二：判别性对照（最关键，约 6–10 h）

**这一批决定论文主线。** 它回答的是："报警之后做动作，到底值不值？"

```bash
python scripts/run_multiseed.py --dataset_source real --datasets insects --aligned_v2 --label_scheme pair_A_vs_B --segments d3_33240,d4_double,d0_control --configs phase4a --seeds 42 --n_parallel 2 --variant_tag v2AvsB_oracle_ctxreset --partial_tag _p55_or --extra_args "--detector_impl river --trigger_source oracle --action_on_alarm context_reset"
```

四个动作分支各跑一次（`--action_on_alarm` 换成 `route_adapter` / `buffer_clear` / `none`，
`--variant_tag` 相应改名），再把 `--trigger_source` 换成 `detector` 重跑一遍同样四个分支。

**检测输入也要扫**（Step 4 诊断结论，见 `todo.md` §5.1）：`--detector_input` 取
`pred1`（**首选**：召回 2/3、延迟最短、无需第二路预测）、`contrast_prob`（导师路径 A）、
`indicator`（现状对照，召回仅 1/3）。只在 `--trigger_source detector` 时有意义。
`none` + `oracle` 是关键对照组：它与 `none` + `detector` 应当完全一致，可用来验证
动作分派没有意外副作用。

**三种结局，各自的下一步**：

| 结局 | 含义 | 下一步 |
|---|---|---|
| oracle 即时触发也无增益 | 问题在**动作**本身 | 优化信号救不了；转维度 C（遗忘）作论文主线 |
| oracle 有效、实际报警无效 | 瓶颈是**检测时机** | 路径 A 才有意义，去压检测延迟 |
| 实际报警也有效 | 方法真的可行 | 直接扩规模，补 seeds |

分析时用 `shared_target_summary(..., reference_predictions=<phase1 同段预测>,
alarm_times=<npz 里的 alarm_events>)`，看 `errors_avoided_from_alarm` ——
这是"动作值不值"的直接数字，不依赖任何恢复阈值。

---

## 4. 批次三：三层激活对比（约 6–8 h）

批次二有增益才跑。phase4a（检测器驱动）vs phase1 vs 双记忆基线，3 个 seed。

```bash
python scripts/run_multiseed.py --dataset_source real --datasets insects --aligned_v2 --label_scheme pair_A_vs_B --segments d3_33240,d4_double,d0_control --configs phase1,phase3,phase4a --seeds 42,123,456 --n_parallel 2 --variant_tag v2AvsB_river_full --partial_tag _p55_full --extra_args "--detector_impl river"
```

---

## 5. 批次四：适应-遗忘前沿图（约 8–12 h，可与批次三并行）

**不依赖检测器是否触发**，所以不被前面任何批次阻塞。导师点名这是最可能出正面结果的点。

`--context_loader` 扫 `sliding` / `composite --fixed_ratio {0.33,0.67}` / `dual`，
每档都要同时跑"有校正"和"无校正"（`--action_on_alarm none`）两版 ——
否则分不清是先制造损失再补回，还是真有净增益。

回测用 `src/utils/forgetting.py`：`carve_holdout()` 先切留出集（**不能进 context /
buffer / 训练**），`ForgettingTracker` 在检查点回测，产出横轴适应、纵轴遗忘的前沿图。

---

## 6. 批次五：收口（约 6–8 h）

胜出配置补齐 5 seeds，做 paired t-test。若批次二/三有增益，此时再评估要不要投
45–60 工时做 6 类原生改造（见 `todo.md` §0 第 8 条）。

---

## 6.5 ⚠️ 只能用有效段

`--segments` **必须**显式写 `d3_33240,d4_double,d0_control`。
`d1_14352`（两种标签方案）与 `d2_19500`（`pair_A_vs_B`）漂移前是单一类别 ——
官方变点 14,352 落在一段 1,754 条连续 class 5 的末尾，那里温度漂移与标签构成变化完全混淆。
加载器默认会对这些组合直接报错，不必担心误用，但段列表写全五段会中途失败浪费时间。

---

## 7. 运行守则

- **每批跑完停下汇报**，不要连着开下一批。
- 长任务放后台，`run_multiseed.py` 会增量写 `results/*.partial.md`，随时可看进度。
- 已存在的 npz 会被自动跳过（可断点续跑）；**不同变体务必用不同 `--variant_tag`**，
  否则 skip-existing 会把新变体误认为已跑过而跳过。
- 用了 `--extra_args` 就**必须**显式给 `--variant_tag`，脚本会强制要求。
- `--extra_args` 只会传给认识该 flag 的脚本（`phase1` = `run_baselines.py` 没有
  detector/action 相关参数，会被自动过滤掉）。
- 结果 PNG 进 git，`.npz` 被 gitignore。
- **绝不 `git add -A`** —— 会把导师私人录音转写推到公开仓库。

---

## 8. 耗时估算

MacBook CPU 实测 1.2 s/步（`n_estimators=4`）、0.38 s/步（`n_estimators=1`）。
ROG 预计快 4–6×，下表按 4× 保守估计。

| 批次 | runs | MacBook CPU | ROG（估计） |
|---|---|---|---|
| 一 新基线 | 10 | 8–16 h | 2–4 h |
| 二 判别性对照 | 40 | 24–40 h | 6–10 h |
| 三 三层激活 | 45 | 24–32 h | 6–8 h |
| 四 前沿图 | 40 | 32–48 h | 8–12 h |
| 五 收口 | 30 | 24–32 h | 6–8 h |

**首次在 ROG 上跑完批次一后，用实测值回来更新这张表。**
