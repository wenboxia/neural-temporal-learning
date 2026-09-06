#!/usr/bin/env bash
# 打包一个**离线自足**的实验包，拷到 ROG 幻 14 上直接跑。
#
# 为什么要自带数据和权重：ROG 那边没有 AI 协助，也不保证能联网下载。
# TabPFN 首次调用会下载 ~41MB 权重，Insects 数据要从 Google Drive 取 ~14MB，
# 两样都打进包里，用环境变量指向，跑的时候就不需要网络。
#
# 用法（在 MacBook 的项目根目录）：
#     bash scripts/make_rog_bundle.sh
# 产出：  ../rog_bundle_<日期>.zip   （约 60MB）

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
STAMP="$(date +%Y%m%d)"
OUT="$ROOT/../rog_bundle_${STAMP}"
ZIP="${OUT}.zip"

echo "=== 1/5 清理旧包 ==="
rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"/{code,data,weights,results_out}

echo "=== 2/5 代码 ==="
for d in src scripts tests configs; do cp -R "$d" "$OUT/code/"; done
cp pyproject.toml "$OUT/code/"
cp ROG_RUNBOOK.md "$OUT/" 2>/dev/null || true
find "$OUT/code" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find "$OUT/code" -name '.DS_Store' -delete 2>/dev/null || true
echo "  code: $(du -sh "$OUT/code" | cut -f1)"

echo "=== 3/5 Insects 数据（免去 Google Drive 下载）==="
SRC_CSV="$HOME/.cache/insects_drift/abrupt_balanced.csv"
if [ -f "$SRC_CSV" ]; then cp "$SRC_CSV" "$OUT/data/"; echo "  data: $(du -sh "$OUT/data" | cut -f1)"
else echo "  !! 找不到 $SRC_CSV —— ROG 上首次运行需要联网下载"; fi

echo "=== 4/5 TabPFN 权重（免去首次运行下载）==="
CACHE=""
for c in "$HOME/Library/Caches/tabpfn" "$HOME/.cache/tabpfn"; do [ -d "$c" ] && CACHE="$c" && break; done
if [ -n "$CACHE" ]; then cp -R "$CACHE"/* "$OUT/weights/"; echo "  weights: $(du -sh "$OUT/weights" | cut -f1)"
else echo "  !! 找不到 TabPFN 权重缓存 —— ROG 上首次运行需要联网下载"; fi

echo "=== 5/5 生成 ROG 上的一键脚本 ==="

cat > "$OUT/SETUP.md" <<'MD'
# ROG 幻 14 —— 照着做就行

> 这个包是**离线自足**的：数据和 TabPFN 权重都已打包，跑实验不需要联网。
> 只有安装 Python 依赖那一步需要网络（只做一次）。
>
> 全程复制粘贴命令即可，不需要判断。**每一批跑完，把 `results_out/` 打包发回。**

## 第 0 步：只做一次的环境安装（需要联网）

在 Windows 上装 WSL2 的 Ubuntu（PowerShell 管理员权限，装完会要求重启）：

```powershell
wsl --install -d Ubuntu
```

重启后打开 Ubuntu，确认能看到显卡：

```bash
nvidia-smi
```

> 看不到显卡 → 先在 Windows 侧装/更新 NVIDIA 驱动，再回来。

把这个包放到 Ubuntu 能访问的地方，然后：

```bash
cd ~/rog_bundle_YYYYMMDD          # 换成实际目录名
bash run_setup.sh
```

这一步会建虚拟环境、装 PyTorch（CUDA 版）和依赖、放好数据与权重、跑一遍测试。
**看到最后打印 `SETUP OK` 才算成功。**

## 第 1 步：GPU 校准（约 15 分钟）

```bash
bash run_batch.sh calib
```

对照 `results_out/calib_report.txt`：准确率应与 MacBook 接近。
**若准确率差异超过 0.5 个百分点，先停下，把报告发回，不要继续。**

## 第 2 步：批次一（约 3–5 小时）

```bash
bash run_batch.sh batch1
```

跑完看 `results_out/batch1_report.txt` 里的"判据"两行。
**两条都是 PASS 才继续下一批**；否则把报告发回。

## 第 3 步：批次二（约 6–10 小时，最关键）

```bash
bash run_batch.sh batch2
```

这一批决定论文主线。跑完**一定把结果发回再继续**。

## 怎么把结果发回

```bash
bash pack_results.sh
```

会生成 `results_to_send_<日期>.zip`，把它发回即可。
里面只有数字、图和日志，体积很小。

## 出问题了怎么办

任何一步报错，把屏幕上的完整报错文字连同 `results_out/` 一起发回。
不要自己改命令重试。
MD

cat > "$OUT/run_setup.sh" <<'SH'
#!/usr/bin/env bash
# 一次性环境安装。需要联网。
set -euo pipefail
cd "$(dirname "$0")"
BASE="$(pwd)"

echo "[1/6] 检查显卡"
nvidia-smi -L || { echo "!! 看不到 NVIDIA 显卡，先装驱动"; exit 1; }

echo "[2/6] 建虚拟环境"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip

echo "[3/6] 装 PyTorch (CUDA 12.1)"
pip install -q --index-url https://download.pytorch.org/whl/cu121 torch

echo "[4/6] 装项目依赖"
pip install -q -e ./code
pip install -q river openml

echo "[5/6] 放置数据与权重（免联网下载）"
mkdir -p "$HOME/.cache/insects_drift"
[ -f data/abrupt_balanced.csv ] && cp data/abrupt_balanced.csv "$HOME/.cache/insects_drift/" && echo "  Insects 数据就位"
mkdir -p "$BASE/weights"
echo "export TABPFN_MODEL_CACHE_DIR=$BASE/weights" > "$BASE/.env_tabpfn"
echo "  TabPFN 权重目录: $BASE/weights"

echo "[6/6] 跑测试"
cd code && python -m pytest -q tests/ 2>&1 | tail -3 && cd ..

python - <<'PY'
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available(): print(f"显卡: {torch.cuda.get_device_name(0)}")
PY

echo
echo "SETUP OK"
SH

cat > "$OUT/run_batch.sh" <<'SH'
#!/usr/bin/env bash
# 跑一批实验。用法: bash run_batch.sh {calib|batch1|batch2}
set -euo pipefail
cd "$(dirname "$0")"
BASE="$(pwd)"
source .venv/bin/activate
source .env_tabpfn
export PYTHONPATH="$BASE/code"
cd code
mkdir -p "$BASE/results_out"

SEGS_A="d3_33240,d4_double,d0_control"
SEGS_P="d2_19500,d3_33240,d0_control"
LOG="$BASE/results_out/$1_console.log"

case "${1:-}" in
calib)
  echo "=== GPU 校准（约 15 分钟）===" | tee "$LOG"
  python scripts/run_phase4_a.py --dataset insects --dataset_source real \
    --segment_id d3_33240 --aligned_v2 --label_scheme pair_A_vs_B \
    --context_size 200 --n_estimators 1 --max_eval_steps 1800 \
    --detector_impl river --detector_input pred1 \
    --action_on_alarm context_reset --trigger_source oracle \
    --results_dir "$BASE/results_out/calib" 2>&1 | tee -a "$LOG"
  { echo "=== 校准报告 ==="; grep -E "准确率|耗时|detector" "$LOG" || true; } \
    > "$BASE/results_out/calib_report.txt"
  echo; echo ">>> 看 results_out/calib_report.txt，准确率与 MacBook 差异 >0.5pp 就停下发回"
  ;;
batch1)
  echo "=== 批次一：新基线（约 3–5 小时）===" | tee "$LOG"
  python scripts/run_multiseed.py --dataset_source real --datasets insects \
    --aligned_v2 --label_scheme pair_A_vs_B --segments "$SEGS_A" \
    --configs phase1 --seeds 42 --n_parallel 2 \
    --variant_tag v2AvsB_base --partial_tag _p55_base 2>&1 | tee -a "$LOG"
  python scripts/run_multiseed.py --dataset_source real --datasets insects \
    --aligned_v2 --label_scheme pair_parity --segments "$SEGS_P" \
    --configs phase1 --seeds 42 --n_parallel 2 \
    --variant_tag v2parity_base --partial_tag _p55_parity 2>&1 | tee -a "$LOG"
  python scripts/run_multiseed.py --dataset_source real --datasets insects \
    --aligned_v2 --label_scheme pair_A_vs_B --segments "$SEGS_A" \
    --configs phase1 --seeds 42 --n_parallel 2 \
    --variant_tag v2AvsB_dual --partial_tag _p55_dual \
    --extra_args "--context_loader dual --short_ratio 0.5 --long_max_age 2000" 2>&1 | tee -a "$LOG"
  python "$BASE/check_batch1.py" > "$BASE/results_out/batch1_report.txt" 2>&1 || true
  cat "$BASE/results_out/batch1_report.txt"
  ;;
batch2)
  echo "=== 批次二：判别性对照（约 6–10 小时）===" | tee "$LOG"
  for TRIG in oracle detector; do
    for ACT in context_reset route_adapter buffer_clear none; do
      echo "--- trigger=$TRIG action=$ACT ---" | tee -a "$LOG"
      python scripts/run_multiseed.py --dataset_source real --datasets insects \
        --aligned_v2 --label_scheme pair_A_vs_B --segments "$SEGS_A" \
        --configs phase4a --seeds 42 --n_parallel 2 \
        --variant_tag "b2_${TRIG}_${ACT}" --partial_tag "_p55_b2" \
        --extra_args "--detector_impl river --detector_input pred1 --trigger_source $TRIG --action_on_alarm $ACT" 2>&1 | tee -a "$LOG"
    done
  done
  echo ">>> 批次二跑完，运行 bash pack_results.sh 把结果发回" | tee -a "$LOG"
  ;;
*)
  echo "用法: bash run_batch.sh {calib|batch1|batch2}"; exit 1;;
esac

echo
echo "完成。结果在 results_out/  —— 运行 bash pack_results.sh 打包发回"
SH

cat > "$OUT/check_batch1.py" <<'PY'
"""批次一的中止判据自动检查 —— 不需要人判断，直接看 PASS/FAIL。"""
import glob, os, sys
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(base, "code", "results", "multiseed_phase1_real_insects_*seed42.npz")))
print("=== 批次一报告 ===\n")
if not files:
    print("FAIL: 没找到结果文件，实验可能没跑起来"); sys.exit(0)

accs, drops = [], []
print(f"{'段':34s} {'总体acc':>8s} {'漂移前':>8s} {'漂移后':>8s} {'落差pp':>8s}")
for f in files:
    d = np.load(f, allow_pickle=True)
    name = os.path.basename(f).replace("multiseed_phase1_real_insects_", "").replace("_seed42.npz", "")
    acc = float(d["overall_acc"][0]); accs.append(acc)
    p, l, ctx = d["predictions"], d["labels"], 200
    dps = [int(x) for x in d["drift_points"]]
    if not dps:
        print(f"{name:34s} {acc*100:7.2f}%        —        —        —   (对照段)"); continue
    for dp in dps:
        i = dp - ctx
        if i <= 0 or i >= len(p): continue
        w = 300
        pre = (p[max(0, i-w):i] == l[max(0, i-w):i]).mean()
        post = (p[i:i+w] == l[i:i+w]).mean()
        drops.append((pre - post) * 100)
        print(f"{name:34s} {acc*100:7.2f}% {pre*100:7.2f}% {post*100:7.2f}% {(pre-post)*100:7.2f}")

print("\n=== 判据 ===")
ok1 = bool(accs) and max(accs) < 0.92
print(f"判据1 有 headroom（最高准确率 < 92%）：{'PASS' if ok1 else 'FAIL'}  实测最高 {max(accs)*100:.2f}%" if accs else "判据1 FAIL")
n_big = sum(1 for d in drops if d >= 5)
ok2 = n_big >= 2
print(f"判据2 至少 2 个漂移落差 ≥5pp：{'PASS' if ok2 else 'FAIL'}  实测 {n_big} 个")
print("\n" + ("两条都 PASS → 可以跑批次二" if (ok1 and ok2) else "有 FAIL → 停下，把本报告发回"))
PY

cat > "$OUT/pack_results.sh" <<'SH'
#!/usr/bin/env bash
# 把结果打包发回。只含数字/图/日志，不含数据与权重。
set -euo pipefail
cd "$(dirname "$0")"
STAMP="$(date +%Y%m%d_%H%M)"
OUT="results_to_send_${STAMP}"
rm -rf "$OUT" "${OUT}.zip"; mkdir -p "$OUT"
[ -d results_out ] && cp -R results_out "$OUT/"
mkdir -p "$OUT/from_code"
for pat in 'results/*.md' 'results/*.json' 'results/*.png' 'results/*.npz' 'logs'; do
  cp -R code/$pat "$OUT/from_code/" 2>/dev/null || true
done
nvidia-smi > "$OUT/gpu_info.txt" 2>&1 || true
zip -qr "${OUT}.zip" "$OUT" && rm -rf "$OUT"
echo "打包完成: ${OUT}.zip  ($(du -h "${OUT}.zip" | cut -f1))"
echo "把这个 zip 发回即可。"
SH

chmod +x "$OUT"/*.sh
cd "$(dirname "$OUT")" && zip -qr "$ZIP" "$(basename "$OUT")" && rm -rf "$OUT"
echo
echo "=== 完成 ==="
echo "离线包: $ZIP  ($(du -h "$ZIP" | cut -f1))"
echo "把这个 zip 拷到 ROG，解压后先读里面的 SETUP.md。"
