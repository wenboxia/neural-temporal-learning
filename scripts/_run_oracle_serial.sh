#!/usr/bin/env bash
# Day 0.5 实验 0a 串行驱动：5 seeds × {baseline, oracle} = 10 runs
# 由 main session 后台启动；每个 run 25-30 min，全部约 4-5 小时
set -u
cd "$(dirname "$0")/.."

LOG="logs/oracle_runs.log"
mkdir -p logs results
echo "===== 0a serial driver started @ $(date) =====" | tee -a "$LOG"

SEEDS=(42 123 456 789 1024)
FAIL=0

for S in "${SEEDS[@]}"; do
    for KIND in baseline reset; do
        TAG="oracle_${KIND}_seed${S}"
        if [[ -f "results/${TAG}.npz" ]]; then
            echo "[skip] results/${TAG}.npz already exists" | tee -a "$LOG"
            continue
        fi
        echo "----- $(date) :: ${TAG} -----" | tee -a "$LOG"
        if [[ "$KIND" == "baseline" ]]; then
            python scripts/run_baselines.py \
                --dataset regime_switching \
                --n_samples 3000 --context_size 200 \
                --seed "$S" \
                --out_tag "$TAG" 2>&1 | tee -a "$LOG"
        else
            python scripts/run_baselines.py \
                --dataset regime_switching \
                --n_samples 3000 --context_size 200 \
                --seed "$S" \
                --oracle_context_reset --reset_size 50 \
                --out_tag "$TAG" 2>&1 | tee -a "$LOG"
        fi
        STATUS=${PIPESTATUS[0]}
        if [[ $STATUS -ne 0 ]]; then
            echo "[FAIL] ${TAG} exited $STATUS" | tee -a "$LOG"
            FAIL=$((FAIL+1))
        fi
    done
done

echo "===== 0a serial driver done @ $(date), failures=$FAIL =====" | tee -a "$LOG"
exit $FAIL
