# Detector replay — own vs river ADWIN on saved indicator streams

**零 CPU 成本诊断**：检测器只消费 1D 的 0/1 错误指示器流，该流已存在 Phase 5 的 npz 里（`indicator_history`），因此换实现 / 扫 δ 无需重跑 TabPFN。

- 流数：35（Insects B1+ 20 + Electricity A+ 15）
- Insects 变点用 **Souza 2020 Table 2 官方坐标** [14352, 19500, 33240, 38682, 39510]，不是仓库此前使用的 P(y) 构成变化点
- Electricity 无 documented 变点（渐进漂移），任何报警都计为误报
- 命中容差 = 变点后 600 步；cooldown = 80

| impl | δ | dataset | runs with ≥1 alarm | alarms | recall | false alarms | median delay |
|---|---|---|---|---|---|---|---|
| own | 0.002 | insects | 0/20 | 0 | 0.00 | 0 | — |
| own | 0.002 | electricity | 1/15 | 1 | — | 1 | — |
| own | 0.01 | insects | 0/20 | 0 | 0.00 | 0 | — |
| own | 0.01 | electricity | 1/15 | 1 | — | 1 | — |
| own | 0.05 | insects | 1/20 | 1 | 0.00 | 1 | — |
| own | 0.05 | electricity | 1/15 | 1 | — | 1 | — |
| own | 0.2 | insects | 2/20 | 2 | 0.00 | 2 | — |
| own | 0.2 | electricity | 1/15 | 1 | — | 1 | — |
| river | 0.002 | insects | 16/20 | 26 | 1.00 | 16 | 266 |
| river | 0.002 | electricity | 1/15 | 1 | — | 1 | — |
| river | 0.01 | insects | 16/20 | 27 | 1.00 | 17 | 170 |
| river | 0.01 | electricity | 1/15 | 1 | — | 1 | — |
| river | 0.05 | insects | 16/20 | 28 | 1.00 | 18 | 87 |
| river | 0.05 | electricity | 2/15 | 2 | — | 2 | — |
| river | 0.2 | insects | 20/20 | 34 | 1.00 | 24 | 52 |
| river | 0.2 | electricity | 11/15 | 17 | — | 17 | — |

## Per-stream detail (river, δ=0.002)

| dataset | segment | seed | err rate | official drifts (local) | alarms (local) | delays |
|---|---|---|---|---|---|---|
| insects | early | 1024 | 0.029 | [4352] | [3420, 4785] | [433] |
| insects | early | 123 | 0.029 | [4352] | [3420, 4785] | [433] |
| insects | early | 42 | 0.028 | [4352] | [3430, 4785] | [433] |
| insects | early | 456 | 0.029 | [4352] | [3410, 4799] | [447] |
| insects | early | 789 | 0.032 | [4352] | [3074, 4713] | [361] |
| insects | mid | 1024 | 0.040 | [3500] | [2807, 3672] | [172] |
| insects | mid | 123 | 0.043 | [3500] | [2672, 3672] | [172] |
| insects | mid | 42 | 0.041 | [3500] | [2488, 3663] | [163] |
| insects | mid | 456 | 0.043 | [3500] | [2425, 3645] | [145] |
| insects | mid | 789 | 0.041 | [3500] | [2514, 3665] | [165] |
| insects | late_pre | 1024 | 0.018 | [] | [] | [] |
| insects | late_pre | 123 | 0.019 | [] | [] | [] |
| insects | late_pre | 42 | 0.022 | [] | [1362] | [] |
| insects | late_pre | 456 | 0.018 | [] | [] | [] |
| insects | late_pre | 789 | 0.018 | [] | [] | [] |
| insects | late_post | 1024 | 0.033 | [] | [4968] | [] |
| insects | late_post | 123 | 0.036 | [] | [4900] | [] |
| insects | late_post | 42 | 0.036 | [] | [4898] | [] |
| insects | late_post | 456 | 0.033 | [] | [4968] | [] |
| insects | late_post | 789 | 0.034 | [] | [4952] | [] |
| electricity | start | 1024 | 0.037 | [] | [] | [] |
| electricity | start | 123 | 0.037 | [] | [] | [] |
| electricity | start | 42 | 0.037 | [] | [] | [] |
| electricity | start | 456 | 0.037 | [] | [] | [] |
| electricity | start | 789 | 0.038 | [] | [] | [] |
| electricity | middle | 1024 | 0.067 | [] | [] | [] |
| electricity | middle | 123 | 0.070 | [] | [] | [] |
| electricity | middle | 42 | 0.068 | [] | [] | [] |
| electricity | middle | 456 | 0.073 | [] | [279] | [] |
| electricity | middle | 789 | 0.067 | [] | [] | [] |
| electricity | end | 1024 | 0.053 | [] | [] | [] |
| electricity | end | 123 | 0.056 | [] | [] | [] |
| electricity | end | 42 | 0.052 | [] | [] | [] |
| electricity | end | 456 | 0.052 | [] | [] | [] |
| electricity | end | 789 | 0.052 | [] | [] | [] |
