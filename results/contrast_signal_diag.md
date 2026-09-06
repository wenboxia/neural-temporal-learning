# Contrast-signal diagnostic (Phase 5.5 Step 4 / 导师路径 A)

把 **stale（不适应，context 固定在段首）vs sliding（适应）两路预测的差值** 喂给检测器，对比现有的 0/1 错误指示器。

- label_scheme = `pair_A_vs_B`，context = 200，stale context = 200，n_estimators = 1
- 检测器 = river ADWIN，δ = 0.002，cooldown = 80，命中窗口 = 变点前 100 步 ~ 变点后 600 步（允许早报：标注点是温度**设定**的切换时刻，读数会提前变）
- Insects 变点用 **Souza 2020 官方坐标**；`d0_control` 段无变点，那里的报警全是误报

| segment | signal | alarms | recall | false alarms | median delay | max shift |
|---|---|---|---|---|---|---|
| d3_33240 | `contrast_prob` | 1 | 1/1 | 0 | -45 | 0.028 |
| d3_33240 | `contrast_hard` | 1 | 1/1 | 0 | -22 | 0.085 |
| d3_33240 | `indicator` | 0 | 0/1 | 0 | — | 0.165 |
| d3_33240 | `pred1` | 1 | 1/1 | 0 | -31 | 0.340 |
| d4_double | `contrast_prob` | 2 | 1/2 | 1 | 68 | 0.278 |
| d4_double | `contrast_hard` | 2 | 1/2 | 1 | 355 | 0.120 |
| d4_double | `indicator` | 2 | 1/2 | 1 | 48 | 0.295 |
| d4_double | `pred1` | 2 | 1/2 | 1 | 26 | 0.480 |
| d0_control | `contrast_prob` | 0 | — | 0 | — | — |
| d0_control | `contrast_hard` | 0 | — | 0 | — | — |
| d0_control | `indicator` | 0 | — | 0 | — | — |
| d0_control | `pred1` | 0 | — | 0 | — | — |

## 每段报警时刻

| segment | signal | drifts (local) | alarms (local) | delays |
|---|---|---|---|---|
| d3_33240 | `contrast_prob` | [1540] | [1495] | [-45] |
| d3_33240 | `contrast_hard` | [1540] | [1518] | [-22] |
| d3_33240 | `indicator` | [1540] | [] | [] |
| d3_33240 | `pred1` | [1540] | [1509] | [-31] |
| d4_double | `contrast_prob` | [880, 1432] | [458, 948] | [68] |
| d4_double | `contrast_hard` | [880, 1432] | [525, 1235] | [355] |
| d4_double | `indicator` | [880, 1432] | [512, 928] | [48] |
| d4_double | `pred1` | [880, 1432] | [451, 906] | [26] |
| d0_control | `contrast_prob` | [] | [] | [] |
| d0_control | `contrast_hard` | [] | [] | [] |
| d0_control | `indicator` | [] | [] | [] |
| d0_control | `pred1` | [] | [] | [] |
