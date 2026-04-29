# Phase 4 A multi-seed partial summary (live)

Increment-appended after each finished seed. Re-run safe (no header dedup needed if file exists).

| seed | dataset | overall_acc | post_drift_acc | n_routes | n_adapters | wall_time |
|---|---|---|---|---|---|---|
| 123 | regime_switching | 0.7950 | 0.6960 | 0 | 1 | 2364s |
| 42 | regime_switching | 0.7839 | 0.6260 | 3 | 4 | 2619s |
| 456 | regime_switching | 0.8057 | 0.6920 | 1 | 2 | 2978s |
| 789 | regime_switching | 0.7975 | 0.6680 | 1 | 2 | 3088s |
| 1024 | regime_switching | 0.7979 | 0.6500 | 4 | 5 | 1841s |
| 42 | rotating_boundary | 0.8418 | 0.8300 | 0 | 1 | 1340s |
| 123 | rotating_boundary | 0.8336 | 0.8575 | 0 | 1 | 1384s |
| 456 | rotating_boundary | 0.8396 | 0.8575 | 0 | 1 | 1336s |
| 789 | rotating_boundary | 0.8279 | 0.8350 | 0 | 1 | 1320s |
| 1024 | rotating_boundary | 0.8364 | 0.8325 | 0 | 1 | 1206s |
| 123 | combined_drift | 0.8179 | 0.5800 | 1 | 2 | 3622s |
| 42 | combined_drift | 0.8185 | 0.6900 | 0 | 1 | 3677s |
| 789 | combined_drift | 0.8235 | 0.5900 | 1 | 2 | 3632s |
| 456 | combined_drift | 0.8058 | 0.5900 | 1 | 2 | 3794s |
| 1024 | combined_drift | 0.8229 | 0.5700 | 2 | 3 | 2972s |
