# Phase 4 A multi-seed partial summary (live)

Increment-appended after each finished seed. Re-run safe (no header dedup needed if file exists).

| seed | dataset | overall_acc | post_drift_acc | n_routes | n_adapters | wall_time |
|---|---|---|---|---|---|---|
| 42 | regime_switching | 0.7875 | 0.6320 | 2 | 3 | 2139s |
| 123 | regime_switching | 0.7896 | 0.6940 | 0 | 1 | 2173s |
| 456 | regime_switching | 0.8046 | 0.6800 | 1 | 2 | 2071s |
| 789 | regime_switching | 0.8075 | 0.6760 | 1 | 2 | 2083s |
| 1024 | regime_switching | 0.7907 | 0.6500 | 3 | 4 | 1799s |
| 123 | rotating_boundary | 0.8289 | 0.8525 | 0 | 1 | 1276s |
| 42 | rotating_boundary | 0.8439 | 0.8300 | 0 | 1 | 1294s |
| 456 | rotating_boundary | 0.8296 | 0.8600 | 1 | 2 | 1308s |
| 789 | rotating_boundary | 0.8282 | 0.8325 | 0 | 1 | 1304s |
| 1024 | rotating_boundary | 0.8386 | 0.8250 | 0 | 1 | 1175s |
| 42 | combined_drift | 0.8171 | 0.6700 | 0 | 1 | 3423s |
| 123 | combined_drift | 0.8152 | 0.5800 | 1 | 2 | 3442s |
| 789 | combined_drift | 0.8179 | 0.5800 | 1 | 2 | 3431s |
| 456 | combined_drift | 0.8140 | 0.6000 | 1 | 2 | 3549s |
| 1024 | combined_drift | 0.8204 | 0.5600 | 2 | 3 | 2843s |
