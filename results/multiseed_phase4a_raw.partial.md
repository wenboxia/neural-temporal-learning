# Phase 4 A multi-seed partial summary (live)

Increment-appended after each finished seed. Re-run safe (no header dedup needed if file exists).

| seed | dataset | overall_acc | post_drift_acc | n_routes | n_adapters | wall_time |
|---|---|---|---|---|---|---|
| 123 | regime_switching | 0.7893 | 0.7040 | 0 | 1 | 2230s |
| 42 | regime_switching | 0.7811 | 0.6300 | 0 | 1 | 2268s |
| 456 | regime_switching | 0.8068 | 0.6960 | 0 | 1 | 2161s |
| 789 | regime_switching | 0.8014 | 0.6600 | 0 | 1 | 2133s |
| 1024 | regime_switching | 0.7943 | 0.6500 | 0 | 1 | 1703s |
| 123 | rotating_boundary | 0.8300 | 0.8525 | 0 | 1 | 1304s |
| 42 | rotating_boundary | 0.8443 | 0.8250 | 0 | 1 | 1309s |
| 789 | rotating_boundary | 0.8243 | 0.8325 | 0 | 1 | 1294s |
| 456 | rotating_boundary | 0.8339 | 0.8600 | 0 | 1 | 1327s |
| 1024 | rotating_boundary | 0.8386 | 0.8250 | 0 | 1 | 1233s |
| 42 | combined_drift | 0.8177 | 0.6700 | 0 | 1 | 3600s |
| 123 | combined_drift | 0.8165 | 0.5800 | 0 | 1 | 3638s |
| 456 | combined_drift | 0.8135 | 0.6100 | 0 | 1 | 3436s |
| 789 | combined_drift | 0.8219 | 0.5800 | 0 | 1 | 3456s |
| 1024 | combined_drift | 0.8256 | 0.5800 | 0 | 1 | 2948s |
