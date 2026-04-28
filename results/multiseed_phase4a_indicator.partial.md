# Phase 4 A multi-seed partial summary (live)

Increment-appended after each finished seed. Re-run safe (no header dedup needed if file exists).

| seed | dataset | overall_acc | post_drift_acc | n_routes | n_adapters | wall_time |
|---|---|---|---|---|---|---|
| 42 | regime_switching | 0.7882 | 0.6460 | 2 | 3 | 2519s |
| 123 | regime_switching | 0.7871 | 0.7000 | 1 | 2 | 2519s |
| 456 | regime_switching | 0.7954 | 0.6660 | 1 | 2 | 2190s |
| 789 | regime_switching | 0.7964 | 0.6720 | 1 | 2 | 2203s |
| 1024 | regime_switching | 0.7943 | 0.6520 | 2 | 3 | 2172s |
| 42 | rotating_boundary | 0.8439 | 0.8275 | 0 | 1 | 1538s |
| 123 | rotating_boundary | 0.8318 | 0.8575 | 0 | 1 | 1552s |
| 456 | rotating_boundary | 0.8361 | 0.8625 | 0 | 1 | 1445s |
| 789 | rotating_boundary | 0.8243 | 0.8275 | 0 | 1 | 1489s |
| 1024 | rotating_boundary | 0.8425 | 0.8275 | 0 | 1 | 1515s |
| 42 | combined_drift | 0.8169 | 0.6600 | 0 | 1 | 5421s |
| 123 | combined_drift | 0.8150 | 0.5800 | 1 | 2 | 5430s |
| 789 | combined_drift | 0.8231 | 0.5800 | 1 | 2 | 4188s |
| 456 | combined_drift | 0.8158 | 0.6000 | 1 | 2 | 4250s |
| 1024 | combined_drift | 0.8275 | 0.5800 | 2 | 3 | 3852s |
