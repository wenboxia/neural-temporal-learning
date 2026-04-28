# Phase 4 A multi-seed partial summary (live)

Increment-appended after each finished seed. Re-run safe (no header dedup needed if file exists).

| seed | dataset | overall_acc | post_drift_acc | n_routes | n_adapters | wall_time |
|---|---|---|---|---|---|---|
| 42 | regime_switching | 0.7821 | 0.6320 | 0 | 1 | 2364s |
| 123 | regime_switching | 0.7918 | 0.6940 | 0 | 1 | 2429s |
| 456 | regime_switching | 0.8064 | 0.6840 | 0 | 1 | 2240s |
| 789 | regime_switching | 0.8082 | 0.6700 | 0 | 1 | 2211s |
| 1024 | regime_switching | 0.7871 | 0.6540 | 0 | 1 | 1706s |
| 123 | rotating_boundary | 0.8318 | 0.8550 | 0 | 1 | 1258s |
| 42 | rotating_boundary | 0.8386 | 0.8200 | 0 | 1 | 1287s |
| 456 | rotating_boundary | 0.8379 | 0.8550 | 0 | 1 | 1280s |
| 789 | rotating_boundary | 0.8236 | 0.8350 | 0 | 1 | 1301s |
| 1024 | rotating_boundary | 0.8439 | 0.8250 | 0 | 1 | 1140s |
| 123 | combined_drift | 0.8181 | 0.5800 | 0 | 1 | 3509s |
| 42 | combined_drift | 0.8163 | 0.6800 | 0 | 1 | 3593s |
| 456 | combined_drift | 0.8185 | 0.6100 | 0 | 1 | 3637s |
| 789 | combined_drift | 0.8223 | 0.5800 | 0 | 1 | 3557s |
| 1024 | combined_drift | 0.8221 | 0.5700 | 0 | 1 | 3668s |
