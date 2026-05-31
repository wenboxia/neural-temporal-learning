# Archive — Stage B 原始 A+ 协议（misaligned）

**归档日期**：2026-05-29
**原因**：Phase 5 Stage B 第一次运行采用 phase5_plan §3.2 的 A+ subsampling 协议（uniform start/middle/end），post-hoc 诊断发现 15 个 segment 中 **14 个不含任何 documented drift event**，导致 F3 detector 0/15 触发是 trivially correct（没 drift 可触发）而非 valid 负面结果。

## A+ 协议 misalignment 分析

Insects abrupt_balanced（N=52,848）的 5 个 documented abrupt drift 位置（empirical 50-chunk P(y) L1 shift 分析）：

```
12,672   14,256   17,952   46,728   52,008
```

A+ 协议 3 个 segment 实际覆盖：

| segment | 范围 | 含 drift |
|---|---|---|
| `start` | [0, 5000) | 0（全部 drift ≥ 12,672 > 5000）|
| `middle` | [23,924, 28,924) | 0（5 个 drift 全在区间外）|
| `end` | [47,848, 52,848) | 1（只含 51,800 那个） |

**结果**：15 runs（3 segments × 5 seeds）里只有 5 个 end runs 真有 drift 可被 detector 触发。F3 0/15 因此是 protocol artifact，不是 detector 性能。

## 修复方案：Stage B1+

重新设计 4 个 drift-aligned 非重叠 5000-sample segments，覆盖全 5/5 documented drifts：

| segment | 范围 | 覆盖 drift |
|---|---|---|
| `early` | [10,000, 15,000) | 12,672 + 14,256 |
| `mid` | [16,000, 21,000) | 17,952 |
| `late_pre` | [42,500, 47,500) | 46,728 |
| `late_post` | [47,848, 52,848) | 52,008 |

Stage B1+ 跑完后 F3 仍 0/20，但**此时 0/20 是 valid 数据点**——γ 诊断进一步定位 confound #2（TabPFN sliding-context 在 ~10-20 步 in-context relearn 内吸收 P(y) shift，indicator |Δ| ≤ 0.019 vs 合成 0.20，10× 稀释）。

## 此目录保留的目的

1. **Paper methodology narrative**：论文 methodology section 写 "we initially used uniform partitioning, post-hoc identified misalignment, redesigned protocol" 的诚实叙事
2. **审稿人 reproducibility**：reviewer 可重跑验证两版协议差异
3. **教训记录**：subsampling 协议必须与漂移结构对齐，否则 detector 实验 invalid

## 文件清单

- `multiseed_phase{1,3,4a}_real_insects_{start,middle,end}_seed{42,123,456,789,1024}.npz` × 45 — 原始 npz 结果
- `multiseed_phase{1,3,4a}_real_insects_{start,middle,end}_seed{42,123,456,789,1024}.png` × 45 — 对应 plots
- `multiseed_phase5_insects.partial.md` — 当时实时 partial summary

## 当前 active 数据位置

Stage B1+ 重跑后的有效数据在 `results/` 根目录：
- `multiseed_phase{1,3,4a}_real_insects_{early,mid,late_pre,late_post}_seed*.npz/png`
- `phase5_real_summary_insects.md` — B1+ 完整 verdict
- `phase5_confound2_diagnostic.md` — γ 诊断
- `phase5_real_summary.md` — Stage A + B1+ combined verdict
