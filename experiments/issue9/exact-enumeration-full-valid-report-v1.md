# Issue #9 full-valid exact-enumeration benchmark

**Evidence label: Experimentally observed.** On 2026-10-23, all eight serial cases completed with exact Fraction objectives and tie sets matching the frozen valid-only references. The run used the `exact_enumeration` backend at Advisor source commit `d28756e71d28484add4f287dec244b0814e1126c`; the authoritative run configuration is [exact-enumeration-full-valid-config-v1.json](exact-enumeration-full-valid-config-v1.json). Compact case outputs and `/usr/bin/time -v` logs are retained under `$ESS_RUNTIME_PARENT/ess-advisor/issue9/runs/exact-enumeration-20261023/`.

The question was whether the budget-pruned exact outer-subset backend reproduces the archived Nominal and task-wise PESS global optima on all 10,000 valid tasks for Forest10 (45 candidates) and Census13 (78 candidates), at B=2 and B=3. This reused the pinned exact coefficient artifacts, observations, coverage, truth counts, and frozen selection files from Research Issues #64/#71. It made no PostgreSQL calls, accessed no test outcomes, and changed no model, workload, priority, or cost assumptions. The additive unit-resource cost is an experimental proxy; the Census13 clamped-log-v2 and Forest10 positive-preclamp-v1 model versions remain distinct.

## Results

| Data / case | Feasible subsets evaluated | Exact backend time | Process wall time | Peak RSS | Frozen objective / optimum set |
|---|---:|---:|---:|---:|---|
| Forest10 Nominal B2 | 1,036 | 3.767 s | 4.71 s | 123.8 MiB | `4374828918765339127297/200000000000000000`; `{e01_06,e02_07}` |
| Forest10 PESS B2 | 1,036 | 2.537 s | 3.53 s | 123.7 MiB | same objective and unique set |
| Forest10 Nominal B3 | 15,226 | 58.334 s | 59.26 s | 123.9 MiB | `21512755121155336377967/1000000000000000000`; `{e01_06,e02_07,e03_04}` |
| Forest10 PESS B3 | 15,226 | 92.365 s | 93.32 s | 123.6 MiB | same objective and unique set |
| Census13 Nominal B2 | 3,082 | 12.212 s | 20.15 s | 228.0 MiB | `733156246462852301047/125000000000000000`; `{e02_03,e04_06}` |
| Census13 PESS B2 | 3,082 | 7.661 s | 15.58 s | 227.8 MiB | same objective and unique set |
| Census13 Nominal B3 | 79,158 | 356.402 s | 6:04.08 | 227.8 MiB | `5486660099511517437079/1000000000000000000`; `{e01_05,e02_03,e04_06}` |
| Census13 PESS B3 | 79,158 | 556.058 s | 9:24.21 | 228.0 MiB | same objective and unique set |

Each case completed the full budget-feasible subset search, used all tasks and the complete candidate universe, and matched the archived exact objective and all archived minimizing sets (one set in each case). No case timed out or required a retry. Aggregate driver wall time was 18:44.91; peak RSS across the serial driver was 233,504 KiB. The configured address-space limit was 12 GiB and the API limit was 895 seconds per case; outer timeout was 910 seconds. These are observed results on Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, NetworkX 3.7, an Intel Core i7-14650HX host, and the pinned local source commits in the config.

## Historical HiGHS comparison and limits

The separate Issue #74 Forest10 Nominal B2 HiGHS recovery attempt used the same archived case but the prior MILP backend. It did not return an Advisor result before being stopped around 607 seconds into the solver-call phase; its observed peak RSS was about 2.27 GiB. The exact-enumeration case completed in 3.767 seconds of backend time and 4.71 seconds for the whole process. This is an operational contrast, **not a controlled speedup estimate**: the backends and measured timing boundaries differ, and the censored HiGHS attempt produced no result to compare. It does not explain why that HiGHS call failed to return.

The exact result is a completed Fraction enumeration for these two frozen model instances. It does not establish PostgreSQL-selected statistics, held-out test performance, physical maintenance cost additivity, or general scalability beyond the 45/78-candidate, B<=3 cases. No solver-reported floating-point optimality or HiGHS status is involved in this benchmark.
