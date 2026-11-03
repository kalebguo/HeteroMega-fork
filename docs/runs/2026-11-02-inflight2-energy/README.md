# 2026-11-02: two-inflight comparison, energy: paused, only the coverage test ran (AUP)

**No energy per prefill was measured.** The user stopped this job at 21:53 CST, in the first energy run. What is here is the test of what the
power reading covers.

Job `aup-runs/inflight2-energy`, 03:49-03:53 UTC Nov 3 (21:49-21:53 CST Nov 2), commit `5bd4648`, `exit=143` (killed). An earlier attempt at
03:34 UTC failed at once because `DRIVER_PATH` was not set. `tools/power_probe/power_probe.py coverage`; the partial first run was not kept.

## What the reading covers (measured, `coverage.json`)

The reading is `/sys/class/hwmon/hwmon5/power1_input` (amdgpu, label `PPT`), which `rocm-smi` reports as socket graphics package power.
Sampled at 100 ms, with the first 5 s of each phase dropped. A 5 ms poll shows the value updating about every 101 ms under load and about every 7 ms at idle.

| Phase | Mean W |
| --- | ---: |
| Idle | 3.41 |
| CPU, one thread | 16.96 |
| Idle after | 3.85 |
| CPU, all threads | 47.43 |
| Idle after | 3.87 |
| NPU only (`k14336_n4096` native GEMM in a loop) | 18.06 |
| Idle after | 3.83 |
| GPU only (same GEMM on the GPU, in a loop) | 47.16 |
| Idle after | 4.01 |

The reading responds to CPU, NPU and GPU load, so it is usable as a package-level reading. Limits:

- The NPU phase also keeps one host thread submitting and waiting, so its +14.65 W is not the NPU alone. The two cannot be separated with this reading.
- It does not cover DRAM or the rest of the board. `/sys/class/powercap` has no readable `energy_uj`, `amd-smi metric --power` gives N/A, and
  `xrt-smi` estimated power gives N/A. Nothing was installed.
- The `rocm-smi` snapshots at the start of other jobs read about 5 W, right after a process exit. The idle baseline for energy is 3.41 W from this test.

To resume: `bg inflight2-energy -- 'bash /notebooks/hm-fix/tools/aup-kit/jobs/inflight2-energy/cmd.sh'`.
