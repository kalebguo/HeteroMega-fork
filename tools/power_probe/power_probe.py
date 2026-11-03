#!/usr/bin/env python3
"""Socket power around a prefill loop, from the amdgpu hwmon reading (label PPT). Needs no package and no root.

The reading is what `rocm-smi --showpower` calls socket graphics package power. `coverage` tests what it covers, with idle,
CPU-only, NPU-only and GPU-only phases, and how often it updates. `prefill` samples in a separate process around one
method's generate(1) loop, the timer of kernels/gpu/tools/run_model_ffn.py, so run it apart from timing jobs. `report`
turns the runs into energy per prefill: mean power over the loop times the loop's period per prefill.

    python3 tools/power_probe/power_probe.py coverage --out RESULT_DIR/coverage.json
    python3 tools/power_probe/power_probe.py prefill --method heteromosaic --model MODEL_DIR --inflight 2 \
        --merged-ffn-config "$PWD/deployment.json" --out RESULT_DIR/heteromosaic-i2-r0.json
    python3 tools/power_probe/power_probe.py report RESULT_DIR --json RESULT_DIR/report.json
"""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "npu_error_probe"))
SETTLE_S = 5  # dropped from the start of each phase, because the reading is smoothed


def power_file():
    for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        if (hwmon / "name").read_text().strip() == "amdgpu" and (hwmon / "power1_input").exists():
            return hwmon / "power1_input"
    raise RuntimeError("no amdgpu hwmon power1_input on this machine")


def sample(out, interval_s):
    """Writes monotonic seconds and watts until SIGTERM."""
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    with power_file().open() as reading, open(out, "w", buffering=1) as f:
        f.write("t,w\n")
        while not stop:
            reading.seek(0)
            f.write(f"{time.monotonic():.4f},{int(reading.read()) / 1e6:.3f}\n")
            time.sleep(interval_s)


class Sampler:
    """The sampling loop in its own process, so that it does not share the measured process's interpreter."""

    def __init__(self, out, interval_ms):
        self.command = [sys.executable, str(Path(__file__).resolve()), "sample", "--out", str(out), "--interval-ms", str(interval_ms)]

    def __enter__(self):
        self.process = subprocess.Popen(self.command)
        return self

    def __exit__(self, *_):
        self.process.terminate()
        self.process.wait()


def read_samples(path):
    with open(path) as f:
        next(f)
        return [tuple(float(v) for v in line.split(",")) for line in f if line.count(",") == 1]


def mean_watts(samples, start, end):
    values = [w for t, w in samples if start <= t < end]
    return statistics.fmean(values) if values else None


def update_period_ms(samples):
    """Median time between changes of the reading, in milliseconds."""
    changes = [t for (_, before), (t, after) in zip(samples, samples[1:]) if before != after]
    gaps = [(b - a) * 1000 for a, b in zip(changes, changes[1:])]
    return statistics.median(gaps) if gaps else None


def spin(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass


def phase(name, seconds, phases, work=None):
    start = time.monotonic()
    cpu = time.process_time()
    if work is None:
        time.sleep(seconds)
    else:
        work(start + seconds)
    phases.append(dict(name=name, start=start, end=time.monotonic(), main_process_cpu_fraction=(time.process_time() - cpu) / (time.monotonic() - start)))


def run_coverage(args):
    """Each load runs alone, with an idle phase after it. The NPU and GPU phases use one native GEMM of the error probe."""
    import numpy as np
    import probe_cases as pc
    import probe_sweep
    import torch
    from experiments.ffn_partition.gpu_backend import pack_heteromosaic_weights

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fast = args.out.with_suffix(".fast.csv")
    case = next(c for c in pc.cases() if c["name"] == "k14336_n4096")
    fw_path = pc.artifact(ROOT, case["m"], case["k"], case["n"])
    config = args.out.with_suffix(".native.json")
    config.write_text(json.dumps(pc.native_config(case, fw_path), indent=2) + "\n")
    module, _ = probe_sweep.load_binding(ROOT, probe_sweep.default_library(ROOT))
    module.initialize(str(config), case["m"], case["k"], case["n"])
    a, weight = pc.make_inputs(case)
    packed = np.ascontiguousarray(pack_heteromosaic_weights(weight)).reshape(-1)
    device = torch.device("cuda")
    x = torch.from_numpy(a).to(device=device, dtype=torch.bfloat16).contiguous()
    w = torch.from_numpy(packed).to(device).contiguous()
    out = torch.zeros((case["m"], case["n"]), dtype=torch.bfloat16, device=device)
    for tensor in (x, w, out):
        module.import_tensor(tensor)

    def loop(on_npu):
        def work(end):
            calls = 0
            while time.monotonic() < end:
                module.run(x, out, w, on_npu)
                calls += 1
            print(f"on_npu={on_npu} calls={calls}", flush=True)
        return work

    def cores(end):
        with multiprocessing.Pool(os.cpu_count()) as pool:
            pool.map(spin, [end - time.monotonic()] * os.cpu_count())

    with Sampler(fast, args.fast_interval_ms):
        time.sleep(10)
    phases = []
    samples_path = args.out.with_suffix(".power.csv")
    with Sampler(samples_path, args.interval_ms):
        phase("idle", 30, phases)
        phase("cpu_one_thread", 20, phases, lambda end: spin(end - time.monotonic()))
        phase("idle_after_cpu", 20, phases)
        phase("cpu_all_threads", 20, phases, cores)
        phase("idle_after_all_threads", 20, phases)
        phase("npu_only", 20, phases, loop(True))
        phase("idle_after_npu", 20, phases)
        phase("gpu_only", 20, phases, loop(False))
        phase("idle_after_gpu", 20, phases)
    samples = read_samples(samples_path)
    for p in phases:
        p["mean_w"] = mean_watts(samples, p["start"] + SETTLE_S, p["end"])
    result = dict(power_file=str(power_file()), interval_ms=args.interval_ms, fast_interval_ms=args.fast_interval_ms,
                  idle_update_period_ms=update_period_ms(read_samples(fast)), active_update_period_ms=update_period_ms(samples),
                  settle_s=SETTLE_S, phases=phases)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps([dict(name=p["name"], mean_w=p["mean_w"], cpu=p["main_process_cpu_fraction"]) for p in phases], indent=1))
    print("update period ms", result["idle_update_period_ms"], result["active_update_period_ms"], flush=True)


def run_prefill(args):
    import path_logits

    heterogeneity, merged = path_logits.METHODS[args.method]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples_path = args.out.with_suffix(".power.csv")
    with Sampler(samples_path, args.interval_ms):
        box_idle = dict(start=time.monotonic())
        time.sleep(args.idle_s)
        box_idle["end"] = time.monotonic()
        model, torch = path_logits.load_model(args, heterogeneity, merged, args.prompt_tokens, args.out)
        counter = model.model.merged_ffn_completed_dispatches
        per_prefill = 32 * (args.prompt_tokens // path_logits.CHUNK) if merged else 0
        ids_all = json.loads(args.ref.read_text())["input_ids"][0]
        ids = torch.tensor([ids_all[:args.prompt_tokens]], dtype=torch.int64, device="cuda")

        def timed():
            torch.cuda.synchronize()
            start = time.monotonic()
            with torch.no_grad():
                tokens = model.generate(ids, max_new_tokens=1, temperature=0.0)[0, args.prompt_tokens:].tolist()
            torch.cuda.synchronize()
            return start, time.monotonic(), tokens

        timed()  # cold pass, outside every window
        loaded_idle = dict(start=time.monotonic())
        time.sleep(args.idle_s)
        loaded_idle["end"] = time.monotonic()
        windows, tokens = [], set()
        before = int(counter())
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            start, stop, ids_out = timed()
            windows.append((start, stop))
            tokens.add(tuple(ids_out))
        completions = int(counter()) - before
    if completions != per_prefill * len(windows):
        raise RuntimeError(f"{args.method}: {completions} merged FFN completions over {len(windows)} prefills")
    record = dict(method=args.method, inflight=args.inflight, prompt_tokens=args.prompt_tokens, interval_ms=args.interval_ms,
                  box_idle=box_idle, loaded_idle=loaded_idle, windows=windows, tokens=[list(t) for t in tokens],
                  merged_ffn_completions=completions, samples=samples_path.name)
    args.out.write_text(json.dumps(record))
    print(json.dumps(summarize_run(record, read_samples(samples_path))), flush=True)


def summarize_run(record, samples):
    """Energy per prefill: mean power from the settled loop start to its end, times the loop's period per prefill."""
    windows = record["windows"]
    first = next(i for i, w in enumerate(windows) if w[0] >= windows[0][0] + SETTLE_S)
    start, end = windows[first][0], windows[-1][1]
    count = len(windows) - first
    period_s = (end - start) / count
    active = mean_watts(samples, start, end)
    box_idle = mean_watts(samples, record["box_idle"]["start"] + SETTLE_S, record["box_idle"]["end"])
    loaded_idle = mean_watts(samples, record["loaded_idle"]["start"] + SETTLE_S, record["loaded_idle"]["end"])
    return dict(method=record["method"], inflight=record["inflight"], prompt_tokens=record["prompt_tokens"], prefills=count,
                prefill_median_ms=statistics.median((b - a) * 1000 for a, b in windows[first:]), period_ms=period_s * 1000,
                active_w=active, box_idle_w=box_idle, loaded_idle_w=loaded_idle, energy_j=active * period_s,
                net_of_loaded_idle_j=(active - loaded_idle) * period_s, samples_in_loop=sum(start <= t < end for t, _ in samples))


def run_report(args):
    rows = []
    for path in sorted(args.directory.glob("*.json")):
        record = json.loads(path.read_text())
        if "windows" in record:
            rows.append(dict(summarize_run(record, read_samples(path.with_name(record["samples"]))), run=path.stem))
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["inflight"], row["prompt_tokens"]), []).append(row)
    summary = [dict(method=m, inflight=i, prompt_tokens=n, runs=len(group), **{key: statistics.median(r[key] for r in group) for key in
                    ("prefill_median_ms", "period_ms", "active_w", "box_idle_w", "loaded_idle_w", "energy_j", "net_of_loaded_idle_j")},
                    energy_j_runs=[round(r["energy_j"], 2) for r in group])
               for (m, i, n), group in sorted(groups.items(), key=lambda item: (item[0][1] or 0, item[0][0]))]
    if args.json:
        args.json.write_text(json.dumps(dict(runs=rows, summary=summary), indent=1))
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--interval-ms", type=float, default=100)
    c = sub.add_parser("coverage")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--interval-ms", type=float, default=100)
    c.add_argument("--fast-interval-ms", type=float, default=5)
    f = sub.add_parser("prefill")
    f.add_argument("--method", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--build", default="build")
    f.add_argument("--config", default="py/unified_llm_w4a16/configs/configs_strixP_llama3_8b.json5")
    f.add_argument("--merged-ffn-config", type=Path)
    f.add_argument("--ref", type=Path, default=ROOT / "docs/runs/2026-10-12-aup/gpu-model-result.json")
    f.add_argument("--prompt-tokens", type=int, default=1024)
    f.add_argument("--inflight", type=int)
    f.add_argument("--seconds", type=float, default=45)
    f.add_argument("--idle-s", type=float, default=20)
    f.add_argument("--interval-ms", type=float, default=100)
    r = sub.add_parser("report")
    r.add_argument("directory", type=Path)
    r.add_argument("--json", type=Path)
    args = p.parse_args()
    if args.command == "sample":
        sample(args.out, args.interval_ms / 1000)
    elif args.command == "coverage":
        run_coverage(args)
    elif args.command == "prefill":
        run_prefill(args)
    else:
        run_report(args)


if __name__ == "__main__":
    main()
