#!/usr/bin/env python3
"""Hardware NPU error probe: one native packed W4A16 GEMM per case, NPU vs GPU vs FP32.

Order: K sweep (M=512; N=2048 then N=4096), sign sweep and metadata sweep (K=N=2048).
Each case runs in a fresh process (the native runtime is single-initialization), with
all 512 rows on the NPU through the original M-split dispatcher and the original GPU
kernel on the same packed bytes as a control. Each case appends one JSON line.

    python3 tools/npu_error_probe/probe_sweep.py --dry-run                 # CPU: commands and artifacts
    python3 tools/npu_error_probe/probe_sweep.py --out RESULT_DIR          # AUP: all cases
    python3 tools/npu_error_probe/probe_report.py RESULT_DIR/probe.jsonl

Requirements on the AUP box: the root build's model extension (build/py/unified_llm_w4a16/
unified_llm_w4a16_hetero_libtorch*.so), the pinned XDNA headers (DRIVER_PATH or the
build's CMakeCache), hw_bins/npu2 and ROCm Torch. The binding is compiled with
torch.utils.cpp_extension; no CMake file, kernel or artifact is changed. Requested cases
without a native artifact are listed in probe_cases.BLOCKED and written as blocked lines.
Errors are reported, never compared against a loosened tolerance.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_cases as pc  # noqa: E402

BINDINGS = HERE / "native_matmul_bindings.cpp"
CLEARED_ENV = ("HETEROMEGA_FFN_CONFIG", "HETEROMEGA_GPU_FFN", "HETEROMEGA_GPU_FFN_SPLIT", "HSA_OVERRIDE_GFX_VERSION")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_library(root):
    found = sorted((root / "build/py/unified_llm_w4a16").glob("unified_llm_w4a16_hetero_libtorch*.so"))
    return found[0] if found else None


def load_binding(root, library, driver_include=None, verbose=False):
    """Compile/load the thin adapter against the existing native extension (cached by content)."""
    import torch
    from torch.utils.cpp_extension import load
    sys.path.insert(0, str(root))
    from experiments.ffn_partition.native_heteromosaic import find_driver_include
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("requires ROCm Torch and an AMD GPU")
    if not Path("/dev/accel/accel0").exists():
        raise RuntimeError("XDNA device /dev/accel/accel0 is unavailable")
    include = find_driver_include(root, driver_include)
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm")).resolve()
    os.environ["ROCM_HOME"] = str(rocm)
    info = dict(native_library=str(library), native_library_sha256=file_hash(library),
                binding_sha256=file_hash(BINDINGS), driver_header_sha256=file_hash(include / "amdxdna_accel.h"),
                torch=str(torch.__version__), hip=str(torch.version.hip))
    tag = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:16]
    build = Path(os.environ.get("HETEROMEGA_PROBE_BUILD_ROOT", Path.home() / ".cache/heteromega/npu_error_probe")) / tag
    build.mkdir(parents=True, exist_ok=True)
    includes = [str(root / "include"), str(include), str(rocm / "include")]
    cflags = ["-O3", "-std=c++17", "-D__HIP_PLATFORM_AMD__=1"]
    ldflags = [str(library), f"-Wl,-rpath,{library.parent}", f"-L{rocm / 'lib'}", f"-Wl,-rpath,{rocm / 'lib'}", "-lamdhip64"]
    name = "heteromega_npu_error_probe_" + tag
    if shutil.which("ninja") is None:
        return compile_without_ninja(root, name, build, includes, ldflags), info
    module = load(name=name, sources=[str(BINDINGS)], extra_include_paths=includes, extra_cflags=cflags,
                  extra_ldflags=ldflags, with_cuda=False, build_directory=str(build), verbose=verbose)
    return module, info


def compile_without_ninja(root, name, build, includes, ldflags):
    """cpp_extension.load's compile for a box without ninja (the AUP box has none, and jobs install no packages).

    Uses the compile flags of the model extension's merged FFN target, so the adapter sees the same Torch and HIP ABI
    and the same torch211-hip-compat.hpp.
    """
    import torch
    from torch.utils.cpp_extension import include_paths
    output = build / (name + sysconfig.get_config_var("EXT_SUFFIX"))
    if not output.exists():
        flags = (root / "build/src/unified_llm_w4a16_hetero/CMakeFiles/heteromega_merged_ffn.dir/flags.make").read_text()
        values = dict(line.split(" = ", 1) for line in flags.splitlines() if " = " in line and not line.startswith("#"))
        torch_lib = Path(torch.__file__).parent / "lib"
        temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")  # a killed build leaves no partial module
        subprocess.run(["c++", "-shared", f"-DTORCH_EXTENSION_NAME={name}", "-DTORCH_API_INCLUDE_EXTENSION_H",
                        *shlex.split(" ".join(values.get(k, "") for k in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"))),
                        *[f"-I{p}" for p in [sysconfig.get_path("include"), *include_paths(), *includes]],
                        str(BINDINGS), f"-L{torch_lib}", f"-Wl,-rpath,{torch_lib}", "-lc10", "-ltorch_cpu", "-ltorch",
                        "-ltorch_python", *ldflags, "-o", str(temporary)], check=True)
        temporary.rename(output)
    spec = importlib.util.spec_from_file_location(name, output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(case, args):
    """Executed in a fresh child process on the AUP box; returns one JSON-serializable record."""
    import torch
    from experiments.ffn_partition.gpu_backend import pack_heteromosaic_weights
    record = dict(case, status="starting", label="measured on AUP (NPU and GPU control)")
    fw_path = pc.artifact(args.root, case["m"], case["k"], case["n"])
    stem = f"{case['m']}x{case['k']}x{case['n']}_{pc.TILE}_{pc.COL}_{pc.DTYPE}"
    record.update(fw_path=fw_path, pdi_sha256=file_hash(args.root / fw_path / f"final_{stem}.pdi"),
                  insts_sha256=file_hash(args.root / fw_path / f"insts_{stem}.txt"))
    config = args.out / "configs" / f"{case['name']}.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps(pc.native_config(case, fw_path), indent=2) + "\n")
    module, record["binding"] = load_binding(args.root, args.native_library, args.driver_include)
    record["selected"] = dict(module.initialize(str(config), case["m"], case["k"], case["n"]))
    a, weight = pc.make_inputs(case)
    expected = pc.reference(a, weight)
    packed = np.ascontiguousarray(pack_heteromosaic_weights(weight)).reshape(-1)
    record.update(input_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                  packed_weight_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
    device = torch.device("cuda")
    x = torch.from_numpy(a).to(device=device, dtype=torch.bfloat16).contiguous()  # exact: values are BF16
    w = torch.from_numpy(packed).to(device).contiguous()
    out_npu = torch.zeros((case["m"], case["n"]), dtype=torch.bfloat16, device=device)
    out_gpu = torch.zeros_like(out_npu)
    for tensor in (x, w, out_npu):
        module.import_tensor(tensor)
    gpu_run = dict(module.run(x, out_gpu, w, False))
    first = dict(module.run(x, out_npu, w, True))
    npu_first = out_npu.float().cpu().numpy().copy()
    out_npu.fill_(float("nan"))  # a stale buffer would make the repeat check fail
    second = dict(module.run(x, out_npu, w, True))
    npu = out_npu.float().cpu().numpy()
    gpu = out_gpu.float().cpu().numpy()
    record.update(pc.compare(npu, gpu, expected), npu_repeat_bit_exact=bool(np.array_equal(npu, npu_first)),
                  completed_npu_dispatches=int(first["completed_npu_dispatches"] + second["completed_npu_dispatches"]),
                  diagnostic_wall_us=dict(gpu=gpu_run["wall_us"], npu_first=first["wall_us"], npu_second=second["wall_us"]))
    if case["series"] == "sign":  # probe_report checks NPU(-a) == -NPU(a) bit for bit
        (args.out / "outputs").mkdir(parents=True, exist_ok=True)
        np.save(args.out / "outputs" / f"{case['name']}.npy", npu.astype(np.float32))
    record["status"] = "ok" if record["completed_npu_dispatches"] == 2 and record["npu"]["finite"] else "failed"
    return record


def child_command(case, args):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--run-case", case["name"],
               "--out", str(args.out), "--root", str(args.root)]
    if args.native_library:
        command += ["--native-library", str(args.native_library)]
    if args.driver_include:
        command += ["--driver-include", str(args.driver_include)]
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("build/npu-error-probe"))
    ap.add_argument("--root", type=Path, default=pc.ROOT)
    ap.add_argument("--native-library", type=Path)
    ap.add_argument("--driver-include", type=Path)
    ap.add_argument("--cases", help="comma-separated case names (default: all, in order)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-case", help=argparse.SUPPRESS)
    args = ap.parse_args()
    args.out, args.root = args.out.resolve(), args.root.resolve()
    if args.native_library is None and not args.dry_run:
        args.native_library = default_library(args.root)
    all_cases = {c["name"]: c for c in pc.cases()}
    if args.run_case:
        record = dict(all_cases[args.run_case], status="failed")
        try:
            record = run_case(all_cases[args.run_case], args)
        except Exception as exc:  # recorded, never hidden: the line keeps the error
            record.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=5))
        print("PROBE_RESULT " + json.dumps(record, allow_nan=False if record["status"] == "ok" else True), flush=True)
        sys.exit(0 if record["status"] == "ok" else 1)
    selected = list(all_cases) if not args.cases else args.cases.split(",")
    unknown = [name for name in selected if name not in all_cases]
    if unknown:
        ap.error(f"unknown cases: {unknown}")
    missing = [n for n in selected if pc.artifact(args.root, 512, all_cases[n]["k"], all_cases[n]["n"]) is None]
    if missing:
        ap.error(f"native M-split artifacts missing under {args.root}/hw_bins/npu2 for {missing}")
    for blocked in pc.BLOCKED:
        print(f"BLOCKED {blocked['series']} K={blocked['k']} N={blocked['n']}: {blocked['reason']}")
    if not args.dry_run and (args.native_library is None or not args.native_library.is_file()):
        ap.error("native model extension not found; build unified_llm_w4a16_hetero_libtorch or pass --native-library")
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        jsonl = args.out / "probe.jsonl"
        with jsonl.open("a") as f:
            for blocked in pc.BLOCKED:
                f.write(json.dumps(dict(blocked, status="blocked", label="not executed")) + "\n")
    env = {k: v for k, v in os.environ.items() if k not in CLEARED_ENV}
    env["HETEROMOSAIC_ROOT"] = str(args.root)  # native fw_path is relative to this root
    env["PYTHONPATH"] = os.pathsep.join([str(args.root)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    failures = 0
    for name in selected:
        case, command = all_cases[name], child_command(all_cases[name], args)
        fw_path = pc.artifact(args.root, 512, case["k"], case["n"])
        print(f"{name} [{case['series']} M={case['m']} K={case['k']} N={case['n']} act={case['act']} "
              f"scales={case['scales']} zeros={case['zeros']} fw={fw_path}]: {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        start = time.perf_counter()
        result = subprocess.run(command, env=env, cwd=args.root, capture_output=True, text=True)
        (args.out / "logs").mkdir(exist_ok=True)
        (args.out / "logs" / f"{name}.log").write_text(result.stdout + result.stderr)
        lines = [line[len("PROBE_RESULT "):] for line in result.stdout.splitlines() if line.startswith("PROBE_RESULT ")]
        record = json.loads(lines[-1]) if lines else dict(case, status="failed", error=f"no result; exit {result.returncode}")
        record["process_seconds"] = round(time.perf_counter() - start, 2)
        with (args.out / "probe.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        failures += record["status"] != "ok"
        summary = {k: record.get(k) for k in ("status", "npu_vs_gpu_max_abs", "error")}
        print(f"  -> {summary} npu={record.get('npu', {}).get('mean_abs')} gpu={record.get('gpu', {}).get('mean_abs')}",
              flush=True)
    if not args.dry_run:
        print(f"wrote {args.out / 'probe.jsonl'}; {failures} failed case(s)")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
