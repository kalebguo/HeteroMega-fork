#!/usr/bin/env python3
"""End-to-end Llama 3.1 8B W4A16 runs with and without the HeteroMega GPU FFN (the model switch of kernels/gpu).

Needs:
  - the root project built with -DHETEROMEGA_GPU_FFN=ON (target unified_llm_w4a16_hetero_libtorch) into --build,
  - the safetensors shards of hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 (revision
    db1f81ad4b8c7e39777509fac66c652eb0a52f91) in --model.
It does not need transformers, safetensors or huggingface_hub:
  - the prompt is the fixed 1024 token IDs of docs/runs/2026-10-12-aup/gpu-model-result.json (--ref),
  - this script merges the shards into the file that the frontend would write (py/unified_llm_w4a16/model_weights/),
  - stub modules stand in for the frontend's imports of those packages when they are missing (no tokenizer).

Usage, from the repo root (HETEROMEGA_GPU_FFN selects the FFN path; the binary is the same):
  HETEROMEGA_GPU_FFN=0 python3 kernels/gpu/tools/run_model_ffn.py --model DIR --ref REF.json --out off-1.json
  HETEROMEGA_GPU_FFN=1 python3 kernels/gpu/tools/run_model_ffn.py --model DIR --ref REF.json --out on-1.json

Per process: load the model once, run one warm-up generation with as many tokens as the reference and compare its
tokens, then run --reps pairs of generate(1 token) and generate(--new-tokens). The warm-up also builds the GPU FFN
weights. prefill_s is the wall time of generate(1). decode_tok_s = (new_tokens - 1) / (wall(new_tokens) - wall(1)).

--dummy uses the backend's seeded random weights (initialize_dummy_weights, seed 42) instead of --model: no download,
same timing work, but the tokens are not meaningful. Compare the tokens of the ON and OFF runs with each other only.

Merged GPU+NPU FFN (HETEROMEGA_FFN_CONFIG, set by evaluate_prefill.py for its merged methods only): every generate call
(warm-up and each repetition) is one prefill pass. The process reads model.model.merged_ffn_completed_dispatches()
before and after each call, outside the synchronized timer window, and stores the per-pass difference. The counter is
recorded for every method when the binding exists, so a control job that inherited the candidate shows completions.
--gpu-chunk-size and --inflight override the configuration's chunk schedule for every method that a batch compares.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def stub_missing_modules():
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("not available in run_model_ffn.py")

    class NoAuto:
        from_pretrained = staticmethod(unavailable)

    stubs = {
        "transformers": {"AutoTokenizer": NoAuto, "AutoModelForCausalLM": NoAuto},
        "huggingface_hub": {"snapshot_download": unavailable},
        "safetensors": {},
        "safetensors.torch": {"load_file": unavailable, "save_file": unavailable},
    }
    for name, attrs in stubs.items():
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            sys.modules[name] = module
            print(f"stub module: {name}", flush=True)
    if isinstance(sys.modules.get("safetensors"), types.ModuleType) and "safetensors.torch" in sys.modules:
        sys.modules["safetensors"].torch = sys.modules["safetensors.torch"]


def read_header(path):
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    header.pop("__metadata__", None)
    return 8 + size, header


def merge_shards(shards, out):
    """Concatenates safetensors files into one file, tensor data unchanged."""
    entries, layout, offset = [], {}, 0
    for path in shards:
        data_start, header = read_header(path)
        for name, info in header.items():
            if name in layout:
                sys.exit(f"tensor {name} is in more than one shard")
            begin, end = info["data_offsets"]
            layout[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + end - begin]}
            entries.append((path, data_start + begin, end - begin))
            offset += end - begin
    blob = json.dumps(layout, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = out.with_name(out.name + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for path, start, size in entries:
            with open(path, "rb") as src:
                src.seek(start)
                shutil.copyfileobj(SizedReader(src, size), f, 64 << 20)
    tmp.rename(out)
    print(f"merged {len(shards)} shards, {len(layout)} tensors, into {out}", flush=True)


class SizedReader:
    def __init__(self, f, size):
        self.f, self.left = f, size

    def read(self, n=-1):
        n = self.left if n < 0 else min(n, self.left)
        data = self.f.read(n)
        self.left -= len(data)
        return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="directory with the *.safetensors shards")
    p.add_argument("--dummy", action="store_true", help="seeded random weights instead of --model")
    p.add_argument("--ref", required=True, help="gpu-model-result.json with input_ids and generated_ids")
    p.add_argument("--out", required=True)
    p.add_argument("--build", default="build-model", help="build directory, relative to the repo root")
    p.add_argument("--config", default="py/unified_llm_w4a16/configs/configs_strixP_llama3_8b.json5")
    p.add_argument("--heterogeneity", choices=("gpu", "hetero"), default="gpu")
    p.add_argument("--prompt-tokens", type=int, help="use a prefix of --ref; never repeat or pad token IDs")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--gpu-chunk-size", type=int, help="override gpu_chunking.gpu_chunk_size (GPU attention backend)")
    p.add_argument("--inflight", type=int, help="override gpu_chunking_inflight and this prompt length's hetero_inflight")
    args = p.parse_args()
    if not args.dummy and not args.model:
        p.error("--model is required without --dummy")
    if any(v is not None and v < 1 for v in (args.gpu_chunk_size, args.inflight)):
        p.error("--gpu-chunk-size and --inflight must be positive")
    ref = json.loads(Path(args.ref).read_text())
    full_length = len(ref["input_ids"][0])
    prompt_length = full_length if args.prompt_tokens is None else args.prompt_tokens
    if not 0 < prompt_length <= full_length:
        p.error(f"--prompt-tokens must be between 1 and the reference length ({full_length})")

    sys.path.insert(0, str(ROOT / args.build / "py" / "unified_llm_w4a16"))
    sys.path.insert(1, str(ROOT / "py" / "unified_llm_w4a16"))
    stub_missing_modules()
    import torch
    import unified_llm_w4a16_hetero_libtorch as backend  # before the frontend, so that its sys.path entries do not win
    from llama3_8b_w4a16_model import LLaMA3W4A16Model, _model_name_from_path, load_config_with_comments

    out = Path(args.out).resolve()
    model_dir = None if args.dummy else Path(args.model).resolve()
    if model_dir is not None:
        merged = ROOT / "py/unified_llm_w4a16/model_weights" / f"{_model_name_from_path(str(model_dir))}.safetensors"
        if not merged.exists():
            merge_shards(sorted(model_dir.glob("*.safetensors")), merged)

    config = load_config_with_comments(str(ROOT / args.config))
    config.update(heterogeneity=args.heterogeneity, prompt_len=prompt_length, usePreSavedWeights=False, warmup=False, debug_verbosity=1,
                  dummy_weights=args.dummy)
    if args.gpu_chunk_size is not None or args.inflight is not None:
        chunking = dict(config.get("gpu_chunking") or {})
        if args.gpu_chunk_size is not None:
            chunking["gpu_chunk_size"] = args.gpu_chunk_size
        if args.inflight is not None:
            chunking["gpu_chunking_inflight"] = args.inflight
        config["gpu_chunking"] = chunking
    if args.inflight is not None:
        for group in config.get("kernels_gemm_chunked", []):
            if group.get("prompt_len") == prompt_length:
                group["hetero_inflight"] = args.inflight
    config_path = out.with_name(out.stem + ".config.json")
    config_path.write_text(json.dumps(config, indent=2))

    ids = torch.tensor([ref["input_ids"][0][:prompt_length]], dtype=torch.int64, device="cuda")
    sha = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    if prompt_length == full_length:
        assert sha == ref["input_sha256"], sha

    record = {"gpu_ffn_env": os.environ.get("HETEROMEGA_GPU_FFN", ""), "backend": backend.__file__,
              "config": str(config_path), "input_sha256": sha, "new_tokens": args.new_tokens, "dummy": args.dummy,
              "merged_ffn_config_env": os.environ.get("HETEROMEGA_FFN_CONFIG", ""),
              "merged_ffn_finite_check_env": os.environ.get("HETEROMEGA_FFN_FINITE_CHECK", ""),
              "merged_ffn_timers_env": os.environ.get("HETEROMEGA_FFN_TIMERS", ""),
              "heteromosaic_root_env": os.environ.get("HETEROMOSAIC_ROOT", ""),
              "gpu_chunk_size": args.gpu_chunk_size, "inflight": args.inflight}
    record.update(prompt_tokens=prompt_length, heterogeneity=args.heterogeneity, model=str(model_dir),
                  torch_version=torch.__version__, gpu_name=torch.cuda.get_device_name())
    t = time.perf_counter()
    path = None if model_dir is None else str(model_dir)
    model = LLaMA3W4A16Model(model_path=path, tokenizer_path=path, max_seq_len=max(args.max_seq_len, prompt_length + args.new_tokens), device="cuda",
                             config_path=str(config_path))
    record["load_s"] = time.perf_counter() - t
    # Host-side atomic counter of successful merged-FFN NPU runs; absent from bindings older than the merged FFN.
    counter = getattr(getattr(model, "model", None), "merged_ffn_completed_dispatches", None)
    record["merged_ffn_counter"] = counter is not None

    def completions():
        return None if counter is None else int(counter())

    def timed(n):
        before = completions()  # read outside the synchronized timer window
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            result = model.generate(ids, max_new_tokens=n, temperature=0.0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        after = completions()
        return elapsed, result[0, prompt_length:].tolist(), None if before is None else after - before

    want = ref["generated_ids"]
    record["warmup_s"], record["warmup_ids"], record["warmup_merged_ffn_completions"] = timed(len(want))
    record["matches_reference"] = None if args.dummy or prompt_length != full_length else record["warmup_ids"] == want
    verdict = {None: "no matching reference (dummy weights or shortened prompt)", True: "matches the reference", False: "DIFFERS from the reference"}
    print("WARMUP", record["warmup_s"], record["warmup_ids"], verdict[record["matches_reference"]], flush=True)

    record["reps"] = []
    for rep in range(args.reps):
        prefill_s, prefill_ids, prefill_merged = timed(1)
        gen_s, gen_ids, gen_merged = timed(args.new_tokens)
        decode = (args.new_tokens - 1) / (gen_s - prefill_s) if gen_s > prefill_s else None
        record["reps"].append({"rep": rep, "prefill_s": prefill_s, "generate_s": gen_s, "decode_tok_s": decode,
                               "generated_ids": gen_ids, "prefill_ids": prefill_ids,
                               "prefill_merged_ffn_completions": prefill_merged, "generate_merged_ffn_completions": gen_merged})
        print(f"REP {rep} prefill_s={prefill_s:.4f} generate_s={gen_s:.4f} decode_tok_s={decode} "
              f"merged_ffn_completions={prefill_merged},{gen_merged}", flush=True)
    record["peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / 2**30
    out.write_text(json.dumps(record, indent=2))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
