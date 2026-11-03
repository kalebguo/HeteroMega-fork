#!/usr/bin/env python3
"""One inference path per fresh process on real weights: step-1 logits, greedy tokens, teacher-forced perplexity.

token_agreement.py switches the merged FFN inside one GPU-attention process, so it cannot reach the native
heterogeneous path, whose backend is fixed when the model is built. Here each process runs one method and writes an
.npz; --compare reports every run against gpu_original across processes. A second process of a method is the control.

    python3 tools/npu_error_probe/path_logits.py --method heteromosaic --model MODEL_DIR --build build \
        --merged-ffn-config "$PWD/deployment.json" --prompt-tokens 1024 --out RESULT_DIR/heteromosaic-1024-r0.npz
    python3 tools/npu_error_probe/path_logits.py --compare RESULT_DIR --json RESULT_DIR/path-comparison.json

Every run records last-position logits of the prefill, twice in one process, and greedy tokens. When --ref has tokens
after the prompt, a run also prefills the prompt and scores the next 512 with single-token decode steps. Those decode
steps use the NPU on native methods (32 calls each) and the original GPU FFN on GPU-attention methods, so native scores
include native decode deviations. Numbers are reported without a tolerance. Small sample; not a model-quality benchmark.

The fixed reference prompt has 12 distinct tokens and a period of 10, and every path predicts it with a perplexity near
1, so it barely tests numerics. --sample-prompt writes a --ref file of model-sampled tokens instead (gpu_original, fixed
seed, temperature 1). The box has no tokenizer, and this needs none.

    python3 tools/npu_error_probe/path_logits.py --sample-prompt RESULT_DIR/sampled.json --seed 1 --model MODEL_DIR
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from token_agreement import agreement, logit_stats, nll, perplexity  # noqa: E402

ROOT = HERE.parents[1]
CHUNK = 512
MAX_SEQ_LEN = 1600
BOS = 128000
# method: (heterogeneity, merged GPU+NPU FFN); same meaning as kernels/gpu/tools/evaluate_prefill.py METHODS
METHODS = {"gpu_original": ("gpu", False), "merged_ffn": ("gpu", True), "heteromosaic": ("hetero", False),
           "merged_ffn_hetero": ("hetero", True), "heteromosaic_mdown": ("hetero", False),  # mdown: pass its --config
           "heteromosaic_gateup": ("hetero", False), "heteromosaic_gateup16384": ("hetero", False),
           "merged_ffn_p8192": ("gpu", True)}  # pass --config, --merged-ffn-config
REFERENCE = "gpu_original"
SWITCHES = ("HETEROMEGA_FFN_CONFIG", "HETEROMEGA_GPU_FFN", "HETEROMEGA_GPU_FFN_SPLIT")


def relative_l2(a, reference):
    a, reference = np.asarray(a, np.float64).reshape(-1), np.asarray(reference, np.float64).reshape(-1)
    return float(np.linalg.norm(a - reference) / np.linalg.norm(reference))


def kl_nats(reference, other):
    """KL(reference || other) of the two next-token distributions, in nats."""
    def log_softmax(logits):
        row = np.asarray(logits, np.float64).reshape(-1)
        row = row - row.max()
        return row - np.log(np.exp(row).sum())
    log_p, log_q = log_softmax(reference), log_softmax(other)
    return float(np.sum(np.exp(log_p) * (log_p - log_q)))


def scored_stats(logits, reference):
    """Agreement over all scored positions; each row is one next-token distribution."""
    kl = [kl_nats(r, l) for r, l in zip(reference, logits)]
    return dict(positions=len(kl), argmax_agreement=float(np.mean(logits.argmax(axis=1) == reference.argmax(axis=1))),
                mean_kl_nats=float(np.mean(kl)), median_kl_nats=float(np.median(kl)), max_kl_nats=float(np.max(kl)),
                median_relative_l2=float(np.median([relative_l2(l, r) for r, l in zip(reference, logits)])))


def run_stem(method, prompt_tokens, repeat):
    return f"{method}-{prompt_tokens}-r{repeat}"


def compare(directory):
    """One row per run against the first gpu_original process of the same prompt length."""
    runs = {path.stem: np.load(path, allow_pickle=False) for path in sorted(Path(directory).glob("*.npz"))}
    rows = []
    for stem, run in runs.items():
        method, prompt_tokens, _ = stem.rsplit("-", 2)
        reference = runs[run_stem(REFERENCE, prompt_tokens, 0)]
        row = dict(run=stem, method=method, prompt_tokens=int(prompt_tokens),
                   meta=json.loads(str(run["meta"])), step1=logit_stats(run["logits"], reference["logits"]),
                   step1_relative_l2=relative_l2(run["logits"], reference["logits"]),
                   step1_kl_nats=kl_nats(reference["logits"], run["logits"]),
                   within_process_repeat_max_abs=float(np.max(np.abs(run["logits"].astype(np.float64) - run["logits_repeat"]))),
                   greedy=agreement(run["tokens"].tolist(), reference["tokens"].tolist()))
        if "nlls" in run:
            row["perplexity"] = perplexity(run["nlls"].tolist())
            row["reference_perplexity"] = perplexity(reference["nlls"].tolist())
            row["mean_abs_nll_difference"] = float(np.mean(np.abs(run["nlls"] - reference["nlls"])))
        if "scored_logits" in run:
            row["scored"] = scored_stats(run["scored_logits"], reference["scored_logits"])
        rows.append(row)
    return rows


def load_model(args, heterogeneity, merged, prompt_tokens, out):
    for name in SWITCHES:
        os.environ.pop(name, None)
    if merged:
        os.environ["HETEROMEGA_FFN_CONFIG"] = str(args.merged_ffn_config.resolve())
    if heterogeneity == "hetero":
        # The native loader otherwise looks in the original author's checkout and silently falls back to the GPU.
        os.environ.setdefault("HETEROMOSAIC_ROOT", str(ROOT))
    sys.path.insert(0, str(ROOT / args.build / "py" / "unified_llm_w4a16"))
    sys.path.insert(1, str(ROOT / "py" / "unified_llm_w4a16"))
    sys.path.insert(2, str(ROOT / "kernels" / "gpu" / "tools"))
    from run_model_ffn import stub_missing_modules
    stub_missing_modules()
    import torch
    import unified_llm_w4a16_hetero_libtorch  # noqa: F401  before the frontend, as run_model_ffn.py does
    from llama3_8b_w4a16_model import LLaMA3W4A16Model, load_config_with_comments

    config = load_config_with_comments(str(ROOT / args.config))
    config.update(heterogeneity=heterogeneity, prompt_len=prompt_tokens, usePreSavedWeights=False, warmup=False,
                  debug_verbosity=1, dummy_weights=False)
    chunking = dict(config.get("gpu_chunking") or {})
    chunking["gpu_chunk_size"] = getattr(args, "chunk", CHUNK)
    inflight = getattr(args, "inflight", None)  # as run_model_ffn.py --inflight
    if inflight is not None:
        chunking["gpu_chunking_inflight"] = inflight
        for group in config.get("kernels_gemm_chunked", []):
            if group.get("prompt_len") == prompt_tokens:
                group["hetero_inflight"] = inflight
    config["gpu_chunking"] = chunking
    out.parent.mkdir(parents=True, exist_ok=True)
    config_path = out.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, indent=2))
    model_dir = str(Path(args.model).resolve())
    return LLaMA3W4A16Model(model_path=model_dir, tokenizer_path=model_dir, max_seq_len=getattr(args, "max_seq_len", MAX_SEQ_LEN), device="cuda",
                            config_path=str(config_path)), torch


def sample_prompt(args):
    """Temperature-1 tokens from gpu_original, from BOS; one single-token decode step per token."""
    model, torch = load_model(args, "gpu", False, 1, args.sample_prompt)
    rng = np.random.default_rng(args.seed)
    ids = [BOS]
    with torch.no_grad():
        for t in range(args.sample_tokens - 1):
            out = model.model.forward(torch.tensor([[ids[-1]]], dtype=torch.int64, device="cuda"), t)
            row = out.float().reshape(-1, out.shape[-1])[-1].cpu().numpy().astype(np.float64)
            p = np.exp(row - row.max())
            ids.append(int(rng.choice(len(p), p=p / p.sum())))
    args.sample_prompt.write_text(json.dumps(dict(input_ids=[ids], seed=args.seed, temperature=1.0, source="gpu_original sampled")) + "\n")
    print(json.dumps(dict(tokens=len(ids), distinct=len(set(ids)), first=ids[:16], seed=args.seed)), flush=True)


def run_method(args):
    heterogeneity, merged = METHODS[args.method]
    ids_all = json.loads(args.ref.read_text())["input_ids"][0]
    start = time.perf_counter()
    model, torch = load_model(args, heterogeneity, merged, args.prompt_tokens, args.out)
    meta = dict(method=args.method, heterogeneity=heterogeneity, merged=merged, prompt_tokens=args.prompt_tokens,
                new_tokens=args.new_tokens, load_s=time.perf_counter() - start, torch=torch.__version__)
    counter = model.model.merged_ffn_completed_dispatches
    per_prefill = 32 * (args.prompt_tokens // CHUNK) if merged else 0

    def checked(expected_prefills, fn):
        before = int(counter())
        with torch.no_grad():
            value = fn()
        torch.cuda.synchronize()
        done = int(counter()) - before
        if done != per_prefill * expected_prefills:  # a wrong count means this run did not exercise the method
            raise RuntimeError(f"{args.method}: {done} merged FFN completions, expected {per_prefill * expected_prefills}")
        return value

    ids = torch.tensor([ids_all[:args.prompt_tokens]], dtype=torch.int64, device="cuda")

    def last_row(out):
        return out.float().reshape(-1, out.shape[-1])[-1].cpu().numpy()

    def last_logits():
        return last_row(model.model.forward(ids, 0))

    arrays = dict(logits=checked(1, last_logits), logits_repeat=checked(1, last_logits))
    arrays["tokens"] = np.asarray(checked(1, lambda: model.generate(ids, max_new_tokens=args.new_tokens, temperature=0.0)[
        0, args.prompt_tokens:].tolist()), dtype=np.int64)

    def scored():
        rows = [last_logits()]
        for t in range(args.prompt_tokens, args.prompt_tokens + args.ppl_tokens - 1):
            step = torch.tensor([[ids_all[t]]], dtype=torch.int64, device="cuda")
            rows.append(last_row(model.model.forward(step, t)))
        return np.stack(rows)

    if args.prompt_tokens + args.ppl_tokens <= len(ids_all):
        arrays["scored_logits"] = checked(1, scored)
        arrays["nlls"] = np.asarray([nll(row, ids_all[args.prompt_tokens + i]) for i, row in enumerate(arrays["scored_logits"])])
    np.savez(args.out, meta=json.dumps(meta), **arrays)
    print(json.dumps(dict(meta, tokens=arrays["tokens"].tolist()[:8])), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--model", help="directory with the AWQ *.safetensors shards")
    ap.add_argument("--build", default="build")
    ap.add_argument("--config", default="py/unified_llm_w4a16/configs/configs_strixP_llama3_8b.json5")
    ap.add_argument("--merged-ffn-config", type=Path)
    ap.add_argument("--ref", type=Path, default=ROOT / "docs/runs/2026-10-12-aup/gpu-model-result.json")
    ap.add_argument("--prompt-tokens", type=int, choices=(512, 1024, 2048, 4096), default=1024)
    ap.add_argument("--chunk", type=int, default=CHUNK, help="prefill chunk rows; merged methods need 512")
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN, help="capacity of the KV caches and scratch buffers")
    ap.add_argument("--inflight", type=int, help="inflight chunk slots, as run_model_ffn.py --inflight")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--ppl-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", type=Path, help="directory of .npz runs; prints the comparison")
    ap.add_argument("--json", type=Path, help="with --compare: write the rows here")
    ap.add_argument("--sample-prompt", type=Path, help="write a --ref file of model-sampled tokens here")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample-tokens", type=int, default=1536)
    args = ap.parse_args()
    if args.sample_prompt:
        if not args.model:
            ap.error("--sample-prompt needs --model")
        sample_prompt(args)
        return
    if args.compare:
        rows = compare(args.compare)
        if args.json:
            args.json.write_text(json.dumps(rows, indent=2) + "\n")
        for r in rows:
            print(r["run"], f"max_abs={r['step1']['max_abs_diff']:.4f} rel_l2={r['step1_relative_l2']:.5f}",
                  f"kl={r['step1_kl_nats']:.6f} argmax_equal={r['step1']['argmax_equal']} top5={r['step1']['top5_overlap']}",
                  f"repeat={r['within_process_repeat_max_abs']:.4f} greedy={r['greedy']['agree']}/{r['greedy']['total']}",
                  f"ppl={r['perplexity']['perplexity']:.4f}" if "perplexity" in r else "",
                  f"scored_argmax={r['scored']['argmax_agreement']:.4f} scored_kl={r['scored']['mean_kl_nats']:.6f}" if "scored" in r else "")
        return
    if not (args.method and args.model and args.out) or (METHODS[args.method][1] and args.merged_ffn_config is None):
        ap.error("--method, --model and --out are required; merged methods also need --merged-ffn-config")
    run_method(args)


if __name__ == "__main__":
    main()
