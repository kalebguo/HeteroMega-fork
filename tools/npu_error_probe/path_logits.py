#!/usr/bin/env python3
"""One inference path per fresh process on real weights: step-1 logits, greedy tokens, teacher-forced perplexity.

token_agreement.py switches the merged FFN inside one GPU-attention process, so it cannot reach the native
heterogeneous path, whose backend is fixed when the model is built. Here each process runs one method and writes an
.npz; --compare reports every run against gpu_original across processes. A second process of a method is the control.

    python3 tools/npu_error_probe/path_logits.py --method heteromosaic --model MODEL_DIR --build build \
        --merged-ffn-config "$PWD/deployment.json" --prompt-tokens 1024 --out RESULT_DIR/heteromosaic-1024-r0.npz
    python3 tools/npu_error_probe/path_logits.py --compare RESULT_DIR --json RESULT_DIR/path-comparison.json

Every run records last-position logits of the prefill, twice in one process, and greedy tokens. A 512-token run also
prefills the first 512 real tokens and scores the next 512 with single-token decode steps. Those decode steps use the
NPU on native methods (32 calls each) and the original GPU FFN on GPU-attention methods, so a native perplexity includes
native decode deviations. Numbers are reported without a tolerance. Small sample; not a model-quality benchmark.
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
# method: (heterogeneity, merged GPU+NPU FFN); same meaning as kernels/gpu/tools/evaluate_prefill.py METHODS
METHODS = {"gpu_original": ("gpu", False), "merged_ffn": ("gpu", True), "heteromosaic": ("hetero", False),
           "merged_ffn_hetero": ("hetero", True)}
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


def run_stem(method, prompt_tokens, repeat):
    return f"{method}-{prompt_tokens}-r{repeat}"


def compare(directory):
    """One row per run against the first gpu_original process of the same prompt length."""
    runs = {path.stem: dict(np.load(path, allow_pickle=False)) for path in sorted(Path(directory).glob("*.npz"))}
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
        rows.append(row)
    return rows


def run_method(args):
    heterogeneity, merged = METHODS[args.method]
    ref = json.loads(args.ref.read_text())
    ids_all = ref["input_ids"][0]
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
    config.update(heterogeneity=heterogeneity, prompt_len=args.prompt_tokens, usePreSavedWeights=False, warmup=False,
                  debug_verbosity=1, dummy_weights=False)
    chunking = dict(config.get("gpu_chunking") or {})
    chunking["gpu_chunk_size"] = CHUNK
    config["gpu_chunking"] = chunking
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.out.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, indent=2))
    model_dir = str(Path(args.model).resolve())
    start = time.perf_counter()
    model = LLaMA3W4A16Model(model_path=model_dir, tokenizer_path=model_dir, max_seq_len=1536, device="cuda",
                             config_path=str(config_path))
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

    def last_logits():
        out = model.model.forward(ids, 0)
        return out.float().reshape(-1, out.shape[-1])[-1].cpu().numpy()

    arrays = dict(logits=checked(1, last_logits), logits_repeat=checked(1, last_logits))
    arrays["tokens"] = np.asarray(checked(1, lambda: model.generate(ids, max_new_tokens=args.new_tokens, temperature=0.0)[
        0, args.prompt_tokens:].tolist()), dtype=np.int64)

    def scored():
        out = model.model.forward(ids, 0)
        nlls = [nll(out.float().reshape(-1, out.shape[-1])[-1].cpu().numpy(), ids_all[args.prompt_tokens])]
        for t in range(args.prompt_tokens, args.prompt_tokens + args.ppl_tokens - 1):
            step = torch.tensor([[ids_all[t]]], dtype=torch.int64, device="cuda")
            out = model.model.forward(step, t)
            nlls.append(nll(out.float().reshape(-1, out.shape[-1])[-1].cpu().numpy(), ids_all[t + 1]))
        return nlls

    if args.prompt_tokens + args.ppl_tokens <= len(ids_all):
        arrays["nlls"] = np.asarray(checked(1, scored), dtype=np.float64)
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
    ap.add_argument("--prompt-tokens", type=int, choices=(512, 1024), default=1024)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--ppl-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", type=Path, help="directory of .npz runs; prints the comparison")
    ap.add_argument("--json", type=Path, help="with --compare: write the rows here")
    args = ap.parse_args()
    if args.compare:
        rows = compare(args.compare)
        if args.json:
            args.json.write_text(json.dumps(rows, indent=2) + "\n")
        for r in rows:
            print(r["run"], f"max_abs={r['step1']['max_abs_diff']:.4f} rel_l2={r['step1_relative_l2']:.5f}",
                  f"kl={r['step1_kl_nats']:.6f} argmax_equal={r['step1']['argmax_equal']} top5={r['step1']['top5_overlap']}",
                  f"repeat={r['within_process_repeat_max_abs']:.4f} greedy={r['greedy']['agree']}/{r['greedy']['total']}",
                  f"ppl={r['perplexity']['perplexity']:.4f}" if "perplexity" in r else "")
        return
    if not (args.method and args.model and args.out) or (METHODS[args.method][1] and args.merged_ffn_config is None):
        ap.error("--method, --model and --out are required; merged methods also need --merged-ffn-config")
    run_method(args)


if __name__ == "__main__":
    main()
