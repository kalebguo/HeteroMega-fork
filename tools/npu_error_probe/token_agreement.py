#!/usr/bin/env python3
"""Original FFN vs merged GPU+NPU FFN on real weights: tokens, step-1 logits, small perplexity.

    python3 tools/npu_error_probe/token_agreement.py --dry-run
    python3 tools/npu_error_probe/token_agreement.py --model MODEL_DIR --build build \
        --merged-ffn-config "$PWD/deployment.json" --out RESULT_DIR/token-agreement.json

One process loads the model once with kernels/gpu/tools/run_model_ffn.py's loader
(stubbed transformers/safetensors, merged shards, the fixed 1024 reference token IDs of
docs/runs/2026-10-12-aup/gpu-model-result.json; no tokenizer). GPU attention with
512-row chunks, so every prefill row of the merged pass goes through the M512 artifact.
HETEROMEGA_FFN_CONFIG is set only for merged passes (the C++ hook reads it on each
forward, as experiments.ffn_deploy.model_smoke relies on) and removed for original passes.
merged_ffn_completed_dispatches() is checked for every pass: 32 per 512-row prefill
chunk for merged passes, 0 for original passes; a wrong count fails the run.

Reported (no tolerance is applied; the numbers are the result):
- greedy tokens: generate(prompt, 64, temperature 0) per path; positionwise agreement
  count and first disagreeing position;
- step-1 logits: last-position logits of the 1024-token prefill; max absolute
  difference, argmax equality, plus an original-vs-original repeat as a control;
- perplexity: prefill the first 512 real tokens (the only place the merged FFN runs),
  then teacher-force the next 512 real tokens with single-token decode steps (original
  FFN, attending to the prefill's KV cache) and score each next token. The model is
  built with LM_HEAD_ONLY_LAST=1, so prefill exposes only last-position logits; this is
  why scored tokens come from decode steps. Small sample; not a model-quality benchmark.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 512
SWITCHES = ("HETEROMEGA_FFN_CONFIG", "HETEROMEGA_GPU_FFN", "HETEROMEGA_GPU_FFN_SPLIT")


def agreement(a, b):
    a, b = list(a), list(b)
    total = min(len(a), len(b))
    mismatches = [i for i in range(total) if a[i] != b[i]]
    return dict(agree=total - len(mismatches), total=total, first_disagreement=mismatches[0] if mismatches else None,
                lengths=(len(a), len(b)))


def nll(logits, target):
    """Negative log-likelihood of one target under one logits row (float64, stable)."""
    row = np.asarray(logits, dtype=np.float64).reshape(-1)
    top = row.max()
    return float(top + math.log(np.exp(row - top).sum()) - row[int(target)])


def perplexity(nlls):
    return dict(tokens=len(nlls), mean_nll=float(np.mean(nlls)), perplexity=float(math.exp(np.mean(nlls))))


def logit_stats(merged, original):
    merged, original = np.asarray(merged, np.float64).reshape(-1), np.asarray(original, np.float64).reshape(-1)
    top_m, top_o = set(np.argsort(merged)[-5:].tolist()), set(np.argsort(original)[-5:].tolist())
    return dict(max_abs_diff=float(np.max(np.abs(merged - original))), argmax_equal=int(merged.argmax()) == int(original.argmax()),
                top5_overlap=len(top_m & top_o))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="directory with the AWQ *.safetensors shards")
    ap.add_argument("--build", default="build")
    ap.add_argument("--config", default="py/unified_llm_w4a16/configs/configs_strixP_llama3_8b.json5")
    ap.add_argument("--merged-ffn-config", type=Path)
    ap.add_argument("--ref", type=Path, default=ROOT / "docs/runs/2026-10-12-aup/gpu-model-result.json")
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--ppl-prefix", type=int, default=512)
    ap.add_argument("--ppl-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, default=Path("build/npu-error-probe/token-agreement.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ref = json.loads(args.ref.read_text())
    ids_all = ref["input_ids"][0]
    if args.prompt_tokens % CHUNK or not 0 < args.prompt_tokens <= len(ids_all):
        ap.error(f"--prompt-tokens must be a multiple of {CHUNK} within the {len(ids_all)} reference tokens")
    if args.ppl_prefix % CHUNK or args.ppl_prefix + args.ppl_tokens > len(ids_all) or args.ppl_tokens < 1:
        ap.error("--ppl-prefix must be a multiple of 512 and prefix + scored tokens must fit the reference")
    plan = [f"load model (GPU attention, gpu_chunk_size={CHUNK}), prompt {args.prompt_tokens} reference tokens",
            "step-1 logits: original, original (repeat control), merged",
            f"greedy generate({args.new_tokens}): original, merged",
            f"perplexity: prefill {args.ppl_prefix} tokens then score {args.ppl_tokens} teacher-forced tokens: original, merged"]
    if args.dry_run:
        print("\n".join(["DRY RUN (no Torch import, no model load):"] + [f"  {i + 1}. {s}" for i, s in enumerate(plan)]))
        return
    if not args.model or args.merged_ffn_config is None:
        ap.error("--model and --merged-ffn-config are required")
    merged_config = args.merged_ffn_config.resolve()
    for name in SWITCHES:
        os.environ.pop(name, None)

    sys.path.insert(0, str(ROOT / args.build / "py" / "unified_llm_w4a16"))
    sys.path.insert(1, str(ROOT / "py" / "unified_llm_w4a16"))
    sys.path.insert(2, str(ROOT / "kernels" / "gpu" / "tools"))
    from run_model_ffn import merge_shards, stub_missing_modules
    stub_missing_modules()
    import torch
    import unified_llm_w4a16_hetero_libtorch as backend  # before the frontend, as run_model_ffn.py does
    from llama3_8b_w4a16_model import LLaMA3W4A16Model, _model_name_from_path, load_config_with_comments

    model_dir = Path(args.model).resolve()
    merged_weights = ROOT / "py/unified_llm_w4a16/model_weights" / f"{_model_name_from_path(str(model_dir))}.safetensors"
    if not merged_weights.exists():
        merge_shards(sorted(model_dir.glob("*.safetensors")), merged_weights)
    config = load_config_with_comments(str(ROOT / args.config))
    config.update(heterogeneity="gpu", prompt_len=args.prompt_tokens, usePreSavedWeights=False, warmup=False,
                  debug_verbosity=1, dummy_weights=False, chunking=True)
    config["gpu_chunking"] = dict(gpu_chunk_size=CHUNK, gpu_chunking_inflight=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config_path = args.out.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, indent=2))
    record = dict(label="measured on AUP; numbers reported without tolerance", backend=backend.__file__,
                  merged_ffn_config=str(merged_config), prompt_tokens=args.prompt_tokens, new_tokens=args.new_tokens,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(), plan=plan, status="starting")
    try:
        start = time.perf_counter()
        model = LLaMA3W4A16Model(model_path=str(model_dir), tokenizer_path=str(model_dir),
                                 max_seq_len=max(1536, args.prompt_tokens + args.new_tokens), device="cuda",
                                 config_path=str(config_path))
        record["load_s"] = time.perf_counter() - start
        counter = model.model.merged_ffn_completed_dispatches

        def checked(merged, expected_completions, fn):
            if merged:
                os.environ["HETEROMEGA_FFN_CONFIG"] = str(merged_config)
            else:
                os.environ.pop("HETEROMEGA_FFN_CONFIG", None)
            before = int(counter())
            with torch.no_grad():
                value = fn()
            torch.cuda.synchronize()
            done = int(counter()) - before
            if done != expected_completions:
                raise RuntimeError(f"merged={merged}: {done} merged FFN completions, expected {expected_completions}")
            return value

        ids = torch.tensor([ids_all[:args.prompt_tokens]], dtype=torch.int64, device="cuda")
        per_prefill = 32 * (args.prompt_tokens // CHUNK)

        def last_logits():
            out = model.model.forward(ids, 0)
            return out.float().reshape(-1, out.shape[-1])[-1].cpu().numpy()

        original = checked(False, 0, last_logits)
        repeat = checked(False, 0, last_logits)
        merged = checked(True, per_prefill, last_logits)
        record["step1_logits"] = dict(merged_vs_original=logit_stats(merged, original),
                                      original_repeat_control=logit_stats(repeat, original))

        def greedy():
            return model.generate(ids, max_new_tokens=args.new_tokens, temperature=0.0)[0, args.prompt_tokens:].tolist()

        tokens = dict(original=checked(False, 0, greedy), merged=checked(True, per_prefill, greedy))
        record["greedy_tokens"] = dict(tokens, **agreement(tokens["original"], tokens["merged"]))

        def scored():
            prefix = torch.tensor([ids_all[:args.ppl_prefix]], dtype=torch.int64, device="cuda")
            out = model.model.forward(prefix, 0)
            nlls = [nll(out.float().reshape(-1, out.shape[-1])[-1].cpu().numpy(), ids_all[args.ppl_prefix])]
            for t in range(args.ppl_prefix, args.ppl_prefix + args.ppl_tokens - 1):
                step = torch.tensor([[ids_all[t]]], dtype=torch.int64, device="cuda")
                out = model.model.forward(step, t)
                nlls.append(nll(out.float().reshape(-1, out.shape[-1])[-1].cpu().numpy(), ids_all[t + 1]))
            return nlls

        ppl_completions = 32 * (args.ppl_prefix // CHUNK)
        record["perplexity"] = dict(prefix_tokens=args.ppl_prefix,
                                    original=perplexity(checked(False, 0, scored)),
                                    merged=perplexity(checked(True, ppl_completions, scored)))
        record["status"] = "completed"
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        os.environ.pop("HETEROMEGA_FFN_CONFIG", None)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        summary = {k: record.get(k) for k in ("status", "error", "step1_logits", "perplexity")}
        if "greedy_tokens" in record:
            summary["greedy"] = {k: record["greedy_tokens"][k] for k in ("agree", "total", "first_disagreement")}
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
