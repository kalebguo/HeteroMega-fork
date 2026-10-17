#!/usr/bin/env python3
"""Summary of run_model_ffn.py results: prefill time and decode rate with and without the GPU FFN.

Usage: python3 summarize_model_runs.py run1-ffn0.json run2-ffn1.json ...
Prints one row per process and one per mode (median over all repetitions of all processes of that mode), the ratio
ON / OFF, and whether the generated tokens of all repetitions are the same within each mode and between the modes.
"""
import json
import statistics
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs = [(Path(p).name, json.loads(Path(p).read_text())) for p in sys.argv[1:]]
    print("run,gpu_ffn,dummy,matches_reference,load_s,warmup_s,prefill_s_median,decode_tok_s_median,peak_GiB")
    by_mode = {}
    for name, r in runs:
        mode = "on" if r["gpu_ffn_env"] == "1" else "off"
        prefill = [x["prefill_s"] for x in r["reps"]]
        decode = [x["decode_tok_s"] for x in r["reps"] if x["decode_tok_s"]]
        by_mode.setdefault(mode, {"prefill": [], "decode": [], "tokens": set()})
        by_mode[mode]["prefill"] += prefill
        by_mode[mode]["decode"] += decode
        by_mode[mode]["tokens"].update(tuple(x["generated_ids"]) for x in r["reps"])
        print(f"{name},{mode},{r.get('dummy', False)},{r['matches_reference']},{r['load_s']:.1f},{r['warmup_s']:.3f},"
              f"{statistics.median(prefill):.4f},{statistics.median(decode):.2f},{r['peak_allocated_GiB']:.2f}")
    print()
    print("mode,processes_x_reps,prefill_s_median,prefill_s_min,prefill_s_max,decode_tok_s_median,decode_min,decode_max,"
          "distinct_token_sequences")
    for mode, m in sorted(by_mode.items()):
        print(f"{mode},{len(m['prefill'])},{statistics.median(m['prefill']):.4f},{min(m['prefill']):.4f},"
              f"{max(m['prefill']):.4f},{statistics.median(m['decode']):.2f},{min(m['decode']):.2f},{max(m['decode']):.2f},"
              f"{len(m['tokens'])}")
    if {"on", "off"} <= by_mode.keys():
        on, off = by_mode["on"], by_mode["off"]
        print()
        print(f"prefill ON/OFF = {statistics.median(on['prefill']) / statistics.median(off['prefill']):.3f}")
        print(f"decode tok/s ON/OFF = {statistics.median(on['decode']) / statistics.median(off['decode']):.3f}")
        print(f"same tokens ON and OFF: {on['tokens'] == off['tokens']}")


if __name__ == "__main__":
    main()
