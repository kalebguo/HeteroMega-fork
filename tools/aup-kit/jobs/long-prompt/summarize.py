#!/usr/bin/env python3
"""Pool the result JSON of the long-prompt timing stages into one CSV row per method and length.

    python3 summarize.py RESULTS_DIR OUT.csv

RESULTS_DIR holds timing2048/ and timing4096/ as cmd.sh wrote them. Token columns compare with gpu_original of the same length.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

CASES = [("timing2048", 1024, "p8192-1024", "merged_ffn P8192, all switches"),
         ("timing2048", 2048, "gpu-2048", "gpu_original"),
         ("timing2048", 2048, "hetero-2048", "heteromosaic, 1024-row chunks, 2 inflight"),
         ("timing2048", 2048, "hetero-sched-2048", "heteromosaic, four 512-row chunks, 4 inflight"),
         ("timing2048", 2048, "p8192-2048", "merged_ffn P8192, all switches"),
         ("timing2048", 2048, "p7168-2048", "merged_ffn P7168, all switches"),
         ("timing2048", 2048, "p6400-2048", "merged_ffn P6400, all switches"),
         ("timing4096", 4096, "gpu-4096", "gpu_original"),
         ("timing4096", 4096, "hetero-4096", "heteromosaic, 2048-row chunks, 2 inflight"),
         ("timing4096", 4096, "p8192-4096", "merged_ffn P8192, without GEMM_UNCACHED"),
         ("timing4096", 4096, "p7168-4096", "merged_ffn P7168, without GEMM_UNCACHED"),
         ("timing4096", 4096, "p6400-4096", "merged_ffn P6400, without GEMM_UNCACHED")]


def common_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    reference = {}
    rows = []
    for stage, length, label, name in CASES:
        directories = sorted((root / stage).glob(f"{label}-r[0-9]")) or [root / stage / label]
        results = [json.loads(path.read_text()) for d in directories for path in sorted(d.glob("*.result.json"))]
        reps = [rep for result in results for rep in result["reps"]]
        prefill = [rep["prefill_s"] * 1000 for rep in reps]
        if label.startswith("gpu-"):
            reference[length] = reps[0]["generated_ids"]
        prefixes = sorted({common_prefix(reference[length], rep["generated_ids"]) for rep in reps}) if length in reference else []
        gtt = [int(line.split(",")[0]) for d in directories for line in Path(f"{d}.gtt.csv").read_text().splitlines() if "," in line]
        rows.append(dict(method=name, prompt_tokens=length, processes=len(results), samples=len(prefill),
                         prefill_median_ms=round(statistics.median(prefill), 1), prefill_min_ms=round(min(prefill), 1), prefill_max_ms=round(max(prefill), 1),
                         process_medians_ms=" ".join(f"{statistics.median(rep['prefill_s'] * 1000 for rep in result['reps']):.1f}" for result in results),
                         ms_per_token=round(statistics.median(prefill) / length, 3),
                         decode_tok_s=round(statistics.median(rep["decode_tok_s"] for rep in reps), 2),
                         generate_s=round(statistics.median(rep["generate_s"] for rep in reps), 2),
                         peak_gtt_mb=max(gtt), tokens_same_as_gpu_prefix=" ".join(map(str, prefixes))))
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(",".join(str(value) for value in row.values()))


if __name__ == "__main__":
    main()
