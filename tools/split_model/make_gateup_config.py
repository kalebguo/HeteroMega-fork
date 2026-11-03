#!/usr/bin/env python3
"""Copy the shipped Llama 3 configuration with every gate/up row on the NPU at 1024 tokens (512-row chunks).

The shipped entries give the NPU 256 of 512 rows. The copy uses the existing
hw_bins/npu2/16384x4096x14336/bf16_int4AWQ_bf16_M/final_512x4096x14336_* artifact, so the GPU slice has zero rows and
the NPU call count per prefill stays the same.
"""
import argparse
import json
from pathlib import Path
import re

FW_PATH = "hw_bins/npu2/16384x4096x14336/bf16_int4AWQ_bf16_M/"
NPU_ROWS = 512


def enable_all_rows(config, fw_path=FW_PATH):
    """Moves every 1024-token gate/up entry to the NPU in place and returns how many there were."""
    enabled = 0
    for group in config["kernels_gemm_chunked"]:
        if group.get("prompt_len") != 1024:
            continue
        for entry in group["kernels"]:
            if entry["layer"] in ("gate", "up"):
                entry.update(use=True, npuM=NPU_ROWS, fw_path=fw_path)
                enabled += 1
    return enabled


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("py/unified_llm_w4a16/configs/configs_strixP_llama3_8b.json5"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fw-path", default=FW_PATH)
    args = p.parse_args()
    text = re.sub(r"/\*.*?\*/", "", re.sub(r"//.*", "", args.base.read_text()), flags=re.DOTALL)
    config = json.loads(text)
    print(f"moved {enable_all_rows(config, args.fw_path)} gate/up entries from {args.base}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
