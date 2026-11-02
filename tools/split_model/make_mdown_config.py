#!/usr/bin/env python3
"""Copy the shipped Llama 3 configuration with native M-split down enabled at 1024 tokens (512-row chunks).

The shipped entries keep down on the GPU (use=false, npuM=0). The enabled entries give 256 of 512 rows to the existing
hw_bins/npu2/16384x14336x4096/bf16_int4AWQ_bf16_M/final_256x14336x4096_* artifact, for +64 NPU calls per prefill.
"""
import argparse
import json
from pathlib import Path
import re

FW_PATH = "hw_bins/npu2/16384x14336x4096/bf16_int4AWQ_bf16_M/"
NPU_ROWS = 256


def enable_down(config):
    """Enables every 1024-token down entry in place and returns how many there were."""
    enabled = 0
    for group in config["kernels_gemm_chunked"]:
        if group.get("prompt_len") != 1024:
            continue
        for entry in group["kernels"]:
            if entry["layer"] == "down":
                entry.update(use=True, npuM=NPU_ROWS, fw_path=FW_PATH)
                enabled += 1
    return enabled


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("py/unified_llm_w4a16/configs/configs_strixP_llama3_8b.json5"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    text = re.sub(r"/\*.*?\*/", "", re.sub(r"//.*", "", args.base.read_text()), flags=re.DOTALL)
    config = json.loads(text)
    print(f"enabled {enable_down(config)} down entries from {args.base}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
