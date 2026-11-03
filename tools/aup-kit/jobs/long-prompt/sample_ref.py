#!/usr/bin/env python3
"""Write a --ref file of model-sampled tokens for prompts longer than the fixed 1024-token reference.

Same sampler as step 5b (gpu_original, temperature 1, from BOS); only the model's capacity is larger. The tokens are longer
than any prompt used, so run_model_ffn.py treats each prompt as a prefix: nothing is repeated or padded and matches_reference stays
None. generated_ids only sets how many tokens the warm-up generates (8, as with the fixed reference). Run from the repository root.
"""
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path.cwd() / "tools/npu_error_probe"))
import path_logits

out, model, tokens, earlier = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
seed = sys.argv[5] if len(sys.argv) > 5 else "1"
raw = out.with_suffix(".raw.json")
path_logits.MAX_SEQ_LEN = tokens + 64
sys.argv = ["path_logits.py", "--sample-prompt", str(raw), "--seed", seed, "--model", model, "--build", "build", "--sample-tokens", str(tokens)]
path_logits.main()
ids = json.loads(raw.read_text())["input_ids"][0]
old = json.loads(earlier.read_text())["input_ids"][0]
same = sum(a == b for a, b in zip(ids, old))
print(f"tokens equal to {earlier.name}: {same} of {min(len(ids), len(old))}, distinct tokens {len(set(ids))} of {len(ids)}", flush=True)
sha = hashlib.sha256(struct.pack(f"<{len(ids)}q", *ids)).hexdigest()
out.write_text(json.dumps(dict(input_ids=[ids], generated_ids=ids[-8:], input_sha256=sha, seed=int(seed), temperature=1.0, source="gpu_original sampled")) + "\n")
