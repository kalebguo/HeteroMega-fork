# aup-kit: driving the AUP cloud box from anywhere with the JupyterHub API

The AUP box (Ryzen AI 9 HX 370, Radeon 890M `gfx1150`, NPU Strix `aie2p`, ROCm 7.2, XRT 2.26)
is a JupyterHub server at `https://dub.aupcloud.io/aipc-16/`. There is no SSH. Everything goes
through the Jupyter Server REST API (files) and its terminal websocket (commands), authenticated
with a hub API token. Verified 2026-11-02 from a Mac.

## 1. Token

- The user creates it at `https://dub.aupcloud.io/aipc-16/hub/token` while signed in, and gives it
  to you. Save it to `~/.config/aup-kit/token` (one line, `chmod 600`). Never print or commit it.
- `AUP_TOKEN_FILE` overrides the path. The hub user is `jinjieg2-illinois`; the box user is
  `mliraie`. The Jupyter root is `/notebooks`, which persists; `/home/mliraie` may not.
- The browser session must already be running (the user signs in; `servers` shows `ready=True`).

## 2. Tools (this directory)

```bash
A=tools/aup-kit/aup_cli.py            # needs uv (it declares websocket-client); or: pip install websocket-client
uv run --script $A servers            # ready=True means the server is up
uv run --script $A exec -- 'rocminfo | grep -m1 gfx'        # fresh terminal, output back as text, exit code
uv run --script $A bg NAME -- 'cd /notebooks/x && make'     # detached job: /notebooks/aup-runs/NAME/run.log ends with exit=N
uv run --script $A status NAME --lines 8                    # RUNNING or DONE exit=N plus the log tail (no terminal)
uv run --script $A cat REMOTE --lines 40                    # tail of any remote text file; REMOTE relative to /notebooks
uv run --script $A put LOCAL REMOTE   /   get REMOTE LOCAL
bash tools/aup-kit/aup_curl.sh status NAME                  # curl-only fallback: cat/status/ls/put/get
```

- `exec` has about 7 s of overhead (terminal creation, banner). Keep its output short (`tail`, `cut -c1-150`).
- Cloudflare fronts the hub: it rejects Python's default User-Agent with HTTP 403 `error code: 1010`.
  The tools send `User-Agent: aup-cli/1.0`. Do not change that to imitate a browser.
- Long jobs: write the script to `jobs/NAME/cmd.sh`, upload it, start it with `bg` (or the launcher below),
  then poll `status`. A browser disconnect does not kill a `setsid nohup` job.
- Manual launcher (what `bg` does): `cd /notebooks/aup-runs/NAME && rm -f run.log pid &&
  (setsid nohup bash -c 'bash cmd.sh >> run.log 2>&1; echo exit=$? >> run.log' >/dev/null 2>&1 </dev/null & echo $! > pid)`.

## 3. What is on the box (read 2026-11-02)

| Path | What |
| --- | --- |
| `/notebooks/hm-eval-main` | clone of glassescrab/HeteroMega, remote `fork` = kalebguo/HeteroMega-fork, branch `eval-runner` (fork `claude/exciting-faraday-95a0c1` on top of `main` `8e5c804`); built with `jobs/build-main/cmd.sh` (`build/`, `deployment.json`) |
| `/notebooks/hm-fix` | branch `merged-ffn-fix` (remote `origin` here = the fork) with its `-O3` build; `jobs/fix-env.sh` is its environment. `hm-eval-main` stays untouched as earlier evidence |
| `/notebooks/hm-fix-noopt` | attribution tree for `jobs/step2c`: no `-O3`, original scan loop. Never use it for headline numbers |
| `/notebooks/hm-results/count_ioctls.so`, `profile_ioctls.so` | `LD_PRELOAD` counters: submissions and contexts, or a timed trace (`jobs/step3c`) |
| `/notebooks/heteromosaic-venv` | Torch 2.11.0+rocm7.2, pybind11 3.1.0, numpy 2.5.4. No `transformers`, no `safetensors` package |
| `/notebooks/hm-model/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` | 2 safetensors shards + index only (no tokenizer, no config.json) |
| `/notebooks/hm-libs/xdna-driver-05-19-25` | pinned xdna-driver sparse clone (`0e6d303b`), `HOME_LIBS=/notebooks/hm-libs` |
| `/notebooks/hm-libs-sys` | 2026-10-17 GPU-only header shortcut (`SYSTEM_XDNA_HEADER=1`); not a git clone; do not use for NPU builds |
| `/notebooks/heteromega-gpu` | the user's October working copy (`gpu-p0p1`, uncommitted edits): do not modify |
| `/notebooks/hm-results/` | results; `2026-11-02/eval-merged-gpu` is the first paired evaluation |
| `/notebooks/aup-runs/<name>/` | job scripts and logs (`build-main`, `smoke-main`, `eval-main`) |
| `/home/mliraie/mlir-aie/ironenv/bin/python` | IRON (MLIR-AIE `7e00b579`) for rebuilding NPU artifacts |
| `/opt/rocm-7.2.0`, `/opt/xilinx/xrt` | ROCm 7.2, XRT 2.26; cmake 4.4.2; OpenSSL headers present; no ninja (`torch.utils.cpp_extension.load` fails) |

Environment for every job (from `docs/ffn-deployment.md`): see `jobs/eval-main/cmd.sh`. Always unset
`HETEROMEGA_FFN_CONFIG HETEROMEGA_GPU_FFN HETEROMEGA_GPU_FFN_SPLIT HSA_OVERRIDE_GFX_VERSION PYTHONPATH`
first; the evaluation runner sets the switch per job.

## 4. Rules

1. One GPU/NPU job at a time. Before starting one, check every `aup-runs/*/run.log` for a job without
   `exit=` (`status NAME` prints RUNNING). A second job shares the GPU and the NPU execution lock and
   corrupts both timings.
2. Do not install packages into `/notebooks/heteromosaic-venv` or the system without the user's OK.
   Record what is missing and stop that step.
3. Do not touch `/notebooks/heteromega-gpu`, host drivers, firmware, or `/notebooks/mlir-aie`.
4. Disk: 915 GB with 214 GB free; the model shards are merged once into
   `py/unified_llm_w4a16/model_weights/` (5.4 GB) by `run_model_ffn.py`.
5. Timestamps in logs are UTC; Chicago is UTC-6 (CST) since 2026-11-01.
6. Keep the result JSON/CSV/logs on the box under `/notebooks/hm-results/<date>/<name>/` and copy the
   small ones (summary.csv, result JSON, run.log) into the repo under `docs/runs/<date>-<name>/`.
