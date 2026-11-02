#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = ["websocket-client>=1.6"]
# ///
"""AUP cloud (dub.aupcloud.io JupyterHub) from the Mac: files, commands and detached jobs through the Jupyter Server API.

Token: ~/.config/aup-kit/token (mode 600, one line; AUP_TOKEN_FILE overrides), made at
https://dub.aupcloud.io/aipc-16/hub/token while signed in. The session must already be running in the browser;
this never starts, stops or changes it. REMOTE paths are relative to the Jupyter root, /notebooks on AUP.

    A=~/code/aup-kit/aup_cli.py
    uv run --script $A servers
    uv run --script $A exec -- 'rocminfo | grep -m1 gfx'       # fresh terminal, prints output, exits with its code
    uv run --script $A bg NAME -- 'cd /notebooks/x && make'     # detached; aup-runs/NAME/run.log ends with exit=N
    uv run --script $A status NAME [--lines 8]                  # RUNNING or DONE exit=N, plus the log tail
    uv run --script $A cat REMOTE [--lines 40]                  # tail of a remote text file (no terminal needed)
    uv run --script $A put LOCAL REMOTE
    uv run --script $A get REMOTE LOCAL

This is a separate tool from ~/.claude/skills/icrn-run/scripts/icrn_cli.py, which other work still uses.
"""
import argparse
import base64
import json
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ORIGIN = os.environ.get("AUP_ORIGIN", "https://dub.aupcloud.io")
PREFIX = os.environ.get("AUP_PREFIX", "/aipc-16")           # the hub lives under ORIGIN + PREFIX + /hub
ROOT = os.environ.get("AUP_ROOT", "/notebooks")              # the Jupyter root as an absolute path on the box
RUNS = "aup-runs"                                            # detached-job logs, under ROOT: survive a disconnect
TOKEN_FILE = os.path.expanduser(os.environ.get("AUP_TOKEN_FILE", "~/.config/aup-kit/token"))
UA = os.environ.get("AUP_UA", "aup-cli/1.0")                  # Cloudflare in front of AUP rejects Python-urllib (error 1010)
ANSI =re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[=>]|\r")
CHUNK = 8 * 2**20


def read_token():
    try:
        mode = stat.S_IMODE(os.stat(TOKEN_FILE).st_mode)
    except FileNotFoundError:
        sys.exit(f"no token file {TOKEN_FILE}: create a token at {ORIGIN}{PREFIX}/hub/token and save it there (mode 600)")
    if mode & 0o077:
        sys.exit(f"{TOKEN_FILE} has mode {oct(mode)}: run chmod 600 {TOKEN_FILE}")
    tok = open(TOKEN_FILE).read().strip()
    if not tok:
        sys.exit(f"{TOKEN_FILE} is empty")
    return tok


class Aup:
    def __init__(self, server=None):
        self.tok = read_token()
        self.user = self.req("GET", f"{ORIGIN}{PREFIX}/hub/api/user")
        servers = self.user.get("servers") or {}
        self.ready = {k: v for k, v in servers.items() if v.get("ready")}
        self.servers = servers
        self.server = server
        self.base = None

    def pick(self):
        s = self.server
        if s is None:
            if len(self.ready) == 1:
                s = next(iter(self.ready))
            elif not self.ready:
                sys.exit("no ready AUP session. Sign in and start the server in the browser first.")
            else:
                sys.exit(f"several ready sessions {list(self.ready)}: pick one with -s ('' is the default session)")
        if s not in self.ready:
            sys.exit(f"session {s!r} is not ready. Ready: {list(self.ready)}")
        self.server, self.base = s, ORIGIN + self.ready[s]["url"].rstrip("/")   # url already carries PREFIX
        return self

    # --- HTTP
    def req(self, method, url, body=None, raw=False, timeout=120):
        h = {"Authorization": f"token {self.tok}", "User-Agent": UA}
        data = None
        if body is not None:
            data, h["Content-Type"] = json.dumps(body).encode(), "application/json"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=h),
                                        timeout=timeout) as r:
                b = r.read()
        except urllib.error.HTTPError as e:
            sys.exit(f"{method} {url.replace(ORIGIN, '')}: HTTP {e.code} {e.read()[:300]!r}")
        return b if raw else (json.loads(b) if b.strip() else None)

    def contents(self, path):
        return f"{self.base}/api/contents/{urllib.parse.quote(path)}"

    def mkdirs(self, remote):
        parts = remote.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            p = "/".join(parts[:i])
            try:
                urllib.request.urlopen(urllib.request.Request(self.contents(p), headers={
                    "Authorization": f"token {self.tok}", "User-Agent": UA}), timeout=60).read()
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                self.req("PUT", self.contents(p), dict(type="directory"))

    def put(self, local, remote):
        self.mkdirs(remote)
        size, sent, i = os.path.getsize(local), 0, 1
        with open(local, "rb") as f:
            while True:
                b = f.read(CHUNK)
                sent += len(b)
                last = sent >= size
                body = dict(type="file", format="base64", content=base64.b64encode(b).decode())
                if size > CHUNK:   # Jupyter's chunked upload: 1, 2, ..., then -1 for the last piece
                    body["chunk"] = -1 if last else i
                self.req("PUT", self.contents(remote), body, timeout=600)
                if last:
                    break
                i += 1
        print(f"put {local} -> {ROOT}/{remote} ({size / 1e6:.2f} MB)")

    def put_text(self, remote, text):
        self.mkdirs(remote)
        self.req("PUT", self.contents(remote), dict(type="file", format="text", content=text))

    def get(self, remote, local):
        url = f"{self.base}/files/{urllib.parse.quote(remote)}"
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"token {self.tok}",
                                                                        "User-Agent": UA}), timeout=600)
        n = 0
        with open(local, "wb") as f:
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                f.write(b)
                n += len(b)
        print(f"get {ROOT}/{remote} -> {local} ({n / 1e6:.2f} MB)")

    def cat(self, remote, lines=None):
        """The text of a remote file through the contents API: no terminal, about one second."""
        try:
            d = self.req("GET", self.contents(remote) + "?content=1&type=file&format=text")
        except SystemExit as e:
            if "HTTP 404" in str(e):
                return None
            raise
        txt = d.get("content") or ""
        return txt if lines is None else "\n".join(txt.splitlines()[-lines:])

    # --- commands through a Jupyter terminal
    def exec(self, cmd, timeout=1800, show=True):
        """Run cmd in a new bash terminal of the session. Returns (exit code, output). The end marker is printed
        by printf from two pieces, so the terminal's echo of the command line never contains it."""
        import websocket
        term = self.req("POST", f"{self.base}/api/terminals")["name"]
        ws_url = re.sub(r"^http", "ws", self.base) + f"/terminals/websocket/{term}"
        ws = websocket.create_connection(ws_url, header=[f"Authorization: token {self.tok}", f"User-Agent: {UA}"],
                                         timeout=30)
        tag = uuid.uuid4().hex[:8]
        end = f"__AUP_END_{tag}__"
        out, buf, code = [], "", None
        try:
            ws.send(json.dumps(["set_size", 50, 300, 800, 1200]))
            # no echo, no prompts: the output is only what the command prints
            ws.send(json.dumps(["stdin", "stty -echo; PS1=''; PS2=''; unset PROMPT_COMMAND\r"]))
            t0 = time.time()                       # drop the banner and prompt that come before the command
            ws.settimeout(1.0)
            while time.time() - t0 < 5:
                try:
                    ws.recv()
                except websocket.WebSocketTimeoutException:
                    break
            ws.settimeout(timeout)
            ws.send(json.dumps(["stdin", f"{cmd}\nprintf '\\n%s%s %s\\n' '__AUP_' 'END_{tag}__' \"$?\"\r"]))
            while code is None:
                msg = json.loads(ws.recv())
                if msg[0] != "stdout":
                    if msg[0] == "disconnect":
                        raise RuntimeError("terminal disconnected")
                    continue
                buf += ANSI.sub("", msg[1])
                *lines, buf = buf.split("\n")
                for ln in lines:
                    if ln.startswith(end):
                        code = int(ln.split()[1])
                        break
                    if ln.endswith("printf '\\n%s%s %s\\n' '__AUP_' 'END_" + tag + "__' \"$?\"") or not (out or ln.strip()):
                        continue                   # an echo that slipped through, or a leading blank line
                    out.append(ln)
                    if show:
                        print(ln, flush=True)
        finally:
            ws.close()
            try:
                self.req("DELETE", f"{self.base}/api/terminals/{term}", raw=True)
            except SystemExit:
                pass
        return code, "\n".join(out)

    # --- detached jobs: cmd.sh under ROOT/aup-runs/NAME, run with setsid nohup, log ends with exit=N
    def bg(self, name, cmd):
        d = f"{RUNS}/{name}"
        self.put_text(f"{d}/cmd.sh", "set -o pipefail\n" + cmd.rstrip("\n") + "\n")
        here = f"{ROOT}/{d}"
        launcher = (f"cd {here} && rm -f run.log pid && "
                    f"setsid nohup bash -c 'bash cmd.sh >> run.log 2>&1; echo exit=$? >> run.log' "
                    f"> /dev/null 2>&1 < /dev/null & echo $! > {here}/pid; sleep 1; "
                    f"echo \"started {here} pid $(cat {here}/pid)\"")
        code, _ = self.exec(launcher, timeout=60)
        return code

    def status(self, name, lines=8):
        log = self.cat(f"{RUNS}/{name}/run.log")
        if log is None:
            print(f"NO LOG: {ROOT}/{RUNS}/{name}/run.log does not exist yet")
            return 2
        rows = log.splitlines()
        done = [r for r in rows if r.startswith("exit=")]
        state = f"DONE {done[-1]}" if done else "RUNNING"
        print(f"{state}  ({len(rows)} log lines)")
        for r in rows[-lines:]:
            print("  | " + r)
        return 0 if done and done[-1] == "exit=0" else (1 if done else 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--server", default=None, help="session name; '' is the default session")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("servers")
    p = sub.add_parser("exec"); p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("bg"); p.add_argument("name"); p.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("status"); p.add_argument("name"); p.add_argument("--lines", type=int, default=8)
    p = sub.add_parser("cat"); p.add_argument("remote"); p.add_argument("--lines", type=int, default=40)
    p = sub.add_parser("put"); p.add_argument("local"); p.add_argument("remote")
    p = sub.add_parser("get"); p.add_argument("remote"); p.add_argument("local")
    a = ap.parse_args()

    ic = Aup(a.server)
    if a.cmd == "servers":
        print(f"user {ic.user['name']}")
        for k, v in ic.servers.items():
            print(f"  {k!r:18s} ready={v.get('ready')} url={v.get('url')} started={v.get('started')} "
                  f"last_activity={v.get('last_activity')}")
        return
    ic.pick()
    if a.cmd in ("exec", "bg"):
        cmd = " ".join(a.command[1:] if a.command[:1] == ["--"] else a.command)
        if not cmd.strip():
            sys.exit("no command given")
        if a.cmd == "exec":
            code, _ = ic.exec(cmd)
        else:
            code = ic.bg(a.name, cmd)
        sys.exit(code)
    if a.cmd == "status":
        sys.exit(ic.status(a.name, a.lines))
    if a.cmd == "cat":
        txt = ic.cat(a.remote, a.lines)
        if txt is None:
            sys.exit(f"{ROOT}/{a.remote}: not found")
        print(txt)
        return
    if a.cmd == "put":
        return ic.put(a.local, a.remote)
    if a.cmd == "get":
        return ic.get(a.remote, a.local)


if __name__ == "__main__":
    main()
