#!/usr/bin/env bash
# AUP Jupyter server from the Mac with curl only (files and logs; no command execution).
# Token: ~/.config/aup-kit/token (mode 600), made at https://dub.aupcloud.io/aipc-16/hub/token.
# REMOTE paths are relative to the Jupyter root, /notebooks on AUP. Commands still go through the
# JupyterLab terminal in the browser; a detached job writes /notebooks/aup-runs/NAME/run.log ending in exit=N.
#
#   aup_curl.sh cat REMOTE [N]        last N lines (default 40) of a remote text file
#   aup_curl.sh status NAME [N]       RUNNING or DONE exit=K for aup-runs/NAME/run.log, plus its tail
#   aup_curl.sh ls REMOTE_DIR         names and sizes in a remote directory
#   aup_curl.sh put LOCAL REMOTE      upload a text file (creates parent directories)
#   aup_curl.sh get REMOTE LOCAL      download any file
set -eo pipefail
ORIGIN=${AUP_ORIGIN:-https://dub.aupcloud.io}
PREFIX=${AUP_PREFIX:-/aipc-16}
USER_NAME=${AUP_USER:-jinjieg2-illinois}
TOKEN_FILE=${AUP_TOKEN_FILE:-$HOME/.config/aup-kit/token}
B="$ORIGIN$PREFIX/user/$USER_NAME"
[ -r "$TOKEN_FILE" ] || { echo "no token file $TOKEN_FILE" >&2; exit 2; }
T=$(cat "$TOKEN_FILE")
api() { curl -sS -m "${AUP_TIMEOUT:-60}" -H "Authorization: token $T" "$@"; }
json() { api -H "Content-Type: application/json" "$@"; }

content() {   # print the text of REMOTE, or exit 4 when it does not exist
    local out
    out=$(api "$B/api/contents/$1?content=1&type=file&format=text")
    python3 -c 'import sys,json
d=json.loads(sys.argv[1])
if "content" not in d: sys.exit(4)
sys.stdout.write(d["content"])' "$out"
}

case "${1:-}" in
cat)
    content "$2" | tail -n "${3:-40}" ;;
status)
    log=$(content "aup-runs/$2/run.log" || true)
    if [ -z "$log" ]; then echo "NO LOG: /notebooks/aup-runs/$2/run.log"; exit 2; fi
    last_exit=$(printf '%s\n' "$log" | grep '^exit=' | tail -1 || true)
    if [ -n "$last_exit" ]; then echo "DONE $last_exit"; else echo "RUNNING"; fi
    printf '%s\n' "$log" | tail -n "${3:-8}" | sed 's/^/  | /'
    [ "$last_exit" = "exit=0" ] && exit 0; [ -n "$last_exit" ] && exit 1; exit 3 ;;
ls)
    api "$B/api/contents/$2?content=1" | python3 -c 'import sys,json
for e in sorted(json.load(sys.stdin).get("content") or [], key=lambda e: e["name"]):
    print(e["type"][:3].ljust(4), str(e.get("size") or 0).rjust(12), e["name"])' ;;
put)
    rel=$3; dir=${rel%/*}; [ "$dir" = "$rel" ] && dir=""
    acc=""; for part in ${dir//\// }; do acc="${acc:+$acc/}$part"
        json -X PUT "$B/api/contents/$acc" -d '{"type":"directory"}' > /dev/null; done
    python3 -c 'import json,sys; print(json.dumps({"type":"file","format":"text","content":open(sys.argv[1]).read()}))' "$2" \
        | json -X PUT "$B/api/contents/$rel" --data-binary @- \
        | python3 -c 'import sys,json; d=json.load(sys.stdin); print("put", d.get("path"), "bytes", d.get("size"))' ;;
get)
    api -o "$3" "$B/files/$2" && ls -l "$3" ;;
*)
    sed -n '2,12p' "$0"; exit 2 ;;
esac
