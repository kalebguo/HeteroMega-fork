"""Summarize HETEROMEGA_FFN_TIMERS CSVs: median milliseconds per phase of MergedFfn::forward.

Diagnostic timing only; the timers add host waits that the normal path leaves asynchronous.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics


def load(paths, skip):
    """All rows of each CSV after its first `skip` rows, which belong to the cold warmup pass."""
    rows = []
    for path in paths:
        with open(path, newline='') as f:
            rows += [{k: int(v) for k, v in row.items()} for row in list(csv.DictReader(f))[skip:]]
    return rows


def medians(rows):
    fields = [k for k in rows[0] if k.endswith('_ns')]
    return {k[:-3] + '_ms': statistics.median(r[k] for r in rows if r[k] >= 0) / 1e6 for k in fields} | {'calls': len(rows)}


def summarize(rows):
    """Overall, per (slot, serial), and per layer."""
    groups = {}
    for row in rows:
        groups.setdefault(f"slot{row['slot']}_{'serial' if row['serial'] else 'concurrent'}", []).append(row)
    layers = {}
    for row in rows:
        layers.setdefault(row['layer'], []).append(row)
    return dict(all=medians(rows), groups={k: medians(v) for k, v in sorted(groups.items())},
                layers={str(k): medians(v) for k, v in sorted(layers.items())})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path, help='the HETEROMEGA_FFN_TIMERS directory')
    ap.add_argument('--skip', type=int, default=0, help='rows to drop from the start of each CSV (the cold warmup pass)')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    paths = sorted(args.directory.glob('merged_ffn_phases_*.csv'))
    if not paths:
        ap.error(f'no merged_ffn_phases_*.csv in {args.directory}')
    report = dict(files=[p.name for p in paths], skip=args.skip) | summarize(load(paths, args.skip))
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + '\n')
    print(json.dumps({'files': report['files'], 'all': report['all'], 'groups': report['groups']}, indent=2))


if __name__ == '__main__':
    main()
