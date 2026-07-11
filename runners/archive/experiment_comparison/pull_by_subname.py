"""
Pull result_*.mat files from Athena into PC with descriptive subname filenames.

Usage:
    python -m runners.experiment_comparison.pull_by_subname --batch 500
    python -m runners.experiment_comparison.pull_by_subname --batch 516 --parallel 4

What it does, safely:
  * For each card in `it11_devices_<batch>.RECORDS`, derive a "pretty" filename
    from `record.subname` (e.g. "corr300,Np160,ts50" → "corr300_Np160_ts50.mat").
  * SSH to Athena, list which devices have a `result_*.mat` on the server.
  * For each missing-local but present-on-server, scp into a `.downloading`
    temp file and atomically rename to the final name only after a size check
    (> 300 MB — well below stripped size, well above any partial).
  * Two cards can share the same parsed subname (e.g. right_down_1B2 and 2B1
    both produce `corr250,Np200,ts70`); in that case both are pulled with the
    label appended, e.g. `corr250_Np200_ts70__right_down_1B2.mat`.

Safety contract — NEVER deletes or overwrites existing local files:
  * If `<pretty>.mat` already exists locally and > 300 MB, it's left alone.
  * If the file is smaller (suspected partial), it's *skipped, not deleted* —
    you'd have to remove it manually if you want a fresh pull.
  * All scp output goes to a temp file. If scp fails or the result is below
    threshold, the temp is removed and the run continues.
  * 500-batch results only land in `results_from_athena/it11_devices_500/by_subname/`,
    and likewise for 516 — the source/dest paths come from a single batch-string,
    so cross-contamination is structurally impossible.

Requires:
  * `ssh` and `scp` on PATH (Git Bash on Windows is fine).
  * passwordless ssh to `evyatarrubin@athena.technion.ac.il`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ATHENA_SSH   = 'evyatarrubin@athena.technion.ac.il'
ATHENA_RESULTS_ROOT = '/home/evyatarrubin/bragg_sim_athena/results'
SIZE_FLOOR_BYTES = 300_000_000   # smaller than any complete result (~632 MB stripped)


def _pretty(subname: str) -> str:
    return subname.replace(',', '_').replace(' ', '')


def _load_records(batch: int):
    mod = importlib.import_module(f'runners.experiment_comparison.it11_devices_{batch}')
    return list(mod.RECORDS)


def _ssh_list_done(batch: int) -> set[str]:
    """Return the set of device labels that have a result_*.mat on Athena."""
    cmd = ['ssh', '-o', 'ConnectTimeout=20', ATHENA_SSH,
           f"find {ATHENA_RESULTS_ROOT}/it11_devices_{batch} "
           "-name 'result_*.mat' -printf '%P\\n' 2>/dev/null | "
           "awk -F/ '{print $1}' | sort -u"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return set(line.strip() for line in out.stdout.splitlines() if line.strip())


def _scp_pull(batch: int, label: str, out_path: str) -> tuple[bool, int, str]:
    """scp -> tmp -> verify -> atomic rename. Returns (ok, size, message)."""
    tmp = out_path + '.downloading'
    if os.path.exists(tmp):
        os.remove(tmp)
    src = f'{ATHENA_SSH}:{ATHENA_RESULTS_ROOT}/it11_devices_{batch}/{label}/results/result_*.mat'
    cmd = ['scp', '-q', '-o', 'ConnectTimeout=30', src, tmp]
    try:
        rc = subprocess.run(cmd, capture_output=True, timeout=600).returncode
    except subprocess.TimeoutExpired:
        rc = -1
    if rc != 0 or not os.path.exists(tmp):
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, 0, f'scp rc={rc}'
    sz = os.path.getsize(tmp)
    if sz < SIZE_FLOOR_BYTES:
        os.remove(tmp)
        return False, sz, f'below threshold ({sz} < {SIZE_FLOOR_BYTES})'
    os.replace(tmp, out_path)  # atomic on same filesystem
    return True, sz, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, required=True, choices=[500, 516],
                    help='IT11 batch pitch (nm).')
    ap.add_argument('--parallel', type=int, default=4,
                    help='Number of concurrent scp connections.')
    ap.add_argument('--dest', default=None,
                    help='Override destination dir (default: results_from_athena/'
                         'it11_devices_<batch>/by_subname under project root).')
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)

    dest = args.dest or os.path.join(
        'results_from_athena', f'it11_devices_{args.batch}', 'by_subname')
    os.makedirs(dest, exist_ok=True)

    print(f'Batch        : {args.batch}')
    print(f'Destination  : {dest}')
    print(f'Parallel scp : {args.parallel}')

    records = _load_records(args.batch)
    print(f'Cards in batch: {len(records)}')

    # Detect collisions (multiple cards mapping to the same pretty name).
    by_pretty = defaultdict(list)
    for r in records:
        by_pretty[_pretty(r.subname)].append(r)
    collisions = {k: v for k, v in by_pretty.items() if len(v) > 1}
    if collisions:
        print(f'Collisions: {len(collisions)} pretty names cover multiple cards '
              f'(will use __<label> suffix to disambiguate):')
        for k, recs in collisions.items():
            print(f'  {k}.mat -> {[r.label for r in recs]}')

    # List what's done on the server.
    print('Querying Athena for completed devices ...')
    done = _ssh_list_done(args.batch)
    print(f'Server has {len(done)} devices with result_*.mat')

    # Plan the pulls.
    plan = []  # list of (label, out_path)
    for r in records:
        if r.label not in done:
            continue
        pretty = _pretty(r.subname)
        if pretty in collisions:
            out = os.path.join(dest, f'{pretty}__{r.label.rsplit("_pitch", 1)[0]}.mat')
        else:
            out = os.path.join(dest, pretty + '.mat')
        if os.path.exists(out) and os.path.getsize(out) >= SIZE_FLOOR_BYTES:
            continue   # already have a usable copy
        plan.append((r.label, out))

    print(f'Files to pull: {len(plan)}')
    if not plan:
        print('Nothing to do.')
        return 0

    # Run in parallel.
    t0 = time.time()
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(_scp_pull, args.batch, lbl, out): (lbl, out)
                for lbl, out in plan}
        for fut in as_completed(futs):
            lbl, out = futs[fut]
            success, sz, msg = fut.result()
            tag = 'OK  ' if success else 'FAIL'
            name = os.path.basename(out)
            print(f'{tag} {name:<60} {sz:>11,} bytes  ({msg})')
            ok += 1 if success else 0
            fail += 0 if success else 1

    dt = time.time() - t0
    have = sum(1 for f in os.listdir(dest)
               if f.endswith('.mat') and
               os.path.getsize(os.path.join(dest, f)) >= SIZE_FLOOR_BYTES)
    print(f'\nDone in {dt:.0f}s. Pulled {ok} new (failed {fail}). '
          f'Total local: {have} files.')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
