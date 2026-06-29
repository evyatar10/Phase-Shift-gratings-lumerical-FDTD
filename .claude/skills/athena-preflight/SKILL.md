---
name: athena-preflight
description: Pre-dispatch safety check for Athena — license seats, home-disk quota, and the job queue — before submitting an FDTD run. Use before deploying/dispatching to Athena, or when a job hangs / finishes implausibly fast / returns empty results.
---

# athena-preflight

Run the three checks that, in this project's history, have silently wasted GPU hours when
skipped: license outage (silent no-op `fdtd.run()`), >300 GB quota (jobs hang at container
init), and clobbering a queue that already has pending tasks. All use existing tooling —
do not write new scripts.

## Steps

1. **License seats** — a license outage makes `fdtd.run()` return instantly with no
   results:
   ```bash
   bash athena/deploy_athena.sh --license-probe
   ```
   If no FDTD seats are reported / the probe errors, stop and surface it — do not dispatch.

2. **Queue** — never launch a second `--option3` sweep while another has pending tasks
   (shared `data/sweep_list.txt` and `results/` get clobbered):
   ```bash
   bash athena/deploy_athena.sh --status
   ```
   If a sweep is RUNNING/PENDING, serialize: wait or confirm with the user before adding
   another that shares mutable state.

3. **Home quota** — home has a ~300 GB cap; over it, jobs hang at
   "Setting --writable-tmpfs". Check usage (uses the configured Athena host/user):
   ```bash
   ssh "$ATHENA_USER@$ATHENA_HOST" "du -sh ~ 2>/dev/null; quota -s 2>/dev/null || true"
   ```
   If near 300 G, clean `.h5` scratch before submitting (`.h5` is not kept by default).

## Report

Summarize the three results in a line each (seats / queue / quota) and give a clear
go / no-go. On any red flag, recommend the fix rather than dispatching.
