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

1. **License reachability** — a license outage makes `fdtd.run()` return instantly with
   no results:
   ```bash
   bash athena/deploy_athena.sh --license-probe
   ```
   **Do NOT treat an `lmstat` error as authoritative.** On Athena `lmstat` returns `-96`
   ("lmgrd is not running"; locally `HOST_NOT_FOUND`) *even when the license works* — it
   enumerates by the server's FQDN `lumerical-lm.ece.technion.ac.il`, which doesn't
   resolve, while real jobs check out **by IP** via the deploy's `ANSYSLMD_LICENSE_FILE`/
   `ANSYSLI_SERVERS` env vars. So `-96` alone is a **false negative — do not block on it**.
   Instead confirm reachability by IP (this is the real signal):
   ```bash
   ssh evyatarrubin@athena.technion.ac.il "for p in 1055 2325; do timeout 8 bash -c \"cat </dev/null >/dev/tcp/132.68.48.51/\$p\" 2>/dev/null && echo \"port \$p OPEN\" || echo \"port \$p CLOSED\"; done"
   ```
   Ports `1055` (lmgrd) and `2325` (vendor) **OPEN** ⇒ server reachable → **proceed**.
   Only if a port is CLOSED, or a real run no-ops in seconds, treat it as an outage. A
   genuine outage no-ops `fdtd.run()` in seconds, so an empirical single-sim / the first
   array task's log is the final word. See `memory/project_athena_lmstat_false_negative.md`.

2. **Queue** — never launch a second `--option3` sweep while another has pending tasks
   (shared `data/sweep_list.txt` and `results/` get clobbered):
   ```bash
   bash athena/deploy_athena.sh --status
   ```
   If a sweep is RUNNING/PENDING, serialize: wait or confirm with the user before adding
   another that shares mutable state.

3. **Home quota** — home has a ~300 GB cap; over it, jobs hang at
   "Setting --writable-tmpfs". Check usage (host/user mirror `athena/athena.conf`):
   ```bash
   ssh evyatarrubin@athena.technion.ac.il "du -sh ~ 2>/dev/null; quota -s 2>/dev/null || true"
   ```
   If near 300 G, clean `.h5` scratch before submitting (`.h5` is not kept by default).

## Report

Summarize the three results in a line each (seats / queue / quota) and give a clear
go / no-go. On any red flag, recommend the fix rather than dispatching.

## Permissions

The project permission policy (`.claude/settings.json`) allows all Bash/ssh commands
without prompting, so every preflight step runs non-interactively. The one guardrail:
any command containing `scancel` is on the **ask** list — it always prompts the user
first (this is the harness-level encoding of CLAUDE.md §6 "stopping runs is a
confirm-first action"). Keep it that way, and always write ssh commands in the plain
`ssh evyatarrubin@athena.technion.ac.il "..."` form (no `SSHHOST=...` env-var prefixes,
which evade the pattern match).
