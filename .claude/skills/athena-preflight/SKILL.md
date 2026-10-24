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
confirm-first action"). The list matches only the LITERAL command, so a `scancel`/`rm`
hidden in a script or `subprocess` call slips past it. The user-level hook
`~/.claude/hooks/remote_destructive_guard.py` closes that gap (incident 2026-10-04).
Keep it that way, and always write ssh commands in the plain
`ssh evyatarrubin@athena.technion.ac.il "..."` form (no `SSHHOST=...` env-var prefixes,
which evade the pattern match).

## 1b. License SEATS, not just reachability (user rule 2026-08-16 — mandatory
## for any dispatch of more than one task)

Ports open ≠ seats free. Probe the actual count from IGUM (reliable vantage;
Athena lmstat is the documented false negative):
```bash
ssh evyatarrubin@132.68.58.101 "LUM=~/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261; \$LUM/licensingclient/linx64/lmutil lmstat -c 1055@132.68.48.51 -f lum_fdtd_solve 2>&1 | grep 'Users of lum_fdtd_solve'"
# (host = the IP from igum/igum.conf IGUM_HOST; `igum.ece.technion.ac.il` is the docs host and fails host-key verification — burned 2026-10-20)
```
Budget by simultaneous solves and GPU SM count: under standard GPU solve
licensing, ceil(SMs/16) seats per solve (A100 7, L40S 9; also check CPU license
requirements). LocalRunner capacity 1 runs the forward/adjoint/FD legs
sequentially. The previous 1-seat/task and concurrent-fwd/adj estimates were
wrong. Bands (measured oscillation 39-46/50 in single hours):
≥35/50 in use = HIGH → hold new fan-outs; ≥45/50 = CRITICAL → no new
dispatches, tighten the trouble-finder. The pool is faculty-shared: a snapshot
is not a promise — for long batches keep the seat-band monitor running.

## Model routing (CLAUDE.md §12, 2026-09-11)

This skill is routine, recipe-driven server work: when the session runs on Fable,
execute it in an **Opus subagent** (`Agent`, `model: "opus"`) with the steps above
as the brief, and have it return verdict lines only (queue state / job IDs /
result paths / PASS-FAIL), never raw logs. Fable reads the verdict and decides.
Exception: a single ssh line the manager needs right now is cheaper inline.


## Seat budget of your OWN fleet (incident 2026-10-20, job 265204 died 23 s in: `FlexNet Licensing error:-4,132` in the
layout's `*_p0.log`): each GPU solve holds ceil(SMs/16) seats — H200 9, A100 7, L40S 9, the 188-SM node n317 12 (FlexNet -4,132 "only 10 of the requested 12" in the layout _p0.log; pin --gpu=a100 under seat pressure) — and lumopt2 lanes hold one solve's
worth continuously. Sum RUNNING + about-to-start tasks (array %N, pending lanes) BEFORE dispatching: 4 H200 rows + 2 lanes +
2 A100 rows = 59 > 50 killed the extra row. Loud `-4` in the engine log on Athena = seat starvation from your own fleet;
the silent 1-s no-op is the shared-pool starvation signature. Recover failed array indices with `--array-tasks=<dead>`.
