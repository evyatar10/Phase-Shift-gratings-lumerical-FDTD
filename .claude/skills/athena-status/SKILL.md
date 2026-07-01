---
name: athena-status
description: One-shot answer to "how is the run doing?" — Athena queue state, latest job-log tail, and freshly produced result files. Use whenever the user asks about job progress, whether a run finished, or whether results exist ("how doing", "is it running", "did we get results", "now?").
---

# athena-status

The single most repeated request in this project's history (~68 asks). Answer it with
one filtered ssh round-trip, not a hand-assembled block each time.

## Conventions (important)

- Always use the plain host-first form `ssh evyatarrubin@athena.technion.ac.il "..."`
  — never `SSHHOST=... ssh` env-prefixed forms (they evade the permission-rule matching).
- Technion's login banner spams every ssh call. Pipe remote output through
  `grep -vE "post-quantum|openssh|may need to be upgraded"` to strip it.
- Remote base is `/home/evyatarrubin/bragg_sim_athena` (from `athena/athena.conf`):
  logs in `jobs/logs/`, results in `results/<study>/results/`.

## Steps

1. **Queue** — what is running/pending:
   ```bash
   ssh evyatarrubin@athena.technion.ac.il "squeue -u evyatarrubin -o '%.12i %.30j %.8T %.10M %R'" 2>&1 | grep -vE "post-quantum|openssh|may need to be upgraded"
   ```

2. **Latest log tail** — progress of the newest (or user-named) job. One combined call:
   ```bash
   ssh evyatarrubin@athena.technion.ac.il "cd ~/bragg_sim_athena/jobs/logs && ls -t lum_*.out 2>/dev/null | head -5 && echo '--- newest ---' && tail -30 \$(ls -t lum_*.out | head -1)" 2>&1 | grep -vE "post-quantum|openssh|may need to be upgraded"
   ```
   If the user asked about a specific job/array, tail that job's `lum_array-<ID>_<task>.out` instead.

3. **Fresh results** — new `.mat` files for the relevant study:
   ```bash
   ssh evyatarrubin@athena.technion.ac.il "ls -lt ~/bragg_sim_athena/results/<study>/results/result_*.mat 2>/dev/null | head -15" 2>&1 | grep -vE "post-quantum|openssh|may need to be upgraded"
   ```
   Compare the count against the expected number of sweep tasks when known.

## Report

Three lines minimum: queue state (N running / N pending, or empty), what the newest log
says the job is doing (solve progress, or errors — quote the error verbatim if present),
and how many result files exist vs expected. Then the verdict: still running / done —
offer to download / **stalled or suspicious** (log silent for a long time, run finished
implausibly fast, T≈0 results) — in that case apply CLAUDE.md §6 (license silent no-op,
quota hang) and say which failure it looks like.

Red flags to check without being asked: job in state `PD` with reason `(QOSMaxJobsPerUserLimit)` or
`(Priority)` is normal waiting; a log stuck at "Setting --writable-tmpfs" = quota hang;
a task that ended in seconds with an empty result = license no-op.
