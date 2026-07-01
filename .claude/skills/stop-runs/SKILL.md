---
name: stop-runs
description: Safely stop/cancel/pause Athena SLURM jobs — resolve the exact job IDs, state them back, cancel (the scancel command itself prompts for confirmation), and verify. Use whenever the user says stop / kill / cancel / pause a run or "cancel and resubmit".
---

# stop-runs

"Stop the run" needs a job ID, not speed (CLAUDE.md §6). Real incident history: a
blanket cancel is unrecoverable for a long optimization. The `scancel` command is on
the permission **ask** list, so the actual cancel always shows the user a prompt —
that prompt is the confirmation step; everything before it should make the prompt
trivially verifiable.

## Steps

1. **Resolve** — list the queue and identify exactly which job(s) match what the user
   asked to stop:
   ```bash
   ssh evyatarrubin@athena.technion.ac.il "squeue -u evyatarrubin -o '%.12i %.30j %.8T %.10M %R'" 2>&1 | grep -vE "post-quantum|openssh|may need to be upgraded"
   ```
   Match by job name / study, not by position. If ambiguous (several candidates, or
   the user said "stop all" while unrelated jobs are queued), ask which ones.

2. **State it back** — one line: "cancelling job(s) <ID list> = <job names>". For an
   array, cancel the array ID (kills all tasks) or `<ID>_<task>` for a single task.

3. **Cancel** — targeted, never blanket `scancel -u`:
   ```bash
   ssh evyatarrubin@athena.technion.ac.il "scancel <ID> [<ID2> ...]"
   ```
   (This command triggers the permission prompt — that's by design; the user approving
   it is the confirmation.)

4. **Verify** — re-run the squeue from step 1 and confirm the jobs are gone / in `CG`.
   Report what remains running.

## After cancelling

- If the user said "cancel and resubmit", run the athena-preflight skill before the
  resubmit (queue is now free, but check quota/license as usual).
- A cancelled `--option3` sweep leaves its `data/sweep_list.txt` and partial results
  on the server; note that a re-dispatch of the same study overwrites them (shared
  mutable state — serialize).
