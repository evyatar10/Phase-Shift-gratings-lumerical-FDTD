---
name: safe-compact
description: Checkpoint the session so compaction/handoff loses nothing — snapshot server job state, persist program state + next steps to memory, refresh todos. Invoke when the user says "make it safe to compact", before/after long autonomous stretches, after any dispatch or verdict, and proactively whenever context is getting long. Do NOT wait to be asked.
---

# safe-compact

Compaction keeps a summary, not the conversation. Anything that exists ONLY in chat
(a job ID, a measured number, a decision and its reason, the exact next command) is
at risk. This skill moves all of it into files that survive: the memory directory,
the todo list, and on-disk results. The test at the end: **a fresh session with zero
conversation must be able to resume from files alone.**

## Steps

1. **Snapshot server state** — one ssh round-trip, so the memory records reality,
   not stale beliefs:
   ```bash
   ssh evyatarrubin@athena.technion.ac.il "squeue -r -u evyatarrubin -o '%.14i %.30j %.8T %.10M %R' | head -25; sacct -j <active_ids> --format=JobID%-16,State%-12,Elapsed -n | grep -v '\.' | awk '{print \$2}' | sort | uniq -c" 2>&1 | grep -vE "post-quantum|openssh|may need to be upgraded"
   ```
   Note counts (COMPLETED / RUNNING / PENDING / FAILED) per active job ID.

2. **Update the active program memory file(s)** in
   `C:\Users\evyat\.claude\projects\c--Users-evyat-Lumerical-phase-shift-grating-FTDT-codes\memory\`
   (usually one `project_*` file per active program). It must contain, current as of
   the snapshot timestamp:
   - stage/phase, job IDs + task counts + states, watcher/background-task IDs and
     what each watches;
   - every MEASURED number quoted to the user this session, with its source file;
   - decisions taken + one-line rationale (especially anything the user approved or
     rejected — dropped parameters stay dropped);
   - **exact next-step commands** (copy-pasteable), including env vars like
     `ARRAY_TIME`/`PRELIM_TIME` and any `%N` throttle;
   - operational rules newly learned this session;
   - uncommitted-files inventory (never commit without permission).

3. **Update the `MEMORY.md` index line** for that file (one line, current state).

4. **Refresh the todo list** to the actual phase — completed items marked done,
   the in-progress item named after the real current step.

5. **Sweep for orphans** — scan the session for load-bearing content not yet in a
   file: numbers cited from analysis, error messages diagnosed, paths of downloaded
   results, scratchpad scripts that became important (promote or note them). If a
   deliverable figure/report exists only as chat text, write it to a file.

## Report

One short line: "Checkpointed: <memory file> updated (jobs <IDs>: <state counts>),
todos current." Do not dump the memory file into chat.
