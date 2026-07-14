---
name: work-alone
description: Autonomous-session mode — the user is away for hours and wants the multi-stage pipeline driven to completion without questions. Invoke when the user says they're unavailable / "continue with the stages" / "keep working while I'm gone". Defines what to decide alone vs park, watcher discipline, and periodic safe-compact checkpoints.
---

# work-alone

The user is not watching and cannot answer. A question asked mid-run blocks the
pipeline for hours; an unjustified irreversible action is worse. This skill is the
contract for the hours in between.

## At the start (once)

State in ONE short message: the pipeline stages you will drive, the decision points
you will take autonomously (with the rule you'll apply at each), and the decisions
you will PARK for the user. Then stop asking — everything after this is action.

## Decision policy while alone

- **Proceed without asking:** downloads, offline solves, plots, memory updates, and
  dispatches that are part of the already-approved pipeline at standard knobs
  (correct study module, `%3` throttle, `ARRAY_TIME`/`PRELIM_TIME` sized to measured
  task times, preflight via `athena-preflight`).
- **Decide by stated rule:** at each go/no-go gate, apply the quantitative rule from
  the program memory (e.g. "ceiling below the measured jitter floor → stop = valid
  negative"). Record the number, the rule, and the verdict in the memory file.
- **PARK (never do alone):** deleting anything, mutating git, `scancel`, changing
  physics scope or geometry beyond the approved plan, spending GPU budget on stages
  the user hasn't approved, and anything CLAUDE.md §8 reserves for the user. Parked
  items go in the final report under "waiting for you", with a recommendation.
- **Gate failures:** a failed §2 sanity check or a dead stage stops THAT branch —
  report it first (§9), continue independent branches if any, don't improvise a
  replacement study.

## Keeping the loop alive

- Never end a turn "waiting" without a mechanism that re-invokes you: a background
  watcher script (`run_in_background`) that polls the queue and **exits only on
  ssh-success AND condition met** (ssh failure ≠ queue empty — VPN blips), with an
  early-exit on >N FAILED tasks.
- After each milestone, immediately arm the next watcher or start the next step.
  Poll interval ~300 s; never tight-loop.
- If a run fails while alone: diagnose from task LOGS (not just sacct states), apply
  the known failure signatures (license cascade → `%3` throttle + resubmit failed
  range; quota hang; stale server code), resubmit the targeted range once. If the
  same failure repeats, stop that branch and park it.

## Checkpoint discipline

Invoke the **safe-compact** skill after every milestone (dispatch, verdict,
download, solve) and at least every ~2 h of autonomous work, so a compaction
mid-run loses nothing. This is not optional — long unattended sessions WILL compact.

## The returning-user report

Maintain one running summary and end every autonomous burst with its current
version: what ran (job IDs, task counts, states), what was measured (numbers +
file paths, labeled MEASURED/DERIVED/EXPECTED), decisions taken + the rule applied,
failures and how they were handled, and the parked list. Full absolute local paths
for every artifact. The user should need to read exactly one message to catch up.
