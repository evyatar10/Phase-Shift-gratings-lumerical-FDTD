---
name: plan-check
description: Direct "are we on plan?" review of a long-running programme — the main goal in the user's words, every plan item with its real status, what actually blocks the goal, drift and waste (diagnostic chains, token/GPU/seat use), and the decisions taken. Invoke when the user asks "are we going according to plan / review everything / clear our head", at every milestone, after any incident, and at least once per autonomous work block. Output is one short table plus corrections, never a narrative.
---

# plan-check

A programme that runs for days drifts in three ways: the work shifts from the
deliverable to diagnostics of a sub-question; cost (model tokens, GPU hours,
license seats, disk) grows without a decision depending on it; and lanes keep
running after their question is answered. This skill is the fixed protocol for
catching all three. It generalises CLAUDE.md §9 "MAIN-GOAL CHECKPOINT" and the
2026-10-07 incident that created it (a whole day of calibration forensics while
no optimizer iterate ran).

## When

- The user asks any form of "are we on plan?", "review everything", "are we
  focused?", "clear our head".
- Every milestone: a verdict, a job finishing, a gate failing, a lane closed.
- Before any new diagnostic or any dispatch that costs more than ~1 GPU-hour.
- At least once per autonomous work block (work-alone), and after any incident
  (crash, seat starvation, quota alarm, wrong dispatch).

## The protocol (do it in this order, write it down)

1. **Goal, in the user's words.** One line. Not your paraphrase. If the user has
   stated priorities between sub-goals (e.g. "S2 first"), include them.

2. **Plan inventory.** One table row per plan item the user asked for or
   approved. Columns: item | status | verdict. Status is exactly one of
   `done` / `running (what is pending)` / `queued` / `blocked (by what)` /
   `drifted` / `parked (why)`. Verdict is one clause: on plan, cost too much,
   answered, needs the user. Include the item "new result toward the goal
   delivered in this block" — if the honest status is `none`, write `none`.

3. **What actually blocks the goal now.** One line. "Nothing structural, waiting
   on X" is a valid answer; "many things" is not.

4. **Drift and waste audit.** Check each, answer yes/no with the evidence:
   - A chain of diagnostics on one sub-question while the deliverable (an
     optimizer step, a device, a measurement the user asked for) has not moved
     for hours.
   - Lanes or jobs still running after their question is answered (each one
     costs seats and disk).
   - Resource ceilings: license seats budgeted against ALL users of the pool,
     disk quota trend, QOS job caps.
   - Token economy: is the session model doing polling, log reading, mechanical
     edits or watcher re-arming that a subagent or a single watcher should do?
     Count the wake-ups that carried no decision.
   - Scope creep: any geometry, physics or objective change the user did not
     ask for or approve.
   - Dropped items: anything the user asked for that has no row in the table.

5. **Decisions, each with the rule applied.** For every drifted/wasteful item:
   stop it, park it, or continue with a stated stop rule ("close after step 2
   if ΔT < 0"). Decisions that are the user's (geometry, scope, deleting,
   cancelling outside an authorised class) go to a "waiting for you" list with
   a one-line recommendation — never a menu.

6. **Record.** Put the checkpoint (steps 1, 3, 5) in the programme's memory file
   with a timestamp. The next plan-check starts by diffing against it.

## Output shape

- Lead with the goal and the blocker (two lines).
- The inventory table (≤ 8 rows; merge minor items).
- "Where I am not on plan" — the audit hits only, each with the correction
  already taken or the stop rule set.
- "Waiting for you" — the parked decisions, one line each with a recommendation.
- Under ~300 words. No history, no justification of past choices, no praise.
  Numbers carry their label (MEASURED / DERIVED / EXPECTED) per CLAUDE.md §9.

## Drift signatures worth naming (from this project's history)

- "Calibrating the gradient" for a day while acceptance is already on measured
  observables — the live gain ratio was the gate all along.
- A new FOM built three times before the quantity it measures was checked
  against the user's words ("sum the FAR field", not |E|² on a plane).
- Fallback watchers that wake the manager on rows it has already read.
- Seven concurrent solves on a 50-seat pool shared with another lane.
- A measurement convention (auto-shutoff) that silently biased every number of a
  seed by more than the gains being chased — found only when the scatter was
  measured instead of explained.

## Related

CLAUDE.md §9 (MAIN-GOAL CHECKPOINT), `work-alone` (decision policy while
unattended), `safe-compact` (checkpoint before compaction), memory
`feedback_work_automatically.md`.
