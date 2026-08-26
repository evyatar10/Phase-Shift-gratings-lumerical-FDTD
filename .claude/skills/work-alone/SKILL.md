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
- **★Monitor wakes must carry INFORMATION (2026-08-16 token lesson): every
  monitor event costs a full model turn, so the change-detection key must
  contain ONLY decision-relevant state** — job IDs + run-states, result/eval
  counts, last-result physics, error counts, resource bands. NEVER include
  always-changing fields (elapsed time, timestamps, load) in the key: a
  clock in the key = a wake every sweep = ~90 no-op turns/day (measured;
  the fix cut wakes ~75% with zero protection lost). Same principle for the
  wake handler: a no-change wake gets a one-line hold, not a re-analysis.
  Polls themselves (ssh/shell) are token-free — only wakes cost; so poll
  as often as robustness wants, but WAKE only on change. Debounce the
  unreachable state (one wake per outage, not per sweep).
- **★POLLS ARE TOKEN-FREE BUT NOT SERVER-FREE — keep a connection budget
  (burned 2026-08-17).** A monitor raised to 24 ssh/h against IGUM got our
  key REFUSED ~80 min later ("Permission denied (publickey,password)" with
  port 22 open and the key offered = server-side rejection, not a network
  or VPN fault); ~45 min of ZERO connections restored it untouched — a
  rate-limit/fail2ban trip. Budget ≤3-6 connections/hour per host, make
  ONE ssh per poll (fold extra probes such as lmstat into that same
  connection, never open a second), and on ANY auth refusal STOP automated
  contact for ≥45 min instead of retrying — retries deepen a ban. Cluster
  JOBS are unaffected by login-node auth (they run on compute nodes and
  afterok chains still fire), so an outage costs visibility, not science.
- **★An unreachable cluster must be LABELLED, never omitted (burned
  2026-08-17).** A monitor that drops a cluster's block when its ssh fails
  produces an event that is INDISTINGUISHABLE from "the job disappeared" —
  a false FATAL alarm (seedA looked preempted; it was RUNNING 11 h with
  Restarts=0, the ssh had blipped). Emit an explicit
  `<CLUSTER>_UNREACHABLE` token into the change key instead: it reads
  correctly AND debounces for free, since the key stays constant for the
  whole outage. Handle EACH source separately — the common bug is guarding
  only the all-sources-down case (`if [ -z "$A" ] && [ -z "$B" ]`), which
  leaves single-source outages silently mangling the event.
- **★A DOWN SCHEDULER IS A THIRD STATE — ssh-up + slurm-down looks exactly like
  "all jobs finished" (burned 2026-08-26).** `squeue` returns EMPTY (not an
  error the shell sees) when slurmctld is unreachable, so a drain condition of
  "no jobs in squeue" fires a FALSE DRAIN while jobs are still solving; `sacct`
  and `scontrol` are dark at the same time, so the natural follow-up
  ("what state did they end in?") returns nothing and invites a
  preempted/crashed story that is pure fiction. Acting on it means RESUBMITTING
  JOBS THAT ARE STILL RUNNING — duplicate GPU + doubled license draw.
  Guard explicitly: capture squeue's stderr (`2>&1`) and test for
  `Unable to contact|connect failure`, emitting a `SLURM_CTL_DOWN` token into
  the change key. Then fall back to a scheduler-free liveness signal — job-log
  BYTE COUNT and terminal markers (`Simulation time` / `Exit code`) — which
  keeps working through a controller outage. NEVER resubmit or cancel while the
  controller is unreachable: job state is unknowable, and an unknowable state is
  not an idle one. Note that FDTD tasks emit NOTHING between "Saved layout" and
  completion, so a log that stopped hours ago is normal mid-solve, not a crash.
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

## Live monitoring doctrine (user rule 2026-08-16 — supersedes drain-only watchers)

Drain-watchers detect COMPLETION, not TROUBLE: a requeued job still "is in the
queue" (B4's 8.9 h preemption sat invisible until the user asked), and an
early crash sits undiscovered for hours. For every active batch run a LIVE
EVENT MONITOR instead (Monitor tool, event stream), emitting on:
- any per-job STATE/NODE change (R→PD = requeue; node swap = migration),
- any NEW error signature in the task logs (Traceback | TASK FAILED |
  LumApiError | Unable to checkout | dead device | DIVERGED),
- implausibly early completion (license no-op suspect),
- final drain.
Plus the T+5-10 min post-dispatch log peek for every new-code dispatch (the
first minutes catch build/import/geometry errors before GPU-hours burn).
Errors are to be SEEN AS THEY HAPPEN, not reconstructed when the user asks —
"the user asking is what surfaced the problem" counts as a monitoring failure
to fix, not a status quo.

## The trouble-finder (user doctrine 2026-08-16) — triage table + lesson capture

The live monitor is a TROUBLE-FINDER with decision authority, not a pager.
On every event, investigate within one cycle and classify:

| Class | Signature examples | Prescribed response |
|---|---|---|
| benign-recovered | license blip + LocalRunner retry succeeded; transient ssh loss | log it, note the systemic signal (e.g. seat pressure), no action |
| degraded-retrying | repeated retries, slow node, requeue of a resume-protected job | keep watching at tighter cadence; pre-stage the recovery command |
| FATAL-branch | task FAILED, dead-device guard, DIVERGED, exhausted retries, requeue of an UNPROTECTED long job | stop that branch NOW, root-cause from logs (free diagnostics first), fix, redispatch; report severity-first (§9) |

License seats are part of the watch (bands ≥35/50 HIGH, ≥45/50 CRITICAL from
the IGUM lmstat probe): before any fan-out, seat-probe; during, the monitor
bands it; LocalRunner's 2 retries are blip-cover only.

★LESSON-CAPTURE DUTY (user, 2026-08-16 — "so I don't have to keep telling
you"): during intensive phases, every incident, surprise, measured limit, or
correction gets written into the relevant rule/skill/memory THE SAME SESSION,
unprompted. At every safe-compact checkpoint ask explicitly: "what did we
learn since the last checkpoint that is not yet in a rule or skill?" — and
write it. The user prompting a lesson that was already visible in the data
counts as a capture failure.
