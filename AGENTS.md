# Pi-shift Bragg grating project

Read `docs/TE_CONTINUATION_2026-10-07.md` first for the current continuation and
isolated restart candidate, then `docs/TE_RECOVERY_2026-10-06.md` for the recovery,
job state and normalization fix, then `docs/CODEX_AUDIT_2026-10-06.md` for the
initial audit. Read `CLAUDE.md` for project conventions,
and the relevant live code. `docs/HANDOFF_FOR_NEW_AI.md` is a historical
snapshot: §6.2.7 supersedes its earlier running-job instructions.
For current TE method read `runners/lumopt2_design/THEORY.md` §13.
For the source/seed map, both server environments, current access checks and
remaining restart bugs, read `docs/INVERSE_DESIGN_READINESS_2026-10-06.md`.

The user retained TE, 98 periods per side, 60 free periods per side, no
surrounding posts, and spatial width near 20 µm on 2026-10-06. Each seed's
existing ±2% width band remains unchanged. The user subsequently authorized
S2 calibration/gradient recovery on Athena; see the recovery record for its
jobs. Unrelated stopped campaigns remain stopped. Ask about unresolved physical/algorithmic choices,
not routine reversible checks already within the requested task.

## Continue authorized work (user instruction, 2026-10-07)

Continue through the authorized workflow until its intended outcome is reached.
Do not end work just because a job was submitted, a test phase passed, or the
next routine step would otherwise prompt “shall I continue?”. During server
waits, advance independent work and check results at a reasonable cadence;
after completion, inspect the evidence and proceed to the next authorized
step. Persist state and next actions so context changes do not lose the task.
Give concise progress updates; distinguish queued, running, tested and complete.
Pause only for a genuine blocker or a material decision requiring user input,
and preserve the existing limits on destructive actions and server changes.

Current continuation authorized October 7: finish S2 calibration on Athena,
fix/test the campaign restart bugs, then run the small hardware smoke after
the calibration gates pass. A full optimization campaign is not yet authorized.

Overnight extension (October 7): the user requested autonomous work until
ready for full optimization, with no routine questions while they sleep.
Continue calibration, evidenced bug fixes, comparison of Claude's stored
results and bounded hardware validation needed for readiness. Do not bypass
failed scientific gates or launch the full campaign prematurely. Preserve the
confirmed width bands while explicitly auditing the stricter stated goal of
no width increase; report the distinction rather than calling +2% unchanged.

Sol subagents are authorized for bounded audits, implementation and tests.
The lead owns cross-cutting physics decisions, integration and server dispatch;
assign separate file ownership and prevent duplicate jobs or shared edits.
Claude consultation through a verified read-only skill is authorized when
useful. Do not assume its CLI, tools or login work without testing access.

A 30-minute thread heartbeat `s2-overnight-inverse-design-readiness` is active
to resume this work. It must read current state and avoid duplicating an active
turn/job. The local host must remain available. Stop the heartbeat when work
is complete or a genuine blocker prevents all useful authorized progress;
record any unfinished work honestly. The read-only Athena watcher is separate.

Preserve existing data and source history. No file deletion, destructive git
operation, cancellation, container replacement or remote cleanup without
specific authorization. Do not test destructive guards on live servers.
Read-only server checks are allowed. Source editing does not imply upload or
dispatch. The user explicitly authorized periodic local Git commits on
October 7. Commit focused, reviewed work at meaningful validated milestones;
inspect the index and diff first, stage explicit paths, and record test status
and pending hardware gates honestly. Preserve unrelated edits and exclude
credentials, bulky simulation results and scratch artifacts. This authorization
does not include pushing, branch changes, history rewriting or destructive Git
operations. Keep candidate changes isolated while calibration dependencies
require frozen production sources; a documentation checkpoint is appropriate
while hardware validation is pending.
Local lumapi sessions must use `hide=True`; no local FDTD solve unless asked.

Convergence extension (October 7): independently assess the approximately
50 nm mesh and N98-per-side spatial-width saturation, including whether a
shorter device is suitable. Keep current production settings until measured
comparisons justify a change. Separate temporal, mesh, transverse domain/PML
and physical-length sensitivity; unchanged width does not mean unchanged
transmission. Reuse surviving solved field-profile data first. Follow the
inverse-design skill's convergence-and-length reference and preserve the
original per-seed width bands while reporting strict no-widening separately.

## Parallel clusters and maintained knowledge (October 7 user instruction)

The user authorized remaining needed gradient/diagnostic jobs on Athena and
IGUM in parallel. Choose independent, scientifically justified work and use
both when useful; this does not authorize a full optimization campaign or
bypass failed calibration. Do not move or duplicate an active job just to use
another server. Before dispatch, inspect both queues and the shared Lumerical
seat pool; account for queued commitments that can start together, actual GPU
seat demand, memory and applicable QOS/account limits. A scheduler GPU cap is
not a license-seat budget. Verify code/runtime/data parity and unique outputs
before splitting a numerical comparison across clusters.

Continually maintain relevant project rules and installed/project skills at
validated milestones: add confirmed reusable discoveries, correct stale
procedures, and keep measurements/job IDs in dated evidence records. Clearly
separate user requirements, measured behavior and unverified hypotheses.
Preserve previous material/history; validate updated skills and record changed
paths. Do not encode a transient failure or advisory model claim as settled
physics. See the cluster-ops and inverse-design skills for current procedures.

Codex skills installed for this project:
- `C:/Users/evyat/.codex/skills/pi-shift-inverse-design/SKILL.md`
- `C:/Users/evyat/.codex/skills/pi-shift-cluster-ops/SKILL.md`
- `C:/Users/evyat/.codex/skills/ask-claude/SKILL.md`
- `C:/Users/evyat/.codex/skills/budget-aware-research/SKILL.md`

The 13 project Claude skills are adapted for Codex under `.agents/skills/`.
Read `.agents/skills/project-context.md` for the shared runtime and historical
memory path. `docs/CLAUDE_SKILL_TRANSFER_2026-10-07.md` records the transfer,
preserved originals and 18 personal skill packages already available in Codex.

## Budget-aware autonomous work (user instruction, October 7)

Use the installed budget-aware-research skill for this workflow. Its tested
usage summarizer preserves unknown limits and never infers tokens remaining.

Prefer Astra for planning, scientific decisions and integration review; use
Sol for bounded implementation and testing with explicit file ownership.
Provide focused context and evidence paths instead of forking the entire
history into every worker. Delegate once, verify the result, and avoid
duplicate audits unless new evidence warrants them. Model labels alone do
not establish correctness; measured physics and tests remain the gates.

Read actual usage limits before a long model-intensive stage. Remaining
percentage is an account allowance, not a count of tokens or a promised
number of future runs. Treat missing/model-specific limits as unknown and
do not assume Sol has a separate allowance. At low remaining allowance,
prioritize critical fixes, concise decisions and durable checkpoints; use
the authorized bounded Claude consultation for useful independent reasoning.
Never redeem a reset or purchase credits without explicit user authorization.
Do not claim that an automation can run after account capacity is exhausted.
Server jobs and the read-only watcher can continue independently of model
availability; preserve exact next actions before capacity becomes critical.

Claude may return analysis or a proposed patch through the tested text-only
helper. It is not automatically a repository/server operator or a native
Codex subagent. Verify the requested model in response metadata, report
access failures, and test proposed patches through the normal workflow.

Existing `.claude/skills/` procedures are useful references, but their older
physics and resource claims need checking against current code and evidence.
Claude-specific model routing, tool names, memory loading and permission
hooks are not automatically available or enforced in Codex. Do not spawn
agents merely because a historical Claude routing rule names them.

Run relevant offline gates on code changes. Check exit codes. Build/monitor
changes also need hidden CAD verification; optimizer changes need a small
authorized hardware smoke before a long campaign. Preserve temp artifacts
when useful and keep scratch probes outside the repo. Report limits plainly:
an offline or build-only pass does not establish new FDTD physics.

## Plan expensive simulations before dispatch (October 7 user instruction)

The user requires physical gradient validation, including critical review of
Claude's original checks and independent reproduction where needed. Maintain
an evidence map distinguishing historical claims, reproduced arithmetic,
matched forward/adjoint tests, failed gates and unavailable raw artifacts.
A prior PASS label applies only to its tested objective, geometry, numerical
settings and held-out scope; it is not transferable calibration.

Before each expensive stage, write a short decision plan: the hypothesis and
observable, reusable evidence, dependencies, proposed independent run groups,
measured runtime range, queue/setup/post-processing overhead, resource/seat
budget and the action each possible result would justify. Identify the
critical path and prepare its readers, tests and next-stage inputs during
current solves. Prefer experiments that separate competing explanations;
do not accumulate runs with no decision attached. Estimate from comparable
completed jobs, revise from actual logs, and report ranges rather than promise
finish times. Include adjoint/twin forwards and CPU post-processing in cost.

Batch independent jobs across verified Athena/IGUM environments within the
shared license budget. Dependent physical decisions still require result
review; do not prequeue them merely to fill GPUs. Do useful independent work
during waits without repeating completed audits. When only external results
remain, maintain the already-authorized continuation and precise next actions.
Preserve files and active source identities; planning never bypasses a failed
gradient gate or authorizes the full optimization campaign.


## Analysis files and parallel servers (user, October 7)

Keep MAT as the user's normal exported/downloaded analysis format. Reuse
existing MAT results and do heavy postprocessing on servers; preserve original
solver FSP/H5 and checkpoint evidence. Export coordinates, spectra and profiles
with definitions/units/provenance so widths can be recomputed. Read the
pi-shift-inverse-design convergence-and-length reference for this workflow.
Use Athena and IGUM concurrently when tasks are independent and resources fit.
Prefer measured turnaround for sequential optimization and available slots for
sweeps; check actual hardware and shared Lumerical seats across both clusters.
