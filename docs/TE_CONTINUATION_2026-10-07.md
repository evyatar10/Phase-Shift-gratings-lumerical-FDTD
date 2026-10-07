# S2 calibration and restart correction continuation

User authorization: finish S2 calibration on Athena, correct/test campaign
restart persistence, then run a small hardware smoke after the calibration
gates pass. Full optimization is not authorized by this continuation.
The user also explicitly requested continuous progress through routine steps;
the project `AGENTS.md` records that instruction.

## Overnight extension and current continuation

03:40 Israel heartbeat: the watcher snapshot `20261007T002253Z` confirms
task170750_9 completed exit0 and170750_10 running on n306; nine of twelve
FD legs complete. Watcher42384 is alive with empty stderr, and no extra SSH
connection was needed. Task09 scalar checks pass (finite and inside scan):
peak3(T_out)=0.9254833733305935, modal_peak3=0.971286134143864,
lambda=1561.0582396498414nm. Tasks11/12, peak reconstructions and fit remain
pending. Prepared N98 packet/doc are being preserved in a focused local Git
checkpoint; this remains offline preparation, not hardware validation.

03:10 Israel heartbeat: bounded read-only SSH succeeded; task170750_9 is
still RUNNING on n306 (launched02:40), eight of twelve FD legs complete.
No failure or new immutable checkpoint appeared; tasks10-12 and final
reconstruction/fit remain pending as expected. Watcher42384 is alive with
empty stderr. Account-wide weekly allowance is 9% remaining, ordinary usage
allowed; conserve it for final-fit review, integration and hardware evidence.
No new dispatch, source change or repeated audit was warranted at this check.

N98 preparation completed after the 02:40 check: read
`TE_N98_PILOT_PREPARATION_2026-10-07.md`. Six focused offline tests passed;
the exact original S2 seed/bounds and pilot options are preserved in its
JSON/archive. It is a blocked preparation helper, not a launchable study or
hardware pass. The worker has released ownership. Calibrated perturbed
`te_point` differs from the original seed; directional qualification at the
actual starting geometry remains necessary. For measured width comparisons,
use the total v3 moving-width prediction, not `dw_pred` alone. Archive SHA256
`b35cf728d62f462a5da29327d48c10a0b06c3131830c4e30073fb7e13fe5f979` was
independently checked. The 02:40 status checkpoint is committed as d23e6ac;
no server/source changes accompanied this preparation.

02:40 Israel heartbeat: task170750_8 completed exit0 and task170750_9 is
running on n306; eight of twelve FD legs complete. The new task08 scalar
checkpoint is saved, with peak3(T_out)=0.9258425299757719,
modal_peak3=0.9719109009213397 and lambda=1560.8782189638007nm. Final
reconstruction/fit jobs remain dependency-pending. Watcher42384 is alive;
one bounded read-only SSH check was used to refresh the older snapshot.
No integration or dispatch occurred. The existing Sol worker
`/root/objective_code_audit` now owns a NEW isolated fail-closed N98 pilot
preparation and `TE_N98_PILOT_PREPARATION_2026-10-07.md`; do not duplicate
that work or modify its files while active. Scope is offline runner/launch
packet preparation only, no CAD/server action, calibration hash unset and
qualified field-gradient evidence must remain visible. Prior candidates and
production remain untouched. Inspect the worker's handoff before integration.

02:10 Israel heartbeat: one bounded read-only SSH check succeeded. Task
170750_7 completed with exit0 and task170750_8 is running on n306; seven of
twelve FD legs are complete. Tasks9-12 are queued at the serial throttle,
170752_0/1 and170754_2 remain dependency-pending. New immutable task07 was
saved by the existing watcher helper. All seven completed legs have finite
scalar observables and resonance inside their configured scan; this is not
the pending calibration fit or gradient-validation verdict. Task07 has
peak3(T_out)=0.9258033986058473, modal_peak3=0.971867567526548,
lambda=1560.8782189638007nm. Quota221G/300G soft, watcher42384 alive with
empty stderr. Ordinary Codex usage is allowed with 10% weekly allowance
remaining. No new job, upload, cleanup, or production edit occurred.

At 01:43 the user reaffirmed overnight continuation toward morning TE
readiness. The existing heartbeat was updated, not duplicated, to include
the morning evidence handoff and the user's periodic local-commit permission.
The read-only watcher is alive with no stderr; calibration still has six of
twelve FD legs complete. Full-campaign dispatch remains outside this scope.
Morning handoff must distinguish completed checks from pending/failed gates,
and include tested code identity, calibration verdict, objective/adjoint
decisions, actual hardware evidence and exact launch/resume commands when
available. Do not convert the morning target into an unsupported readiness claim.

Heartbeat at 01:40 Israel time found no change since the fresh 01:37 watcher
snapshot. Account-wide weekly Codex allowance is 11% remaining; ordinary
usage is still allowed. Preserve capacity for the final calibration decision
and integration, rather than repeating completed audits. No Git mutation,
server connection or new dispatch was made in this follow-up. The next
scientific action remains inspection of the full peak3 calibration vectors
and unchanged held-out gates after the dependency chain completes.

Readiness status at 01:39 Israel time: the fresh read-only watcher snapshot
`20261006T223738Z` has task170750_6 complete and task170750_7 running;
six of twelve FD legs are complete. Peak reconstruction and final fitting
remain pending. The watcher process42384 is alive and its stderr is empty.
The small hardware smoke and N98 validation have not run. Thirteen offline
suites passing is not full optimization readiness. This check used the fresh
local snapshot and did not create an additional SSH connection or submit jobs.

Latest formulation review: read `TE_OBJECTIVE_DECISION_2026-10-07.md` for the
user's transmission/constant-width priority, wavelength freedom, the corrected
AL formulation, actual solve count and the reuse recommendation. Claude Opus
provided a substantive review, then its claims were checked/corrected against
live code and numerical algebra tests. The new opt-in measured-modal guard
passed all 13 offline suites, including ten focused driver regressions. Read
`TE_MODAL_GUARD_CANDIDATE_2026-10-07.md` and
`te_modal_guard_candidate_location.json` for the preferred isolated candidate.
Its engine SHA is `d11f34db9777dd1d9deba6ab19372238c046e8d705a5c9c8c4281702e88b9559`.
**Apply only `te_modal_guard_candidate_20261007.patch`, not both patches.**
It consolidates the previous fixes; the earlier candidate/archive remain
preserved as historical evidence. The new archive is
`readiness_evidence_20261007/modal_guard_candidate_20261007_52ed8b9e.zip`
(SHA256 `82ea60faa70780021183be2b8c8c04296b4baca48585dcee34eda45e84b83187`).
The production engine remains frozen; integration and hardware validation are
pending. After dependencies finish, recheck the manifest, integrate this one
patch, and rerun all thirteen gates from the actual integrated tree. Pilot
options are `wgp_modal_guard=True`, `best_metric="modal_peak3"`; defaults stay
off and existing S1/S2 configurations are not activated by this record.

Read-only Athena access at 01:30 succeeded: five FD legs complete,
170750_6 RUNNING on n306, remaining chain pending, quota234G/300G soft.
The task6 wrapper confirms eight threads and build4572. Its launch-time GPU
header is not a current GPU-utilization measurement. No submission, source
upload, cancellation or cleanup was performed by this check.

At 01:33 the account-wide weekly allowance was12% remaining (same reset,
ordinary usage allowed). Conserve model capacity for calibration decisions,
integration and hardware evidence. Use brief status-only follow-ups while
the unchanged serial dependency chain is running; do not repeat these audits.

The user subsequently requested autonomous overnight readiness work, Sol
workers, and a tested Claude fallback, with no routine questions. Full
optimization is still not authorized. The completion criteria are now explicit
in `TE_READINESS_GATES_2026-10-07.md`; preserve all failed scientific gates.

At 01:07 Israel time, the watcher confirmed task170750_5 completed and
170750_6 running: **five of twelve FD legs complete**. Remaining FD tasks,
170752_0/1 peak reconstruction and170754_2 fit are still pending. Watcher
PID42384 is alive; its stdout is change-only, so an old printed timestamp is
not proof polling stopped. No new job was dispatched during this overnight
audit. Production source hashes remain frozen.

Native thread heartbeat `s2-overnight-inverse-design-readiness` is ACTIVE every
30 minutes for this thread. Its prompt reads these records, avoids duplicate
work and now includes budget-aware routing. It is separate from the read-only
Athena watcher. Host connectivity and account allowance limit future model
execution; neither an automation nor a skill guarantees unlimited operation.

Independent Sol reports:

- `TE_ALGORITHM_AUDIT_2026-10-07.md`: confirmed acceptance-reference poisoning,
  stale-width lookup, repeating missing-width trials, missing-gLambda step
  protection, stricter-band selection trap and global evaluation-budget gap.
- `TE_SAVED_RESULTS_REVIEW_2026-10-07.md`: stored S1 modal T0.95276031 at
  W19.385006µm is +1.38% wider than its anchor. Its original bands allow that;
  a literal no-growth claim does not. Missing final raw profiles and the
  recorded proxy residuals remain evidence limits.

Candidate fixes for accepted-only FOM reference, newest-row measurement,
bounded missing-width retries and fail-closed v3 wavelength-gradient handling
passed v3, projection and12 persistence tests at engine SHA
`a8ec9be98ee49ba0651526bd0c551f87b7dd4c5e20140e535177041104e42cb8`.
That intermediate hash is superseded by the completed candidate below.
The shared hard-width option and persisted global forward-evaluation budget
are now integrated **in the isolated candidate only**. All twelve offline
suites pass (exit0), including fourteen persistence tests, fourteen budget
helper tests, six real-campaign budget tests, six smoke guards and strict
width acceptance/selection regressions. Defaults and existing S1/S2 seed
configurations remain unchanged. Budget units are geometry forward batches,
not individual engine jobs or adjoints. Interrupted reservations stay spent.

Previously tested 12-suite candidate engine SHA256 (preserved, superseded by
the consolidated 13-suite candidate above):
`2558954f23df132a57f6de259e2a60ee43bf62012ce87617913b288117a38587`.
Regenerated patch SHA256:
`d4cf42eec8844e4a87c2d3dbe7d321ff1d608bdbbde0b795f1206286469dc14e`.
`git apply --check` passed without applying it. The expanded location manifest
records six candidate files and seven root dependencies with hashes; the new
campaign-budget gate is included in the patch. Full commands, exit codes,
hashes and logs are in
`readiness_evidence_20261007/budget_integration_gate_results.json`.
Recheck manifest hashes, preserve the frozen baseline, and apply/integrate
only after every dependent recovery task has finished. Rerun the actual
aggregate from the integrated tree before upload. No source was uploaded and
no additional solve was started during this candidate work.

The complete tested candidate, shared dependencies, patch, manifest and gate
record are also preserved byte-for-byte in
`readiness_evidence_20261007/tested_candidate_0118.zip` (16 files, SHA256
`a550be9f508c08d15e27a4185a9246a008b909fe337f9df231f551e68f73f78a`).
This removes reliance on the temporary candidate directory for recovery.

The prepared smoke now fingerprints both new helpers and its own runner,
and checks the persisted budget as well as evaluation/profile/state evidence.
Its calibration hash is still unset, deliberately blocking CAD/solves until
the actual peak3 fit passes review. Existing hidden-CAD generation established
geometry/plumbing only; all subsequent edits were optimizer/persistence logic.

The installed `budget-aware-research` and `ask-claude` skills are linked from
AGENTS.md. Ten usage-summary tests and a live exact `claude-opus-5-5`
text-only consultation passed. Claude's review is advisory, not a physics
gate. `BUDGET_WORKFLOW_2026-10-07.md` records the exact evidence. The account
reported18% weekly Codex allowance remaining, reset October12 at10:44 Israel
time; no Astra-specific allowance or Claude remaining allowance is known.
No reset credit was redeemed. The sanitized actual snapshot is
`readiness_evidence_20261007/usage_snapshot_0106.json`.
At the 01:19 stage-boundary check the same account bucket had **15% remaining**,
with the same October12 reset; ordinary usage was still allowed and the reset
credit remained unused. Do not attribute the account-wide change exclusively
to this task. The watcher PID was alive with empty stderr, and the native
automation view plus saved configuration confirmed the existing ACTIVE
30-minute heartbeat. Candidate work is now checkpointed; avoid new broad
audits while waiting for the calibration evidence.

Latest user steering: prioritize Claude reasoning/reviews now because Codex
capacity is scarce, while expecting intermittent Claude failures. The global
rules and budget-aware skill record this priority. A failure must be visible,
without repeated immediate probes; continue independent work or a bounded
Sol fallback where needed. Claude availability does not gate unrelated work.

VPN note from the user: access may intermittently become unavailable and
require reconnection; this was a future contingency, not a reported current
outage. The cluster skill now records bounded connection attempts, preserving
jobs/last known state, and reconciliation after access returns. The current
watcher exits on an SSH error, so check its error log and process liveness
before restarting one copy after connectivity is restored. Never infer job
failure or resubmit solely because the VPN is disconnected.

## Calibration evidence at 00:37 Israel time

170746 tasks 1, 2, 13 and 14 completed. Both stored-field gradient
reconstructions match the original printed gradients. 170750 tasks 3 and 4
completed; task 5 is running. Four of twelve FD legs are available. Tasks
6–12, peak3 reconstruction 170752_0/1 and fit 170754_2 remain queued behind
the serial throttle/dependencies. No final calibration verdict exists yet.
Quota is 237G/300G soft (330G hard). Scalar evidence is preserved under
`results_from_athena/recover_te_s2_20261006/watch/`.

The frozen shared sources remain unchanged. Do not deploy the candidate
below while those jobs depend on their current source fingerprints. Refer to
`TE_RECOVERY_2026-10-06.md` for the job chain, paths and resource checks.

## Isolated restart candidate

Candidate location is recorded in `te_restart_candidate_location.json`;
`te_restart_candidate_20261007.patch` carries the changes to existing source
files. New shared helper: `runners/lumopt2_design/persistence.py`.
New failure-injection gate: `runners/lumopt2_design/gates/gate_persistence.py`.
The production engine is not yet wired to that helper.

Changes in the candidate:

- Immutable evaluation/profile IDs, so repeated eval0000 cannot replace an
  earlier profile; MAC uses each row's recorded profile path. A missing
  required profile rejects the trial, even if a stale legacy file exists.
- Recoverable unterminated JSONL tail: readers preserve the original; the
  next writer archives it before atomically publishing the repaired log.
  Corruption in a completed record is fatal. All driver append paths use
  this behavior, including the missing-width retry branch.
- Missing optstate means fresh state; corrupt or invalid state fails with
  an explicit recovery message. Atomic writes retain previous/new versions.
- Committed acceptance IDs prevent a logged but unclassified trial from
  becoming the restart best after an interruption. Legacy rows remain
  readable under their old eligibility rules; historical missing decisions
  cannot be reconstructed retrospectively.
- Gradient artifact names use UUIDs, avoiding same-second restart collisions.
- The final reported best FOM comes from the same eligible row as its
  parameter vector; previously it could report the last accepted FOM beside
  a better historical vector. A real `run_campaign` regression test covers it.

The original candidate passed all seven existing offline suites. Committed
acceptance IDs passed the complete nine-suite run (including the new
persistence and smoke guard suites). After fixing the final best-FOM report,
eleven persistence tests pass, including real callback/driver tests and
deliberate publication failures. Rerun the aggregate after integration.
Logs live beside the candidate as `*.candidate.log`. These are offline tests,
not evidence of a new FDTD solve or a completed hardware smoke.

Persistence assumes one writer per campaign directory. State/profile history
is intentionally retained. Do not silently recover a corrupt active state
from an arbitrary history file: inspect matching evidence first.

Original frozen sources have been preserved and byte-verified in
`readiness_evidence_20261007/frozen_recovery_sources_20261006.zip`
(SHA256 `339c182a464eb8830232359527af4a06a3d6cc493c980b88e5fa9dc496992cc6`).
The engine remains `8f09350583a53146091faf590c2c93e8d93ac01a46b48006890db62c72a295e5`.

## Prepared hardware smoke

`runners/lumopt2_design/smoke_te_s2_restart.py` derives the existing task38
pipeline surrogate: N70/free60, 491.06nm pitch, 350nm height, TE bare,
1560.464nm center, 2nm/501-point scan. Two evaluations per invocation and no
automatic recenter retries. This is explicitly not N98 physics validation.
It refuses CAD/solves until an exact reviewed calibration-file hash is filled
in, rechecks the actual peak3 vectors/10% held-out gate, and audits immutable
profiles, committed state, MAC, v3 step and FOM=T_out peak3 before completion.
Five offline smoke guards passed. No server dispatch has occurred.

Hidden local CAD generated this N70 project successfully with provisional
(1,0) port coefficients for build testing only. Monitor assertions passed:
ports at +/-44.318um outside grating +/-34.497um, TE symmetry and frequency
sampling correct, single-frequency twin explicitly pinned. No solve ran.
Artifacts: `C:/Users/evyat/AppData/Local/Temp/s2_restart_smoke_build_szfccjdw/`;
`build_only_pass.json` records completion. The PowerShell `*>` wrapper returned
1 on benign Python INFO stderr; a no-CAD probe reproduced that wrapper error
with an actual Python exit0. Use subprocess capture for future exit reporting.

## Next actions

1. Keep the existing read-only watcher active; inspect each new scalar result.
2. Review both final calibration fits, including peak3 per-class and
   leave-one-out errors at the existing 10% gate. Do not relax or transplant
   coefficients. Save full precision and provenance when installing a fit.
3. Preserve a snapshot of the original frozen sources; integrate the tested
   restart candidate only after dependent recovery tasks finish.
4. Run the bounded hardware smoke with explicit calibration and fresh output
   identity; inspect width, wavelength, MAC, step and persistence evidence.
   Calibration failure is a scientific blocker to that smoke, not a reason
   to launch a full campaign or substitute S1 coefficients.
