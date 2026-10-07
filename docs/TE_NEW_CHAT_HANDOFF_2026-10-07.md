# TE inverse-design handoff — October 7, 2026

## Read this first

Work is **paused at the user's requested 2% remaining account allowance**.
The existing automation `s2-overnight-inverse-design-readiness` is PAUSED.
Do not resume autonomous work or the schedule without user direction. No reset
was redeemed and no capacity purchased. Account allowance is not a token count;
model-specific and Claude allowances are unknown.

The main goal is a working TE optimization that increases transmission while
maintaining spatial envelope FWHM near 20 microns. **The next useful step is a
bounded current-code S1 optimizer/restart smoke. It is not yet prepared or
submitted.** S2 has a separate failed transmission-gradient calibration.
Do not describe that failure as disproving the successful historical S1 lane.
No full optimization campaign has been released or authorized.

This file supersedes older broad statements that all TE work must wait for S2,
older 3% pause thresholds, and proposals for more N80 qualification. Detailed
dated evidence remains in `TE_CONTINUATION_2026-10-07.md`; its top is newest.

## Decisions to retain

- TE, no surrounding posts, 60 free/apodized periods per side. **N80 is accepted
  for practical TE work**, leaving 20 outer periods. Do not spend more time on
  length certification unless an observed consequential anomaly changes a decision.
  Preserve historical N98 configurations and artifact identities when resuming.
- Spatial envelope FWHM is the width acceptance authority. Second-moment sigma
  is diagnostic only and must not substitute for FWHM. Existing seed-specific
  +/-2% bands remain; acknowledge that +2% is not literally “no width increase.”
- Wavelength may move modestly; current v3 local wavelength trust control remains.
- The intended smoother mesher is **precise volume average (PVA)**. S1 nominal
  optimization mesh is 50 nm; S2 is pitch-locked 49.106 x 50 x 50 nm. Native
  saved/reloaded CAD confirmed PVA. An overlapping device dy=57.915 nm override
  exists, so nominal optimization settings do not describe every realized cell.
  No demonstrated wrong mesh choice or misplaced TE monitor was established.
- Reuse saved data before expensive solves. Every further check should be able
  to change an actual decision. At milestones and direction changes, reassess
  the main outcome, actual blocker, and whether the next action advances it.

## Algorithm and implemented fixes

Current code uses `campaign_te_s1.SPEC_V3` / `campaign_te_s2.SPEC_V3` and the
v3 bounded QP update, with variable bounds, trust caps, width band and wavelength
trust control. There are 296 stored vector slots and 181 physically active
variables: 60 corrugation, 60 average-width, 60 shift, and cavity.

The transmission objective is `peak3(abs(T_out))`, a three-sample parabolic
selected-resonance estimate. Modal `abs(S21)**2` is also reported; these are
different quantities, without a universal conversion. One forward and separate
port/width adjoints are used. `softW` supplies the width gradient, but measured
envelope FWHM gates acceptance. Moving-resonance width differentiation uses
`gW_res = gW_fixed + cW*gLambda`. Width-gradient reuse remains off. A combined
adjoint or augmented-Lagrangian replacement was not implemented or validated.

Confirmed fixes integrated and tested:

- Wavelength-gradient normalization previously mixed a Tout adjoint numerator
  with a modal-power spectral slope, producing about a 5% mismatch. It now
  differentiates Tout consistently; regression checked.
- Restart/persistence/forward-budget fixes cover unique evaluation identity,
  profile binding, accepted/rejected state, interrupted JSONL/atomic state,
  measured-width guarding, persistent global forward budget and cold restart.
- All 13 offline suites passed at the integration milestone **1b2d75e**.
  Do not reapply the old isolated candidate patch. Current-code hardware
  optimizer/restart validation is still pending.

Frozen engine SHA recorded in the evidence:
`53108bbd368dd699179ab65029628aa4c4b2d8ad687777f88b82f404a0952569`.
Recheck exact identities before any new release; do not assume all files are
unchanged merely because this handoff quotes a historical hash.

## S1: useful historical validation and shortest next route

Original uniform-seed S1 reached modal transmission **0.9527603093** with spatial
FWHM **19.38500619 um**, near 1561.044 nm (seed about 0.9054 and 19.12 um).
Its measured constants in `campaign_te_s1.py` are port `(0.945335, +0.117012)`
and field `(0.966720, +0.036560)`. Recorded port fit errors were <=5%, with a
known held-out shift30 discrepancy of 16.5%; width fit errors were <=0.2%.
These qualifications remain visible.

Six independently recomputed chronological N98 peak3 directional gain ratios
(measured / predicted) were:

| Saved run | Iterations 0, 1, 2 |
| --- | --- |
| Toy | 1.012327, 0.887844, 0.897119 |
| Campaign | 0.982931, 0.951232, 0.842508 |

All had the predicted sign. Copied toy-prefix records and reused evaluation IDs
were disambiguated by timestamps and matching FOM. Historical toy acceptance/
marker problems were not relabeled as clean passes. N70 plumbing smoke was
excluded from this physical evidence. The evidence supports a bounded current
S1 smoke; it does not establish every component gradient or release a full run.

On user-directed resume:

1. Read AGENTS.md, this handoff, top continuation and current readiness gates.
   Check actual allowance and preserve the user's reserve instruction.
2. Prepare an isolated S1-specific runner using current SPEC_V3, measured S1
   constants, **historical N98/free60**, and one exactly pinned historical start
   vector. Keep the original campaign untouched and use a new output/label.
3. Bound iterations and persistent `max_feval`; no automatic recenter loop.
   Reuse existing budget/persistence tests and the S2 smoke audit structure,
   without importing S2's failed constants, calibration gate or N70 geometry.
4. Verify start-vector, FOM/profile identity and source fingerprints offline.
   Fresh shared-seat, storage and queue checks precede any authorized dispatch.
5. Run the small readiness test, stop normally at its budget, and cold-verify
   state/results. No forced kill or cancellation. Inspect evidence before
   proposing campaign release; do not silently launch a full campaign.

Exact evidence: `readiness_evidence_20261007/s1_directional_readiness_2240.json`,
`s1_readiness_review_2240.json`, `s1_readiness_consult_2240.zip`.
The bounded text-only Claude review verified `claude-opus-5-5`; its proposed
forced kill, generic L-BFGS description and invented repeatability tolerance
were rejected/corrected. Reviewer optimism cannot waive scientific gates.

## S2: unresolved calibration, not a reason to stop independent S1 work

Claude's original S2 nonuniform/overshoot lane had port calibration OPEN;
`campaign_te_s2.py` still records `ADJ_FIX_PORT=None`. Field constants were
`(0.983623, -0.088013)`. All 12 recovered finite-difference legs and both
adjoint quadratures completed, but both legacy and peak3 global fits FAILED.
Peak fit residuals for corr1/corr30/avg1/shift1/shift30/cavity were respectively
+16.70%, +162.11%, -8.31%, +13.34%, +0.64%, -31.95%.
The held-out checks failed their 10% gate; minimax worst error was 58.1015%.
Do not install failed coefficients or patch individual parameters to force a fit.

Finite-difference step and duration sensitivity remain. At shift30 h=0.0625 nm,
Tout derivatives at 100/150/200 ps were -0.000208331154 / -0.000153814689 /
-0.000383580910 per nm. This is not temporal convergence. Original adjoint
fields are approximately 67.4 ps; longer forward runs lack optimization-volume
DFT fields and cannot become replacement adjoint pairs. Geometry/dEps/grid
provenance audits did not establish a geometry/monitor bug. The Port2 displacement
of -49.106 nm is intentional vendor numerical propagation phase, not evidence
of accidental placement. Failure alone does not identify adjoint versus FD cause.

Saved-data source normalization showed unit-magnitude ratios to about 2e-15,
linear phase equivalent to **0.198736 fs delay per additional 50 ps**, and
unchanged sourcepower. This explains phase-like raw trace changes, not normalized
power drift or the gradient error. Do not call it a measured noise floor.
If S2 is resumed separately, next diagnosis should examine original paired
forward/adjoint normalization and phase using saved evidence/vendor source;
do not automatically run more long forwards.

## Other completed measurements and review

**40 FDTD solves were verified complete.** Later extraction jobs were CPU-only,
not additional FDTD solves. IGUM native N98/N80 pair 101434 and profile reader
101720 completed and artifacts were preserved. Production crop FWHM was
19.643715344 / 19.473301618 um; common N80 crop gave 19.492190806 /
19.473301618 um. Crop/baseline effects explain much of that difference.
Length is now settled by the user's practical decision; do not repeat this study.

Time-monitor raw `getdata` export job 171516 failed because its API path did not
expose Ey. This was **not an FDTD failure**. Alternate E/H `getresult` inventory
171733 and trace export 171739 succeeded. Source reader Athena **171768_0**
completed exit 0 in 28 seconds, with inputs/class/constructor preserved.
Its source SHA was `4597f1fbb9a2510fa846a8da0a38459db8b46022c58e76f5ebf065f255ae54b2`;
remote output `/work/results/s2_source_norms_20261007_2224`.

Q3dB/CMT saved-data audit and fixes are committed in **197d343**:
selective MAT loading, false feasibility-boundary root rejection using actual
residuals, honest single-anchor uncertainty, conditioning reporting, invalid
T>=1 rejection without clipping, and retained width-only apodized rows.
Six regressions passed; isolated calibration passed 48/50 with two preserved
stress failures. A TM N100-to-N165 holdout missed transmission by about 0.07
while width agreed within about 0.5%; adding N120 improved transmission greatly.
No matched TE N<=100 holdout established short-device extension accuracy.
No further Q3dB solves are needed for the current optimizer path. Production
calibration CSV was unchanged. See `Q3DB_AUDIT_2026-10-07.md`.

## Durable memory, skills and evidence locations

- Detailed chronology: `TE_CONTINUATION_2026-10-07.md` (newest first).
- Gates/failure: `TE_READINESS_GATES_2026-10-07.md`,
  `TE_S2_CALIBRATION_FAILURE_2026-10-07.md`, `TE_RECOVERY_2026-10-06.md`.
- Theory/current code: `../runners/lumopt2_design/THEORY.md` section 13,
  campaign S1/S2 modules and live optimizer code.
- Skill transfer: `CLAUDE_SKILL_TRANSFER_2026-10-07.md`; 13 project skills
  adapted under `.agents/skills/`, preserving originals. Personal skills under
  `C:/Users/evyat/.codex/skills/`: pi-shift-inverse-design,
  pi-shift-cluster-ops, budget-aware-research and ask-claude.
- Rules now cover continued authorized work, main-goal checks, proportionate
  validation, N80 acceptance, resource/storage checks, server-side processing,
  saved-data reuse, bounded Claude/Sol delegation and account reserve.
- Compact scratch/reviewer archives in `readiness_evidence_20261007/`:
  `saved_data_review_and_readers_2220.zip`,
  `saved_source_and_proportionality_2228.zip`,
  `s1_readiness_consult_2240.zip`. Scripts/reports survive temporary paths.
- Original S1 results: `../results_from_athena/campaign_te_s1/` and `validate_te/`.
  Traces/source data: `../results_from_athena/te_readiness_20261007/`.
  Native profiles: `../results_from_igum/te_readiness_20261007/length_profiles_2222/`.
- Pause receipt: `readiness_evidence_20261007/user_budget_pause_2percent.json`.

Validated checkpoints include 1b2d75e (integrated fixes), 197d343 (Q3dB),
c983ee7 (N80), b9c3aaa (S1/S2 distinction and 2% reserve), and 60cc47e
(main-goal rule and measured S1 directions). This handoff is committed separately.
Large results and some evidence remain preserved locally but untracked;
**Git is not a complete backup of raw simulation data**. Preserve unrelated
dirty files in CLAUDE.md, README files, igum config and original Claude skills.

## Operations and safety at pause

Athena and IGUM may run concurrently after checking their actual queues,
resources and shared Lumerical seats; do not assume a fixed slot limit.
MAT is preferred for transfer; perform heavy reads on servers. Preserve FSP/H5.
Use the cluster skill's tested launch commands rather than inventing dispatch.

Last dated snapshots, NOT fresh live guarantees: queues empty after the last
CPU reader; at 22:23 GUI 0/50 and solver 0/50, Athena soft-quota headroom about
105.80 GiB. IGUM at 22:15 had about 25.904 TB filesystem free, enforceable quota
headroom unknown; local at 21:42 had about 88.22 GiB free. Earlier VPN/SSH
timeouts recovered without authentication/VPN changes. If access fails, make
one bounded check, preserve unknown job state, and avoid retry loops/duplicates.
The old recovery watcher exited as designed; it is separate from the paused
heartbeat. No new job or cancellation was performed during final handoff.

Preserve all failed evidence and frozen inputs. No deletion, cancellation,
cleanup, container replacement, pushing, branch changes or destructive Git.
Focused reviewed local commits are authorized; inspect/stage explicit paths.
No local FDTD. No full campaign until its relevant gates and authorization pass.
