# Independent TE v3 algorithm audit — October 7, 01:05 Israel time

Read-only audit for the overnight readiness task. No solver, dispatch, remote
mutation, shared engine/candidate edit, deletion or Git mutation was performed.
The only project file written by this audit is this report. Reproductions and
their source snapshot are preserved in
`C:/Users/evyat/AppData/Local/Temp/te_algorithm_audit_20261007_i_dcyokz/`.

The reviewed restart candidate snapshot has SHA256
`130b8fd44a9460312e9686d4636626e005d1e7179228c02c5ea8febf5ee3a2fb`.
Root may subsequently correct it; findings below describe this exact snapshot,
not the integration status of later changes. Production engine remains the
frozen recovery source at review time.

## Confirmed implementation findings

### 1. A width-rejected objective can poison future acceptance

`run_projected` correctly rejects a v3 trial when width violation grows, but
its final `fom_best` update excludes only wavelength jumps and mode hops.
A width-rejected trial can therefore raise the slack reference. Subsequent
feasible improvements are judged against an infeasible objective and rejected.
Production reference: `lumopt2_design.py` lines 3148–3161 and 3574.

Real driver reproduction (`reproduce.py`): accepted FOM 0.700 at W=19.121;
trial FOM 0.800 at `W_HI+0.200` is rejected with width violation 0.150 µm;
`fom_best` nevertheless becomes 0.800. A subsequent in-band FOM 0.705 is
rejected. All geometry/gradient math is the actual driver/QP with fake scalar
forward results. This establishes the coding behavior, not a new FDTD result.

Negative control (`poison_negative_control.py`): adding `w_inf` to that final
guard leaves the reference 0.700 after rejection and accepts 0.705 next.
Better general rule: commit the objective reference only for accepted points,
with feasibility/restoration status explicit. Acceptance IDs alone do not fix
the in-process reference.

### 2. A missing fresh width can silently reuse an older row

`_row_of_params` filters by required fields before selecting the newest
matching geometry. Thus a latest row with missing width is skipped and an
older width-valid row for the same geometry is returned. That defeats the
first-evaluation width failure and missing-width retry logic. A recentered
window or changed twin wavelength also makes old diagnostics inappropriate
beside fresh gradients. Production reference: lines 3794 onward.

`stale_row_probe.py` writes identical parameter rows: ID `old` with W=19.121,
then ID `new` with missing width and a diagnostic error. The candidate selects
`old`. Fix: choose the newest geometry match first, then check required keys;
for the immediate driver callback, prefer the current evaluation ID as the
identity rather than an approximate vector match.

### 3. Persistent missing-width errors repeat the same trial at the cap floor

The missing-width branch halves the cap to a 2 nm floor and continues before
the ordinary rejection branch's duplicate/stall safeguards. It neither limits
consecutive missing-width retries nor checks whether the next geometry was
already attempted. Production reference: lines 3059–3091.

`missing_width_probe.py` runs the real driver with one good width followed by
six deliberately missing widths. Cap sequence: 10, 5, 2.5, 2, 2, 2, 2 nm.
The final two differences between consecutive trial vectors are exactly zero;
all six are `no-width-retry`. Those repeated forwards cannot repair a broken
post-processing pipeline, and consume campaign time until the iteration limit.

Fix: bounded missing-width retry count plus duplicate/floor stop that retains
the accepted state. Report an unresolved measurement failure, not convergence.

## Width contract: retained band versus latest no-increase request

Current retained bands permit width growth. With margin 0.100 µm:

| Seed | Anchor | Hard final-selection band | Effective measured acceptance upper |
|---|---:|---:|---:|
| S1 | 19.121 µm | 18.73858–19.50342 µm | 19.45342 µm |
| S2 | 19.636 µm | 19.24328–20.02872 µm | 19.97872 µm |

QP bands are inset by the full margin; acceptance permits half that margin
beyond the QP edge. These choices are compatible with the previously retained
±2% bands, but do not mean width never increases. The newest user wording asks
for transmission growth without width growth. Preserve old results and
calibration while making any stricter future contract explicit.

Every improved row in the stored eight-row S1 log exceeds the 19.121 µm anchor.
The latest is 19.38500619 µm, a 0.26400619 µm / 1.381% increase. It is compliant
with the old band. Only initial row 0 (19.11980134 µm, modal T=0.90544418) meets
the literal seed-upper criterion among these eight stored rows. Do not present
the later T=0.95276031 result as improvement at unchanged width.

**Confirmed configuration trap:** setting `wgp_v3_band_um` alone does not make
restart/final selection use that band. `_best_from_log` still uses only
`RHO_DN/RHO_UP*fwhm0_um`. `strict_band_select_probe.py` configures inner QP band
[18.83858,19.021] for a seed hard upper 19.121, but a committed W=19.300 row is
selected. A stricter option needs one shared hard observable band used by QP
insetting, measured acceptance, restart/final selection and validation. Merely
changing the QP tuple is incomplete. No band configuration was changed here.

## Readiness policies and scientific limitations

1. **Missing gLambda does not retain a wavelength bound.** In `_step_of`,
   `cw_state=no_glam` computes a halved allowance, then omits the wavelength row
   entirely. The message says the bound is halved although there is no row to
   enforce it. The driver also allows normal ascent and radius growth. The
   `missing_glambda` reproduction gives one-row ascent, fallback=true and
   cap 10→15 nm on the next accepted point. This is a demonstrated policy gap;
   it does not prove a physical wavelength failure in a particular device.
   Production should refresh/recenter or fail with preserved state before
   making an unrestricted step when this essential derivative is missing.

2. **T_out ascent is distinct from modal transmission ascent.** Peak3 and its
   port adjoint consistently consume abs(T_out). Callback width, resonance and
   reported T use modal abs(S)^2. The October 6 slope-denominator correction
   removes a real normalization mismatch, but gLambda remains a translating
   lineshape approximation, not the exact derivative of the modal peak.
   Available p0/p29 recovery FD pairs improve both peak metrics in the same
   direction; this limited evidence does not prove monotonic modal T for all
   campaign directions. Before a full campaign, require a modal-T acceptance
   audit or derive/calibrate a modal objective consistently. Do not multiply
   the objective by a universal normalization correction.

3. **Width derivatives are a surrogate.** Fixed-wavelength softW plus measured
   `cW*gLambda` steers production envelope FWHM. Its true trial measurement is
   still required. S2 field calibration reports a near-zero shift component
   error 23.8% despite vector error 1.25%; large relative error in a very small
   component is not automatically a fatal vector error, nor a per-component
   PASS. Inspect step-level moving-width predictions on N98 physics before
   declaring full readiness. N70 pipeline smoke alone cannot certify N98.

4. **Projected evaluation budget is per restart.** `run_projected` uses
   `max_iter`; it does not consult `max_feval`. `run_campaign` allows 13 attempts
   via MAX_RESTARTS=12. A nominal max_iter60 can therefore permit up to780
   forward evaluations, with adjoints at accepted points, instead of a global
   120-evaluation cap. This is an operational budget gap, not a numerical QP
   failure. The bounded smoke correctly disables automatic restart retries.

5. **Stops do not establish an optimum.** Three small-scale noise rejections
   produce a noise-limited stop; duplicate collapse is explicitly unresolved.
   Neither proves constrained stationarity or global optimality. QP status,
   LP feasibility, measured width/wavelength agreement, surrogate residuals,
   modal/T_out gains and gradient cancellation should remain visible in the
   final verdict. Restoration uses a transmission-ratio radius policy; a
   width-violation reduction ratio would provide a better restoration signal.

6. **Other useful criteria are diagnostics, not silently new objectives.**
   Record reflection, residual1−T−R, spectral linewidth/Q, peak resolution and
   mode identity alongside transmission and spatial width. Q_i derived from
   Q_L/(1−sqrt(T)) depends on coupling assumptions. A larger transmission
   alone does not independently establish radiation-Q improvement. Domain,
   PML and conformal confirmation remain separate physical checks.

## What appears structurally sound

The v3 bounded QP handles physical parameter bounds and one infinity-norm
radius together; it avoids the old projection-then-clipping loss of tangency.
Driver restoration explicitly retains the available wavelength row rather
than delivering the raw dropped-row solution. Peak3's autograd weights match
its three-sample functional; symmetry/linewidth assumptions and peak switching
still limit its global smoothness. Width-row reuse is disabled in current v3
specs, so old Broyden/reuse incidents do not presently justify extra adjoints.
The candidate's immutable profile IDs, interrupted-tail preservation,
fail-closed state loading and committed acceptance IDs address real recovery
hazards, subject to the additional row/reference findings above.

No alternative optimizer or metric weighting was introduced. A paired study
of better conditioning, stricter width upper, or a consistently modal objective
should follow calibrated N98 local evidence, rather than being justified by a
large amount of unchecked overnight optimization.

## Implementation follow-up in the isolated candidate

Root subsequently delegated exclusive edits of the candidate engine and v3
gate to this audit agent. The production engine and server were untouched.
Before editing, the candidate was preserved as
`engine_before_missing_width_glambda_20261007.py` beside the candidate root.

Implemented bounded missing-width recovery: three consecutive failed/nonfinite
width reads raise an explicit unresolved-measurement error. A retry that would
repeat any rejected geometry or collapse onto the accepted point stops before
another forward. The accepted state and failure journal remain available;
transient failures can still recover. The missing-width radius shrinks the
radius actually used, including any temporary noise shrink factor.

V3 now refuses a missing/nonfinite gLambda or missing/nonfinite resonance
before acceptance, radius updates or a geometry proposal. The journal records
`unresolved-gradient`; nonfinite gradient indices are recorded without writing
invalid JSON values. A second guard in the QP caller defends retry paths.
Non-v3 legacy fallback behavior is retained. Missing/curved cW with a valid
gLambda still uses the documented smaller wavelength allowance.

Root's acceptance-reference and newest-row fixes were retained. Existing V18
poison regression remains passing. New V22/V23 tests cover sustained and
transient missing-width errors, cap-floor duplicates, missing gLambda on the
first/later point, and nonfinite gLambda. Final exit codes: v3 0, projection 0,
persistence 0 (12 tests). Evidence:
`<candidate>/missing_glambda_width_gate_results.json` and
`gate_*.py.missing_glambda_width_final.log`.

This is offline candidate validation. Production integration and hardware
validation are owned by root. No strict width option or global evaluation
budget was introduced by this subtask; those are separate bounded changes.

### Shared width contract follow-up

Root then delegated width-contract wiring into the isolated candidate. Added
optional `CampaignSpec.fwhm_bounds_um`; its default None retains the old anchor
band. Root's `width_contract.py` now supplies the QP inset, measured acceptance,
best/restart selection, callback tripwire, FWHM wall and AL hinges/multiplier
updates. Explicit hard bounds apply even without fwhm0. An inconsistent inner
QP band fails before a forward. Hard violation from a previously compliant
accepted point is rejected even below the inner model's1e-6 tolerance.

No S1/S2 configuration activates a stricter contract. Tests demonstrate strict
seed upper rejects +0.030 µm and +1e-7 µm trials while the default accepts both
under the same inner band; the actual selector chooses initial historical S1
row under strict upper and the latest row under default. Additional tests cover
explicit bounds without anchor, callback tripwire, AL and FWHM wall. Exit0:
v3, projection, persistence (14 tests), width helper (4 tests), TM identity
(7 specs+module constants). Logs are `*.width_contract.log` beside the candidate;
summary is `width_contract_integration_gate_results.json`. Global budget wiring
remains a separate task. Production engine is still the frozen recovery version.
