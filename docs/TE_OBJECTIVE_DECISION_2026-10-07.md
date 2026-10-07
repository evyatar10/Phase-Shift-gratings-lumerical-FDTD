# Transmission at nearly constant spatial width: formulation decision

User clarification, October 7: transmission and nearly constant spatial FWHM
are the priorities; wavelength may move. Reassess width-gradient reuse and
combined adjoints/augmented Lagrangians before the next campaign. This document
supersedes recommendations in historical AL notes where they conflict below.
It does not modify running calibration or authorize a full optimization run.

## Recommended physical problem

### Width authority: sigma is a rejected proxy

Rechecked after the user's October 7 clarification: constant second-moment
sigma does not imply constant spatial FWHM. THEORY.md section 4a records
the historical sigma constraint missing substantial FWHM growth; its fitted
linear surrogate compounded the error. Do not revive either as the TE width
constraint, acceptance criterion, length-convergence criterion or success metric.

Read-only imports of both current TE `SPEC_V3` objects confirmed
`sigma_wall=False`, `width_grad=True`, `wg_project=True`, and measured FWHM
anchors (S1 19.121 um, S2 19.636 um). Neither entry point passes `sigma0_um`.
The engine reads `fwhm_env_um` into W for the projected/QP decisions and
filters eligible final/restart designs using the measured width band.
The differentiable soft-level-set width supplies a gradient direction;
it is not sigma and does not replace measured envelope FWHM. Its physical
gradient still requires the outstanding validation gates.

The saved 100/150/200 ps spatial report contains sigma only as a supplemental
profile diagnostic. Its reported Tout-anchored FWHMs are respectively
19.6441074142, 19.6437170638 and 19.6442773610 um; the separate sigma values
are 12.4373100359, 12.4372615345 and 12.4373150458 um. The reported near-19.64
um widths are envelope half-height crossing distances, not sigma conversions.
Future summaries must label sigma diagnostic-only and never use its stability
to certify FWHM stability. Original evidence and frozen sources are preserved.
The engine's opening docstring describes the historical TM/sigma formulation;
use the live TE branches and THEORY section 13 for the current method.

### Physical objective

Let p denote the retained TE N98/free60/bare geometry. Track the same defect
resonance branch, rather than taking the largest peak anywhere in the scan.
Let lambda*(p) be that branch's measured resonance and W(p) its production
y-integrated spatial-envelope FWHM at resonance. The physical goal is

    maximize M(p) = peak modal transmission |S21|^2 on the tracked branch
    subject to W_lo <= W(p) <= W_hi and the existing fabrication/geometry bounds.

Use the retained bands: S1 [18.73858,19.50342] um, S2
[19.24328,20.02872] um. These mean nearly constant width, not exact equality.
An optimizer may use the allowed widening. Do not call that strictly no growth.
The tested optional hard-width contract can express a seed-width upper bound,
but existing seeds/calibration retain their confirmed bands. No new global
wavelength target or penalty is introduced by this formulation.

Use the resonance peak value, not a spectral average or the old power-mean
softmax that can reward linewidth growth within a fixed window. For a smooth
isolated continuous peak, the envelope theorem removes the resonance-motion
term from the derivative of its peak transmission because dM/dlambda=0 there.
It does not remove the width chain term: dW/dlambda need not be zero. Finite
sampling, peak switching and the port/modal normalization difference still
require the stated tests; the three-point estimator is a local approximation.

Mode identity, valid peak extraction, adequate spectral sampling and measured
width are required evidence. Q, reflection and 1-T-R remain diagnostics;
they are not silently new competing objectives. Exact width equality would
unnecessarily replace the user's tolerance band and is not the default.

## Match the search quantity to the reported success criterion

The current adjoint differentiates J=peak3(abs(T_out)); plots often report
sampled |S21|^2. These are different observables. The actual callback now also
reports modal_peak3, allowing consistent peak-estimator comparisons. Official
[lumopt2 documentation](https://lumerical.docs.pyansys.com/version/stable/user_guide/lumopt2/figure_of_merit.html)
defines PortResults transmission as T_out. Saved spectra independently confirm
the discrepancy and rule out a universal offset or scale conversion.

For the first bounded pilot, retain the calibrated J derivative as a search
direction, while enforcing measured modal non-regression on feasible ascent
steps and ranking eligible final designs by modal_peak3. This is explicitly a
proxy-gradient method with a measured performance guard, not a claim that the
J adjoint is the exact gradient of M. A guard adds no FDTD solve: the forward
already contains the modal spectrum.

Use the best committed, width-feasible modal value as the reference, rather
than the last accepted value; repeated small losses must not accumulate.
Missing/nonfinite modal data fail closed. Start with zero allowed modal drop
for this opt-in guard; the historical FOM slack is not automatically a modal
noise standard deviation. Feasibility restoration from an out-of-band point
may lose transmission to recover width, but such a point cannot raise the
feasible reference or be advertised as an improved design. Historical rows
without committed IDs may supply a starting geometry; remeasure them before
using them as the guarded pilot's certified reference.

If J-directed steps repeatedly oppose M, stop using J as a useful proxy and
derive/validate a consistent modal-S functional. Source-port normalization and
its geometry dependence must be included; simply dividing by T_in is not an
exact fix on the stored spectra. Do not use a correlation coefficient over a
few steps as a substitute for directional finite differences or acceptance
evidence. Report changes smaller than established repeatability as unresolved.

The opt-in candidate implementation passed all thirteen offline suites,
including ten focused modal-guard driver regressions. State, hashes and
integration instructions are in `TE_MODAL_GUARD_CANDIDATE_2026-10-07.md`.
The bounded pilot must explicitly set `wgp_modal_guard=True` and
`best_metric="modal_peak3"`. Defaults remain off. Production integration completed in commit1b2d75e;
hardware validation remains blocked by failed S2 calibration. Offline tests do not establish modal-gradient
accuracy or successful physical optimization.

## Wavelength: allow drift, retain local model validity

Current v3 already uses a per-step wavelength trust row, not a global lock:

    |g_lambda^T d| <= delta_lambda,k.

Current normal values are 0.25 nm for S1 and 0.05 nm for S2, with stricter
fallbacks when the width-versus-wavelength estimate is unreliable. Cumulative
drift is permitted, and the scan follows accepted points. Keep acceptance and
mode checks before recentering. Log cumulative drift; do not penalize it just
because it differs from the seed wavelength. A future relaxation must be
earned by measured prediction accuracy, not by deleting all step protection.

The width derivative still needs the moving-resonance correction

    g_W,res approximately equals g_W,fixed_lambda + c_W * g_lambda.

Wavelength freedom makes this chain term more relevant. g_lambda is presently
a translating-lineshape approximation. Its selector passes reuse the port
adjoint fields and do not add FDTD solves. Removing this row would therefore
not save the alleged third adjoint; no such extra solve exists here.

## Most efficient first method: the existing constrained QP

Keep the tested bounded QP/filter with separate objective and width derivatives
for the first validated pilot. It directly expresses a width band, handles
geometry bounds and checks actual measured width after each trial. Linearized
feasibility is not a guarantee of nonlinear feasibility; measured acceptance
and restoration remain essential. Rejected trials already avoid new adjoints
and retry from stored accepted-point gradients.

Source inspection establishes one shared default forward, one broadband port
adjoint, and one tiled single-frequency width adjoint. There is no separate
wg_single forward configuration in the live code. Four spatial source tiles
are injected in one width-adjoint run. Thus fresh cost is

    C_fresh = t_forward + t_port_adj + t_width_adj + t_setup/extract/IO.

Selector assembly has CPU cost but no extra solve. Measure actual timings and
feasible modal gain per total solver-hour, including rejections and restart
overhead. Three files do not imply three equal-duration jobs or a guaranteed
33% saving from deleting one.

Independent adjoints could also be scheduled concurrently after the shared
forward. This can reduce elapsed time toward t_forward+max(t_port,t_width),
but does not reduce total solver work and requires additional GPU/license
capacity and a tested scheduler path. The current serialized runner is not
made parallel by calling the adjoints mathematically independent. Treat this
as a separate measured resource option, not a promised overnight speedup.

## Width-gradient reuse: conditional, not a fixed promise to skip loops

The reusable expensive object is the fixed-wavelength width row. Keep modal
transmission, actual width/profile, c_W and the inexpensive port-derived
g_lambda fresh on each evaluated geometry. Reconstruct the moving-width row
from the current quantities. Do not cache the whole corrected row unchanged
while the resonance moves.

Current eligibility includes width drift <=0.025 um since the last fresh row,
bounded cumulative geometry travel, age, and clean state. The 0.025 um number
is a **width-change** gate, not a geometry-travel radius. Current TE toy steps
changed W by 0.08-0.12 um, so this gate did not open and reuse was disabled.
Historical successful TM reuse does not validate TE reuse.

Keep reuse off for the calibration/smoke baseline. Then evaluate one skipped
width solve in a bounded TE comparison, with a fresh width-gradient endpoint
to measure error. Refresh on rejection, mode change, recentering, uncertain
cache identity, excessive twin lag, accumulated travel, or failed width
prediction. Use both absolute and relative secant errors so nearly zero width
changes do not produce meaningless relative tests. Any Broyden update must
account for the actual twin-wavelength motion and the observable used.

Near a width boundary, limit the proposed step conservatively using remaining
width margin and measured model error. An unconstrained fixed schedule of two
or three skipped solves is not justified yet. If validated reuse succeeds on
a fraction q of accepted steps, the gross saving is approximately
q*t_width_adj, reduced by refresh/rejection/assembly overhead. Shorter steps
may improve reuse but require more iterates; compare total time to a feasible
gain rather than savings per iterate alone.

## Combined adjoint and proper augmented Lagrangian

Claude's historical decision to start with separate adjoints is reasonable for
this implementation. It does not establish that combined-source optimization
is impossible or inherently inferior. Historical fixed-penalty failures do
not disprove AL for this constrained problem.

For normalized inequality constraints
g_hi=(W-W_hi)/W0 <=0 and g_lo=(W_lo-W)/W0 <=0, a standard maximization-form
Powell-Hestenes-Rockafellar merit is

    A(p;mu,rho) = J(p) - sum_j ([max(0,mu_j+rho*g_j(p))]^2-mu_j^2)/(2*rho),
    mu_j,new = max(0,mu_j + rho*g_j(p_outer)), rho >0.

Hold multipliers/penalty fixed during an inner solve, then update them on a
defined outer schedule. Increase the penalty when constraint progress stalls,
not automatically at every iteration. This equation is the sign-adjusted
inequality form in the [PHR literature](https://optimization-online.org/wp-content/uploads/2024/09/ALAKKT2_rev2.pdf).
Using a surrogate W for derivatives still requires checking actual FWHM;
updating multipliers on measurements does not magically make a wrong surrogate
an exact AL derivative. Normalize units consistently when choosing rho.

The historical lambda*max(0,g)+rho/2*max(0,g)^2 expression is not this shifted
PHR penalty. When lambda>0 and -lambda/rho<g<0, its gradient is zero while
the PHR gradient is (lambda+rho*g)*grad(g). The historical expression can be
studied as a different nonsmooth penalty, but should not inherit PHR claims.
It is inactive in current projected v3, so this is not an active v3 failure.

For one compatible linear Maxwell operator and frequency, adjoint sources for
a scalar merit can be superposed. The scalar gradient then combines objective
and constraint gradients. It does not identify the separate rows required by
the current QP. Source-derived details are in `TE_OBJECTIVE_CODE_FACTS_2026-10-07.md`:
the installed implementation applies scalar Jacobian weights after separate
solves, and `Project.set_adj` saves a file per adjoint entry. Changing MixedFom's
formula therefore does not implement a one-solve adjoint.

A real combined-source implementation must handle frequency-dependent port
weights, the single-frequency width source, the moving-resonance chain term,
and distinct complex calibration conventions before summing contributions.
Combined fields cannot simply be reused as uncontaminated port selector fields.
Compatible spectral encoding/separation must be demonstrated, or extra basis
solves retained. AL also needs a genuine outer loop, persistent multipliers,
merit-consistent line searches and stopping tests. Its iterates need not all
be feasible; keep the best measured feasible design throughout.

AL can solve a fixed constrained problem; it is **not restricted to trade-off
curve exploration**. Defer it as the first production method because its
source machinery and gradient validation are unfinished here. After the
baseline, compare true combined-source AL and guarded QP at equal total
solver cost, matched physical bands and the same measured modal endpoint.

## Verification and decisions from the independent review

Claude Opus 5.5 supplied the main review through the restricted helper; exact
model metadata and answer are preserved in
`readiness_evidence_20261007/claude_objective_review.json`. A focused Sol audit
checked live code, installed vendor code and the original Claude memory.
The following Claude suggestions were corrected or not adopted:

- No invented extra forward configuration: live entries share config=None.
- No conversion of the 0.025 um width-change gate into a travel radius.
- No added linewidth-ratio constraint, guessed 2-sigma modal tolerance or
  short-history correlation gate. The supplied brief did not establish these.
- The old +2%/-5% penalty docstring is stale; live RHO constants are ±2%.
- AL is not confined to trade-off curves, and historical S1 calibration has
  qualifications; it is not an automatically fully calibrated ablation host.

Offline numerical checks pass: PHR derivative versus central FD has maximum
error1.69e-11; a shared complex linear operator reproduces the combined adjoint
to6.21e-17, while showing the separate rows cannot be uniquely recovered from
their sum. These are mathematical checks, not a Lumerical merged-source test.
Evidence: `readiness_evidence_20261007/objective_math_review.json`.

## Execution order

1. Finish existing S2 peak3 calibration and inspect unchanged scientific gates.
2. Integrate the verified restart/budget candidate after frozen dependencies
   finish; perform the prepared bounded hardware smoke with fresh width rows.
3. Validate the modal guard and N98 moving-width/wavelength predictions in a
   small pilot; report both observables and actual width. Investigate the S2
   field-component qualification before calling its gradient fully validated.
4. Start a full campaign only after these gates and the user's full-run
   authorization. Retain ±2% bands, local wavelength trust and fresh width rows
   initially. Consider conditional reuse only when the measured ablation pays
   for its added complexity; combined-source AL follows its own validation.

Morning readiness is an intended outcome, not permission to turn pending or
failed calibration into a pass. Preserve all results and report blockers.
