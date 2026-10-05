**Verdict — [FROM-CODE + INFERENCE]: The QP core is substantially better than projection followed by clipping, but the integration is not yet reliable enough to interpret a four-iterate gain as validated constrained optimization.** The main problems are: ignored `c_W` quality flags; width-model inconsistency from the lagged twin and reuse; a radius-growth argument contradicted by the implemented objective; and retries/stopping that can abandon the QP’s feasibility guarantees. The toy can validate plumbing and expose model errors, but its current markers do not validate those guarantees.

References: **Q** = `runners/lumopt2_design/v3_step.py`; **E** = `runners/lumopt2_design/lumopt2_design.py`; **G** = `runners/lumopt2_design/gates/gate_v3_local.py`; **V** = `runners/lumopt2_design/validate_te.py`; **S1** = `runners/lumopt2_design/campaign_te_s1.py`.

**Verification — [FROM-CODE]:** I reviewed commit `975032a` against `c8e1057`. I reproduced the 216 QP/reference comparisons: worst reported relative objective discrepancy \(5.4\times10^{-11}\), normalized violation \(1.9\times10^{-16}\). Pure-math checks and V1 passed. I inspected V2–V7 but could not rerun them: this session denied temporary-directory access, including the designated scratch directory. No repository changes or electromagnetic solves were made.

| Priority | Finding | Evidence / label | Required change |
|---|---|---|---|
| H | `cw_curved` is recorded but ignored; missing `c_W` silently gives a fixed-wavelength row. | **FROM-CODE:** E:1864–1870, E:2899–2908, E:3329. | Explicit derivative-validity state; shorten the spectral stencil or refresh, rather than silently accepting the approximation. |
| H | Physical width feasibility is not an unconditional acceptance requirement. | **FROM-CODE:** E:3043–3046 accepts sufficient FOM regardless of width violation; callback width guards run only on callback-best FOM, E:1921–1939. | Separate feasible ascent from restoration. Do not accept a physically infeasible ascent merely because transmission passes. |
| H | “Inactive cap means increasing it cannot change the step” is false because \(\tau\) depends on cap. | **FROM-CODE:** Q:201–202, Q:239–258. **INFERENCE, reproduced below.** | Measure sensitivity to radius or separate proximal strength from the box radius. |
| H | Duplicate-retry halving can destroy restoration feasibility; noise stopping does not check feasibility or refresh. | **FROM-CODE:** E:3095–3107, E:3432–3435; G:530–540 explicitly expects convergence outside the internal band. | Use feasibility-preserving retry subproblems; classify unresolved restoration separately. |
| M | Missing-width retries bypass duplicate protection and leave `pred_step` stale. | **FROM-CODE:** E:2961–2991 versus E:3107. | Route all retries through one proposal/diagnostic function. |
| M | Markers count engagement, not correctness; skipped Broyden counts as a Broyden marker. | **FROM-CODE:** V:217–230. | Validate every evaluated displacement, its derivative provenance, feasibility and prediction errors. |

**G1. QP formulation, dual, metric and small gradients**

**FROM-CODE + INFERENCE:** The stated primal and dual signs are correct:
\[
\min_{\ell\le d\le u}\frac{d^\top D^{-1}d}{2\tau}-g_T^\top d,\qquad L\le Ad\le U.
\]
The clipped primal recovery, positive upper-bound multiplier and negative lower-bound multiplier are consistent. The restoration conjugate term \(\|\mu\|^2/4\) correctly represents squared interval distance; setting \(\tau=1/\mathrm{RESTORE\_EPS}\) supplies the stated proximal regularizer. Dropping wavelength before width is appropriate for its status as an aid. [Q:85–161, Q:178–224.]

**FROM-CODE:** Important limits on the guarantee:
- Box/cap compliance follows from primal clipping for finite, valid inputs. Restoration deliberately need not meet the width band; dropped-row mode deliberately need not meet wavelength.
- `ascent()` treats numerical dual failure like infeasibility, even after the LP found the rows reachable. This can unnecessarily drop wavelength or enter restoration. Distinguish `infeasible` from `solver_failed`. [Q:207–221.]
- Restoration lacks a final success check; after the iteration limit, `_dual_solve` returns the newly updated multiplier with the preceding iteration’s `d,G`. Recompute all returned quantities and validate primal residuals in physical units. [Q:159–160, Q:222–235.]
- Guard nonfinite gradients/rows/scales explicitly. Do not rely on the current assertions, which do not cover all these inputs. [Q:193–205.]

**INFERENCE:** A \(D\)-metric objective with a physical-nm infinity cap is mathematically coherent. It is **different from** the scaled-coordinate trust region I proposed, not intrinsically wrong. Keep it if a common maximum wall displacement is intentional. Scaled coordinates instead impose \(|d_i|/\sqrt{D_i}\le\Delta\), allowing different physical movements per block. Document that choice and keep metric scales independent of later bound changes.

**INFERENCE, reproduced:** Scaling \(g_T=(1,0.1)\) to \(10^{-15}(1,0.1)\) produces the same cap-sized unconstrained step. Add a **directional uncertainty guard**: compute the candidate, then require predicted gain to exceed an estimate of objective-change and gradient-direction uncertainty. If unresolved, refresh/check the direction or perform necessary restoration; do not normalize numerical residue into an exploratory full-radius step. A raw gradient-norm threshold alone is units-dependent.

**G2. Total width row, lagged twin and delta anchoring**

**FROM-CODE:** Same-forward softW samples provide the correct *kind* of partial derivative: fixed geometry and the differentiated observable. However, the quarter-linewidth fit is a finite-stencil estimate, not an exact derivative. `cw_curved` is ignored, and missing `cw` falls back without a specific warning. [E:1847–1870, E:2899–2908.] The curvature test itself can flag a nearly zero central slope because of symmetric curvature; that should trigger a narrower-stencil check, not an assumption that the central derivative is wrong. [Q:63–82.]

**INFERENCE:** Let \(\delta\lambda=\lambda_*(p_k)-\lambda_{\rm twin}\), and \(w\) denote softW. The lagged fixed-wavelength row differs from the desired row by
\[
e_g=-\delta\lambda\,\partial_\lambda\nabla_p w(p_k,\lambda_*)+O(\delta\lambda^2).
\]
Thus its **Jacobian error is first order in the wavelength lag**. Its error in predicting the next displacement is bounded approximately by
\[
|e_g^\top d|\le|\delta\lambda|\,\|\partial_\lambda\nabla_p w\|_*\,\|d\|.
\]
With comparable successive small steps, that is second order in step size; with a fixed preceding lag while the next retry shrinks, it is first order in the retry displacement. A numerical bound requires the **unmeasured mixed derivative**—`c_W` alone cannot supply it.

**FROM-CODE + INFERENCE:** The tracker advances after every evaluation, including rejected trials, so the twin need not correspond to the previous **accepted** resonance. Moreover, 0.25 nm limits the model prediction, not necessarily the measured move. [E:1765–1769, E:2907–2908.] Log actual twin detuning and restrict reuse using accumulated wavelength travel as well as geometry travel; the current reuse conditions do not do that. [E:3114–3136.]

**INFERENCE:** Delta anchoring,
\[
W_{\rm model}(p_k+d)=W_{\rm measured}(p_k)+g_{\mathrm{softW,res}}^\top d,
\]
is a legitimate local surrogate model, but anchoring fixes only its intercept—not its slope. It needs measured error control against `fwhm_env`. With gain correction \(\eta\), apply \(\eta\) to the **whole total row**, not only its fixed-wavelength part.

**FROM-CODE + INFERENCE:** Broyden currently uses twin-width differences but gates them using resonance differences. These are different wavelength changes. Correct the secant by subtracting \(c_W\Delta\lambda_{\rm twin}\), or obtain both widths at one common wavelength. Otherwise the update can absorb spectral dependence and the subsequent chain addition can miscount it. [E:2763–2778, E:3142–3151.]

**G3. `peak3`, switching, fallback and slack**

**FROM-CODE + INFERENCE:** Within a fixed valid local-maximum stencil, the formula and autograd weights are correct. One side weight can be negative; that is expected interpolation algebra, not a sign defect. Frequency-uniform indexing is appropriate. [Q:31–60; E:857–873.]

Index switching is not generally continuous. **Reproduced:** for samples \((0,0.9,0.9,0.8)\), choosing the left tied maximum gives \(1.0125\), while choosing the right gives \(0.9125\). This is a deliberately non-parabolic example, not a prediction for S1. Smooth, well-resolved Lorentzians have much smaller errors, as the existing gate demonstrates. Add asymmetric/flat-top switch tests. [Q:43–49; G:32–70.]

**FROM-CODE:** Selection uses a scored resonance finder, not necessarily the tallest spectral peak. Therefore branch-score changes can also switch the objective; MAC is subsequent protection, and its missing-data result does not reject. [E:794–805, E:863–871, E:3020–3025; `sim_helpers.py:231–245`.]

**INFERENCE:** At a true sampled maximum, \(B\ge0\) principally indicates a flat/tied stencil. Returning the sample is numerically reasonable but should mark the peak derivative unresolved. An edge fallback must not silently certify a resonance peak. Require adequate curvature, interior coverage and branch identity before using the derivative.

**FROM-CODE + INFERENCE:** Slack is now directly in transmission units. S1 uses \(5\times10^{-4}\), and radius comparisons become unresolved below \(10^{-3}\) predicted gain. This is much larger than the cited \(10^{-5}\) perturbation scale; distinguish conservative acceptance slack from measured numerical uncertainty. Best-FOM anchoring limits downhill drift but can retain an interpolation/branch artifact as the reference. [S1:63,110; E:3037–3046; Q:248–249.]

**G4. Wavelength drift and recentering**

**INFERENCE:** Cumulative wavelength drift is allowed by the user’s ruling. It can be systematic ascent rather than random walk. A local wavelength bound is sufficient only when spectral coverage and branch identity remain independently enforced.

**FROM-CODE:** The present recenter guard is insufficient as a universal protection: it runs only when the callback FOM reaches a new callback best, and runs before driver eligibility classification. Accepted slightly downhill steps may therefore cross its threshold without recentering, while an ineligible high-FOM trial may trigger a restart first. [E:1921–1932, E:2958–2959, E:3043–3046.]

**INFERENCE:** Check coverage on every evaluation. Require a safe distance from both recording edges for the peak and all derivative stencils; classify the evaluation before selecting a restart geometry. S1’s ±5-nm recording window and 2-nm recenter threshold provide useful nominal room, not protection against arbitrary jumps. [FROM-CODE: S1:98.]

Adapt \(\delta_\lambda\) using
\[
e_\lambda=|\Delta\lambda_{\rm measured}-g_\lambda^\top d|.
\]
Shrink after resolved large error, branch ambiguity or inadequate coverage; enlarge cautiously when the wavelength row binds and predictions remain accurate. Cap it relative to the **current** linewidth, safe recording span and twin-detuning tolerance.

**G5. Radius growth when rows bind**

**INFERENCE, reproduced:** The inactive-cap reasoning is incorrect for this QP. With \(D=I\), \(g=(1,1,0.1)\), rows restricting \(d_1,d_2\in[-0.2,0.2]\), and broad geometry bounds:
\[
R=1:\ d=(0.2,0.2,0.1),\qquad R=2:\ d=(0.2,0.2,0.2).
\]
Both rows bind and `active_cap=0` in both solves. The third component changes because \(\tau\propto R\). [FROM-CODE: Q:201–202, Q:254–255.]

Two binding rows constrain only two combinations of roughly 296 variables; they do not generally determine the entire step. Also, the gate’s vectors are seeded random vectors, not measured device gradients. Its result is a useful fixture, not evidence that actual TE steps are entirely band-limited. [FROM-CODE: G:346–354, G:492–506.]

**INFERENCE:** Either separate proximal strength from radius, or retain the present objective and cheaply solve a hypothetical \(1.5R\) subproblem. Grow when good measured agreement accompanies a meaningful predicted benefit from that larger-radius solution. Adapt wavelength allowance separately when it limits useful motion; never widen the physical width specification to manufacture progress.

**FROM-CODE + INFERENCE:** `radius_update` also accepts negative predicted gain: `pred=meas=-0.01`, active cap, grows 10→15 in my reproduction. Restoration needs a violation-reduction ratio, not this objective-gain rule. Its 2-nm floor can also increase a smaller retry radius; distinguish an operational floor from evidence supporting enlargement. [Q:248–259.]

**G6. Correct retry rule**

**INFERENCE:** Halving a delivered QP step is acceptable as a line-search trial **when the base point satisfies the linear bands**: convexity then preserves feasibility along the segment to the QP solution. It need not remain the QP minimizer. From an infeasible base, however, halving can prevent the trial from reaching the width band. Calling that trial the unchanged QP mode is misleading.

For feasible ascent, use retry factor \(0<\beta<1\):
\[
R_\beta=\beta R_0,\quad \delta_{\lambda,\beta}=\beta\delta_{\lambda,0},
\]
and, optionally, shrink the allowed *increment interval*:
\[
g_W^\top d\in
[\beta(W_{\rm lo}-W_k),\ \beta(W_{\rm hi}-W_k)].
\]
This keeps the predicted endpoint inside the original width band because \(W_k\) is inside it. It does **not** change the physical specification.

For restoration, use a feasibility-only subproblem with an explicit fractional violation-reduction target or slack. Do not blindly multiply an interval excluding zero and then call it hard-band feasibility. Refresh a repeatedly failing restoration model; do not declare optimization converged because transmission changes are small.

**FROM-CODE + INFERENCE:** After every delivered-step modification, recompute all predictions, active constraints and status. Currently duplicate halving updates `pred_step` but leaves `v3_last` describing the unhalved QP proposal, which later supplies radius diagnostics. Missing-width retries update neither `pred_step` nor duplicate history. [E:2978–2991, E:3084–3107, E:3289–3302.]

**G7. Fallbacks and what the markers miss**

| Path | Current behavior and evidence — **FROM-CODE** | Required assertion — **INFERENCE** |
|---|---|---|
| Missing/curved `c_W` | Missing uses raw row; curved value is still used. E:2899–2908,3329. | Every production step has a validated total row or an explicit degraded policy with quantified error. |
| Missing \(g_\lambda\) | Fixed-width-only QP; warning exists, but no wavelength trust row. E:2907–2908,3335–3342. | Do not count this as a fully validated v3 step. |
| Solver trouble / dropped row | Fallback conflates infeasibility and numerical failure. Q:207–224. | Log cause, retained rows, physical residuals, solver convergence and finite outputs. |
| Retry diagnostics | Full v3 diagnostics are logged in the accepted branch; retry prediction provenance is incomplete. E:3106–3107,3390–3396. | One record for every **delivered** candidate, keyed to its accepted base. |
| Legacy controls | `range_alpha` remains a logged marker although the v3 QP does not use it. E:2899–2910,2996–2997. | Report it as inactive under v3; marker presence is not engagement. |
| Stopping | No fresh-gradient or feasibility condition in noise stop. E:3432–3435. | Separate feasible noise-limited progress, failed restoration and numerical stall. |
| Toy assertions | One v3 marker and one measured `cw` suffice; reuse is required, but actual Broyden correction need not occur. V:217–232,323–327. | Check every eligible step; permit a legitimate fresh-only toy rather than forcing unsafe reuse. |

**INFERENCE:** Also align the resonance definitions: the peak objective uses an interpolated value, whereas logged wavelength is a sampled index and \(g_\lambda\) remains the wider matched-stencil estimate. The latter is not generally the derivative of the three-point vertex for a changing asymmetric lineshape. Test these consistently rather than assuming identity. [FROM-CODE: E:804–805,857–873,1788–1808,3213.]

**G8. What would make the four-iterate toy credible?**

**INFERENCE — proposed screening criteria, not measured guarantees:** Four evaluations usually provide at most three tested displacements; the final proposed step is not evidence until evaluated. Compare each trial against its **actual accepted base**, including rejects.

| Observable | Credible initial result | Falsification / unresolved result |
|---|---|---|
| Delivered geometry | Every displacement satisfies its recorded box and effective cap; initial S1 cap is 10 nm. Smaller row-limited steps are legitimate. | Any unexplained cap/bound breach, duplicate solve, or stale prediction record. |
| Width | Every accepted ascent remains physically feasible. For S1 \(W_0=19.121\), physical band is **[18.73858,19.50342] µm**; default internal band is **[18.83858,19.40342] µm**. | An accepted physical violation; repeated total-row errors consuming the 0.1-µm safety buffer. |
| Transmission prediction | On at least two resolved ascent steps, provisionally \(0.5\le\Delta T/(g_T^\top d)\le1.5\); investigate any wider discrepancy. | Statistically resolved negative measured gain for positive predicted gain, especially repeated at half step. |
| Width prediction | For resolved changes, provisionally 0.7–1.3 measured/predicted ratio using the **total** row; otherwise use absolute error against measured width uncertainty and remaining margin. | Wrong-sign resolved width change or persistent slope error; fixed-row agreement alone does not pass. |
| Wavelength prediction | For a predicted 0.25-nm move, approximately 0.20–0.30 nm measured is a reasonable first screen; demand improvement at smaller steps. | Repeated large/sign errors, lost peak identity, or a missing row masquerading as success. |
| Derivative freshness | At least one direction assessed with fresh width information; confirm any apparent reuse success with a fresh comparison before campaigning. | Gains occur only with degraded/unidentified derivative state. |

**FROM-CODE:** Numerical bands above follow E:2796–2805 and S1:100,106,137. Logged peak wavelength remains grid-quantized; S1’s 10-nm/501-point recording implies approximately 0.02-nm spacing, so small wavelength ratios are not meaningful without sub-grid estimation. [E:804–805; S1:98.]

**INFERENCE:** Use independently estimated uncertainty, not `slack`, to decide whether these tests resolve a signal. A four-iterate toy need not produce a particular transmission improvement to pass plumbing; conversely, a large gain with uncontrolled width fails. S2’s smaller transmission headroom makes model validity and efficiency especially important. Before trusting campaign claims, fix the acceptance/diagnostic paths and run the two-forward directional check along an actual v3 displacement.
