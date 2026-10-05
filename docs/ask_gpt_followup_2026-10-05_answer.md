**Verdict — [FROM-CODE + INFERENCE]: The fixes improve the implementation, but the noise-stop rule remains unsafe.** I reproduced three **identical restoration trials** followed by “converged”; a mixed sequence of noise and ordinary rejects can also increase the effective retry radius. Callback-triggered restarts can still bypass the new eligibility filter. The highest-value next change is a bounded, single-radius quadratic subproblem using the **total moving-resonance width derivative**, with wavelength treated as a local search aid. The TE calibration results support continuing experiments, but do not establish gradient-direction accuracy after projection.

References below: **E** = `runners/lumopt2_design/lumopt2_design.py`; **V** = `runners/lumopt2_design/validate_te.py`; **G** = `runners/lumopt2_design/gates/gate_projection_local.py`; **S1** = `runners/lumopt2_design/campaign_te_s1.py`. Review was read-only; numerical reproductions used mocked solves, not FDTD.

**F1. Fix review: several important paths remain open**

| Finding | Evidence and label | Impact | Fix / falsification test |
|---|---|---|---|
| **Restoration can produce identical retries despite `retry_shrink`.** Shrinking the cap does nothing when the restoration is already below it. | **FROM-CODE:** E:2554–2576, E:2919–2935. **INFERENCE / reproduced:** with zero objective gradient, a 0.15-nm restoration and caps 10, 5, 2.5 nm, the driver evaluated the same displacement three times and declared convergence; only one gradient evaluation occurred. | H | Detect duplicate **delivered geometries** before dispatch. Do not count them as independent noise tests. Add this case to G; its current fixture explicitly excludes restoration at G:382. |
| **An ordinary reject after noise rejects can enlarge the effective radius.** | **FROM-CODE:** E:2910–2929. **INFERENCE / reproduced:** base cap 10, retry factor ¼ gives 2.5; an ordinary reject halves the base to 5 and resets the factor to 1, producing 5. | H | Maintain one effective radius. A rejected trial must not increase it. Gate a noise/noise/ordinary-reject sequence. |
| **Restart can precede eligibility classification.** | **FROM-CODE:** callback runs at E:2815; it can raise recenter/width exceptions at E:1860–1868, before hop rejection and blacklisting at E:2898,2917. Recenter selects from the log at E:3358. | H | Log each evaluation initially as **pending**; classify it before triggering restart. Select only validated eligible evaluations. Test a high-FOM mode hop that also triggers recenter. |
| **The blacklist is geometry-wide and persistence-dependent.** A geometry rejected in one continuation context can later be valid; a missing blacklist can admit an unclassified evaluation. | **FROM-CODE:** geometry hashing E:2658; blacklist persistence E:2744,2771–2775; selection E:3458–3463. | M | Store evaluation IDs, acceptance/eligibility state, spectrum/window provenance and branch identity in the evaluation record. Geometry identity is useful for caching, not sufficient for eligibility. |
| **Band filtering fixes the specific within-band ratchet, but not arbitrary acceptance outside it.** | **FROM-CODE:** E:2897–2898 accepts any decrease in excess width violation, regardless of objective loss. | M | Require sufficient violation reduction during restoration, and identify restoration separately. A tiny width improvement should not buy an arbitrarily large transmission loss. |
| **Accepted small retries reset to the larger base cap, potentially causing repeated overshoot.** | **FROM-CODE:** cap growth E:3117–3127; retry reset E:3134. | M | Adapt from the radius actually used; enlarge only when model agreement and boundary activity justify it. |
| **Exact-LSQ calibration fixes the coefficient fitting, but its validation is not a generalization gate.** | **FROM-CODE:** V:176–201 uses `lstsq`; held-out residuals are printed, while the pass/fail criterion uses in-sample residuals. | M | Gate a held-out **direction**, not every cancellation-dominated coordinate percentage. Save returned full-precision coefficients and conditioning diagnostics. |
| **Selection bookkeeping still permits a value/geometry mismatch.** | **FROM-CODE:** parameters come from filtered logs while `_final_fom(result)` can update the reported best value, E:3337–3341. | M | Return one evaluation record containing geometry, objective, width and wavelength together. |

**INFERENCE:** “Three noise rejects at cap, cap/2, cap/4” is defensible only as a **noise-limited stopping heuristic**, not a convergence certificate. Small predicted gains eventually follow from shrinking any step, even with a useful direction remaining.

Before stopping, require: distinct delivered trials; physical width feasibility; healthy peak tracking; gradients tied to the accepted geometry and spectral definition; and a bound-aware model check at a **fixed reference radius**, rather than only at the vanishing retry radius. Refresh lagged/Broyden width information before that check. Re-solving an already fresh deterministic gradient at the identical point adds no information; an independent directional FD check can. Missing diagnostics should interrupt, not advance, the noise-stop sequence.

**F2. One bounded composite subproblem**

**INFERENCE — proposed implementation.** Remove frozen variables. Choose fixed positive parameter scales \(s_i\), let \(S=\operatorname{diag}(s_i)\), and write the physical step as \(d=Sz\). Keep scales separate from physical bounds so changing a fabrication limit does not silently change the optimizer metric.

Define
\[
q=Sg_T,\quad \widehat q=q/\|q\|_\infty,\quad
a_W=Sg_{W,\mathrm{res}},\quad a_\lambda=Sg_\lambda,
\]
with \(\widehat q=0\) if \(q=0\). Let \(v_W=W-W_{\rm tgt}\), \(b_W=\mathrm{marg}/2\), \(v_\lambda=\lambda_*-\lambda_c\), \(b_\lambda=\mathrm{lam\_margin}\). Solve:
\[
\boxed{\min_z\ \frac{1}{2\Delta}z^\top z-\widehat q^\top z}
\]
subject to
\[
-b_W-v_W\le a_W^\top z\le b_W-v_W,\qquad
-b_\lambda-v_\lambda\le a_\lambda^\top z\le b_\lambda-v_\lambda,
\]
\[
\max\!\left((l-p)/s,-\Delta\right)\le z\le
\min\!\left((u-p)/s,\Delta\right).
\]

This is one convex QP with two dense rows and **one total infinity-norm trust radius**. Restoration and objective improvement compete inside the same feasible problem; there is no subsequent clipping. Divide each constraint row **and its limits** by a corresponding physical tolerance for numerical conditioning.

Multiplying \(g_T\) by any positive scalar leaves this step unchanged. Without constraints, \(z=\Delta\widehat q\), so the step reaches the radius. Constraints can legitimately shorten it. Demanding \(\|z\|_\infty=\Delta\) in every case would be wrong: the feasible set may contain no such point, or the current point may already maximize the local model. If boundary-seeking linear ascent is essential, use the corresponding LP, accepting its tendency toward corners.

**FROM-SOURCE:** I would use OSQP with polishing and explicit physical-unit residual checks. It supports precisely this convex QP structure and infeasibility detection. `lsq_linear` alone handles variable bounds, not these coupled inequality rows. [OSQP formulation](https://osqp.org/docs/solver/index.html), [SciPy `lsq_linear`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html).

**INFERENCE:** If infeasible:
- First relax/recenter the wavelength aid; it is not a physical spec.
- If width remains unreachable within the radius, solve a feasibility-only version with nonnegative width slack, minimizing squared normalized violation. Label the step restoration, and require predicted measurable violation reduction.
- If the current point satisfies all bands and bounds, \(z=0\) is feasible: an infeasibility report then indicates inconsistent inputs or numerical tolerances.
- Do not shrink merely because restoration is unreachable; shrinking can only worsen reachability.

Adapt the **effective** \(\Delta\): halve after meaningful model failure; enlarge, for example by 1.5, only after an accepted near-boundary step with good actual/predicted agreement. Use objective agreement for feasible ascent and violation agreement for restoration. Disable ratio decisions when predicted change is below measurement/model uncertainty. A hard 2-nm floor can remain an operational limit, but reaching it means “resolution-limited,” not stationary.

**F3. Peak objective and its derivative**

**INFERENCE — algebra verified numerically:** Your formula is correct for three samples equally spaced in an abscissa \(x\). Put
\[
D=T_+-T_-,\quad B=T_+-2T_0+T_-<0,\quad
r=-D/(2B),\quad x_*=x_0+hr.
\]
Then
\[
T_*=T_0-\frac{D^2}{8B},\qquad
\nabla T_*=
\frac{r(r-1)}2g_-+(1-r^2)g_0+\frac{r(r+1)}2g_+.
\]
Here \(g_i=\nabla_pT(x_i)\). The vertex derivative need not be differentiated separately: the interpolating parabola has zero slope there. My algebraic FD check agreed to approximately \(6.3\times10^{-10}\).

For the frequency-uniform recording grid, use **frequency**, then \(\lambda_*=c/f_*\). The same interpolation provides
\[
\nabla r=-\frac{\nabla D}{2B}+\frac{D\nabla B}{2B^2},
\qquad \nabla\lambda_*=-\frac{c\,h}{f_*^2}\nabla r,
\]
with consistent units. Do not apply the equal-spacing wavelength formula to a frequency-uniform grid.

**FROM-CODE:** Existing selector passes reweight already-solved port fields, E:2390–2425. Replace the objective weighting with the weights above; use another weighted assembly for \(\nabla\lambda_*\). This requires no additional electromagnetic adjoint solve, although it changes field-processing work. The present two selectors alone do not supply three individual sample gradients unless you change their weighting/assembly.

**INFERENCE:** Practical qualifications:
- At a valid sampled local maximum, \(|r|\le½\), and the peak-value weights remain bounded. Peak **position** is much more sensitive to weak curvature than peak value.
- Three-point curvature uses a small difference of nearby noisy values. Start with a **five-point least-squares parabola**, compare against three and seven points, and reject unresolved/nonconcave fits. Avoid a fit span broad enough that Lorentzian non-parabolicity dominates.
- For a fixed fit matrix \(K\), coefficients \((a_0,a_1,a_2)=KT\), and vertex coordinate \(u_*\), the spectral gradient weights are \(K^\top(1,u_*,u_*^2)^\top\).
- Changing the selected triplet can introduce value/gradient discontinuities for a general spectrum. A fixed-grid smooth interpolant with a tracked isolated maximum is cleaner if these switches matter.
- This removes the power-mean’s direct reward for more high-transmission samples. It does **not** eliminate finite-resolution/interpolation bias versus linewidth.
- First test synthetic Lorentzians and asymmetric peaks over width, grid offset and measured noise: compare value error, derivative error and stencil-switch jumps. Cost: CPU only.

**F4. Exact width restoration with the wavelength cross-term**

**INFERENCE — mathematical identity:** For the differentiated width observable \(w(p,\lambda)\),
\[
W_{\rm res}(p)=w(p,\lambda_*(p)),\qquad
g_{W,\rm res}=g_{w|\lambda}+c_Wg_\lambda,\quad
c_W=\left.\frac{\partial w}{\partial\lambda}\right|_p .
\]
Use this **total row directly** in F2. If \(w\) is softW, this is the total derivative of softW; it is not automatically the derivative of authoritative `fwhm_env`. A locally checked surrogate-to-authority slope can improve the model, but does not make the observables identical.

For equality-style restoration, define deadband residuals
\[
h_W=v_W-\operatorname{clip}(v_W,-b_W,b_W),\quad
h_\lambda=v_\lambda-\operatorname{clip}(v_\lambda,-b_\lambda,b_\lambda).
\]
Then the correct equations are
\[
(g_{w|\lambda}+c_Wg_\lambda)^\top d=-h_W,\qquad
g_\lambda^\top d=-h_\lambda.
\]
Equivalently, using the raw rows \(A=[g_{w|\lambda},g_\lambda]\),
\[
A^\top d=-\begin{bmatrix}h_W-c_Wh_\lambda\\h_\lambda\end{bmatrix},
\qquad
d=-DA(A^\top DA)^\dagger
\begin{bmatrix}h_W-c_Wh_\lambda\\h_\lambda\end{bmatrix},
\]
provided the equations are compatible. Bounds/radius belong in the QP instead. With different restoration gains, the first transformed residual becomes \(\beta_Wh_W-c_W\beta_\lambda h_\lambda\).

**FROM-CODE:** E:2510–2512 correctly explains cancellation for tangent directions, but E:2555 uses the untransformed residual for restoration. Cancellation does not hold when restoration moves wavelength.

**INFERENCE:** Estimate \(c_W\) from three same-forward profiles around the peak; use five to check curvature and sensitivity to sample spacing. Apply exactly the same softW definition and spatial processing at each wavelength, fitting against their actual wavelength coordinates. Increase spacing if differences are noise-dominated.

The single-wavelength twin must match the derivative’s evaluation wavelength. A chain term does not repair a fixed-wavelength gradient evaluated at the **previous** resonance. Either construct and validate the width-adjoint path from the current broadband spectral data, or repin and rerun when the mismatch exceeds a measured tolerance. Interpolating broadband data is a useful approximation, not exact recovery of an unrecorded continuous-frequency field. Log twin wavelength, peak wavelength and the resulting width discrepancy separately.

**F5. Wavelength is an aid, not a spec**

**INFERENCE — recommendation:** Optimize peak transmission subject only to the physical moving-resonance width band and geometry bounds. During each local step, impose
\[
|g_\lambda^\top d|\le\delta_\lambda
\]
as an auxiliary wavelength trust bound, centered on the **current accepted resonance**. Include recording-window coverage and branch-tracking checks in trial validation.

Inside this bound, wavelength-changing directions remain available. Recenter the local band after acceptance; do not restore to the original seed wavelength. Always use \(g_{W,\rm res}\), including when the wavelength row is inactive.

Choose and adapt \(\delta_\lambda\) according to peak-tracking/model reliability and linewidth, rather than treating it as a fabrication requirement. Before declaring physical stationarity, check whether relaxing this aid exposes useful feasible ascent. Otherwise the stopping point may simply be optimal under an unnecessary algorithmic restriction.

**F6. Smallest credible experiment matrix**

**INFERENCE:** No fixed matrix guarantees referee acceptance. This is the smallest useful **paired study** I would propose; it supports claims about these two devices, not broad superiority across photonic design.

| Configuration | Purpose |
|---|---|
| A. Scaled SQP/BFGS, fresh derivatives | Established constrained-optimization baseline; same peak objective, total width row, bounds and wavelength aid. |
| B. Normalized bounded QP, fresh derivatives | Tests the proposed step formulation against A. |
| C. B + width-gradient reuse | Isolates reuse. |
| D. C + Broyden | Isolates Broyden; candidate production method. |
| E. D + fixed-wavelength equality | Isolates the cost/benefit of allowing wavelength motion. |
| F. D with conventional radius reduction/stopping | Isolates the noise-aware policy; retain essential safety checks in both. |

Run all six on both seeds with matched per-run **GPU-hour budgets**, not matched iterations. An initial 12 GPU-hours per run costs **144 GPU-hours**, roughly 8–12 fresh-iterate equivalents under the supplied estimate. Reserve another estimated 12–24 GPU-hours for independent final validation and selected directional checks: **156–168 GPU-hours total**, with all costs beyond the supplied iterate estimate unmeasured.

Report best independently validated **feasible** \(t_{\rm pk}\) versus GPU-hours and forward/adjoint counts; width error; wavelength drift; rejection/restoration counts; duplicate trials; gradient freshness; and termination reason. Rank each seed’s methods within the same problem—different seed width targets are not interchangeable benchmarks.

Extend A and D if neither has plateaued; otherwise the result concerns finite-budget performance only. Add an augmented-Lagrangian baseline if claiming an advantage over scalar penalty methods. Add a power-mean objective ablation if claiming measured improvement from the new peak objective. Synthetic noise/solver tests are cheap supporting evidence, not substitutes for physical runs.

**F7. Replacement wording for the THEORY argument**

**FROM-CODE:** The disputed argument is in `THEORY.md:177–215,237–267`. **INFERENCE — proposed replacement text:**

> We seek to maximize resonant transmission while keeping the measured spatial mode width within a prescribed band. Wavelength control is an algorithmic aid for reliable resonance tracking, rather than an independent physical specification.
>
> This constrained problem can be solved using scalar merit functions, penalties, augmented Lagrangians, or explicit constrained optimization. Our unsuccessful scalar formulations do not establish that scalar objectives are incapable of solving it. They demonstrate difficulties with the selected width surrogate, relative scaling, penalty tuning, and local derivative accuracy.
>
> We retain objective and constraint gradients separately so that their scaling, feasibility predictions, and numerical reliability can be inspected independently. The proposed bounded subproblem uses these derivatives to combine transmission improvement with width restoration. Its first-order predictions apply to the actual delivered step and require a consistent moving-resonance width derivative. Measured trial evaluations determine acceptance because surrogate mismatch, finite discretization, and nonlinear effects remain.
>
> The contribution should therefore be evaluated through feasibility, transmission achieved per computational budget, and robustness against established constrained-optimization baselines—not through a claimed impossibility of scalar optimization.

**F8. What the TE results change—and the cheapest decisive direction test**

**FROM-CODE:** S1:59–62 records port in-sample residuals within approximately 5%, held-out `shift_30` error −16.5%, cancellation up to 1300, and field residuals within 0.2%. These are substantially more encouraging than an unresolved parameter-class failure.

**INFERENCE:** They support using the calibrated gradients experimentally and deprioritizing an immediate TE-specific shape-adjoint rewrite. They do not establish a globally valid complex correction. A held-out failure on a cancellation-dominated coordinate can arise from coefficient uncertainty/leverage, not necessarily different boundary physics. More importantly, small coordinate errors can become large **relative errors in the surviving projected direction**.

The cheapest decisive test is:

1. At an accepted point, construct the actual bounded production direction \(d\), including the total width row. Use already available real/imaginary gradient vectors to assess how leave-one-out calibration coefficients change its predicted gain.
2. If both perturbations respect bounds, run \(p+hd\) and \(p-hd\). Compare the measured directional derivatives of **peak transmission, moving-resonance width and peak wavelength** with their three predictions.
3. Choose \(h\) so the objective difference exceeds measured uncertainty while remaining local. If linearity is unestablished, repeat at \(h/2\): four forwards total. At active bounds, use feasible one-sided steps at \(h\) and \(h/2\), reusing the existing baseline.
4. Test a direction representative of the actual constrained ascent, not a conveniently large coordinate derivative. If predicted positive transmission gain becomes statistically resolved negative gain, the production direction fails regardless of the coordinate-fit residuals.

Cost is two additional forwards for the first check, four for a central two-scale check; GPU-hours are unmeasured. This test simultaneously checks the corrected port direction, the wavelength chain, and whether softW predicts the authoritative width along the step that actually matters.
