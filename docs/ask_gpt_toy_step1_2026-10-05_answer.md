**Verdict — [FROM-DATA + INFERENCE]: DATA B is convincing evidence of one useful, locally well-predicted S1 ascent step. It does not yet validate constrained steering: neither row was active, and wavelength moved only one spectral bin.** Continue the toy, permitting one increase to 15 nm if the new model remains safely feasible. Claude’s S2 FD-nonlinearity hypothesis is plausible but unproved: **a narrow resonance does not by itself make normalized spatial width nonlinear**, and the ill-conditioned complex fit should not be deployed. The remaining highest-value checks are a genuinely constraint-active step, an FD step-size convergence test, and reconciliation of FOM with logged transmission.

References: **E** = `runners/lumopt2_design/lumopt2_design.py`; **Q** = `runners/lumopt2_design/v3_step.py`; **V** = `runners/lumopt2_design/validate_te.py`. Hardware conclusions below use the supplied DATA A–C, not independently inspected cluster results.

**H1. What the successful step establishes**

| Observation | What it establishes | What remains untested; next decisive observation |
|---|---|---|
| \(\Delta F/\widehat{\Delta F}=1.012\) | **FROM-DATA:** Correct sign and excellent finite-step prediction along this production direction. **INFERENCE:** Strong evidence against a gross port-gradient scaling/sign error here. | Gradient accuracy in other directions and operating points. Test the next direction after a constraint becomes active; projection can expose errors hidden in unconstrained ascent. |
| Authority-width ratio \(0.89\), softW ratio \(0.96\) | **FROM-DATA:** Both satisfy the proposed first-step screening ranges. **INFERENCE:** Delta anchoring predicts this local width change adequately. | Accuracy near the width boundary. Require a row-active step whose measured endpoint stays feasible and whose width-model error fits inside the safety buffer. |
| Predicted wavelength \(+0.0158\), measured \(+0.0200\) nm | **FROM-DATA:** Compatible with the prediction at approximately 0.02-nm sampling. | A precise wavelength-sensitivity test. Estimate the sub-grid peak from saved spectra; otherwise this difference is too quantized to establish a ratio near one. |
| Both multipliers zero | **FROM-DATA:** This was unconstrained ascent within the local feasible set. | QP constraint steering, restoration, conflicting rows and rejection handling. A constrained step is needed; DATA A exercised the old build, not the corrected restoration path. |
| \(t_{\rm pk}\) rose by \(0.007043\), width by \(0.077780\) µm | **FROM-DATA:** A useful gain inside the prescribed width band. | Improvement at exactly unchanged width. This step spends width allowance; it is not evidence of width-neutral improvement. |
| First production step worked despite curved `c_W` | **INFERENCE:** The degraded model was adequate for this small wavelength move. | Accuracy near the allowed 0.125-nm move. Compare same-forward narrow-stencil slopes and validate a larger resolved wavelength displacement. |

**INFERENCE:** The chain contribution was only
\[
0.29574(0.0158)=0.00467\ \mu{\rm m},
\]
about 5.4% of the predicted \(0.0873\)-µm width change. Consequently, this step weakly tests the chain correction itself. The width prediction could look good even with a materially wrong `c_W`.

**FROM-CODE:** Curved `c_W` is still included in the total row; the fix adds a degraded status and halves the wavelength allowance. It does not discard or repair the slope. [E:2912–2925.]

**H2. Curvature flag and estimator**

**INFERENCE:** Halving the wavelength allowance is reasonable temporary conservatism, but it is not an error estimate. The right next action costs **zero new forward solves**:

1. Extract softW at several wavelengths from the same saved forward.
2. Compare centered slopes using half-spans approximately \(0.25\Gamma,\ 0.125\Gamma,\ 0.0625\Gamma\), where \(\Gamma\) is spectral FWHM, subject to grid resolution.
3. Fit a local quadratic using five nearby samples where available; report the derivative at the estimated resonance, fit residuals and variation with span.
4. Retain the narrowest sufficiently resolved, stable estimate. If none is stable, constrain wavelength motion using a conservative derivative-error bound and mark the prediction unresolved.

**FROM-CODE + INFERENCE:** The present three-sample calculation uses the same-forward softW observable, which is appropriate. Its curvature flag measures slope variation over the stencil, not directly the error in the central derivative. Symmetric quadratic curvature can trigger the flag even when the central slope is estimated correctly. [E:1847–1867; Q:`cw_from_widths`.]

Use actual wavelength coordinates; a nonuniform-grid polynomial fit is valid. Fitting in frequency is also valid if the derivative is converted by \(dW/d\lambda=(dW/df)(df/d\lambda)\). Frequency fitting is not intrinsically a cure. **Keep softW for the adjoint chain**; calculate `fwhm_env` slopes alongside it as a surrogate check, not as an interchangeable replacement.

**INFERENCE:** \(0.25\)–\(0.30\ \mu{\rm m/nm}\) is plausible: it corresponds to only approximately 1.3–1.6% of a 19-µm width over one 1-nm linewidth. Plausibility does not validate its local value. At the degraded 0.125-nm allowance, the nominal chain contribution is approximately \(0.031\)–\(0.037\) µm—large enough to matter.

**FROM-CODE + INFERENCE:** Crucially, softW is invariant under uniform positive scaling \(I(x)\mapsto aI(x)\): its floor, peak, normalization and threshold scale together. [E:2076–2091.] An ideal separable single-mode field \(E(x,\lambda)=A(\lambda)e(x)\) therefore has wavelength-independent softW despite its resonant intensity. Nonzero `c_W` requires spatial-profile variation, background interference or another departure from this ideal. Inspect normalized profiles versus wavelength to identify which is occurring.

**H3. Radius policy for the rest of the toy**

**INFERENCE:** I would allow **10→15 nm once**, provided the newly computed candidate predicts both width feasibility with useful clearance and valid wavelength tracking. Then hold at **at most 15 nm for the remaining toy**, rather than automatically escalating 15→22.5→30 after each good transmission ratio.

The first gain is strongly resolved relative to the supplied \(10^{-5}\) scale. A simple linear extrapolation to 15 nm would predict approximately \(+0.0104\) FOM and \(+0.131\) µm width; from eval 1 that would reach approximately 19.329 µm, leaving approximately 0.075 µm below the internal upper edge. These are **illustrative extrapolations**, not substitutes for the new gradients/QP.

The most informative next event is the width row becoming active. Require its endpoint prediction to work before further growth. A reject should reduce the effective radius; a width-model failure should also trigger a fresh width check rather than only a smaller repeat of a stale model.

**FROM-CODE:** Because the current step already reaches the cap, the 1.5× probe is not the sole growth decision: the driver uses its extra-gain result mainly when `active_cap==0`. [E:3372–3380.]

**INFERENCE:** Before a long campaign, make growth depend on **all relevant model errors**, not only transmission:
\[
e_W=\Delta W_{\rm measured}-g_{W,\rm res}^{T}d,\qquad
e_\lambda=\Delta\lambda_{\rm measured}-g_\lambda^{T}d.
\]
Use observed width-error magnitude to reserve endpoint clearance; adapt the wavelength allowance separately. Reuse gradients when those checks justify it. This preserves inexpensive progress without paying for repeated overambitious rejected forwards. No extra electromagnetic solve is needed for the radius probe or spectral-slope refinement.

**H4. S2 calibration puzzle**

**INFERENCE:** The present evidence does not distinguish a wrong adjoint from a finite-step FD error. The large, unstable fitted coefficients are evidence that these three measurements do not identify a transferable complex correction—not evidence that the underlying adjoint is necessarily wrong.

For a smooth scalar \(J\), central FD satisfies
\[
F_h=J'(0)+\frac{h^2}{6}J'''(0)+O(h^4).
\]
If the 4-nm error is already dominated by that term, reducing to 1 nm reduces its **absolute derivative error by sixteen**, not by four.

| Hypothesis | Quantitative prediction for the ±1-nm rerun | Discriminating follow-up |
|---|---|---|
| Raw real adjoint is correct; dominant \(h^2\) FD error | **INFERENCE:** Predicted \(F_1=[8.2598{\times}10^{-4},\,1.23858{\times}10^{-3},\,1.11024{\times}10^{-2}]\). Thus Re/FD ≈ **[1.019, 1.013, 1.005]**. | Compare with ±0.5 nm on the worst parameter; verify convergence toward the adjoint. |
| S1-corrected adjoint transfers; dominant \(h^2\) error | **INFERENCE:** Using \(A=0.96672\,\mathrm{Re}+0.03656\,\mathrm{Im}\), predicted corrected-A/FD ≈ **[1.018, 1.012, 1.003]**; **raw** Re/FD ≈ **[1.047, 1.042, 1.034]**. | State explicitly which ratio is being tested. Raw ratios need not reach one if the calibration transfers. |
| Genuine class-dependent adjoint error, FD already local | **INFERENCE:** Ratios remain approximately **[1.42, 1.27, 1.08]**, within FD uncertainty, as \(h\) shrinks. | Stable FD limit disagrees with adjoint; then test another operating point and discretization sensitivity. |
| Near-zero-corrugation geometry/mesh effect | **INFERENCE:** Corrugation samples change from **1–9 nm** to **4–6 nm** around the same 5-nm center. Strong improvement isolated to this parameter supports local geometric/discretization nonlinearity. No defensible universal ratio prediction. | Inspect ± leg asymmetry and a smaller-step convergence sequence; compare a more interior corrugation operating point. |
| Off-resonance twin plus spatial-profile dispersion | **INFERENCE:** Shrinking \(h\) can improve FD convergence, but does not remove the center’s detuning. No ratio follows from linewidth alone. | Log twin-to-resonance detuning at the center and both legs; repeat at a resonance-pinned **center**, keeping that wavelength fixed throughout the FD pair. |
| SoftW/profile-processing nonlinearity | **INFERENCE:** Smooth nonlinearity may show \(h^2\) convergence; a spatial maximum/processing switch may show irregular convergence. | Inspect normalized profiles, floor/peak locations and alternative processing **as diagnostics**, while retaining the differentiated observable for the gate. |

These numerical predictions assume the 4-nm FD is in the asymptotic regime. A displacement spanning a substantial fraction of a narrow line may not satisfy that assumption.

**FROM-CODE:** The operating point includes **both `shift_1` and `shift_30` at 5 nm**, plus cavity width +10 nm and the corrugation lift. The unprobed shift also contributes to detuning. [V:86–98.] Measure this point’s linewidth and resonance; the seed’s \(0.203\)-nm linewidth is not automatically its linewidth.

**INFERENCE:** If ±1-nm results agree, call that evidence for finite-step error, not proof of exact adjoint calibration. A cheap confirmation is ±0.5 nm for the worst parameter, reusing the center adjoint. Richardson estimates are
\[
J'\approx(16F_1-F_4)/15,\qquad
J'\approx(4F_{0.5}-F_1)/3.
\]
Agreement between them is stronger evidence than a new two-coefficient fit.

For future gates, initially require each leg’s measured resonance displacement to satisfy
\[
|\lambda_*(p\pm he_j)-\lambda_*(p)|\lesssim0.05\Gamma.
\]
For S2’s quoted linewidth this is approximately **0.010 nm per leg**; 0.1\(\Gamma\) can be used provisionally if an \(h,h/2\) check validates it. Also require enough FD signal above **width-observable** uncertainty and respect geometric/mesh nonlinearities. Linewidth scaling is a useful step-selection heuristic, not a sufficient convergence test.

**FROM-CODE + INFERENCE:** The S2 port gate also merits step convergence: it currently uses ±2 nm. [V:277–283.] Its moving peak/window may reduce pure translation sensitivity, but changing amplitude, linewidth and window membership still matter. Moreover, an FD that reselects a stop-gradient window may differentiate a different function from the adjoint. Compare FD with the center’s selection frozen for the derivative gate, then separately audit the operational objective with reselection.

**H5. Additional warnings**

1. **The “0.95 normalization” explanation is not established by these numbers. — FROM-DATA:**  
   \(F/T\) changes from **0.9497683 to 0.9501363**, while \(T-F\) is nearly constant: **0.045482→0.045500**. Also, \(\Delta F/\Delta T=0.99744\), not 0.95. **INFERENCE:** Do not simply multiply predicted FOM gain by \(1/0.95\). Save the lumopt2 input spectrum, logged modal spectrum and selected peak weights at the same evaluation; evaluate `peak3` on each and reconcile their definitions. **FROM-CODE:** The optimizer reads `PortResults(...,"transmission",...)`, while the callback logs modal \(|S|^2\). [E:1640; E:1758–1764.] Their geometry-dependent relationship remains unverified.

2. **The Broyden correction still trusts a flagged `c_W`. — FROM-CODE:** It subtracts the old `cw` times twin-wavelength change and sets `dlam=0`, without checking `cw_bad` in that block. [E:3220–3229.] **INFERENCE / DATA B:** The correction here would be approximately \(-0.01657\) µm, increasing the twin-width secant from \(0.08107\) to \(0.09764\) µm—about a 20% adjustment using a curvature-flagged slope. Validate a narrow-stencil slope before letting this correction bypass the old wavelength-change guard.

3. **`restore_lam` can stall even when useful restoration is allowed. — FROM-CODE:** Its objective removes the entire wavelength-aligned width component. [E:2937–2950.] **INFERENCE:** If \(g_W=cg_\lambda\), that objective is zero; the minimum-norm solution is zero although the nonzero wavelength allowance permits partial width recovery. Use minimum width-violation slack subject to the hard wavelength/geometry/radius bounds. Add this exact collinearity test.

4. **The growth probe can assess a different feasible problem during restoration. — FROM-CODE:** The 1.5× call uses the original `qp_step`, which can drop its last row; it does not pass through the driver’s `restore_lam` replacement. [E:2961–2963; Q:324–329.] **INFERENCE:** A gain obtained by abandoning wavelength protection must not justify radius growth. Route both probes through identical constraint policy.

5. **The quoted noise floor needs the right interpretation. — FROM-CODE:** The original noise tasks perturb outer teeth rather than repeat identical geometry. [V:267–276.] **INFERENCE:** That measures a small physical response mixed with numerical effects, not independently identified numerical noise. This distinction does not undermine the large DATA B gain; it matters when stopping or rejecting near the claimed floor.
