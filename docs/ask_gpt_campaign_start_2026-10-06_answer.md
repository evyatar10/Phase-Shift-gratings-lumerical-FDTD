**Verdict — [FROM-CODE + FROM-DATA + INFERENCE]: The toy now provides useful evidence for width-constrained ascent: its row-active step gained transmission with only a 0.0071-µm width-model error, safely inside the physical specification. C6 is defensible. However, the warm start exposes a concrete bookkeeping bug: fresh gradients can be combined with the copied toy’s old diagnostics. Fix that before trusting further campaign decisions. S2’s detuning is a plausible contributor, not an explanation that clears the adjoint: a fixed-wavelength adjoint should match converged FD off resonance too.**

References: **E** = `runners/lumopt2_design/lumopt2_design.py`; **Q** = `runners/lumopt2_design/v3_step.py`; **V** = `runners/lumopt2_design/validate_te.py`; **S1** = `runners/lumopt2_design/campaign_te_s1.py`. I inspected the specified changes through `a4b006b`; hardware findings use the supplied data.

**J1. C1–C9 verification and warm-start behavior**

| Change | Verdict and evidence |
|---|---|
| C1: nested `c_W` slopes | **FROM-CODE:** Implemented: distinct integer-grid spans, narrowest adjacent agreement within 20%, otherwise narrowest flagged slope. [E:1849–1888.] **INFERENCE:** Agreement is a consistency check, not a derivative-error bound; nearby spans can share bias. Log actual sample wavelengths and profile-grid indices. |
| C2: growth-probe policy | **FROM-CODE:** Both current and larger-radius modes must be `ascent`. [E:2976–2989.] **INFERENCE:** Also require solver `status=="ok"`; mode alone does not exclude a numerical fallback. |
| C3: flagged-slope correction | **FROM-CODE:** The twin-wavelength correction now requires `not acc["cw_bad"]`. [E:3250–3263.] **INFERENCE:** If correction is skipped, the remaining Broyden eligibility still uses resonance change rather than actual twin-wavelength change. C7 makes this inactive for S1 v3. |
| C4: restoration | **FROM-CODE:** LP reach, half-reach target, minimum-\(D\)-norm equality solve and scaled-LP fallback are implemented. [Q:363–434.] Qualifications below. |
| C5: saved gradients | **FROM-CODE:** Saves \(p,g_T,g_W,g_{W,\mathrm{eff}},g_\lambda\) for accepted iterations. [E:3449–3456.] **Bug:** names use only local iteration number, so campaign restarts can overwrite earlier vectors. Add a run/attempt or unique evaluation ID and spectral provenance. |
| C6: acceptance tolerance | **FROM-CODE:** Uses excess beyond the inner band minus `marg/2`; rejects increases in that excess. [E:3125–3143.] This is relative non-worsening outside the expanded band, not an unconditional absolute feasibility test. |
| C7: reuse off | **FROM-CODE:** S1 `SPEC_V3` sets `wgp_reuse_k=0`; the v3 toy no longer requires reuse. [S1:135–141; V:358–361.] This is a configuration choice, not a global prohibition on v3 reuse. |
| C8: gate recentering | **FROM-CODE:** Both S2 port and field specifications use `GATE_LAM_NM`; separate centered-run labels are supplied. [V:119–160,302–335.] Recompute both quadratures at this new wavelength—do not combine with old Im data. |
| C9: warm start | **FROM-CODE:** Main uses `SPEC_V3`; log selection uses the physical ±2% width band; cap loads from optstate. [S1:149–158; E:2865–2889,3594–3595,3796–3837.] **FROM-DATA:** DATA B supports the claimed starting geometry. I cannot verify the externally copied sidecar or running job’s effective cap from source alone. |

**Critical warm-start defect — FROM-CODE:** After the callback writes the current evaluation, the driver calls `_row_of_params`, which returns the **first** matching geometry in the log. With copied toy records, that is the historical toy record. [E:3038–3040,3774–3786.]

**INFERENCE:** Campaign iteration 0 can therefore combine the new FOM/gradient/selector state with old width, wavelength, twin wavelength and old-estimator `cw_curved=True`. The tiny width discrepancy is harmless by itself; the mixed provenance is not. Return the callback’s current record directly, or identify it by a unique evaluation ID. Selecting the newest matching record is an interim repair.

**J1(a) — INFERENCE:** At \(W=19.410527\), the model width interval is approximately
\[
-0.571947\le g_{W,\mathrm{res}}^\top d\le-0.007107\quad(\mu{\rm m}).
\]
The first QP must predict a small width decrease. It remains `ascent` if that interval is reachable under wavelength, box and radius constraints. The upper width row will bind if unconstrained ascent wants to increase width; it need not bind if the chosen ascent already decreases width sufficiently. `restore_lam` occurs only if the full problem is unreachable under the implemented fallback policy. Neither mode is inherently wrong.

C6 does not change those QP bounds. Also, the first evaluation bypasses trial acceptance because `acc=None`; C6 is therefore not literally what admits the starting point. [FROM-CODE: E:2845,2943–2968,3140.]

**J1(b) — FROM-DATA + INFERENCE:** The three width prediction errors are approximately **−0.00952, −0.00462, +0.00714 µm**. A 0.05-µm tolerance is over five times their largest magnitude and retains 0.05 µm of protection before the physical limit. It is a reasonable provisional buffer, not a statistically calibrated uncertainty bound.

The resulting three bands are:

| Role | Width interval, µm |
|---|---|
| QP target | [18.83858, 19.40342] |
| Expanded acceptance band | [18.78858, 19.45342] |
| Physical specification | [18.73858, 19.50342] |

**INFERENCE:** Keep the physical band as an unconditional eligibility check. A restart selected from the physical band can start outside the expanded acceptance band; the current relative rule then permits non-worsening violation. Label that restoration explicitly.

**J2. Half-attainable restoration and the metric**

**INFERENCE:** The fallback is mathematically sound under its intended assumptions: the step box and wavelength band contain zero, the LP succeeds, and the target is within attainable reach. Convexity then makes scaling an LP vertex toward zero feasible. Minimum-\(D\)-norm restoration is preferable to delivering a large LP vertex merely to obtain a small width correction.

It is not a guarantee of half the required restoration: it targets half the **attainable** change, capped by `need`. Nor does it guarantee nonlinear width improvement; measured acceptance remains essential.

**FROM-CODE:** `need` is taken from the nonzero end of `wband`, while the driver supplies the distance to the **far** inner-band edge. [Q:393–409; E:2958–2959.] **INFERENCE:** In the present truly-unreachable restoration case, half the attainable reach is below the nearest-edge requirement, so this normally does not change the target. Nevertheless, pass nearest-edge violation explicitly; the helper’s contract should not depend on that contextual implication.

**INFERENCE:** With \(D_{ii}=s_i^2\), minimizing \(d^\top D^{-1}d\) penalizes fractional bound-range movement. Wide-bound classes are cheaper. Without active bounds, the one-row solution is proportional to \(Dg_W\), so the preference can be substantial. It is coherent but not physically neutral.

**FROM-DATA:** The toy’s 15-nm corrugation versus 0.21-nm average-width movement is evidence of strong parameter anisotropy, not proof that it is harmful. The measured transmission gain shows that direction was useful.

**INFERENCE:** Freeze metric scales independently of future bound edits. Using saved vectors, compare current-\(D\), equal-nm and moderately capped-scale solutions offline. Report predicted transmission gain, total displacement norm, number of moved teeth and active bounds. Spend a forward comparison only if another metric offers a materially different, credible direction.

**FROM-CODE + INFERENCE:** The scaled-LP fallback assigns `status="ok"` and `kkt=0` without establishing minimum-norm KKT conditions. Feasibility is the defensible claim; distinguish “feasible fallback” from “optimal minimum-norm solution.” [Q:417–430.]

**J3. What S2 detuning explains—and does not**

**FROM-DATA:** The gate was approximately **2.03 linewidths** from the operating point’s resonance. The ±1-nm FD changed the corrugation derivative’s sign and reduced the shift derivative sharply. These are not small residual calibration changes.

**INFERENCE:** Detuning can make background interference and spatial-profile derivatives much more consequential. It can contribute to FD nonlinearity and poor numerical conditioning. However:

- A consistent fixed-wavelength adjoint remains mathematically valid off resonance.
- softW is invariant to uniform intensity scaling; reduced resonant amplitude alone does not explain the discrepancy.
- Nearly parallel Re/Im vectors explain unstable fitted coefficients, but do not establish that FD is wrong.
- Recentering changes the function being differentiated. Success there validates the new on-resonance operating point; it does not retrospectively validate the old derivative.

**FROM-CODE:** The comment describing off-resonance sampling as having “voided” the old gates is too strong. [V:119–124.] Those measurements remain legitimate evidence about the old fixed-wavelength calculation.

**INFERENCE:** If “off-resonance ill-conditioning/nonlinearity” is the whole story, the centered rerun should show:
1. FD convergence under ±1→±0.5 nm, with differences small compared with the former errors.
2. Agreement with newly computed centered Re/Im using a well-conditioned correction, or a stable justified real scale.
3. Better resolved normalized-profile changes and no unexplained sign disagreement.

Do not require its FD vector to match the old vector: changing wavelength changes the derivative. If centered agreement fails, inspect source/forward wavelength consistency, mesh sensitivity, shape-gradient accuracy and softW processing. If centered agreement passes but old-wavelength converged FD still disagrees, the derivative implementation has an operating-point limitation that needs characterization.

**INFERENCE:** An illustrative \(h^2\) extrapolation of the old data predicts an off-resonance limiting derivative near
\[
[-2.5221{\times}10^{-4},\ 1.0484{\times}10^{-4},\ 9.9417{\times}10^{-3}],
\]
still very different from old Re. This is not a trustworthy extrapolation without another step size; it shows why “±1 nm moved closer to truth” cannot be assumed to mean “closer to the adjoint.”

**Port gate — INFERENCE:** Recentering is prudent and now implemented, but the peak’s presence in the old window already matters: approximately \(2.5\Gamma=0.511\) nm around the detuned peak fits inside the old ±1-nm window, with only about **0.074 nm red-side clearance**. The ±2-nm geometry legs may exhaust that clearance. Audit every leg’s selected window and run step convergence. Recentered port success does not resolve frozen-selector versus reselected-FD objective consistency.

**J4. The width-gradient jump**

**FROM-DATA:** The norm grew approximately **8.3×**, with a negative cosine, then stabilized. That deserves investigation, but good directional width predictions show it was not automatically catastrophic.

| Hypothesis — **INFERENCE** | Cheapest discriminating test |
|---|---|
| Symmetry/cancellation at the uniform seed | Inspect blockwise signed components, left/right or mirrored contributions where available, and top contributors to \(g_W^\top d\). Uniformity alone does not require a discontinuity: a smooth derivative should approach the seed continuously. |
| Twin detuning across the resonance | At one fixed geometry, compute width gradients at the two relevant twin wavelengths. Saved vectors from different geometries cannot isolate wavelength dependence. First inspect normalized saved profiles and softW source weights across wavelength without new solves. |
| softW peak/floor/threshold sensitivity | Recompute smoothing, floor, peak weights and spatial sensitivity weights from saved profiles. Look for a change in the intensity maximum or a large set of points near half-height. This is CPU work if profiles are available. |
| Gradient extraction/indexing/provenance error | Verify parameter ordering, physical units, evaluated geometry, twin wavelength, mesh and calibration constants for each vector. Compare component ratios: a scalar change preserves cosine; this anomaly does not. |
| Large changes in irrelevant components | Compute \(g_{W,k}^{T}d_k\), \(g_{W,k+1}^{T}d_k\), blockwise contributions and scaled angles \(\cos(\sqrt Dg_{W,k},\sqrt Dg_{W,k+1})\). Raw Euclidean norm can be dominated by components that barely move. |

**INFERENCE:** Fresh campaign vectors cannot reconstruct the missing toy seed vector. Reproduce that seed gradient only if the anomaly recurs or blocks interpretation; otherwise prioritize the boundary-active operating region. The near-unit second cosine is encouraging but not a substitute for directional accuracy.

**J5. Should job 170253 stop?**

**INFERENCE:** I would not cancel the campaign because of the toy’s 0.007-µm overshoot, DATA B’s small repeat differences, or the separate S2 problem. **I would pause before committing further trials if the running build uses the copied log and first-match lookup unchanged.** That is a concrete model-provenance defect. Let an already-running solve finish and preserve its results; inspect which record supplied iteration 0’s `cw`, `cw_bad` and wavelength before continuing.

For its first three evaluated steps:

| Check | Provisional pass band — **INFERENCE** |
|---|---|
| Record identity | Current geometry **and evaluation/window identity** match the row and saved gradients; no historical-row substitution. |
| Width | Accepted ascent within [18.78858,19.45342] µm; every reported feasible best within [18.73858,19.50342] µm. |
| Width-model error | Prefer \(|\Delta W-\widehat{\Delta W}|\le0.02\) µm. Refresh/reduce after larger error; treat repeated ≥0.05 µm error as a failed model. Use absolute errors when predicted restoration is only 0.007 µm. |
| Transmission | For clearly resolved positive predictions, ratio 0.5–1.5; the toy supports a tighter expectation around 0.8–1.2, not a guaranteed bound. Repeated resolved wrong-sign gains warrant a directional FD check. |
| Wavelength | Compare sub-grid peak shifts. Initially require error ≤max(0.02 nm, 25% of predicted magnitude); monitor the actual effective 0.125/0.25-nm allowance separately. |
| Radius/solver | Confirm initial **effective** cap 11.25 nm, including persisted retry factor. No dropped wavelength row, nonfinite result or unexplained solver fallback; withhold growth until width and wavelength predictions also pass. |
| Derivative health | Fresh width row, recorded nested slopes and actual twin detuning. A large norm/angle jump triggers diagnosis, not automatic rejection if directional tests remain sound. |

**J6. Other warnings**

**FROM-DATA + INFERENCE:** DATA B’s window change produced approximately \(1.26\times10^{-4}\) FOM and \(2.09\times10^{-4}\) sampled-transmission differences—well above the previously quoted \(10^{-5}\) scale. This is not a same-window noise measurement. Grid phase, interpolation, source/normalization and simulation changes must be separated before using \(10^{-5}\) as a stopping threshold.

**FROM-CODE + INFERENCE:** Gradient filenames can overwrite across restarts, and historical best selection compares FOMs across windows without provenance-aware revalidation. [E:3453,3594–3595,3833–3836.] Fix these together with the current-row lookup.

**INFERENCE:** The row-active toy result is real progress: it addresses the main missing test from the previous review. It does not justify automatic radius escalation based only on transmission, nor an ill-conditioned S2 calibration. The next useful evidence is accurate small width restoration followed by further feasible transmission gain at the boundary.
