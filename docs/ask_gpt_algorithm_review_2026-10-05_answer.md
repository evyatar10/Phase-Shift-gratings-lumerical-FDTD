**Verdict — INFERENCE:** Keep the two-gradient architecture, but fix the acceptance, stopping, and delivery logic before a long TE campaign. **FROM-CODE:** the driver can repeat an identical rejected geometry three times and declare convergence; componentwise bounds clipping destroys the null-space guarantee; λ restoration reintroduces the width-chain term; and final selection can recover a mode-hop-rejected trial (E:2872–2893, 3178–3181, 3126, 2546–2567, 3398–3436). **INFERENCE:** the moving-window power mean also gives a misleading linewidth incentive. The reported d1 improvement remains useful evidence, but it does not establish gradient exactness or convergence. The defensible novelty is the application-specific width constraint and its validated implementation—not a new constrained-optimization principle.

References use **E** = [lumopt2_design.py](C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/runners/lumopt2_design/lumopt2_design.py), **G** = `runners/lumopt2_design/gates/gate_projection_local.py`, **S1/S2** = `runners/lumopt2_design/campaign_te_s1.py` / `campaign_te_s2.py`, **V** = `runners/lumopt2_design/validate_te.py`. Other filenames are relative to the repository.

**Verification scope:** I read the requested implementation and historical lessons. I ran the real numerical helpers and a mocked `run_projected` loop without FDTD. The existing projection gate passed its algebraic checks through G:184, then filesystem restrictions blocked its temporary-file test; I do **not** claim the complete gate passed. Reported cluster measurements below are documented measurements, not independently reprocessed raw results. No repository changes or cluster actions were made.

**(a) Mistakes / bugs**

| Finding | Evidence: file:line or reference | Label | Impact | Fix or test (cost) |
|---|---|---|---|---|
| **A1. False convergence after identical retries.** With adaptive cap enabled, α no longer controls the step. A noise reject freezes the cap and retries stored gradients, producing the same geometry. Small predicted gain is called “noise” even when the measured loss is large. | E:2635–2641, 2781–2783, 2872–2893, 3178–3181. Mocked full driver: one gradient calculation, three identical rejected trials, then convergence; measured synthetic loss 0.01 versus slack 0.0005. | FROM-CODE | **H** | Classify using observed **and** predicted changes; refresh suspect gradients before stopping; prohibit duplicate deterministic trials. CPU regression, no GPU. |
| **A2. The delivered step is neither guaranteed tangent nor bounded by the advertised cap.** Null and restoration components are separately capped, then added; componentwise parameter clipping follows. | E:2555–2567, 3126. Real-helper counterexamples: cap 10 produces a 19.975 nm combined step; clipping a tangent step produces nonzero first-order width change. G:163–170 tests only mid-band caps. | FROM-CODE | **H** | Include bounds and a total trust radius in the step subproblem. Test seed-boundary and simultaneous-restoration cases, not just scalar scaling. |
| **A3. “The λ-chain cancels” is true for the tangent component, not λ restoration.** If `gLam·d ≠ 0`, raw fixed-λ `gW` does not predict width at resonance. | E:2521–2546, 2562–2567, 3015–3020. G:131–143 checks cancellation only for a tangent step and restoration against the raw rows separately. | FROM-CODE + INFERENCE | **H** | Use a spectral partial derivative measured at fixed geometry, or transform the restoration residual consistently. CPU analytic test first. |
| **A4. Rejected modes can become the final design.** All evaluations are logged before acceptance; `_best_from_log` filters width but neither mode identity nor λ acceptance. It ranks FOM, not `t_pk`. | E:2785–2787, 2845–2867, 3286–3293, 3398–3436. | FROM-CODE | **H** | Give evaluations unique IDs and explicit eligibility reasons. Select the best eligible measured peak; return its matching FOM and parameters. Zero GPU. |
| **A5. The downhill ratchet is only partly fixed.** Any decrease in `h=abs(W−target)` bypasses the best-FOM condition, however small and even inside the acceptable band. Rejected high-FOM trials can raise `fom_best`; that anchor then disappears on restart. | E:2681, 2819, 2863–2867, 3170; persisted state at 2741–2746. G:218–236 omits the width-improvement branch. | FROM-CODE | **H** | Use actual band violation, sufficient feasibility improvement, and a coherent filter/funnel archive. Persist comparable accepted state. |
| **A6. Rank handling is unit-dependent and can drop a healthy λ constraint.** `cond(AᵀDA)` changes under harmless row rescaling. With zero `gW`, the code first drops λ, then returns a free step. | E:2527–2544. Real-helper test: rescaling the width row by `1e−5` changes `ns2` to `ns2_degraded`; zero width row permits a λ-changing step. G:181–184 checks only finiteness. | FROM-CODE | **H/M** | Normalize constraint rows/residuals; use rank-revealing QR/SVD; retain independent healthy constraints. Zero GPU. |
| **A7. C-fitting precision is inadequate for cancellation-sensitive components.** Phase is searched on a 0.05° grid; coefficients are printed/stored to four decimals. | `fit_c_field.py:49–54,72–75`; V:169–176; S1:59–62. | FROM-CODE + INFERENCE | **H/M** | Solve the two-coefficient linear least-squares problem directly, retain full precision, report conditioning and held-out errors. Refit existing vectors: zero GPU. |
| **A8. The width derivative and Broyden secant use different sampling conventions.** The adjoint uses the single-λ twin pinned from a previous evaluation; Broyden uses broadband `softw_um` at the current selected peak. Rejected evaluations also advance the pin. | E:736–744, 1714–1718, 1790–1802, 2232–2240, 2927–2937. | FROM-CODE | **H/M** | Track the actual twin wavelength; form secants for one consistently sampled functional, correcting its wavelength change. Gate on linewidth-relative detuning. |
| **A9. Failed IFT eligibility can leave stale stencil state.** The callback assigns `_wg_lam_idx/_wg_dTp` only inside its eligibility branch; there is no corresponding reset there. | E:1720–1767; selectors consume the stored indices at 2390–2416. | FROM-CODE | **M** | Clear both fields before processing each spectrum; reject invalid/undersampled selectors explicitly. Fault-injection test, zero GPU. |
| **A10. Diagnostics can be retrieved from the wrong evaluation.** `_row_of_params` returns the first approximate parameter match, with NumPy’s default relative tolerance and no wavelength-window/monitor identity. | E:3376–3388. Numerical check: `10000` and `10000.05` match despite `atol=1e−6`. | FROM-CODE | **M** | Use the current callback row directly; historical lookup needs exact evaluation identity or `rtol=0` plus provenance. |
| **A11. Two smaller mathematical defects:** the logged “shadow price” is not the constraint multiplier; softW has unguarded zero contrast and an overflow-prone exponential. | E:2513–2516 versus multiplier solve at 2537; E:1995–2001. | FROM-CODE | **M/L** | Log the actual multiplier vector with units; stabilize exponentials and reject degenerate profiles. CPU-only tests. |

**(b) Methodological weaknesses**

| Finding | Evidence: file:line or reference | Label | Impact | Fix or test (cost) |
|---|---|---|---|---|
| **B1. Window-adapted FOM values and frozen-window gradients have different linewidth behavior.** For a self-similar Lorentzian, the continuously rescaled-window value is independent of linewidth, but the frozen-window derivative favors broadening. The discrete objective realizes this through jumps in selected samples. | E:797–821. Real-helper Lorentzian test at fixed peak 0.9: γ=0.45/0.50/0.55 nm gives FOM ≈0.7044/0.7042/0.7044, while each local linewidth derivative is positive. | FROM-CODE + INFERENCE | **H** | Test translation, linewidth and asymmetry independently; optimize a differentiable interpolated peak or validated fit amplitude. |
| **B2. A global C is empirically useful, not established as an exact discrete-adjoint correction.** Fitting an integrated FOM does not validate the frequency-resolved derivatives used by `gLam`. | E:1602–1614, 2366–2372, 2410–2416; V:251–275. | FROM-CODE + INFERENCE | **H** | Held-out directional and spectral checks at a displaced operating point; examine the projected gradient error, not only componentwise percentages. |
| **B3. The “noise floor” experiment mixes numerical error with a real geometry response.** Two outer-tooth perturbations do not characterize error along inner-tooth, cavity, shift or grid-translation directions. Slack is then set to 50× the observed peak-T change. | V:241–250; S1:63; S2:54. | FROM-CODE + INFERENCE | **H/M** | Estimate errors for the actual FOM, width and peak estimator separately. Reuse stored spectral FD legs first. |
| **B4. Bounds simultaneously define feasibility, preconditioning and a restart-centered search box.** `trust_nm` is not a trust region recentered after every accepted iterate. Widening λ’s deadband does not release the λ-neutral objective direction. | E:603–633, 2674–2676, 2524–2545, 3199–3207. | FROM-CODE | **H** | Separate physical bounds, fixed parameter scales and movable trust radius; distinguish equality targets from allowed bands. |
| **B5. Reuse policy is extrapolated from one TM trajectory.** Angular rotation in raw Euclidean coordinates is not the relevant error measure for the D-scaled null space. The eligibility check occurs before cap growth and ignores possible extra range-step travel. | E:2901–2922, 2953–2956, 3080–3086, 3130; `HANDOFF_2026-09-01.md:77–79`. | FROM-CODE + INFERENCE | **M/H** | Budget the actual proposed scaled travel and measured constraint-prediction error. Require a fresh row before declaring stationarity. |
| **B6. Intensity MAC is a useful alarm, not a mode-identity certificate.** It loses phase and polarization information, admits gradual drift, and fails open when profiles are missing. | E:2619–2632, 2845–2851; G:342–369. | FROM-CODE + INFERENCE | **M** | Combine branch-continuous spectral tracking with field/envelope diagnostics; missing required identity evidence must be explicit. |
| **B7. The authority itself has exploitable conventions.** It uses a cubic peak envelope, its global minimum as floor, and the first-to-last half-level crossing. SoftW instead integrates all smoothed above-level regions and uses edge means as floor. | `sim_helpers.py:226–276`; E:1998–2003. | FROM-CODE + INFERENCE | **H/M** | Audit disconnected lobes, shoulders, cubic overshoot, floor changes and tails; monitor acoustic overlap without silently changing the agreed width definition. |
| **B8. An approximate κ model remains a hard optimization gate.** It assumes κ proportional to corrugation and ignores local average width and shifts; falling below it raises an uncaught ordinary runtime failure. | E:1103–1110, 1825–1828, 3365–3366. | FROM-CODE | **M/H**, especially S1 | Keep N=98 as ordered; make this model diagnostic unless a separately justified restriction is intended. Use full-wave confinement evidence. |
| **B9. Tests establish local algebra, not the full optimizer’s guarantees.** Several integration checks are source-string searches; noise tests never exercise repeated retries, and Broyden tests use exact linear data. | G:197–236, 292–340; V:193–212. | FROM-CODE | **H** | Add deterministic fake-project trajectories covering bounds, rejection, restoration, restart and final selection. Zero GPU. |
| **B10. Fixed spatial width does not imply fixed loaded Q or pure intrinsic-loss improvement.** The code infers `Qi` from a particular resonant-coupling identity; the theory also calls `1−T` cavity loss. | `THEORY.md:45–46`; `HANDOFF_2026-09-01.md:191–196`; E:1780–1783. | FROM-CODE + INFERENCE | **H** for interpretation | Separate measured T, R, uncollected power, linewidth and coupling changes. Validate the single-mode symmetric-coupling model before inferring intrinsic Q. |

**(c) Novelty**

| Finding | Evidence: file:line or reference | Label | Impact | Fix or test (cost) |
|---|---|---|---|---|
| **Adjoint shape optimization and arbitrary field-functional adjoints are established.** The weighted width source is an application-specific implementation. | E:2288–2301; [Lalau-Keraly et al., 2013](https://doi.org/10.1364/OE.21.021693); [Hammond et al., 2022](https://dspace.mit.edu/entities/publication/37501777-083a-4e72-90c1-74daf47bcdf6). | FROM-SOURCE + FROM-CODE | M | Claim the specific observable, implementation and validation—not the adjoint principle. |
| **Null/range decomposition is established; this implementation omits important inequality/bound handling.** | E:2493–2575; [Feppon–Allaire–Dapogny, 2020](https://www.numdam.org/item/COCV_2020__26_1_A90_0/). | FROM-SOURCE + FROM-CODE | H | Benchmark against an appropriately scaled constrained solver. |
| **Approximate Jacobians and noise-aware constrained trust regions are established.** Three noise rejects are not their convergence theorem. | [Walther–Biegler](https://optimization-online.org/2014/10/4596/); [Sun–Nocedal, noisy equality-constrained optimization](https://arxiv.org/abs/2411.02665); E:3178–3181. | FROM-SOURCE + FROM-CODE | H | Describe reuse and stopping as heuristics until tested against those frameworks. |
| **Resonator inverse design and preserving confinement while improving Q predate this project.** | [Liang–Johnson, 2013](https://dspace.mit.edu/entities/publication/04daf997-a9dc-4fdf-85aa-67c5c32506d5); [Wang et al., 2018](https://arxiv.org/abs/1810.02417); [Ahn et al., 2022](https://web.stanford.edu/group/nqp/jv_files/papers/geun_ho_microres.pdf); [Vučković et al., 2002](https://web.stanford.edu/group/nqp/jv_files/papers/jqe2002.pdf). | FROM-SOURCE | M | Distinguish a prescribed **spatial envelope-width band** from spectral bandwidth or mode volume. Exact priority for this combination remains unverified. |
| **Potential contribution:** constrained transmission optimization using an authoritative envelope-width measurement, a differentiable carrier, resonance sensitivities and economically refreshed adjoints. | E:1930–2014, 2381–2428, 2896–3024. | INFERENCE | H | Demonstrate better feasible T per GPU-hour across seeds/devices, with ablations and independent final validation. |
| **The “a scalar objective cannot do this” argument is wrong.** A scalar constraint can depend on every parameter; adaptive multipliers and exact penalties are counterexamples. The problem was an inadequate observable/model and globalization. | `THEORY.md:177–215,237–267`; its own AL discussion at 560–577. | FROM-CODE + INFERENCE | H for credibility | Rewrite the argument. “One cost function is impossible” will not survive mathematical review. Zero GPU. |

**(d) Prioritized improvements**

Costs below are **INFERENCE / planning estimates**, not measured runtimes. Let **F** denote one additional forward; illustratively budget **0.5 GPU-hour/F**, replacing that with actual TE timing. Fresh-iterate comparisons use the briefing’s approximately **1–1.5 hours/iterate**. Code sizes are rough implementation estimates.

| Finding / proposed change | Evidence motivating it | Label | Impact | Fix or falsification test (cost) |
|---|---|---|---|---|
| **P0. Repair evaluation identity, eligibility and stopping first.** | A1, A4, A5, A9, A10 | INFERENCE | **H** | About 100–200 lines plus fake-project tests; zero GPU. Falsify with a high-FOM mode-hop reject, a large-loss/small-prediction reject and a restart. |
| **P1. Replace projection-then-clipping with a bounded composite step.** Solve a small constrained quadratic/least-squares problem in scaled coordinates; include total radius and physical bounds. | A2, A6; B4 | INFERENCE | **H** | About 200–400 lines; CPU tests, then four matched iterations per seed, approximately 8–12 GPU-hours total. Must improve feasible progress per solve. |
| **P2. Refit C analytically and preserve precision.** Report singular values, uncertainty and held-out errors. | A7; S1:59–62 | INFERENCE | **H/M** | About 30–60 lines; zero GPU using existing vectors. Falsifier: rounding/grid removal does not reduce residuals or improve held-out predictions. |
| **P3. Validate the actual optimization direction and `gLam`.** Add a displaced-point directional check, not another fit to the same coordinates. | B2; V:77–84 | INFERENCE | **H** | Two directions × central FD = 4F, approximately 2 GPU-hours, plus a fresh gradient only if not already available. Include a withheld E-normal/outer-tooth direction. |
| **P4. Use a peak objective whose value and gradient agree.** Start with local polynomial peak interpolation; allow a Lorentzian/Fano fit only with residual diagnostics. | B1; E:761–821 | INFERENCE | **H** | About 100–200 lines; synthetic and stored-spectrum tests first, zero GPU. Then two directional FD legs per candidate estimator. |
| **P5. Measure spectral width partials and repair secants/restoration.** Use same-geometry neighboring-λ profiles already recorded; avoid geometry-path regressions. | A3, A8; E:3107–3124 | INFERENCE | **H/M** | About 100–200 lines; potentially zero additional solves for the spectral partial. Two to four F for an off-target restoration check. |
| **P6. Make reuse conditional on prediction quality.** Damped secants, scaled-angle diagnostics, small-step/noise guards, fresh final stationarity check. | B5; E:2644–2659 | INFERENCE | **M/H** | About 80–150 lines; two extra width refreshes, roughly 1–2 GPU-hours provisionally. Reject reuse if saved time is outweighed by extra rejected forwards. |
| **P7. Strengthen the width and physical-performance audit.** Track crossing count, floor/peak contrast, tails, acoustic overlap, R and linewidth alongside unchanged `fwhm_env`. | B7, B10 | INFERENCE | **H/M** | About 100–200 lines; stored-profile audit costs zero GPU. Confirm finalists at agreed numerical settings; no new production-N campaign is implied. |
| **P8. After correctness, add reduced-space curvature and broader search.** Damped L-BFGS/SQP curvature; smooth geometric basis followed by full-tooth release; short restarts from several feasible candidates. | E:2555–2561; B4; paired-arm map E:707–727 | INFERENCE | **M/H** | About 150–300 lines; matched-budget trial approximately 12–24 GPU-hours. Falsifier: no gain in best feasible T versus solve count. No CMT inside the optimizer. |

**Q1. Is one global complex C legitimate?**

**INFERENCE:** It is legitimate **if the error is truly a common complex scalar multiplying the adjoint excitation/normalization**. It cannot generally repair spatial interpolation, boundary-component weighting, geometry-dependent meshing or frequency-dependent errors. Matching six coordinates establishes local predictive utility, not exactness.

**FROM-CODE:** Application is correctly separated by path: port C and field C are applied in E:2356–2373; the width path is not double-corrected at E:1598–1602. The fit helper’s “Im” is the measured `+i` quadrature response, so its documented coefficient convention matters (`fit_c_field.py:67–75`).

**INFERENCE:** At cancellation ratio 1300, a phase error of approximately `0.1/1300 = 7.7×10⁻⁵ rad = 0.0044°` can cause 10% relative gradient error. The 0.05° fit grid is therefore insufficiently fine in principle. Direct least squares is cheaper and better.

**FROM-SOURCE:** TE boundary-normal fields warrant scrutiny, but do not imply inevitable failure: correct interface perturbations depend on continuous tangential E and normal D, as developed by [Johnson et al.](https://pbg-rle.mit.edu/Documents/PerturbationtheoryforMaxwellsequationswithshiftingmaterialboundaries_JUNE_2002.pdf) and [Kottke et al.](https://math.mit.edu/~stevenj/papers/KottkeFa08.pdf).

**INFERENCE:** Cheapest decisive extension: first test stored FD **spectra**, if retained, against corrected spectral gradients; then use fixed C at a displaced accepted geometry for two held-out directional central differences. Proving exactness requires tracing the discrete forward/adjoint operators and source normalization; finite tests can falsify exactness, not prove it universally.

**Q2. Is `_ns2_step` mathematically correct?**

**FROM-CODE:** Its unconstrained tangent projection is correct for positive D and independent supplied rows:
\[
d_J=Dg_T-DA(A^\top DA)^{-1}A^\top Dg_T .
\]
Here `gLam` has units nm wavelength per nm parameter; the lack of a `2h` factor in the matched ratio is intentional (E:2497–2499, 2999).

**INFERENCE:** D-metric projection followed by a physical-nm cap is a defensible preconditioned method, not a units error. However, it is not the solution of a consistently scaled trust-region subproblem, and post-clipping loses feasibility.

**INFERENCE:** For moving-resonance width, write `gW_res = gW_fixed + Wλ gLam`. Restoration with `gW_fixed·d=−hW` and `gLam·d=−hλ` actually predicts `ΔW_res=−hW−Wλhλ`. Either use the total row or transform the raw-row residual to `hW−Wλhλ`. Damping does not remove this cross-term.

**FROM-CODE + INFERENCE:** Row normalization, rank-revealing linear algebra, bound-active directions and a bounded normal-step least-squares solve are missing (E:2527–2567). Near-collinearity means both corrections may not be achievable within the radius; dropping λ based on an unscaled Gram condition number is not a principled resolution.

**FROM-SOURCE:** Feppon’s treatment includes feasible-cone handling for inequalities; normal/tangential steps alone do not import the complete method or its guarantees. [Feppon et al.](https://www.numdam.org/item/COCV_2020__26_1_A90_0/)

**Q3. Are the filter, cap adaptation, Broyden and MAC sound?**

**FROM-CODE:** Each contains a useful idea, but their combination is not sound enough to justify convergence claims. A1 and A5 are executable counterexamples. Cap growth requires small measured W/λ changes, but neither positive objective progress nor model agreement (E:3075–3086).

**INFERENCE:** The filter should use normalized **violation beyond the allowed bands**, not distance to their centers. Require meaningful feasibility reduction and sufficient objective improvement near feasibility. A genuine filter can work without a merit function; the missing requirement is coherent globalization, not necessarily a penalty.

**FROM-CODE + INFERENCE:** Broyden satisfies its secant equation, but it has no damping, signal threshold or bound on update magnitude (E:2653–2659). Matching a noisy or wavelength-contaminated secant exactly can worsen the row.

**INFERENCE:** MAC should remain a diagnostic/branch guard. Identical intensities do not establish identical complex modes, and small successive changes can accumulate.

**FROM-SOURCE:** Noise-aware trust-region theory uses error-aware acceptance and model assumptions; it does not license “small prediction + three rejects = converged.” The directly relevant comparator is [Sun–Nocedal’s constrained method](https://arxiv.org/abs/2411.02665). Use a fresh-gradient, bound-aware stationarity test and a separately identified restoration phase when objective steps cannot improve feasibility.

**Q4. Is the softW/authority split sound, and what cheats remain?**

**INFERENCE:** Yes—as an inexact constraint model governed by measured feasibility. An accurate value anchor does not establish an accurate derivative; that historical lesson still applies to softW.

**FROM-CODE:** Remaining channels include disconnected above-half lobes, below-half tails, changing edge background, sharp central peaks, cubic-envelope overshoot and changing peak/crossing topology (`sim_helpers.py:226–276`; E:1995–2003). SoftW’s zero-temperature limit is the **measure of the superlevel set**; the authority measures its outer span. Those coincide only for a connected interval with compatible smoothing/floor definitions.

**FROM-CODE + INFERENCE:** Corrugation and average width can redistribute shoulders and transverse confinement; shifts jointly alter tooth lengths and cavity length; cavity width can alter modal composition (E:707–728). Paired arms and frozen outer teeth restrict these channels but do not eliminate them. A fixed central-plane, y-integrated intensity width is not a constraint on full optical energy or acoustic overlap (E:1900–1918).

**INFERENCE:** If these shapes satisfy the agreed width convention, they are not automatically illegitimate designs. They expose whether width alone captures the sensing requirement. Add diagnostics first; obtain a physical justification before introducing another optimization constraint.

**Q5. Is the λ-chain correct, and is there a cleaner formulation?**

**INFERENCE:** The exact identity is
\[
\nabla_p\lambda_*=-T_{\lambda p}/T_{\lambda\lambda}
\]
at a smooth, isolated, nondegenerate maximum. The implementation’s separated-point ratio is an approximation to that identity, with unusually good cancellation for centered symmetric translating lineshapes.

**FROM-CODE:** Descending-wavelength handling and edge bounds are present (E:1752–1767); curvature collapse is checked relative to an initial reference (E:2970–2987). There is no runtime points-per-linewidth assertion in this branch. `measure_peak` returns a sampled wavelength, so “λ held exactly” means the selected grid point did not change (E:771–786).

**INFERENCE:** “Exact for any symmetric lineshape” needs qualifications: truly symmetric offsets about the continuous peak, suitable parameter dependence, and exact spectral slopes. Grid offset, frequency-uniform sampling, changing asymmetry and finite-difference slopes leave error. The existing gate acknowledges grid-offset leakage (`gates/gate_lam_chain.py:131–140`).

**FROM-CODE + INFERENCE:** Regressing W against λ along changing geometries estimates a path slope, not `∂W/∂λ|p` (E:3107–3124). Measure the latter from neighboring spectral field samples instead.

**INFERENCE:** A local differentiable peak fit is the lowest-cost cleaner formulation. QNM sensitivities are attractive for tracking a pole, but the pole frequency and transmission maximum can differ with background interference; adopting QNMs would be a larger method change, not an automatic correction.

**Q6. Is the FOM appropriate for peak transmission?**

**INFERENCE:** It is a reasonable smooth spectral aggregate, but an imperfect peak-transmission objective. The truncation fix helps only while the assumed linewidth remains valid. The runtime condition should explicitly require
`distance_to_each_band_edge ≥ 2.5 × current_spectral_FWHM`, rather than relying on a fixed recenter distance.

**FROM-CODE:** The fallback averages the entire recorded band if a crossing is missing (E:810–821). Its comment that clipping necessarily makes a trial score worse is not generally true: passband content or denominator changes can increase the score.

**INFERENCE:** Frozen selection is the correct local derivative of a discrete branch while its indices stay unchanged. The problem is that the complete objective jumps between branches, and its continuum width-adapted counterpart has different derivatives. My Lorentzian test isolates this mismatch without any adjoint error.

**INFERENCE:** Prefer interpolated `T(λ*)` with a tracked resonance branch. Lorentzian amplitude works only with fit-quality checks; Fano/background fits broaden applicability. Normalized log-sum-exp remains a spectral aggregate and needs a fixed measure/window or consistent moving-window differentiation.

**FROM-CODE + INFERENCE:** Equal sample weights mean frequency weighting, not wavelength quadrature (E:1392–1396, 821). That bias is likely secondary here, but explicit quadrature removes ambiguity. A Q-aware term changes the design objective and should only be added if linewidth is part of the physical requirement.

**Q7. What novelty claims survive peer review?**

**FROM-SOURCE:** Adjoint electromagnetic optimization, field-functional sensitivities, constrained null-space flows, approximate Jacobians, noise-aware trust regions, and resonator Q/V/LDOS optimization all have clear precedents listed in table (c).

**INFERENCE:** The strongest possible claim is: **an experimentally/numerically validated workflow that improves resonant transmission under a prescribed spatial envelope-width band, with robust resonance tracking and demonstrably economical constraint-adjoint reuse.** I could not establish priority for that exact combination; do not claim “first” from this search.

**INFERENCE:** A referee will attack the scalar-objective impossibility argument, empirical C as a claimed root-cause proof, sampled λ as “exact,” baseline methods weakened by bad surrogates, unequal-width comparisons, and claims of physical limits from a small projected gradient.

**INFERENCE:** A method-worthy experiment would compare corrected ns2 against competent scaled SQP or AL under identical observables, geometry bounds, seeds and solve budgets; ablate reuse, Broyden and resonance handling; report best feasible peak T versus GPU-hours, constraint excursions and final independent validation.

**FROM-CODE + INFERENCE:** The documented d1 gain is encouraging (`HANDOFF_2026-09-01.md:51–69`), but two successful trajectories establish capability on those cases, not superiority over published constrained methods.

**Q8. What should change before TE campaigns?**

**FROM-CODE:** The 61-to-60 apodization mapping is correct: entry 61 already equals the frozen bulk widths (`runners/sweeps/itai_hh_nt60w20.py:127–142`; S2:65–75). Zero corrugation at tooth 1 is a legitimate boundary point; it particularly requires bound-aware stepping and one-sided seed checks.

**INFERENCE:** Respect N=98. Being below an empirical 3.5 surrogate rule is not itself an algorithm error. The risky part is treating the approximate κ integral as a hard truth for freely varying corrugation, average width and shifts.

**FROM-CODE:** Both seed windows start at roughly 50 samples per spectral linewidth according to their documented PVA anchors, but S2 has little room after its FOM half-window and recenter allowance are deducted (S1:95–98; S2:95–97). Resolution and full-window coverage must be checked dynamically.

**INFERENCE:** TE-scaled caps are provisional starting values. The 1.8× effective-index argument does not calibrate a 181-active-parameter D-scaled step. Fix clipping and total-cap handling before interpreting cap adaptation physically.

**FROM-CODE + INFERENCE:** S1’s documented E-normal residuals do **not** support an immediate boundary-patch intervention; nevertheless `shift_30` exceeds the advertised 10% gate (S1:59–62). Do not dismiss it merely because the algorithm intends to null λ: the λ row itself depends on corrected spectral gradients.

**INFERENCE:** Before campaigns: repair P0/P1, refit C without quantization, verify S2 transfer for both paths, and test one held-out projected direction plus one restoration step. The toy’s marker check currently asserts reuse, not successful Broyden correction or noise-rule validity (V:193–212, 290–300).

**Q9. What else materially limits ability?**

**FROM-CODE + INFERENCE:** The two TE seeds are not a controlled multistart comparison of one optimization problem: they have different widths, pitches and feasible parameter boxes (S1:57,73–84; S2:50,84–101). Compare gains within each specification; a cross-seed claim needs common final requirements.

**INFERENCE:** A small projected gradient indicates local first-order stationarity only relative to the supplied rows, metric, active bounds and parameterization. It does not show that transmission and width are physically locked. Gradient uncertainty is particularly dangerous when projection subtracts most of the original gradient.

**FROM-CODE + INFERENCE:** The λ deadband affects restoration but λ remains in the objective projection throughout (E:2524–2545). Therefore the documented advice “widen the λ margin before paying T for λ” does not release potentially useful λ-changing ascent directions. Decide explicitly whether λ is an equality, a band, or an algorithmic tracking aid.

**INFERENCE:** Keep paired-arm symmetry initially because it controls search size and coupling balance. Later test a few antisymmetric perturbations, released outer-tooth groups, and smooth geometric basis directions before paying for unrestricted multistart search. At an exactly symmetric stationary point, antisymmetric improvements may first appear at second order.

**INFERENCE:** The 2 nm floor is not a convergence certificate. PVA’s smooth sub-cell response makes smaller steps potentially useful; conversely, unresolved gradient/model errors can make any step size ineffective.

**INFERENCE:** A lower-fidelity Maxwell model may screen seeds or propose directions, provided its ranking is checked against full 3D and every accepted result uses the authoritative solver. It can miss the radiation mechanism being optimized, so start with inexpensive full-wave geometric subspaces and curvature reuse. No CMT model inside the optimizer is needed or recommended.
