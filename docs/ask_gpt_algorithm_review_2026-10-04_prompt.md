TASK: a thorough, adversarial, honest review of the inverse-design ALGORITHM of this project
(the lumopt2-based constrained adjoint optimizer of the pi-shift Bragg grating), on three
axes: (1) mistakes — wrong math, bugs, hidden assumptions, places where the method can be
fooled or silently biased; (2) novelty and capability — what in this formulation is actually
new relative to published photonic inverse design / constrained shape optimization, what is
re-invented under a new name, and what a referee would attack; (3) improvements — concrete,
prioritised changes that would raise the optimizer's ABILITY (better t_pk at fixed mode width
and resonance, fewer wasted solves, robustness), each with a cost estimate and a cheap
falsification test. The user's words: "find mistakes, look at the novelty, improve the inverse
design process and the algorithms and the understanding. Be honest and thorough."

READ (in this order; the briefing above lists what each is):
  runners/lumopt2_design/THEORY.md
  runners/lumopt2_design/HANDOFF_2026-09-01.md
  runners/lumopt2_design/lumopt2_design.py — at least: make_fct and the softmax reader;
     soft_width_of_line / softw_and_weight / _wsmooth_matrix (softW); _ns2_step; run_projected
     (filter with slack anchored to fom_best, cap adapt, width-row reuse + travel budget,
     restoration, λ-chain gLam from the IFT selector passes, dwdlam refit, WidthTrip, recenter);
     _reject_cap, _broyden_update, profile_mac (the 2026-10-04 upgrades); make_project (region,
     dp clamping); run_validate_gradient / run_adjoint_only / detune_params (the C-recipe);
     adj_phase_fix and the MixedFom C_field application; param_bounds and trust_nm (bounds are
     the learning rate — skill items 21/22).
  runners/lumopt2_design/V2_FWHM_PLAN.md (skim: why softW, the width adjoint, gates W0-W6)
  .claude/skills/lumopt2-design/SKILL.md items 6, 15, 21, 22, 24-28, 32-35, 37, 42
  runners/lumopt2_design/gates/gate_projection_local.py (what the local gates actually assert)
  runners/lumopt2_design/campaign_te_s1.py, campaign_te_s2.py, validate_te.py (the new TE lane)

SPECIFIC QUESTIONS (answer each explicitly, then add what we did not think to ask):
 Q1. The adjoint "C factor" recipe: the pipeline's resonant-FOM adjoint gradient is corrected by
     ONE global complex factor C fitted to finite differences (FD ≈ s(cosφ·Re − sinφ·Im)), per
     adjoint path (port C ≈ 1.0561+0.1239i; field C ≈ 0.4554+0.1336i). Is a single global
     complex factor a legitimate correction of a Yee-grid phase/normalisation error, or does it
     hide a parameter-class-dependent error that reappears when the operating point moves (the
     project measured per-class α varying before the fix, and ≤10 % residuals after)? What would
     prove it is exact vs. merely locally fitted? For TE (E normal to the moving walls) do you
     expect the residual to become class-dependent again (Johnson/Kottke E∥–D⊥)? Propose the
     cheapest decisive test beyond what validate_te tasks 4-7 already do.
 Q2. The two-constraint null-space step (_ns2_step): D-metric from bounds half-ranges (so the
     bounds act as a per-block learning rate), ξ_J normalised to the trust cap, range-space
     restoration ξ_C = −D A M⁻¹ h, deadbanded residuals, near-null guard, condition-number
     fallback. Find errors or inconsistencies (e.g. mixing a scaled metric with an unscaled
     cap; the λ constraint row's units; what happens when gW is reused/stale or Broyden-updated
     while gLam is fresh; whether restoring W and λ simultaneously with first-order exact steps
     is well-posed when the two gradients are nearly collinear). Compare with Feppon–Allaire–
     Dapogny null-space flows and with SQP/trust-region practice: what did we get wrong or omit?
 Q3. The acceptance filter (Fletcher–Leyffer-style, slack anchored to fom_best), the adaptive
     trust cap (×1.5 on measured holds, ×0.5 on rejects, floor 2 nm), the noise-aware cap freeze
     with "3 consecutive noise rejects = converged", the Broyden rank-1 update of a reused width
     gradient, the MAC mode-identity reject. Are these sound as specified? Where can they
     deadlock, oscillate, or declare false convergence? What is missing (e.g. a proper merit
     function, second-order information, a restoration phase separated from the objective step)?
 Q4. The width observable: softW (boxcar + Gaussian smoothed, floor-relative soft level-set
     width of the y-integrated |E|² line at the single-λ twin monitor) steers gradients while
     the measured fwhm_env (envelope-peak interpolation, half-max relative to floor) is the
     authority with a ±2 % band. Is this proxy/authority split sound? Can the optimizer still
     cheat (the project found σ-vs-FWHM, ρ-deadband, Σshift=cavity-length cheats before)?
     Enumerate remaining cheat channels given the parametrization (per-tooth corr, avg width,
     x-shift; cavity width; mirrored arms; frozen outer teeth).
 Q5. The λ-chain: ∇W is at fixed λ while W is specced at the moving resonance; the project adds
     the resonance-shift term via gλ = dλ_pk/dp from the implicit-function theorem on ∂T/∂λ=0,
     obtained by two selector passes over the already-solved fields (matched antisymmetric
     stencil). Check the math and the guards (dTp<0, edge wrap, points per linewidth). Is there
     a cleaner formulation (e.g. constrain the resonance via an eigenvalue/QNM sensitivity, or
     optimise at the resonance by construction)?
 Q6. FOM: windowed p=12 softmax of port T over ±2.5 measured FWHM with stop-gradient on the
     selection; the audit found the window is silently TRUNCATED at the band edge, which
     INFLATES the FOM; the fix was a wider recording window + recenter. Is the FOM itself well
     chosen for "peak transmission of a resonance"? Alternatives (Lorentzian fit amplitude,
     T at the IFT-located peak, log-sum-exp with correct normalisation, a Q-aware objective)?
     Any bias from the frequency-uniform grid being converted to wavelength?
 Q7. Novelty: state plainly what is new here (if anything) versus: Lalau-Keraly 2013 lumopt;
     Feppon et al. null-space gradient flows; Hammond/Oskooi/Johnson shape/density adjoint work;
     resonator inverse design with Q/V or LDOS objectives (Liang & Johnson 2013; Wang et al.;
     Ahn et al. "photonic inverse design of on-chip microresonators" 2022); constrained
     "fixed-width"/fixed-bandwidth cavity design; adjoint-gradient reuse / lagged Jacobians;
     noise-aware trust regions (Sun & Nocedal). Which claims would survive peer review, and
     what experiment would make the method paper-worthy?
 Q8. The TE lane specifically (campaign_te_s1/s2, validate_te): errors or risky assumptions in
     the seed construction (Itai's 61-entry apodization → 60 free teeth, corr_min 0 at tooth 1),
     the surrogate N=98 with 2κL 3.36 (S1, below the project's 3.5 rule by user order), the
     per-seed windows (S1 10 nm/501, S2 2 nm/501), the TE-scaled caps, the plan to transfer S1's
     C factors to S2 after a ≤10 % verification. What would you change before the campaigns?
 Q9. Anything we did not ask that materially limits this optimizer's ability — e.g. local vs
     global search (two seeds only), the mirror symmetry constraint, the frozen outer teeth,
     the 2-nm trust floor, the per-iterate cost (~1-1.5 h) and what a cheaper surrogate loop
     (e.g. CMT-informed preconditioning, which the user BANNED inside the optimizer — respect
     that) or a lower-fidelity model could legitimately do.

FORMAT: verdict paragraph first (≤10 lines: the 3-5 most important findings). Then one table per
axis — (a) mistakes/bugs, (b) methodological weaknesses, (c) novelty, (d) improvements — with
columns: finding | evidence (file:line or reference) | label FROM-CODE/FROM-SOURCE/INFERENCE |
impact on t_pk-at-fixed-width (H/M/L) | fix or test (cost). Then the explicit Q1-Q9 answers,
≤12 lines each. ≤300 lines total. Do not pad; "I could not verify" is a valid cell.
