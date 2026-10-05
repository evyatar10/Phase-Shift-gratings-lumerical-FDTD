FOLLOW-UP to your algorithm review of 2026-10-05 (your answer is at
docs/ask_gpt_algorithm_review_2026-10-05_answer.md; the briefing log at the bottom of
docs/ASK_GPT_BRIEF.md says what Claude verified and already changed). Same rules: verdict
first, file:line evidence, FROM-CODE / FROM-SOURCE / INFERENCE labels, <=200 lines.

F1. Re-review the fixes Claude made for your A1, A4, A5, A7 (git diff of
    runners/lumopt2_design/lumopt2_design.py and validate_te.py since commit e121e05; look at
    _reject_cap, retry_shrink, _param_key / the ineligible list, wgp_filter_band, fit_port).
    Is any identical-retry, false-convergence or wrong-selection path still open? Is the
    "three noise rejects at cap, cap/2, cap/4 = converged within noise" rule now defensible,
    or should stopping also require a FRESH gradient at the accepted point?
F2. Specify your P1 precisely enough to implement: the bounded composite step as ONE small
    optimisation problem in scaled coordinates — variables, objective, the two linearised
    constraint rows with their bands (W: +-marg/2 about W_tgt; lambda: +-lam_margin), box
    bounds, a single total trust radius, and the solver you would use for ~296 variables with
    2 dense rows (closed-form active-set, scipy lsq_linear, SLSQP, OSQP?). State what happens
    when it is infeasible, how to keep "the step length is the trust radius, not |grad T|",
    and how the adaptive radius should react (your A2/A6/B4 points).
F3. Your P4: give the exact formula for a peak objective computable from the spectrum we
    already record and whose gradient needs NO extra adjoint solve beyond the selector passes
    we already do (they give dT(lambda_i)/dp at chosen grid points). Candidate: 3-point
    quadratic interpolation T* = T0 - (T+ - T-)^2 / (8 (T+ - 2 T0 + T-)) around the sampled
    maximum, gradient = linear combination of the three dT(lambda_i)/dp. Is that right? Noise
    sensitivity versus the p=12 windowed power mean; how many points; frequency-uniform grid;
    behaviour when the peak moves across a grid point; and does it remove the linewidth
    incentive you found (B1)?
F4. Your A3 / P5: write the exact restoration with the lambda cross-term. We can measure
    dW/dlambda at fixed geometry from neighbouring-lambda field profiles of the SAME forward
    solve (the profile monitor records the whole band). Formula for the total row or the
    transformed residual; how many lambda samples; what to do with the single-lambda twin.
F5. Lambda as equality vs band (your Q9 remark): the user's ruling is that lambda is an
    algorithmic aid, not a spec; only the spatial width is a hard spec. Propose the cleanest
    formulation that lets T use lambda-changing directions inside a band while the width is
    specced at the MOVING resonance (so the width constraint row is the total derivative).
F6. Minimal paper-grade experiment matrix given ~1-1.5 GPU-h per iterate and two seeds (plain
    TE grating; an apodized "overshoot" seed): baselines (scaled SQP? augmented Lagrangian?),
    ablations (reuse, Broyden, lambda handling, noise rule), metrics, budget. What is the
    smallest set a referee would accept?
F7. Propose corrected wording for the THEORY.md argument you called wrong ("a scalar
    objective cannot do this", THEORY.md:177-215, 237-267) that keeps what is true.
F8. Anything in the TE gate results now in the briefing (port C exact-LSQ residuals <=5 %,
    field C 0.2 %, held-out shift_30 -16.5 %, |Z|/|FD| up to 1300) that changes your
    recommendations or suggests a cheaper decisive test of the corrected gradient DIRECTION.
