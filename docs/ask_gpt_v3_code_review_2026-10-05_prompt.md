SECOND FOLLOW-UP (2026-10-05): review of the IMPLEMENTED "v3" step engine. Context is in the
briefing log (docs/ASK_GPT_BRIEF.md, bottom) and your two earlier answers
(docs/ask_gpt_algorithm_review_2026-10-05_answer.md, docs/ask_gpt_followup_2026-10-05_answer.md).
Since then Claude implemented your F2-F5 proposal behind flags and it passes local gates; a
2-iterate pipeline smoke is running on hardware and a 4-iterate toy on the real device is
queued. Review the CODE before the toy's numbers are trusted. Same rules: verdict first,
file:line evidence, FROM-CODE / FROM-SOURCE / INFERENCE labels, <= 200 lines, adversarial.

READ: runners/lumopt2_design/v3_step.py (peak3, cw_from_widths, qp_step, radius_update);
runners/lumopt2_design/gates/gate_v3_local.py (math + driver-level checks V1-V7);
in runners/lumopt2_design/lumopt2_design.py: make_fct_peak, make_fct_v2 (v3 branch), the
"c_W = dW/dlambda at FIXED geometry" block in the log callback, and in run_projected the v3
state block, the v3 branch of _step_of, h/hv/reuse eligibility under v3, the v3 radius rule,
pred_step, and the retry logic (retry_shrink, rej_trials duplicate guard, stalled stop,
ineligible list) as it now interacts with the QP step; campaign_te_s1.py SPEC_V3;
validate_te.py tasks 28/29 and _upgrade_markers. `git log --oneline -6` and
`git diff c8e1057 975032a -- runners/lumopt2_design` show exactly what changed.

QUESTIONS:
G1. qp_step: is the formulation and its dual solve correct (objective, tau rule, box, two-sided
    rows, restoration fallback, dropped-row order)? Any case where it returns a step that
    violates the box/cap/bands, or a wrong "mode"? Is using D = (bounds half-range)^2 as the
    proximal metric together with an inf-norm cap in nm coherent, or should the cap be applied
    in scaled coordinates as you wrote in F2? What does the tau rule do when gT is tiny but
    nonzero (a noise direction amplified to the full radius)? Propose the guard.
G2. Total width row gW_res = gW_fixed + c_W*gLam with c_W from three softW samples of the same
    forward at lambda_pk and +-1/4 linewidth: is that the right partial derivative (fixed
    geometry, softW observable)? The fixed-lambda adjoint gW is evaluated at the single-lambda
    twin monitor pinned to the PREVIOUS evaluation's resonance; quantify the error this
    leaves when the resonance moved by up to wgp_v3_dlam_nm in the last step, and say whether
    it is first-order or second-order in the step. Is the delta-anchored use of softW for a
    band that is defined on the measured fwhm_env sound inside a band constraint?
G3. peak3 objective inside the lumopt2 fct (index selection stop-gradient, three weights from
    autograd): any discontinuity or sign problem when the sampled maximum switches grid
    point between evaluations, or when B >= 0 falls back to the raw sample? Interaction with
    the acceptance slack (FOM now in T units) and with fom_best / the filter.
G4. The lambda trust row (|gLam.d| <= delta, re-centred each accept, no absolute target):
    can the resonance now random-walk out of the recorded window without any restoring
    term, and is the existing recenter guard (callback raises when |lam_pk - scan centre| >
    recenter_nm and the campaign restarts with a re-centred window) sufficient? Should delta
    adapt (e.g. to predicted-vs-measured d lambda)?
G5. Radius rule (v3_step.radius_update + the driver): it grows only when active_cap > 0. The
    driver-level gate found that with realistic gradients BOTH rows bind on every step and
    the cap never binds, so the radius never grows and the step length is set entirely by
    the bands (width band edge + delta_lambda). Is that the intended behaviour? Does it make
    delta_lambda the de-facto step-size parameter, and what should adapt it?
G6. Retry logic with the QP: after a reject the trial is recomputed from stored gradients at
    cap*retry_shrink; if the rows (not the cap) set the step, the QP returns the SAME d and
    only the duplicate guard changes the trial (halving the delivered displacement, which
    no longer solves the QP). Is that acceptable, or should the retry shrink the BANDS
    (delta_lambda and the allowed width move) together with the cap? Give the rule.
G7. Anything in the v3 integration that can silently fall back to the old behaviour or to an
    unconstrained step (e.g. gLam None, cw None/curved, row dropped, restore mode) without a
    loud log line; anything the toy's markers (validate_te._upgrade_markers) fail to assert.
G8. Given TE S1 at T 0.905 / Q 1539 / width 19.12 um with a +-2 % band and S2 at T 0.973:
    what should the 4-iterate toy show for you to believe v3 works (specific predicted-vs-
    measured ratios for dT, dW, d lambda; step sizes), and what result would falsify it?
