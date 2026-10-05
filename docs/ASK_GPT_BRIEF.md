# Standing briefing for GPT (ask-gpt skill) — maintained by Claude

You are GPT (gpt-6-astra), called from Claude Code via the Codex CLI as an independent
reviewer / second opinion for this research project. This file is your memory across
calls: Claude appends a dated entry after every session (bottom). Read it, then the task.

## Who / what
- User: Evyatar Rubin, M.Sc. photonics (Technion). Device: a **pi-shift Bragg grating**
  in SiN (n 1.97 / SiO2 1.444, core 350 nm high, ~800 nm wide, sidewall-corrugated),
  simulated in Ansys Lumerical FDTD (2026 R1.3) on two SLURM GPU clusters (Athena,
  IGUM). Claude manages everything (planning, dispatch, physics); you have the repo
  read-only, no network, no cluster access — never try to run FDTD or ssh.
- Programme you are reviewing: **lumopt2 adjoint inverse design** of the grating —
  maximise peak resonant transmission t_pk while holding the spatial mode FWHM (level-
  set width of the field envelope, an acousto-optic sensing spec) and the resonance λ.
- Terminology: "FWHM" alone = SPECTRAL FWHM (Q = λ/FWHM); "mode width / fwhm_env /
  fwhm_m" = SPATIAL envelope width (µm). Mesher: PVA (precise volume average) inside
  the optimizer, conformal elsewhere — never cross-quote numbers across meshers.

## Reading list (authoritative; read what the task needs, in this order)
1. `runners/lumopt2_design/THEORY.md` — the METHOD: softmax-T FOM, 191/296-param
   layout, projected null-space step, the two adjoints, resonance chain rule.
2. `runners/lumopt2_design/HANDOFF_2026-09-01.md` — the d1 generation: two-constraint
   (W, λ) null-space step (`_ns2_step`), adaptive trust cap, width-row reuse, measured
   results, the three fixed defects, rulings.
3. `runners/lumopt2_design/lumopt2_design.py` — the engine (~3500 lines). Key
   functions: `make_fct` (FOM), `soft_width_of_line`/`softw_and_weight` (softW width
   surrogate), `_ns2_step`, `run_projected` (the optimizer loop: filter, cap, reuse,
   restoration, λ-chain IFT selector `gLam`), `_reject_cap`, `_broyden_update`,
   `profile_mac`, `make_project` (optimization region), `run_validate_gradient` /
   `run_adjoint_only` (the C-recipe gates).
4. `runners/lumopt2_design/V2_FWHM_PLAN.md` — why softW (level-set) replaced σ
   (second moment) and the width-adjoint design; `runners/lumopt2_design/
   LIT_REVIEW_2026-08-29.md`; `docs/novelty_analysis_2026-07-07.md` (older novelty
   assessment of the device-physics programme, not of the optimizer).
5. `.claude/skills/lumopt2-design/SKILL.md` — numbered lessons (items 1-42), every one
   paid for by a measured incident; items 21-22 (bounds = trust region), 24-27
   (proxy traps), 32-35 (surrogate slopes), 37 (λ-chain), 42 (d1 formulation).
6. `runners/lumopt2_design/campaign_te_s1.py`, `campaign_te_s2.py`, `validate_te.py`,
   `gates/gate_projection_local.py` (section 10 = the four 2026-10-04 upgrades).
7. Deep background only if needed: `docs/HANDOFF_FOR_NEW_AI.md` (450 KB, the whole
   programme), `runners/lumopt2_design/HANDOFF.md` (220 KB operational log).

## Live state (refreshed by Claude; 2026-10-04 evening)
- TM lane: best machine-driven design `BEST_D1_T9676` (t_pk 0.96762 PVA at FWHM
  18.29 µm, λ held exactly) from the d1 ns2 formulation; stopped clean 2026-09-01.
- TE lane (started 2026-10-04): engine made device-parametric (spec fields pitch /
  polarization / corr0 / avg / κ / n_free / bounds; TM bit-identical, gate
  `gate_tm_identity.py`). Two seeds, both N=98/side with 60 FREE periods/side, no
  scatterers: S1 plain (pitch 500, corr 250, W800; MEASURED PVA λ 1560.900, T 0.9053,
  Q 1539, fwhm_env 19.121 µm) and S2 = Itai Lev-Ran's Nt60 "overshoot" apodization
  (pitch 491.06, bulk corr 494, avg 1000; λ 1560.407, T 0.9731, Q 7694, 19.636 µm).
  Noise floor MEASURED 1e-5 in T for sub-cell wall moves. C-port / C-field
  calibration gates are running on the cluster (jobs 168581/582, 168641/642,
  168644/645). Four optimizer upgrades implemented 2026-10-04, default-inert, gated:
  noise-aware cap freeze (`wgp_noise_freeze`), Broyden update of the reused width
  gradient (`wgp_reuse_broyden`), MAC mode-identity reject (`wgp_mode_mac`),
  separate null/range caps (`wgp_range_alpha`, `wgp_range_cap_frac`).
- Known TE risk (literature): in TE the E field is NORMAL to the walls that
  corrugation/width move — the hard case for FDTD shape gradients (Johnson/Kottke
  E∥/D⊥). TM had E parallel. The C-port FD gate measures the per-class residual.

## Rules for your answers
- Verdict first, then evidence. Cite file:line for every claim about the code.
- Label each claim: FROM-CODE (you read it), FROM-SOURCE (literature you cite with a
  reference), INFERENCE. Never invent numbers; say "unmeasured" when it is.
- Be adversarial and honest: the user wants mistakes found, not agreement. Rank
  findings by expected impact on the optimizer's result (t_pk gained at fixed width).
- Separate (a) bugs / wrong math, (b) methodological weaknesses, (c) novelty
  assessment vs published inverse-design practice, (d) concrete improvements with
  cost (GPU-hours, code size) and a falsification test for each.
- Keep it ≤ ~250 lines; use tables for lists of findings.
- Claude owns decisions and may disagree; say where you are uncertain.

## Log of GPT sessions (newest last; Claude appends after every call)
- 2026-10-04 16:03 — smoke test of the skill (read CLAUDE.md default mesh; write probe
  correctly denied). No science content.
- 2026-10-04 21:48 — algorithm review requested (mistakes / novelty / improvements of the
  ns2 optimizer, C-recipe, softW, λ-chain, the four upgrades, the TE lane; prompt kept at
  `docs/ask_gpt_algorithm_review_2026-10-04_prompt.md`). FAILED before starting: ChatGPT
  usage limit reached, retry after 2026-10-05 01:13. PENDING — run with:
  `python C:/Users/evyat/.claude/skills/ask-gpt/ask_gpt.py --dir <repo> --timeout-min 45 < docs/ask_gpt_algorithm_review_2026-10-04_prompt.md`
- 2026-10-05 10:45 — STATE UPDATE before the review call (measured overnight, jobs 168581/582/
  641/642/646): TE S1 width-adjoint C_field = (0.9668, +0.0363), residual <=0.2 % per param
  (uncorrected adjoint already within 1.7 % of FD). TE S1 port-adjoint C = (0.9546, +0.1181),
  signs 6/6, residuals corr +4.6/-2.5 %, avg +9.3 %, shift_1 +0.6 %, wcav -3.6 %, shift_30
  -12.3 % (|Z| is ~1300x the gradient there). FD/Re/Im vectors are in the comment block of
  runners/lumopt2_design/campaign_te_s1.py and in Claude's memory. The E-normal classes (corr,
  avg) did NOT show the feared class-dependent failure at dx 50 nm PVA. Noise floor |dT| ~1e-5.
  S1 pipeline smoke (job 168909) and the S2 C-port gate (168644) are running now.
- 2026-10-05 10:21-10:45 — ALGORITHM REVIEW DELIVERED (answer kept verbatim at
  docs/ask_gpt_algorithm_review_2026-10-05_answer.md; prompt at
  docs/ask_gpt_algorithm_review_2026-10-04_prompt.md). GPT's verdict: keep the two-gradient
  architecture; fix acceptance / stopping / delivery logic before a long campaign; novelty =
  the application-specific spatial-width constraint + validated implementation, not a new
  constrained-optimization principle. Claude's VERIFICATION and actions (same day):
  * A7 (C fit grid too coarse) CONFIRMED: exact LSQ gives port C (0.945335, +0.117012), all
    residuals <=5 % (shift_30 -0.1 %; the earlier -12.3 % was the 0.05 deg grid). fit_port now
    exact LSQ + cond + leave-one-out. ADOPTED.
  * A1 (identical retries -> false "converged") CONFIRMED in code: fixed with a temporary
    retry_shrink (cap, cap/2, cap/4 = three DIFFERENT trials) and a noise reject now also
    needs a small OBSERVED loss. ADOPTED.
  * A4 (rejected lambda-jump / mode-hop trial selectable as best) CONFIRMED: ineligible list
    in the optstate sidecar, _best_from_log skips it. ADOPTED.
  * A5 (filter accepts any step closer to the band centre): new spec flag wgp_filter_band
    (violation beyond the deadband); ON for TE, default off for TM. ADOPTED.
  * NOT YET DONE (accepted as real, zero-GPU, queued): A2 (null+range step can reach 2x cap;
    bounds clipping breaks tangency), A3 (lambda-restoration cross term W_lambda*h_lambda),
    A6 (unit-dependent cond / row normalisation), A8 (Broyden secant uses broadband softw_um
    while the gradient is the twin's softW), A9 (stale IFT stencil state), A10
    (_row_of_params relative tolerance), A11 (shadow price is not the multiplier).
  * NEEDS THE USER (method changes): B1/P4 replace the windowed softmax by an interpolated
    T(lambda*) objective; P1 bounded composite QP step; lambda as band vs equality; P8
    curvature; rewrite THEORY.md's "a scalar objective cannot do this" argument; the
    paper-grade comparison vs SQP/AL with ablations.
  * Disagreement / nuance kept by Claude: the measured d1 result (lambda held to the grid,
    W in band, T +0.004) stands as evidence the step works in practice; GPT is right that it
    does not prove convergence or gradient exactness.
  Follow-up questions parked at docs/ask_gpt_followup_2026-10-05_prompt.md (resume the same
  session; coordinate with the benchmark session before calling).

### 2026-10-05 — far-field multipole cancellation (session 01a10af6, 2 turns; a different lane from the TE inverse design)
Q: can Johnson-style multipole cancellation cut the pi-shift cavity leak at fixed width; which perturbation; is there
a basis with one dominant term? Data: results_from_athena/farfield_sph_20um (A TE plain, B TE overshoot, C TM plain).
GPT: (1) no x mirror symmetry in the built grating (both arms narrow-wide) -> Claude's "E_y odd in x" hypothesis wrong;
(2) top/side far fields DISAGREE on the 45-deg seam (corr 0.85/0.80/0.41, power ratio 1.3/1.3/2.3) -> all multipole
fractions indicative; (3) per-harmonic cancellation is the wrong primary coordinate; use min ||A + B N z||^2 and the
SVD of the constrained real Jacobian (a controllable left singular vector = the meaningful "one term");
(4) pure 2G second-harmonic teeth put carriers at +-3 beta, outside the light cone; (5) no FWHM-only leak bound
(sinc^2 envelope); B vs A is ~15x in Q_rad; (6) on-axis dipoles remove <5 %.
Claude verified (2) directly (A 0.847/1.31, C 0.411/2.26) and adopted (1)-(6). Claude's own result: A's leak is
reproduced 75-89 % by sources within +-0.5 um of the pi shift (B control 19-26 %, scrambled null 0.00); GPT's
top->side no-refit test passes one way (0.56-0.80), fails in reverse. GPT calls the ceiling optimistic (expects ~10 %
realisable). OPEN: 4 lateral faces = open tube, endcaps cut the feed guide -> needs a checked projection;
proposed Run 1 (certified far field + complex near field +-2 um) and Run 2 (one constrained perturbation). Nothing
dispatched. Write-up: docs/radiation_cancellation_review_2026-10-05.pdf.
- 2026-10-05 10:44-11:05 — FOLLOW-UP (resume of the review session; prompt
  docs/ask_gpt_followup_2026-10-05_prompt.md, answer docs/ask_gpt_followup_2026-10-05_answer.md).
  GPT: the fixes help but (i) restoration-dominated retries are still identical, (ii) an
  ordinary reject after noise rejects can enlarge the effective radius, (iii) a
  callback-triggered restart (recenter / width trip) can precede eligibility classification,
  (iv) an accepted shrunken retry snaps back to the base cap, (v) the violation filter accepts
  any violation decrease regardless of T loss. It specified: F2 the bounded composite step as
  ONE convex QP in scaled coordinates (two band rows, box, a single inf-norm radius; OSQP);
  F3 the 3-point/5-point parabola peak objective with gradient weights (use FREQUENCY spacing);
  F4 the total moving-resonance width row g_W,res = g_w|lambda + c_W g_lambda with c_W measured
  from same-forward neighbouring-lambda profiles; F5 lambda as a local trust bound re-centred on
  the accepted resonance, not an equality; F6 a 6-configuration paired experiment matrix
  (~156-168 GPU-h); F7 replacement wording for THEORY.md; F8 the cheapest decisive test =
  central difference along the ACTUAL production direction for T, W and lambda (2-4 forwards).
  Claude's actions same day: (i), (ii), (iv) FIXED in run_projected (duplicate-retry guard,
  one effective radius, accept adopts the radius used) with driver-level gate tests; (iii)
  and (v) accepted as real, not yet fixed. F2-F7 are method changes awaiting the user's
  decision; F8 is folded into the toy's readout (predicted vs measured dT, dW, dlambda per step).
