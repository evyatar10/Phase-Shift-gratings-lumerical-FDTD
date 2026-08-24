# HANDOFF — lumopt2 corr-325 inverse design — updated 2026-08-24 IDT

> ## ★★★STATE AS OF 2026-08-24 EVENING — READ THIS WHOLE BOX FIRST.
> ## Every earlier job list in this file is SUPERSEDED, including §0a
> ## ("EVERYTHING IS STOPPED"), which is wrong.
>
> **LIVE (Athena): 136752 `lumopt2_v2_seesaw` · 136753 `lumopt2_v2_uniform_s5`.**
> Both re-dispatched this evening with the corrected wall AND a 30 nm shift
> trust box. CANCELLED today, do not resume: 136465 (converged, its winner is
> preserved as `BEST_T9636`), 136468, 136695, 136708, 136709.
> Full measured state: `memory/project_v2_width_gradient_plan.md`.
> Six fixes committed today: 7eb7d35, 0d2ff88, 30f8f77, 3c10524, 60e57f9, b0102dc.
>
> ### THE PROGRAMME HAS THREE JOBS. In the user's own priority order:
>
> **① THE UNIFORM SEED — TOP PRIORITY (user, 2026-08-24).** Why it matters:
> `BEST_T9636` is NOT a clean research result. Its shifts (e≈130.6) were
> inherited from stage-1 work fitted to a device that no longer exists, and
> 136465 only nudged them 2 nm; the corrugation came through hand retrims. So
> the headline number is a chain of adjustments plus partial gradient runs, not
> one honest optimization. A uniform-seeded campaign that reaches a comparable
> design IS that honest result, and it is what lets us claim we understand the
> patterns rather than having stumbled into them.
> ★**The stalls were NOT local minima** (Fable audit, task a5fddc5457850a073):
> the fixed point was wrong because the PRICES were wrong (rank-deficient
> wall), not because of multimodality. The seed is fixable.
> ★**But it cannot "discover everything" from the seed, structurally.** From
> uniform, the pure-T gradient says "lower every corrugation, spend width" —
> ∂T/∂corr < 0 on every tooth. The see-saw REQUIRES raising outer teeth, which
> locally costs T, so it is justified only once width has a price — and the
> hinge gives width ZERO price until the band edge (MEASURED: penalty gradient
> is exactly 0.000e+00 for e ≤ 81 nm). Therefore the uniform seed can only find
> the see-saw AFTER walking to the band ceiling. That is exactly where 136468
> was pinned, and exactly where the per-tooth fix bites. It also cannot
> discover the comb (frozen) or N (fixed).
> ★**Historical contrast worth keeping:** the sigma-era campaign DID travel
> from uniform to a shaped design — because sigma put no price on the profile
> at all — and its width blew up +14.9%. Sigma era = freedom without control;
> FWHM era = control without freedom. Neither was right.
> ★**If 136753 thrashes at the band edge, the next move is the AUGMENTED
> LAGRANGIAN, not another patch.** The hinge gives width shadow price 0 inside
> the band and ∞ outside, so the optimizer spends blindly then crashes. The AL
> machinery (`wg_mu`, `wg_lam_hi`, `wg_lam_lo`) already exists in the engine —
> it was built for the exact width-gradient path and orphaned when that path
> was priced out at 8.7 h/solve. Applying it to the cheap surrogate is the
> principled fix. A "soft shoulder" is a patch on the wrong architecture.
>
> **② THE BEST DESIGN — and the measurement that decides what to do with it.**
> `BEST_T9636` (in `best_designs.py`): T 0.96361 / λ 1566.444 / fwhm_env
> 18.35309 µm / Q_load 2021.6 / Q_i 110 087 / mcorr 357.95 / e 132.6 /
> wcav 961.1.
> ★**MEASURED 2026-08-24: campaign 136465 moved its seed by ≤0.26 nm in ANY of
> the 191 params** (corr 0.059 / avg 0.263 / shift 0.169 / wcav 0.161 nm)
> against trust radii of 10-12 nm — it used ~2% of its allowed travel.
> **`BEST_T9636` is therefore `BEST_T9635` + the 42 nm hand retrim, and the
> gradient method contributed essentially nothing to it.** That is the concrete
> basis for the user's "this is not a full research way".
> ★**AND THE WALL IS INACTIVE THERE: penalty 0.0016, |grad| 5.0e-4**, because
> 18.353 µm sits well inside the 18.713 ceiling. Two consequences, both
> important, and BOTH were stated wrongly in session before being checked:
>   - "It is converged so there is nothing to do" — misleading, retracted.
>   - "Restart it under the corrected wall and it may find more" — ALSO wrong,
>     retracted. A term that is inactive cannot change the local landscape;
>     a restart would very likely sit still exactly as 136465 did. Do not
>     spend a campaign slot on it.
> What we actually measured is that the optimizer probed and rejected
> everything, NOT that no better nearby point exists — do not overstate it as
> a proven local optimum.
> ★**So the things worth trying on this design are the ones that CHANGE the
> landscape, not search it harder** (all cheap, all forward-solve only):
>   1. **Cavity width probe** — wcav 961 → ~1100 (bound 1150), 2 forwards.
>      189 nm never explored, and it is the most width-efficient lever ever
>      measured here (rtdec: +0.0409 T for +0.0305 µm). Highest value/cost.
>   2. **Comb re-tune** — the 57-site comb is FROZEN at a tune fitted for the
>      uniform seed, but the mode has moved a long way since (mcorr 325→358,
>      e 0→132.6, wcav 800→961). Worth +0.0040 at its current tune; a small
>      δx/r/d scan on the CURRENT mode is untested. NOTE: a second comb ROW is
>      MEASURED DEAD three times — see the correction above.
>   3. **Spend the 0.36 µm of unused headroom deliberately, then re-optimize
>      from the new point.** The design stops short of the ceiling. A hand move
>      (see-saw, or the split test) puts it somewhere the constraint is active,
>      which is the only regime where the corrected wall does anything.
>   4. **PRODUCTION CONFIRM** (see "what the user forgot") — the actual
>      deliverable, and independent of all of the above.
>
> **③ LEARN FROM THE DESIGNS ALREADY RUN.** The measured pattern inventory is
> in §"WHAT WE MEASURED" below and in the plan memory. The two the user singled
> out, both confirmed: (a) **many teeth × tiny shifts costs NO width and raises
> T** — below ~1.3 nm/tooth (e ≤ 65) width is flat and T gains +0.0185;
> (b) **reaching 0.96 needed MORE shift than that** — e = 132.6 is double the
> knee, and the part above the knee cost 1.5 µm of width for +0.0207 T.
>
> ### ★"IS FWHM IN THE GRADIENT?" — SETTLED 2026-08-24, THE RECURRING QUESTION
> **NO — the true FWHM is not. A MODEL of it is, and that model says nothing
> while the device is in spec.** There are TWO width paths in the engine;
> check which one you are looking at before answering this again.
> - **Path A, the real one — BUILT, VALIDATED, SWITCHED OFF.** `softW`
>   (differentiable mode envelope from the field) + `width_band_penalty`
>   (augmented Lagrangian) + `make_fct_v2`. `width_grad=True` appears ONLY in
>   `validate_c325.py` and behind `EXACT_WIDTH_GRAD=False`; **`ADJ_FIX_FIELD`
>   is None — the field-adjoint C was never fitted** (an assert fires if you
>   enable it). MEASURED cost 8.7 h/solve ⇒ priced out.
> - **Path B, what actually runs — `make_fwhm_wall`.** An analytic model of
>   width (per-tooth corr + total elongation). It IS autograd-differentiated
>   and IS added to the FOM gradient. But it is a pure QUADRATIC hinge, so its
>   derivative is exactly zero everywhere inside the band AND zero AT the band
>   edge; force only appears once you are measurably past it. MEASURED:
>   |∂pen/∂shift| = 0.000e+00 for e ≤ 81 nm, 2.1e-3 just past, 0.136 at e=90.
> ★**The AL is NOT the cure for the in-band blindness** (said in session,
> corrected): `λ·max(0,g)` is also zero when the constraint is satisfied —
> correct and unavoidable for an inequality constraint. Its real advantage is
> narrower: the LINEAR term gives a nonzero slope AT the boundary where the
> quadratic hinge gives zero, so it bites on contact instead of only after
> overshoot. Worth adopting for that reason; it will not produce an in-band
> width-reducing direction, because no constraint acts when satisfied.
> ★**What the user asks for — "a step that minimises FWHM", i.e. the see-saw —
> is a move gradient descent CANNOT make from a local optimum.** The see-saw
> narrows AND raises T in order to free width budget to respend on shifts:
> a TWO-STEP move whose first step looks locally worse. `BEST_T9636` sits at a
> local T max with 0.36 µm of slack, so a descent method will not take it.
> Not a bug in the wall — outside what the method does.
> ★**What DOES serve that goal, and shipped today: `FW_TOOTH_W`.** When the
> design is against the ceiling and wants more T, the gradient now knows which
> directions are width-cheap (inner teeth cost ~11× outer). That is the
> see-saw knowledge, in the gradient, engaging exactly where it is needed.
> ★**"CAN'T THE WIDTH ADJOINT JUST RUN ON GPU?" — ASKED AND ANSWERED, NO.**
> The 8.7 h IS a CPU number, so the question is the right one to ask; it was
> chased and it failed for a physical reason, not a cost one. MEASURED (job
> 136108, n310/A100, identical scene):
>   - forward: 3,100.3 s
>   - width adjoint via lumopt2's own **FieldRegion** object: CUDA `invalid
>     configuration argument`, dies in seconds — 3 independent tasks. The GPU
>     engine rejects that object outright.
>   - workaround via a **standard import source**: 3,133.6 s (52 min) on GPU,
>     which looked like a 10-14× win and was written up as
>     "★★★GPU WIDTH-ADJOINT PROVEN".
>   - **THAT CLAIM WAS THEN RETRACTED.** The adjoint source is an imported
>     field sheet at z = 0, and an import source injects through TANGENTIAL
>     components — but at z = 0 this TM mode is tangentially DEAD BY PARITY:
>     MEASURED max|Ex| = 0.0, |Ey| = 0.0 EXACTLY, |Ez| = 8.41. So it injected
>     nothing. The output h5 has EVERY monitor field EXACTLY 0.0 — a dead
>     source at ratio 0, not 1e-4. The 52 min was an EMPTY SCENE integrating
>     the full sim time (hence runtime ≈ the forward's). Every printed
>     "adjoint" vector in that run was the penalty gradient alone.
>   - the CPU FieldRegion timings (8.7-12.1 h) WERE real adjoints — dipole
>     injection is unaffected — so the cost verdict stands.
> ⇒ **Do not resurrect the 10-14× speed claim; it is void.** The one live
> thread is the qualifier: the route is dead **at z = 0**. A weighted sheet on
> a plane where the TM mode HAS tangential field (i.e. off the symmetry plane)
> is untested and is the only remaining way this becomes affordable.
> KEEP-FOREVER byproduct of that failure: job 136189's FD half is genuine
> first-of-its-kind data — MEASURED d(softW)/dp at detune-1 =
> [−0.00365 corr_1, +0.01825 shift_1, +0.02026 wcav]. FD is
> config-independent, so it is valid for any future gate.
>
> ### ★MESHER METHODOLOGY — USER RULE 2026-08-24. Binding.
> "It is okay to use PVA if it is the only one that is differentiable, **as
> long as the best device in PVA is also best in conformal**. If we use PVA,
> **final devices must be verified in conformal against the INITIAL in
> conformal**."
> Both halves are load-bearing and the second is the stricter one: a
> conformal number for the winner alone proves nothing, because the claim is
> always a DELTA. Validation is a conformal-vs-conformal PAIR — final and
> seed, same mesher, same everything — never a cross-mesher subtraction. This
> is §2's identical-numerics rule applied to the mesher axis.
> ★**THE RANKING-TRANSFER ASSUMPTION IS UNTESTED, AND FABLE'S OWN MECHANISM
> SAYS IT IS NOT SAFE BY DEFAULT.** The verdict is that PVA's error comes from
> scalar averaging over-weighting ε at material faces, and the over-weight per
> unit length scales with the LOCAL CORE WIDTH — so wide (tooth) sections gain
> more spurious n_eff than narrow ones. That makes the error a function of the
> CORRUGATION PROFILE, which is exactly what the optimizer changes. A
> profile-dependent error can in principle reorder two designs. Nobody has
> ever measured two DIFFERENT devices under both meshers; the only paired
> measurement is one device (MX-14/15).
> ★**CHEAP CLOSER, serves BOTH halves of the rule at once (3 forwards):** run
> the uniform seed, seesaw_d090 and `BEST_T9636` under CONFORMAL at
> pitch-locked dx. Their PVA numbers already exist (0.90120 / 0.93790 /
> 0.96361), so this immediately (a) tests whether the PVA ordering survives,
> and (b) produces the conformal INITIAL that the user's validation rule
> requires. Do this BEFORE quoting any conformal delta. If the ordering does
> NOT transfer, PVA is disqualified as an optimization mesher and the whole
> architecture needs revisiting — so this is a gate, not a nicety.
> Related but separate: mode width has never been laddered vs dx under EITHER
> mesher. Fable's 6-run closer (bare N=100, dx ∈ {51.68, 32.30, 25.84} ×
> both meshers, ~12 GPU-h, per-hypothesis predictions, no new engine code) is
> designed and unrun — it decides which mesher is right, whereas the 3-forward
> check above decides whether it MATTERS for our rankings. The 3-forward one
> is higher priority.
>
> ### ★UNITS TRAP THAT COST REAL CONFUSION (2026-08-24) — state them always
> **`e` is the TOTAL, not the per-tooth shift**: `e = 2 × sum(25 shifts)`, so
> per-tooth = **e/50**. The knee at e=65 is a per-tooth shift of just
> **1.30 nm** (0.50% of the 258.4 nm segment); the largest rung ever run,
> e=287.5, is still only 5.75 nm/tooth; `BEST_T9636` spans 0.89-6.43 nm/tooth.
> The user reasonably remembered "width increases from the very first point"
> — true if you scan TENS of nm per tooth, because 10 nm/tooth is e=500,
> ~8× past the knee. Both readings are the same curve at different scales.
> ⇒ Whenever quoting `e`, give the per-tooth equivalent alongside it.
> ⇒ And note `shift_bounds` are (0, 200) nm PER TOOTH against a useful range
> of ~0-7 nm — see the conditioning item below.
>
> ### ★THE TRANSMISSION CEILING — what any further gain actually costs
> At the width spec Q_load is pinned near 2020 (MEASURED 1930-2109 across
> every device this programme has built), so T is a function of RADIATION
> Q_i alone. Q_i is DERIVED as `Q_L/(1-sqrt(T))`, so this is an exchange
> rate, not a prediction — but it is the right accounting frame:
> | target T | required Q_i | vs today's 110 087 |
> |---|---|---|
> | 0.965 | 114 500 | ×1.04 |
> | 0.970 | 133 755 | ×1.21 |
> | 0.975 | 160 711 | ×1.46 |
> | 0.980 | 201 144 | ×1.83 |
> | 0.990 | 403 307 | ×3.66 |
> Brutally nonlinear. **0.97 is plausibly reachable** from envelope shape plus
> the cavity-width headroom. **0.98 needs something structurally new**, and
> the one candidate that used to be quoted for it (second comb row) is
> measured dead three times — see the correction above. Device LENGTH is not
> available: Q_i ~ L^2.5-3.6 but L saturates ~20 µm and the width spec pins it.
>
> ### WHAT THE USER FORGOT (raise these unprompted)
> 1. **PRODUCTION CONFIRM is the biggest gap.** 0.96361 is a surrogate-N,
>    PVA-mesh OPTIMIZER number — not a device number. N≈169 + accurate mesh,
>    run OUTSIDE lumopt2 via a plain SweepSpec runner, is what makes it real.
> 2. **Width conversion is now ×1.049** (PVA→conformal). The old 0.92 is
>    RETIRED. Mesher arbitration settled 2026-08-24 (Fable, task
>    a46f03072ddb16c77): conformal variant 0 is presumptively truthful, the
>    ~20 µm spec is NOT an artifact, and the famous "−8%" gap was itself partly
>    a grid-phase artifact — the clean matched-grid gap is −4.4 to −4.7%.
>    Residual: mode width has never been laddered under either mesher; the
>    6-run closer (~12 GPU-h) is designed and unrun.
> 3. **The h5 janitor must be RUNNING** (`~/h5_roll_clean.sh`, nohup on the
>    Athena login node). It was found DEAD today with ~3.5 h of quota runway
>    left. Verify with `pgrep -af h5_roll_clean` at every check.
> 4. **The bounds are badly conditioned.** `shift_bounds` are 0-200 nm PER
>    TOOTH while the entire useful range is ~0-7 nm (BEST_T9636 spans
>    0.89-6.43). That ~30× mismatch caused THREE failures: the 136640
>    line-search abort and both first-step lurches to e≈285. The trust box
>    papers over it; it does not fix it.
> 5. **Two cheap untested experiments, both with plausible FREE gains:**
>    (a) shift DISTRIBUTION in the free zone — nobody has compared shapes at
>    matched e on the same device, and free-zone shifts are pure profit;
>    (b) the SPLIT — above-knee shifts buy at 0.0136 T/µm while the see-saw
>    appears to buy at ~0.021, so spending headroom on apodization instead may
>    beat the current design. Both need ~3 forwards. (b) rests on a rate
>    measured on a DIFFERENT device — treat as hypothesis, not result.
>
> **THE ONE THING THAT CHANGES HOW YOU READ THE WIDTH MODEL:** the
> fwhm_wall's elongation slope `FW_A_ELONG = 0.01355 um/nm` is a SECANT of a
> single pair, and it is WRONG in both directions. The width-vs-elongation
> law was MEASURED on 2026-08-23/24 (IGUM jobs 61742 + 61782, 6 rungs on the
> uniform corr-325 seed, pure common mode, pitch-locked mesh):
>
> | e = 2*sum(shift), nm | 0 | 60 | 120 | 180 | 240 | 287.5 |
> |---|---|---|---|---|---|---|
> | fwhm_env, um | 18.345 | **18.311** | 20.483 | 24.015 | 28.768 | 32.698 |
>
> It is a THRESHOLD: **flat to ~65 nm** (e=60 measures NARROWER than the
> seed), then a knee and a steep, still-accelerating rise. Fitted law
> `dW = 7.8654e-3 * max(0, e-65)^1.39`, max residual **0.106 um** vs the
> 0.367 um half-band. Engine: `_fw_elong_curve`, spec flag **`fw_curve`**
> (default False; ON only for 136640). The older quadratic `fw_convex`
> (FW_C_ELONG=1.07e-4) is ALSO refuted by these data — **do not enable it**.
>
> WHY IT MATTERED: the old linear wall charged +0.813 um of predicted
> widening at e=60 where the true cost is ZERO — a penalty (~0.795) on the
> order of the whole FOM (~0.67). Campaign 136466 was therefore effectively
> FORBIDDEN from using shifts: it oscillated e = 0 → 287 → 0.3 → 144 → 0.7,
> never probing the 1-100 nm range where shifts are free, and gained
> +0.0005 T in ~7 h while shift-FROZEN 136468 gained +0.0076. That is why it
> was restarted as s3/136640.
>
> WHAT ELONGATION PHYSICALLY IS (code-verified, make_func:445-476): it
> LENGTHENS the central cavity block (`cavity::x span = pitch/2 + 2*sum(s)`)
> AND shortens every free period by s (narrow segment `hp - s`, walk
> `2*hp - s`), with both regions walked inward from FIXED outer edges so
> total device length is constant. Cavity lengthening is linear and tiny
> (0.29 um at e=287.5); the width blow-up is MIRROR DETUNING eroding
> kappa_eff, and width ~ 1/kappa_eff is convex — hence the threshold shape.
> ⇒ In this parametrization cavity-lengthening and mirror-detuning are THE
> SAME MOVE, so "are tooth shifts necessary?" tests that COUPLED direction,
> not cavity length in isolation. NOTE `I_CAV` is the cavity **WIDTH (y
> span)**, 750-1150 nm — not a length.
>
> ALSO CLOSED 2026-08-23: the comb is worth **+0.0040 T** at benchmark width
> (not width-driven; it narrows the mode only 0.34%, ~5% of its own gain) ⇒
> a fabrication decision, not a physics necessity. And a CONCENTRATED shift
> pattern widens LESS than a uniform one at matched elongation (0.385 um of
> 14.35, i.e. 2.7%) ⇒ pattern is a minor correction, common mode dominates.

> **UPDATE 2026-08-21 — the §6 fix now has a researched, offline-VALIDATED v2
> design: `runners/lumopt2_design/V2_FWHM_PLAN.md`** (differentiable
> FWHM-tracking width `softW` validated ≤2 pp on the corrected profiles where
> σ errs 24 pp; lumopt2 field-adjoint feasibility read from source; AL
> architecture + gate ladder W0-W6; W0 passed, zero GPU used). Skill item 28.
>
> **LATER SAME DAY — implemented + in cluster validation:** engine carries
> width_grad (MixedFom + single-λ twin monitor + AL penalty + re-anchoring);
> local gates all pass; cluster gates W1r-W3 = Athena job 135971 tasks 10-13
> (two dispatch-time bug classes found & fixed en route, zero GPU lost —
> skill item 28 lessons a-d). ALSO: mesher presumption FLIPPED to conformal
> (see §5-4 below); chirp + sinc CLOSED by calibrated light-cone model;
> N_FREE 25→40 = banked v2.1 candidate (~10× model headroom); ★second comb
> ROW = **MEASURED DEAD, three times — this line previously called it a "live
> candidate with a ~2.2× Q_i phasor ceiling" and that was WRONG** (corrected
> 2026-08-24 after the user pushed back). The 2.2× is a DERIVED bound from a
> model the same handoff flags PRELIMINARY, and it stands against three
> independent measured negatives, the last of which POSTDATES it and tested
> this exact comb family: job 130154 (2026-08-10) multi-row 2D lattice —
> single row **+0.0115**, 2row +0.0099, 4row +0.0087, 4row-r80 +0.0088, i.e.
> EVERY multi-row variant BELOW single row, logged as "user's priority
> question answered NO". Earlier: job 121392 (2026-07-15) row-2 columns ~10×
> weaker than row-1, route CLOSED; job 123563 (2026-07-18) 2-row 1.139 vs
> 1-row 1.150, "NO coherent row buildup ⇒ no N-row/bigger-post path". Only
> surviving thread: those tests fixed the inter-row offset, so a δx fan is
> UNTESTED — that is a speculative retry of a thrice-negative result, not a
> candidate, and must be described as such. ★SEED-B FWHM NEVER
> MEASURED (verified: no fwhm_env in any seedB/A4 log; .fsp gone on IGUM) —
> `seed_width_audit.py` rebuilds ev0/best/seedB2-best from logged params,
> 3 forwards, answers whether seed B started/ended in the FWHM band.

## ►► IF YOU READ ONLY ONE BOX: THE JOB IS TO FIX THE MODE-WIDTH PROBLEM

The campaign produced designs at T≈0.96 that are **15-27% too wide**, using a
width constraint that was measuring the wrong thing through a broken extraction.
All runs are stopped. Nothing is pending. Your job is to re-specify the problem
and re-run it honestly. In order:

1. **§0** — the user's non-negotiables (width growth is a DEFECT; FWHM is the
   `post_processing` convention ONLY; no cheating; never compare across widths).
2. **§0a** — current state (everything stopped, where every file is).
3. **§0c** — every correction/retraction already made. **Do not re-derive a
   claim that is on that list as withdrawn.**
4. **§2** — the profile bug, because it voids every pre-2026-08-18 σ and FWHM.
5. **§6** — the fix: projection/re-trim, not a penalty band.
6. **§6b, §6e** — the two experiments to run first, in that order.

**The single most useful fact for the fix:** mode length **saturates** at
~19.7-20 µm (§6d-v), so length was never a lever. Only envelope **shape** at
fixed length is available — which means the **inner see-saw** (§6e), the
**comb**, and **taper length** (§6d-iii), not tooth shifts.

---

**Point a new chat at this file first.** It is written to be read AFTER the
in-flight jobs have finished, so section 5 tells you how to read them and what
each possible outcome means. Also read `CLAUDE.md` (project rules) and
`.claude/skills/lumopt2-design/SKILL.md` (the living runbook, items 1-27).
Deeper history: memory file `project_lumopt2_campaign_state.md`.

---

## 0. ★★★NON-NEGOTIABLES (user, 2026-08-18) — READ BEFORE TOUCHING ANYTHING

**(a) THE WIDTH GROWTH IS A DEFECT TO BE FIXED, NOT A RESULT TO BE REPORTED.**
The measured designs grew the mode by **+14.9% (best) up to +26.6% (d+80)**.
User's words: *"if fwhm actually changed in 30% its very bad"*. It is. This is
an acousto-optic detector that senses at a FIXED width — the width is a spec,
not a free parameter. A design that reaches T 0.966 by growing the mode a
quarter wider has not solved the problem, it has changed the problem. **The job
of the next phase is to recover the transmission at the ORIGINAL width**, not
to document how much width the old campaign spent.

**(b) FWHM IS MEASURED THE WAY `post_processing` MEASURES IT. FULL STOP.**
The one and only width observable is `sim_helpers.extract_and_process_field_profile`'s
recipe — resonance lambda -> |Ex|²+|Ey|²+|Ez|² -> **integrate over y** -> crop to
the grating -> `extract_envelope_peaks` -> `calculate_fwhm_relative`. This is the
same convention as every other trusted quantity in this project, and the engine
now calls those exact functions (verified to 7e-15 um against stored `fwhm_m`).
Do NOT invent a second convention, a fitted slope, or a theoretical model.
Both were tried on 2026-08-18, both were wrong, and both are deleted.

**(c) NO CHEATING TO MEET THE SPEC.** The baseline's UNIFORM corrugation stays
325 nm. Re-tuning it so a ~20 um target lands conveniently does not count as a
fix. Any claimed gain must be a gain at the SAME measured width.

**(d) NEVER COMPARE TWO DESIGNS AT DIFFERENT WIDTHS.** T_A vs T_B is meaningless
unless FWHM_A == FWHM_B. Re-trim first (section 6), then compare.

---

## 0a. STATE AS OF 2026-08-19 01:30 — EVERYTHING IS STOPPED, NOTHING IS PENDING

**All runs stopped on user instruction.** Both cluster queues are EMPTY.
- Athena `134032` (stage-4 seedA4) — **CANCELLED**. 25 h for nothing: FOM 0.7157612
  at eval 3 → 0.7157579 at eval 9, i.e. it moved BACKWARDS by 3e-6. Its 14-eval
  log was fetched first → `results_from_athena/lumopt2_logs_seedA4_evals.jsonl`.
- IGUM `55801` (bare) — died 2026-08-18 23:54 **on TIME LIMIT**, 12 evals, no
  `_best.json`. Walltime was badly undersized: the log shows one adjoint at
  **3107 s (52 min)**, so ~1.5-2 h per gradient iteration. Long campaigns need
  `--qos=4d_1g`.
- IGUM `56033` (seedB2) — finished cleanly earlier (exit 0, best_fom 0.702857).
- All eval logs, `_best.json` files and corrected `.npz` profiles are LOCAL.
- All watchers/monitors stopped. **No dispatch is pending. Nothing is at risk.**

**The next chat starts from zero running jobs and a clean slate.** The task is
§6: re-specify the problem correctly and re-run. Read §0, §0b, §0c, then §6/6b/6e.

### ADDENDUM 2026-08-19 — two short jobs ran AFTER the stop, both finished
Both were user-authorised, both COMPLETED (~10 min/task), results downloaded,
queue empty again. **Nothing is pending.**
- Athena **134977** — negative-shift mirror, s = −51.68/−103.37 (dispatched by me
  in error: the user asked for the WIDE segment, not a negative shift; it ran to
  completion before it could be cancelled, and the data is kept and reported).
- Athena **134984** — the correct run: positive shift on the **wide** segment,
  s = +51.68/+103.37, via the new `shift_target="wide"` knob.
- Verdict + full table: **§6b**. One-line summary: every shift variant widens the
  TM mode; wide-target is strictly dominated by narrow-target; the shift axis is
  closed for the width problem.
- Files: `results_from_athena/tm_shift_c400/results/` (`_S52w`, `_S103w`,
  `dsh1Sm52sm52`, `dsh1Sm103sm103`).
- **UNCOMMITTED** (user permission needed): `bragg_device.py`,
  `simulation_config.py`, `experiment_card.py`, `runners/sweeps/sweep_spec.py`,
  `sim_helpers.py`, `runners/sweeps/tm_shift_c400.py`.

### ★TOOLING TRAPS FOUND 2026-08-19 (all cost time this session)
- **`getent` is a FALSE-NEGATIVE DNS probe** under Git Bash on Windows — it
  reported even `google.com` unresolvable on a live network. A watcher armed on
  it would never fire. Use PowerShell `Resolve-DnsName` / `Test-NetConnection`.
  `ssh`'s own "Could not resolve hostname" IS trustworthy.
- **`scene_snapshot.py` crashes** with `UnicodeEncodeError` on the two-device
  config (Greek delta → cp1252 console). Run it as
  `PYTHONIOENCODING=utf-8 python debug_fsp_compare/scene_snapshot.py --out ...`.
- **`deploy_athena.sh --results-no-fsp` HANGS forever (0 bytes) as a background
  task** — it blocks on an interactive prompt with no stdin. For a few files use
  `scp` directly; note remote brace expansion (`result_{a,b}.mat`) does NOT
  expand through scp, so loop over names.
- Full detail: `memory/project_getent_false_negative.md`.

## 0c. ★EVERY CORRECTION AND RETRACTION MADE ON 2026-08-18 (user asked for this)

Recorded so nobody rebuilds on a claim that was already withdrawn. Each of these
was **wrong at some point during the session and then corrected**:

| # | What was claimed | What is actually true |
|---|---|---|
| 1 | "Origin FWHM is 17.100 µm, best is 22.210, so FWHM grew +29.9% while σ grew +1.9%" | Both numbers came from a raw-line metric **and** a broken extraction. **Void.** Corrected values: origin 17.7005, best 20.3362 (+14.89%), all PVA. |
| 2 | "19.24 µm is the comb origin's width" | 19.24 is the **BARE** N=100 device, and it is **conformal** mesh. Not comparable to a campaign number. |
| 3 | "The raw-line metric is fine for RELATIVE change even if absolutes differ" | **No.** It reads first/last crossing of an absolute half-max on an *oscillating* line, so it moves with fringe contrast, not just the envelope. Deleted entirely. |
| 4 | "Coupled-mode theory says the true growth is only ~4%, so the 27% may be a metric artifact" | CMT was fitted/validated against the same void numbers. **All CMT deleted on user order.** The measured answer is +14.89%. |
| 5 | "The σ-guarded campaigns should be cancelled immediately" (given, then withdrawn 40 min later, then re-supported) | Correct in the end, but I gave it before the arbiter measurement existed. The rule: **do not restructure a program on a number from a metric you wrote the same day and have not cross-checked.** |
| 6 | "The lumopt2 scene builds a DIFFERENT DEVICE (5.3 nm λ offset)" + a config diff appearing to prove it | **Retracted.** The diff compared against **SweepSpec defaults** (it showed polarization TE and pitch 500 nm — impossible for a TM corr-325 study). The real cause is the **mesher split** (§5). |
| 7 | "The profile bug means we were sampling at the transverse BOX EDGE, so radiation contaminated it" | Overstated a mechanism. The monitor is a narrow 2D Z-normal plane (y span ≈1.5 µm), so row 0 sits ~0.75 µm off-axis in the evanescent skirt — **not** the box edge. What is certain is only that it was one off-axis row instead of the y-integral. |
| 8 | "Q_i ∝ L_mode² is the biggest Q lever" (long-standing project belief) | **MEASURED Q_i ∝ L^2.5-3.6**, and L **saturates** at ~19.7-20 µm, so length was never available at all. Corrected in the memory index too. |
| 9 | "κ ∝ corr, measured" (long-standing project belief) | Holds narrowly, **fails between 325 and 400 nm**: Q_i ∝ corr^−1.8, and L moved only 13% for a 23% corr change. |
| 10 | "Apodization is 1.75× more width-efficient than shifts" | Computed on void widths. Corrected: **2.9×** (0.0483 vs 0.0167 T/µm). |
| 11 | "seedB2 / bare failed" (implied by monitor `err=2`) | Both were **not** failures at that point — seedB2 exited 0; the errors were IGUM ssh flakiness. (bare *later* died on walltime, separately.) |
| 12 | "Both clusters are unreachable → possible outage" | **Local VPN drop.** Athena failed DNS; two raw-IP hosts timed out. Three Technion hosts do not fail together. Jobs were unaffected. |

**The meta-lesson, and it is the reason §0(b) exists:** almost every one of these
came from comparing numbers that were not comparable — two FWHM conventions, two
meshers, two extraction paths, a fitted surrogate vs a measurement. **Before
quoting any two numbers together, check they came from the same convention, the
same mesher, and the same pipeline.**

## 0b. ★★★IS THE INVERSE DESIGN BROKEN? — YES, SUBSTANTIALLY. BE HONEST ABOUT IT.

**Verdict: the optimization MACHINERY works; the PROBLEM SPECIFICATION was
wrong. The optimizer did exactly what it was told, and what it was told was
wrong.** That is the good news and the bad news together — the adjoint stack
does not need rebuilding, but essentially none of the campaign's design output
is usable as a delivered design.

**What is broken (all confirmed by measurement, 2026-08-18):**
1. **The width constraint never constrained anything.** The campaign controlled
   `sigma`, and sigma is BLIND to apodization: corrugation shaping moved the
   real FWHM +4.89% while moving sigma +0.001%. So on the dominant design axis
   the optimizer was effectively unconstrained in width.
2. **The width was measured through a broken extraction the whole time.**
   `profile_line` never integrated over y and always read one off-axis row.
   Every sigma, every FWHM, every sigma-anchor and wall calibrated from them:
   void. The guard was policing a number that was not the mode width.
3. **The objective intrinsically pays for widening.** T rises with Q_i and
   Q_i ~ L_mode². With (1) and (2), riding the width lever was not a bug in the
   optimizer's behaviour — it was the rational response to the stated problem.
   Result: the banked designs bought +15% to +27% width.
4. **A core design variable was mis-chosen for this polarization.** Tooth shifts
   were inherited from TE, where they cost ~0.5% width for +0.085 T. For TM they
   cost 3.6-7.2% for +0.006-0.012 T — about **100x worse** (section 6b). The
   campaign then spent much of its budget on that lever.
5. **Every fitted surrogate was calibrated on void data** (sigma-hat
   coefficients, the FWHM_A_* slopes, the sigma-neutral payback recipe). The
   sigma-neutral probe consequently FAILED to hold width — all four rows came
   out wider.
6. **The base scene may not even be the project's standard device** — 5.3 nm
   resonance offset, unresolved (section 5).

**What still works and does NOT need redoing:**
- The adjoint gradient stack: the 4-fix stack + the measured complex C-fix for
  the ~6.7° phase error (vec_error 11.40 -> 0.144). This was hard-won and is fine.
- All infrastructure: dispatch, per-study sweep lists, trust regions, cold-start
  resume, the `_final_fom` completion path, license/preemption handling.
- Every port quantity ever measured: T, lambda, Q_L, Q_i, R, loss.
- **The comb sub-programme is a genuine success**: +17.1% Q_i and +0.046 T at
  −0.35% width, with a clean with/without control. It is the one part of this
  work that is both real and on-spec.

**Therefore, do NOT:** report the T~0.96 designs as achievements, seed a new
campaign from them, or trust any width-related number produced before
2026-08-18. **Do:** treat the next phase as re-running the optimization against
a correctly specified problem (section 6), and expect the honest answer to land
well below 0.96 — the origin is 0.893 and the only validated fixed-width gain in
the programme is the comb's +0.046.

---

## 1. The device and the goal

Pi-shift Bragg grating, TM, corr-325 family, h350, pitch 516.83 nm, W800,
n 1.97/1.444, plus a 57-post SiN "comb" (an anti-radiation decoration).
Optimization surrogate = **N=100 periods/side**; production confirm is N≈165-169.

Goal: **maximise peak transmission T at the resonance WHILE HOLDING the spatial
mode width fixed.** The width is a hard spec — this is an acousto-optic detector
that senses at a fixed width; widening is forbidden and narrowing does not help
(memory: `project_acoustic_detector_width_spec.md`).

Why the width keeps trying to grow: T rises with Q_i, and **Q_i scales as
mode-length squared**. Widening is therefore the single most profitable move
available to any optimizer, and it will find it unless the constraint is exact.

---

## 2. ★★★THE BUG THAT DOMINATES EVERYTHING (found 2026-08-18, user caught it)

`lumopt2_design.profile_line` **never integrated over y**. It flattened the
(y, lambda) axes into one and indexed with the LAMBDA index — which is always
smaller than n_lambda — so it **always returned y-row 0**, for every design, in
every campaign, for the whole program.

`field_profile` is a 2D Z-normal monitor of y span 1.5*width_wide (~1.5 um,
see `bragg_device.py:1335-1339`), so row 0 sits ~0.75 um off the guide axis in
the evanescent skirt instead of across the mode.

**Measured impact:** the buggy path reports the uniform origin at
`fwhm_env = 16.7224 um` (Athena job 134299 task 0). The same device family
measured correctly is **19.24 um**. That is a **−13% error**, so the bug is not
cosmetic.

**Consequence: every `sigma_um` and every FWHM this engine ever logged is VOID** —
the audit rows, the sigma anchors and sigma-hat walls calibrated from them, the
shift ladder's sigma values, the sigma-neutral probe's widths, all of it.

**FIXED.** `profile_line` now replicates `sim_helpers.extract_and_process_field_profile`
step for step: pick resonance lambda -> |Ex|²+|Ey|²+|Ez|² -> **trapz over y** ->
crop to |x| <= n_side*pitch -> envelope through standing-wave peaks ->
floor-relative half-max.

**VALIDATED to machine precision, no GPU:** `eng.fwhm_env_of_line(x, I)` run on
stored `field_energy_density_1D` reproduces the stored `fwhm_m` exactly —
N100 c325 19.244767 um and N80 c325 18.393528 um, both matching to **7e-15 um**;
re-derived envelope matches stored `field_envelope_1D` to 5e-16 relative.

---

## 3. What is VALID and what is VOID

**VALID (port quantities — unaffected by the profile bug):**
- Every T, lambda_resonance, Q_L, Q_i, R, loss ever measured.
- The whole comb program: phase and pitch are sharp, radius and distance loose,
  post count irrelevant from 29 to 113 (n=29 halves the posts for free).
- The comb's benefit: bare-uniform T 0.88073 vs comb-uniform T 0.89265 at
  identical knobs = **+0.0119**, reproducing the A0 gate's +0.0105.
- Shift ladder's T ordering: T rises monotonically with tooth shift, **no
  interior optimum** — the width constraint is what stops it, not physics.
- All infrastructure: trust regions, cold-start resume, the `_final_fom`
  completion fix (verified on a real completion), license/preemption handling.

**VOID (anything width-related):**
- All sigma and FWHM values in every `*_evals.jsonl`.
- Any "in band" / "width-compliant" label — those meant "in the sigma band",
  measured wrongly.
- The trade line `T = 0.89265 + 0.01549*dFWHM`, the per-lever width
  efficiencies, the "seed B beats the trade line" ranking, the FWHM_hat ratios
  1.17-1.19, the 3-point factorial slopes. Kept in memory only as a record of
  reasoning — **re-derive all of them** from y-integrated data before citing.

**DELETED by user order — do not reintroduce (CLAUDE.md §8 "dropped stays dropped"):**
- the raw-line FWHM metric (`fwhm_raw_of_line` / `mode_fwhm_um`),
- the fitted `FWHM_A_RHO` / `FWHM_A_SHIFT` slopes,
- **all coupled-mode-theory (CMT) width modelling** ("delete all cmt use here
  it is not relevant"). It had been "validated" against the void numbers anyway.

**THE WIDTH OBSERVABLE IS NOW EXACTLY ONE THING:** `fwhm_env_of_line` ==
`post_processing`'s `fwhm_m`, by construction and verified.

---

## 4. Reference numbers you can trust (MEASURED from stored .mat this session)

| quantity | value | source |
|---|---|---|
| bare N=100 corr-325 mode FWHM | **19.244767 um** | `results_from_igum/tm_nladder_c325/results/result_N100_TM_avg_C325_Ybox8p0_Zbox8p8.mat` |
| bare N=80 corr-325 mode FWHM | 18.393528 um | same dir, N80 file |
| FWHM across 7 different boxes | 19.2411 - 19.2471 (spread **0.03%**) | `results_from_athena/tm_span_conv_c325/` |
| T_res across those same 7 boxes | 0.9091 - 0.9194 (spread **0.010**) | same |

Two rules follow, both important:
- The width metric's **noise floor is 0.03%**, and width is box-INSENSITIVE. So
  holding width to a fraction of a percent is measurable and realistic.
- **T is ~30x more numerics-sensitive than width.** Never compare absolute T
  across different boxes/mesh; always use an in-study control at identical
  numerics (CLAUDE.md §2).

Comb-decorated origin should sit near 19.17 um (A0 gate value) — comb is
width-neutral to ~0.4%.

---

## 5. ★IN-FLIGHT JOBS — HOW TO READ THEM WHEN THEY LAND

State at 13:05 IDT. `ssh evyatarrubin@athena.technion.ac.il` (host-first form
only; strip the banner with `grep -vE "post-quantum|openssh|may need"`).

| job | tasks | what it is | trust? |
|---|---|---|---|
| 134032 | stage-4 campaign | long lumopt2 run, 12h+, FLAT progress | widths VOID (pre-fix) |
| 134299 | 3 (t0 done, t1 running, t2 pending) | FWHM audit: origin/best/noshift | **t0,t1 VOID** (pre-fix); **t2 valid** (starts after the fix was deployed) |
| 134334 | 0-2 | width recovery from stored .fsp: origin, best, noshift | **VALID** |
| 134335 | 3-6 | width recovery: d+20, d+40, d+60, d+80 | **VALID** |
| IGUM 55801 | bare campaign | still running | widths VOID |

134334/134335 re-read the stored forward `.fsp` files and recompute width with
the corrected pipeline — **no re-solving**, ~2 min/case. They are the answer.

**Read them with:**
```bash
ssh evyatarrubin@athena.technion.ac.il "grep -h 'fsp_width' ~/bragg_sim_athena/jobs/logs/lum_*13433[45]*.out"
```
Each line prints: `T`, `lam`, `FWHM um`, `sigma um`, and the 19.2448 reference.
Each task also writes `<out>/fspw_*/fspw_*_profile.npz` (x_um, I, fwhm_um,
sigma_um) — keep these, they make every future width question free.

### ★★★THEY LANDED — THE CORRECTED WIDTHS (MEASURED, jobs 134334/134335)

All six from ONE corrected pipeline, so the ratios are apples-to-apples:

| design | T | FWHM um | vs origin | sigma um | vs origin |
|---|---|---|---|---|---|
| origin  (uniform+comb, no shifts) | 0.89265 | 17.7005 | — | 17.2518 | — |
| noshift (apodized+comb, no shifts)| 0.93450 | 18.5664 | **+4.89%** | 17.2520 | **+0.001%** |
| best BEST_T9635 | 0.96404 | 20.3362 | **+14.89%** | 17.5221 | +1.567% |
| d+20 | 0.96587 | 20.6170 | +16.48% | 17.5472 | +1.712% |
| d+40 | 0.96673 | 20.9619 | +18.43% | 17.5834 | +1.922% |
| d+60 | 0.96632 | 21.4767 | +21.33% | 17.6300 | +2.192% |

**VERDICT: the width really did grow ~15-21% on every T~0.96 device.** Not the
+27% the broken metric claimed, not the +4% CMT claimed — about +15% for the
banked best. The gains were substantially bought with mode width.

**★THE CORE DIAGNOSIS SURVIVES THE BUG FIX, and is now airtight:** the
corrugation apodization alone moved FWHM **+4.89%** while moving sigma
**+0.001%** (17.2518 -> 17.2520, identical to four decimals). sigma is not
merely insensitive to apodization — it is *blind* to it. Over the full change
sigma under-reports the width growth by ~10x (+1.57% vs +14.89%).

**★Corrected width efficiency:** corrugation apodization **0.0483 T/um** vs
tooth shifts **0.0167 T/um** — apodization is **2.9x more width-efficient**
(the earlier void estimate said 1.75x; the corrected gap is larger). Spend width
budget on apodization, not on shifts.

**T saturates:** d+40 is the peak at 0.96673; d+60 is lower (0.96632) while
still 2.9% wider. Past d+40 you are paying width for nothing.

(d+80 completed too: T 0.9653, FWHM **22.4013 um**, +26.6% — the widest of all.)

### DID THE COMB ALONE CHANGE THE WIDTH? NO — measured, clean control

`comb_q3db` contains a with/without pair at N=165, IDENTICAL numerics, both
through the SweepSpec/`post_processing` pipeline:

| case | fwhm_m um | dFWHM | T_res | Q_i |
|---|---|---|---|---|
| **no comb** | 19.9702 | — | 0.4906 | 46,499 |
| **winner comb** (r80, 57 posts, d1.9) | 19.9001 | **−0.35%** | 0.5361 | **54,457** |
| comb variant (x-14604..15412) | 19.9601 | −0.05% | 0.5283 | 52,997 |
| comb variant (x-14732..15004) | 20.0468 | +0.38% | 0.4371 | 38,784 |

**The comb is width-neutral: −0.35%, against a 0.03% noise floor.** It buys
**+17.1% Q_i** (46,499 -> 54,457) and +0.046 T for essentially no width. That
makes the comb the ONLY lever in the program measured to deliver a large gain
at constant width — the grating levers cost +4.9% (apodization) and +9.5%
(shifts) of width for theirs. Note the third row: a mis-placed comb both widens
AND loses Q_i, so comb PHASE is what matters, consistent with the comb study.

### ★★RESOLVED 2026-08-18 — IT IS A **MESHER** DIFFERENCE, NOT A GEOMETRY ONE

**The campaign and every stored SweepSpec study use DIFFERENT MESHERS:**
- `lumopt2_design.py:745` — `setnamed("FDTD","mesh refinement","precise volume average")`
- `bragg_device.py:780`  — `set("mesh refinement","conformal variant 0")`

The engine overrides to PVA deliberately (comment at :739-745: conformal
variant 0 **staircases the grid-aligned TOOTH edges**, while the comb cylinders
meshed fine). `CampaignSpec.scan_center_nm = 1564.21` even documents it:
*"MEASURED at PVA (job 132654) — the precise-volume-average mesher shifts λ
+5.2 nm vs the family's 1559.0"*.

That accounts for **both** discrepancies quantitatively:
| | PVA (campaign) | conformal (stored) | Δ |
|---|---|---|---|
| λ_res | 1564.276 | 1559.006 | **+5.27 nm** (documented: +5.2) |
| mode FWHM | 17.7005 | 19.2448 | **−8.0%** |
(The comb −0.35% and the box 0.03% are far too small to matter here.)

**CONSEQUENCES — these matter for every number in this file:**
1. **Never compare a campaign width to a stored width or to the ~20 µm spec.**
   The spec, the 19.24 µm anchor and the 19.91 µm production value are all
   **conformal**; every campaign width is **PVA**, which reads ~8% narrower.
2. **Ratios remain valid within each pipeline** — the §5 table (origin →
   best = +14.89%) is all-PVA and stands; the comb control (−0.35%) is
   all-conformal and stands.
3. Rough conversion from the one paired device: **PVA ≈ 0.92 × conformal**. So
   the campaign's best, 20.34 PVA, is ≈22.1 conformal — against a ~20 µm spec
   and a 19.91 µm production value, i.e. **~10% over spec even after the
   conversion.** The over-width conclusion is unchanged.
4. ★**Which mesher is right — PRESUMPTION FLIPPED 2026-08-21 (researched,
   Ansys KB + Farjadpour/Kottke/Johnson):** presume **conformal variant 0**
   is the better reference. The docs recommend CT0 for dielectric
   high-contrast structures; the only documented staircase reversion is >2
   materials/cell (NO public backing for "CT0 staircases grid-aligned tooth
   edges" — that engine comment is uncorroborated); PVA is documented as a
   GRADIENT-SMOOTHNESS tool ("naive smoothing", first-order, known-sign bias
   matching our +5.3 nm red-shift). The ~20 µm spec stays conformal-defined.
   Cheap arbitration (parked for user): single-period Bloch-cell λ ladder,
   both meshers × dx 50/35/25/17.5/10 (~minutes/run) + one full-device
   PVA-25 vs stored conformal-35 confirm. Full digest: mesher memory file.
5. Any future cross-pipeline comparison must state its mesher.

### (superseded) earlier note — the lumopt2 scene is NOT the stored N=100 device

The lumopt2 "origin" reads **17.7005 um at lambda 1564.276**, while the stored
bare N=100 corr-325 anchor reads **19.2448 um at lambda 1559.006**. Ruled out:
- **crop**: the recovered profiles span ±51.66 um vs the stored ±51.8, and
  cropping the stored profile anywhere ≥51.68 um changes nothing (verified;
  the floor-relative FWHM IS very crop-sensitive below ~45 um, but neither
  dataset is truncated there).
- **the comb**: measured above at −0.35%, nowhere near −8%.
- **box size**: FWHM varies 0.03% across seven boxes.
**The smoking gun is the resonance: 1564.276 vs 1559.006, a 5.3 nm offset**,
where the comb moves lambda by only 0.01 nm. So the lumopt2 base scene builds a
genuinely DIFFERENT device from the standard builder's N=100 corr-325 — not a
measurement artefact. Cavity length, the free/frozen tooth boundary, or an
`avg`-width convention are the candidates.
**Consequence:** ratios WITHIN the lumopt2 set are sound (one scene, one
pipeline). Absolute lumopt2 widths must NOT be compared to the ~20 um spec or
to any stored SweepSpec number until this is explained.
**THE CHECK:** build the lumopt2 uniform seed and the standard N=100 device
locally (`save_fsp`, build-only, <1 min each) and diff the geometry — cavity
length first. This is free and needs no GPU.

---

## 6. THE FIX — projection, not a penalty band

**This section is the actual task.** Per section 0(a), the +15-27% width growth
is the defect. Everything here exists to recover T at the ORIGINAL width.


A band has slack and an optimizer always spends slack; a projection has none.
Since T rises monotonically with width with no interior optimum, the answer
always lies **on** the constraint surface, so you never explore the interior —
you only need the ability to **return** to it.

**The scheme:**
1. Pick ONE scalar width knob — the amplitude of the apodization deviation, or
   the global shift scale. Both move width monotonically.
2. The optimizer proposes a design.
3. **Bisect that one knob on the MEASURED `fwhm_env` until it lands on the
   target width.** (Noise floor is 0.03%, so this converges cleanly; 2-3 sims,
   fewer once you have a local secant estimate.)
4. **Only then** score its transmission.

Widening now cannot pay — it is undone before anyone counts the transmission.
No width model is needed anywhere, only the measurement, which is now correct.

**Rule that follows: never compare two designs at different widths.** T_A vs
T_B is meaningless unless FWHM_A == FWHM_B. Everything reported should be
re-trimmed.

**USER CONSTRAINT (verbatim intent, 2026-08-18):** the baseline's UNIFORM
corrugation stays 325 nm. Re-matching it so that a ~20 um target lands
conveniently **does not count** as a fix — "we do not want to cheat in any way".
The apodization DEVIATION is a design variable; the baseline is not.

**Guards already shipped:** `CampaignSpec.fwhm0_um` — when set, accepted-best
designs must keep `fwhm_env_um / fwhm0_um` inside the deadband (raises
`WidthTrip`), and `_best_from_log` filters restarts and final selection on it.
Default `None` = legacy sigma-only, so live campaigns are requeue-safe.
Every eval also saves its profile to `<out>/profiles/*.npz`.

**The one physics lever with real headroom at fixed width:** the pi-shift mode
is an exponential with a cusp at the centre, and that kink is the natural
radiation source. Smoothing it must be paid for by stronger confinement just
outside, or the mode simply lengthens — i.e. dip-at-the-cusp plus
overshoot-outside, which is what seed B's profile is.
`runners/lumopt2_design/rho_neutral_shape.py` is **written, smoke-tested, NOT
dispatched**: 4 rows at rho = 1.000000 EXACTLY (mean corrugation pinned to the
baseline 325 nm), amplitudes a = 0.5/1.0/1.5/2.0, shifts zero. It asks whether
redistributing a FIXED corrugation budget buys T at constant width. Needs user
approval (new study, CLAUDE.md §8).

---

## 6b. ★THE FIRST EXPERIMENT TO RUN — do tooth shifts help at CONSTANT width?

> ### ★★★UPDATE 2026-08-19 — THE SHIFT AXIS IS NOW CLOSED FOR THE WIDTH PROBLEM
>
> The user asked: does the shift behave differently if it shortens the **wide**
> segment instead of the narrow one? It had never been possible to ask — the wide
> segment was hard-coded to `half_pitch` at both arm build sites. Added the
> `shift_target` knob (`"narrow"` = legacy default, `"wide"` = new) and MEASURED
> it (Athena **134984**), plus the strict negative mirror (Athena **134977**).
>
> **ANSWER: every version of the tooth shift widens the TM mode. There is no
> width-neutral variant.** Do not spend more GPU looking for one.
>
> corr-400 N=80, box y6.8/z8.8, mesh "optimization", ports-only; s=0 anchor =
> stored `asym_dw_study/results/result_N80_TM_avg_Ybox6p8_Zbox8p8.mat`:
>
> | variant | s nm | lambda | T_res | dT | Q_i | mode um | d_mode | dT per %w |
> |---|---|---|---|---|---|---|---|---|
> | control | 0 | 1558.617 | 0.8864 | — | 22404 | 15.532 | — | — |
> | narrow | +51.68 | 1558.946 | 0.9038 | +0.0174 | 27051 | 15.707 | +1.12% | +0.0155 |
> | narrow | +103.37 | 1559.186 | 0.9179 | +0.0315 | 32033 | 16.260 | +4.69% | +0.0067 |
> | narrow | +155.05 | 1559.266 | 0.9279 | +0.0415 | 36692 | 16.983 | +9.34% | +0.0044 |
> | narrow | +206.73 | 1559.196 | 0.9335 | +0.0471 | 39921 | 16.947 | +9.11% | +0.0052 |
> | **wide** | +51.68 | 1558.796 | 0.9006 | +0.0142 | 26090 | 15.783 | +1.62% | +0.0088 |
> | **wide** | +103.37 | 1558.876 | 0.9067 | +0.0203 | 27908 | 16.404 | +5.61% | +0.0036 |
> | neg | −51.68 | 1558.276 | 0.8665 | −0.0199 | 18869 | 15.840 | +1.98% | −0.0100 |
> | neg | −103.37 | 1558.036 | 0.8457 | −0.0407 | 16027 | 16.380 | +5.46% | −0.0075 |
>
> **Wide-target is STRICTLY DOMINATED**: less T *and* more widening than
> narrow-target at both rungs (~1.8x worse dT per % width). Same sign and shape on
> every observable (T, mode, Q_i, lambda) ⇒ one mechanism at two strengths, not a
> new lever. The campaign's narrow-target basis was already the better half; the
> shift bounds do NOT need reopening.
>
> **CONFIDENCE:** the +103.37 narrow-vs-wide gap is 0.0112 ≈ **6.2x** the dx=50 nm
> jitter floor (0.0018) — solid. The +51.68 gap is 0.0032 ≈ 1.8x — consistent but
> not independently decisive. The verdict rests on the +103 pair.
>
> **★THE TRAP (cost a wrong prediction — do not repeat it):** the cavity absorbs
> `2*sum(shift)` whichever segment is shortened, so **cavity lengthening is
> COMMON-MODE** between narrow and wide and is the dominant T lever here; the
> duty-cycle/<n_eff> term is only the ~36% DIFFERENTIAL. I predicted wide-target
> would fall BELOW control on the <n_eff> argument; it GAINED, because I had
> neglected the common-mode term. In this builder the "shift" bundles THREE
> changes — duty cycle + local period + cavity length — always separate them
> before predicting a sign. (Attempted to prove the split from stored data:
> impossible. The only stored cavity-detuning rows are ±20/±40 nm on W1000-W1100
> cavities in `results_from_athena/tm_center_completion/`; we would need −206.7 nm
> at our avg-width cavity. And a bare cavity override is not a clean control
> anyway — the shift REDISTRIBUTES length while an override ADDS it. The 64/36
> split is INTERPRETATION, not measurement.)
>
> **THE <n_eff> SIGNATURE IS REAL** and points as the light-cone argument says:
> lambda rises +0.569 nm for narrow vs only +0.259 nm for wide at s=103.37, and
> FALLS for negative shifts — i.e. lengthening the wide fraction raises <n_eff>,
> widening the (n_eff − n_clad) light-cone margin that limits TM radiation.
>
> **WHY NO SHIFT CAN EVER NARROW (the general no-go):** envelope ~ exp(−∫q dx)
> with q = sqrt(kappa² − delta²) ≤ kappa. Any local detuning — either duty-cycle
> sign, either segment — only lengthens the decay. Narrowing REQUIRES raising
> kappa near the centre, which no shift does. The only routes that raise kappa
> are (a) more corrugation at the centre (the excluded "cheat", and the exact
> opposite of the dip that buys T), (b) a second scattering mechanism = **the
> comb** (the ONLY measured width-negative lever: −0.35% width at +17.1% Q_i, and
> never optimised for width), (c) a perturbation TM is intrinsically strong at =
> a TOP-SURFACE corrugation, since at a sidewall TM's E_z is tangential (plain
> Δε, no enhancement) while at the top surface it is normal — the Johnson
> boundary theorem already coded at `lumopt2_design.py:625-651`. (c) costs a
> second etch depth, so it is a fab question, not a physics dead end.
>
> **CODE ADDED (uncommitted at handoff time):** `shift_target` through
> `bragg_device.py` (both arm loops + validation), `simulation_config.py`,
> `experiment_card.py` `_CARD_FIELD_MAP`, `runners/sweeps/sweep_spec.py`, and a
> `"w"` marker in `sim_helpers.py generate_file_tag` — **without that marker the
> wide rows overwrite the stored `_S52`/`_S103` narrow results.** Runner:
> `runners/sweeps/tm_shift_c400.py`. VERIFIED: all 6 committed scene snapshots
> byte-identical (default path provably unchanged); the new path differs from
> narrow-target in exactly 4 objects (innermost tooth, both arms) with spans
> 258.415/155.045 nm swapping; cavity and total grating length identical.
> Full detail: `memory/project_shift_target_sign_test.md`.

**Status: NEVER MEASURED.** Every shift datapoint in this program was taken at a
different (larger) width, so "shifts raise T" has always meant "shifts raise T
by widening the mode". Whether they add anything at FIXED width is open.
(Still true as written — the 2026-08-19 run above settled WHICH SEGMENT, not the
constant-width question. But note it lowers the priority: since no shift variant
is width-neutral, any constant-width test must pay the width back from another
knob, and §6e's see-saw + the comb are the better places to spend the GPU.)

### ★★★WHAT ACTUALLY NARROWS THE TM MODE — the measured inventory (2026-08-19)

Found by scanning stored studies and comparing every row to **its own in-study
control**. These are the ONLY width-reducing effects this program has ever
measured for TM:

| effect | study dir | d_width | dT | verdict |
|---|---|---|---|---|
| **air trench** (rect L84 um x W800) | `air_trench_w1050` | **−0.85%** | **+0.0157** | WIN-WIN |
| **cavity width W1250** | `cavity_width_ladder` | **−0.29%** | **+0.0178** | WIN-WIN |
| cavity width W1400 | `cavity_width_ladder` | −0.74% | −0.0061 | ~T-neutral |
| cavity **hourglass** pinch 150 | `inner_shape_study` | **−1.05%** | −0.0280 | costs T |
| cavity hourglass pinch 75 | `inner_shape_study` | −0.50% | −0.0147 | costs T |
| **comb** (corr-325 N165) | `comb_q3db` | −0.35% | Q_i +17.1% | WIN-WIN |

Controls: `cavity_width_ladder`/`inner_shape_study` → in-dir
`result_N80_TM_avg_Ybox6p8_Zbox8p8.mat` (15.532 um, T 0.8864); `air_trench_w1050`
→ in-dir `..._ff.mat` (15.622 um, T 0.9218).

**★THE PATTERN — narrowing lives in the CAVITY, never in the teeth.** Every one
is either a change in/near the cavity (where the mode peaks) or an added
scatterer. NOTHING done to the teeth ever narrowed: apodization, shifts (both
duty-cycle signs, both segments), tooth shapes (ellipse/tri/wedge +1.6% to
+9.8%) and the see-saw all WIDEN. That is the general no-go made visible —
tooth-level detuning only lengthens the decay; narrowing needs kappa raised near
the CENTRE. Sign detail: cavity **hourglass narrows, barrel widens** (+0.34% /
+0.58%) — a clean antisymmetric pair. Cavity width is NON-monotonic: W1050
+0.59%, W1150 +0.18%, W1250 −0.29%, W1400 −0.74%.

**CAVEATS — do not overstate:** (1) all corr-400 N=80 (~15.5 um modes), NOT the
corr-325 production family — porting UNVERIFIED, the comb row is the only
corr-325 point; (2) all are **≤1%** against a **+15%** problem — counterweights,
not a solution; (3) stacking unmeasured, and modularity in this program has
sign-inverted under apodization before; (4) no dx=50 width jitter floor exists
for this family, so the −0.29% row is the one most likely near noise.

Full detail + sources: `memory/project_tm_width_reducing_levers.md`.

What the corrected data says INDIRECTLY (all MEASURED, one pipeline):
- apodization lever (origin -> noshift, zero shifts): +0.0419 T for +0.866 um
  = **0.0483 T/um**
- shift lever (noshift -> best, same corrugation): +0.0295 T for +1.770 um
  = **0.0167 T/um**
=> per micron of width spent, shifts are **2.9x worse than apodization**. So at
a fixed width budget you would rather spend it on apodization. That is
suggestive, NOT conclusive — it compares marginal rates at different points on
two curves, and neither curve's shape is known.
- The sigma-neutral probe (d+20..d+80) TRIED to add shifts while paying the
  width back with corrugation. It failed: every row came out WIDER (20.62,
  20.96, 21.48, 22.40 vs best 20.34), because the payback was sized with the
  sigma surrogate — which is blind to apodization AND was computed through the
  broken profile path. Those rows do not answer the question.
- Physics says shifts SHOULD help in principle: they are a distributed phase
  shift, spreading the abrupt pi discontinuity over several periods, which is
  the classic gentle-confinement trick for cutting light-cone radiation. So do
  not assume the answer is "no".

### ★★★TE vs TM: THE SHIFTS ARE NEARLY FREE FOR TE AND EXPENSIVE FOR TM

MEASURED, stored study `results_from_athena/tm_te_shift/results/`, N=80, each
polarization against its OWN S=0 baseline, all via the trusted `post_processing`
pipeline (user recalled this and was right):

| shift nm | TE fwhm_m | TE dFWHM | TE T | TM fwhm_m | TM dFWHM | TM T |
|---|---|---|---|---|---|---|
| 0   | 15.2164 | — | 0.8594 | 17.8992 | — | 0.9739 |
| 50  | 15.0689 | **−0.97%** | 0.9096 | 18.0202 | +0.68% | 0.9768 |
| 100 | 15.2991 | **+0.54%** | **0.9439** | 18.5387 | +3.57% | 0.9801 |
| 150 | 15.6470 | +2.83% | 0.9344 | 19.1702 | +7.10% | 0.9829 |
| 200 | 15.8074 | +3.88% | 0.9057 | 19.1813 | +7.16% | 0.9855 |

- **TE:** +0.0845 T at **+0.54%** width (S=100) => **~1.02 T per um**. The shifts
  are essentially FREE. TE also has a clear INTERIOR OPTIMUM at S≈100 (T falls
  at 150 and 200), and at S=50 the mode even NARROWS.
- **TM:** +0.0062 T at +3.57% width (S=100), +0.0116 at +7.16% (S=200)
  => **~0.0097 T per um**. Monotonic, no interior optimum in range.
- **TE shifts are ~100x more width-efficient than TM shifts.**

**This is the likely root of the whole problem:** the tooth-shift lever was
inherited from TE work where it costs almost nothing, and applied to TM where it
is the single most width-hungry knob available. It is consistent with the
current campaign's own TM numbers (noshift -> best: +0.0295 T for +9.5% width)
and with the corrected finding that for TM, apodization is 2.9x more
width-efficient than shifts.

CAVEATS: N=80, an older study, and the TE and TM devices differ in geometry
(resonances 1570.7 vs 1523.6 nm), so this is NOT a controlled A/B on
polarization alone — but each polarization is internally controlled against its
own S=0 row, so the RELATIVE responses are sound.

**Implication for TM:** expect shifts to be the wrong lever. Do not delete them
on this evidence alone (N=80, different family), but weight the experiment below
accordingly and consider testing S≈50 too, where TM is cheapest (+0.68%).

**THE DECISIVE EXPERIMENT (cheap, ~3-5 forwards, no new physics):**
Take `best` (apodized + shifts 2Ss=130.6, FWHM 20.3362) and raise the
corrugation — **bisecting on the MEASURED `fwhm_env`, never on a surrogate** —
until it lands on **18.5664 um**, which is `noshift`'s width. Then compare its T
against `noshift`'s **T 0.93450**, a shift-free design at exactly that width.
- T > 0.9345 -> shifts DO add something at constant width; keep them, and the
  design axis is shifts + corrugation traded against each other.
- T <= 0.9345 -> shifts contribute nothing that apodization does not do better;
  drop them (they are also the width-hungrier lever) and optimize the
  corrugation SHAPE alone, which is what `rho_neutral_shape.py` explores.

Run the same re-trim against the ORIGIN's width (17.7005 um, control T 0.89265)
for the on-spec version of the same question.
This is the cleanest single answer available and it settles the shift axis.

---

## 6c. ★★★WHY TM ≠ TE — physics, from our own archive + a literature sweep (2026-08-18)

### (i) WE ALREADY FALSIFIED THE DISTRIBUTED SHIFT FOR TM, IN 2026-07
`memory/project_loss_exploration_chain.md`, "FALSIFIED / CLOSED routes":
> *"Distributed pi-shift (job 117530): ALL variants +21..+39% loss, fwhm also
> widens — each shifted gap is its own radiating kink; lumped shift optimal."*
The lumopt2 campaign then adopted **per-tooth shifts over 25 teeth** — i.e. a
distributed pi-shift — as a core design variable. That is a repeat of a closed
negative result. (Not a perfect contradiction: 117530 used much larger shifts on
a corr-400 cavity-scope device, and the campaign's small shifts DID raise T. But
it was a strong recorded prior that was not consulted.)

### (ii) THE QUANTITATIVE REASON: TM HAS HALF THE k-SPACE MARGIN
A first-order grating **cannot radiate**: at Bragg, beta = K/2, so every order
sits at K(1/2 - m), i.e. |k| >= n_eff*k0 > n_clad*k0 — all evanescent. Radiation
exists ONLY where periodicity is broken (the defect), and its strength is the
mode envelope's Fourier weight **inside the cladding light cone**
(Englund/Fushman/Vuckovic, Opt. Express 13, 1202 (2005)).
The margin to that light cone is **dk = (n_eff - n_clad)*k0** — DERIVED from our
own stored resonances:

| | pitch | lambda_B | n_eff | n_eff - n_clad | dk (rad/um) | smoothing length 1/dk |
|---|---|---|---|---|---|---|
| TE | 500 nm | 1570.7 | 1.5707 | 0.1267 | **0.507** | **1.97 um** |
| TM | 500 nm | 1523.6 | 1.5236 | 0.0796 | 0.328 | 3.05 um |
| TM anchored | 516.83 | 1558.6 | 1.5079 | 0.0639 | **0.258** | **3.87 um** |

**TM must smooth a feature over ~2x the length TE does before it stops
radiating.** Against a ~20 um budget that is 10% (TE) vs 20% (TM). This is the
cleanest single-number statement of the whole problem and it comes from our own
measurements.

### (iii) PUBLISHED AND DIRECTLY APPLICABLE: THE TM STOP-BAND COLLAPSES IN THIN CORES
Zhang, McCutcheon, Burgess, Loncar, *"Ultra-high-Q TE/TM dual-polarized photonic
crystal nanocavities"*, Opt. Lett. **34**, 2694 (2009), arXiv:0905.3854:
> *"Decreasing thickness causes the width of the TM bandgap to sharply decrease,
> whereas the width of the TE bandgap remains almost constant."*
> *"The narrowed TM bandgap results in a reduced Bragg confinement, which
> increases the transmission losses through the Bragg mirrors."*
**Our core is 350 nm** — exactly that regime. Weaker TM confinement per period =>
deeper mirror penetration => longer mode => every length-based lever costs more.
(Same paper: with a thick enough core TM is NOT worse — Q_TM 2.4e6 > Q_TE 1.2e6.)

### (iv) ★THE TOOTH SHIFT IS THREE PERTURBATIONS, NOT ONE — AND WE HAVE THE RECEIPT
`bragg_device.py:226-229`: a positive shift SHORTENS THE NARROW GAP and the
cavity absorbs 2*sum(shift). So one "shift" simultaneously applies:
1. the intended local **phase advance**;
2. a local **duty-cycle change** (wide tooth keeps its half-pitch while the
   period shrinks => D rises above 0.5 => local kappa changes as sin(pi*D));
3. a local **DC-index change** (more high-index material per period) => a local
   **Bragg detuning / chirp**.
**Proof that (3) is real and large in our device:** `shift_ladder.py:23-24`
recorded **lambda moving +1.6 nm per +374 nm of 2*Sigma_s**. A pure phase
redistribution cannot move the resonance at all. And the DC-index term scales
like 1/(n_eff - n_clad), which is **~2x larger for TM** by the table in (ii).
So the TE/TM shift comparison is CONFOUNDED: we are not comparing the same
perturbation across polarizations.

### (v) SO IS THERE A "TM VERSION"? — the useful distinction is TRANSVERSE, not wide-vs-narrow
The productive split is **longitudinal (segment lengths) vs transverse (widths)**,
not wide-segment vs narrow-segment. Our own corrected numbers already say
transverse wins for TM (apodization 0.0483 vs shifts 0.0167 T/um = **2.9x**), and
the low-index 1-D-cavity literature apodizes **transversely** as standard:
- McCutcheon & Loncar, Opt. Express **16**, 19136 (2008) — SiN n~2.0, 1-D
  confinement, hole taper; Q 2.3e5 at V ~ 0.55(lambda/n)^3.
- *High-Q asymmetrically cladded SiN 1D photonic crystal cavities*, Nanophotonics
  (2022), PMC9412843 — **quadratically tapered nanostick WIDTHS**, no pitch change.
- Quan & Loncar, Opt. Express **19**, 18529 (2011) — the deterministic recipe:
  **constant pitch, quadratic taper of scatterer size => linear mirror-strength
  ramp => Gaussian envelope**, which is the Fourier-optimal shape.
★**A length-neutral AND index-neutral kappa knob already exists in the builder**:
`wall_phase_offset_deg` (misaligned sidewalls, kappa = kappa0*sin(pi*dP/Lambda);
`bragg_device.py:934`; cf. Wang et al., Opt. Lett. **39**, 5519 (2014); Jiang et
al., Micromachines **15**, 666 (2024)). Ramping it over the inner periods gives a
Gaussian kappa taper with **zero change to average width, average index, or
device length** — it sidesteps confound (3) entirely.
**CAVEAT, verified in code (`bragg_device.py:478-492`):** it is currently a
GLOBAL uniform knob that RAISES ValueError if combined with apodization,
per-tooth arrays, or tooth shifts, and it also breaks the y-mirror plane so
`use_y_symmetry` must be OFF (2x cost). Making it a per-tooth taper is an ENGINE
CHANGE, not a config change. Scope it before promising it.

### (vi) DUTY CYCLE IS NOT THE ANSWER (and here is why)
D = 0.5 maximises kappa (sin(pi*D)) and kills even harmonics — but per (ii), for
a FIRST-ORDER grating no harmonic can reach the light cone anyway, so harmonic
suppression buys **nothing** radiatively. Duty is a kappa + DC-index lever, not a
radiation lever. No polarization-dependent duty optimum found in the literature.

### (vii) Q vs LENGTH — our assumed scaling may be wrong
From the light-cone integral, an exponential (cusped) envelope gives
**Q_rad ~ dk/gamma ~ L*dk — LINEAR in L**, not quadratic. And for SiN slow-light
nanobeams, Zhan et al., APL Photonics **5**, 066101 (2020) MEASURED stored energy
**cubic in cavity length**. We have been assuming Q ~ L^2; it is not established.
**At FIXED length the literature offers exactly two levers**: (a) termination /
mode-profile matching (a kappa taper, not a length change) — Lalanne & Hugonin,
IEEE JQE **39**, 1430 (2003); Sauvan et al., PRB **71**, 165118 (2005); and
(b) **radiation recycling / cancellation** — Lalanne, Mias & Hugonin, Opt.
Express **12**, 458 (2004); Kazarinov & Henry, IEEE JQE **21**, 144 (1985).
Our comb is already (b).

### (viii) ★★A TESTABLE MECHANISM FOR THE COMB (best new idea from the sweep)
DERIVED from our own banked geometry at lambda ~ 1566 nm:
- guided Bloch wavenumber beta = pi/0.51683 = **6.0786 rad/um**
- cladding light cone n_clad*k0 = **5.7940 rad/um**
- comb reciprocal vector K_c = 2*pi/0.53098 = **11.8329 rad/um**
- **beta - K_c = -5.7543**, and |−5.7543| **< 5.7940** => **PROPAGATING in the
  cladding**, at ux = cos(theta) = **0.9932**, theta ~ **6.7deg — grazing**.
- The pitch at which this order goes evanescent is **529.2 nm**. The optimizer
  landed on **530.98 nm — 1.8 nm on the propagating side.**
=> **HYPOTHESIS:** the comb is not a mirror or an index cladding. It is a second,
laterally offset, longitudinally phased radiator, tuned to emit into the SAME
grazing lobe the defect leaks into, ANTI-PHASE — i.e. Kazarinov-Henry
cancellation / Noda's double-lattice (Yoshida et al., Nat. Mater. **18**, 121
(2019)) transplanted to a ridge waveguide.
**Two independent cross-checks that this is not numerology:**
- it predicts our MEASURED sensitivity ranking exactly — longitudinal phase turns
  over 2*pi per **1.03 um** (~2 grating periods) => *sharp*; standoff turns over
  2*pi per **9.2 um** => *loose*; radius sets amplitude only => *loose*. Our comb
  study measured precisely "phase and pitch sharp, radius and distance loose".
- the angle matches the independently measured leak: memory records the
  innermost-tooth leak as *"Lorentzian^2 at grazing ux 0.99"*; this predicts 0.993.
**DISCRIMINATING EXPERIMENTS (cheap):** (1) comb-pitch scan 522/526/529/531/535/540
nm — a sharp feature within ~2 nm of 529 confirms it, a smooth monotone curve
refutes it; (2) comb longitudinal phase scan over one grating period — predict
Q_i oscillates with period 1.03 um and crosses BELOW the no-comb control (we
already have one such point: the mis-placed variant, Q_i 38,784 < 46,499, which
is our strongest existing evidence for coherent interference); (3) standoff scan
1-4 um — predict nearly flat (period ~9 um), which distinguishes this from a
mirror-at-standoff picture (which would give ~542 nm fringes).
**Also predicted:** the comb should be much WEAKER for TE (for TM, E is along the
posts' 350 nm axis => maximal polarizability; for TE it is transverse).
★This configuration appears to be **absent from the literature** — if the
mechanism confirms, it is publishable.

### (ix) WHAT THE LITERATURE DOES *NOT* SUPPORT
No peer-reviewed source measures radiation loss vs phase-shift distribution — DFB
models are 1-D and cannot radiate. Our *"each shifted gap is its own radiating
kink"* reading has **no literature support and no refutation**. The only source
addressing it at all (US Patent 11,125,935, Honeywell) claims the OPPOSITE sign
(distributed shift lowers loss) — consistent with our TE result, not our TM one.
Treat the kink picture as our own hypothesis, not established physics.

---

## 6d. ★★★Q-vs-LENGTH LITERATURE — the trade-off is NOT fundamental, and we have a taper problem

Second literature sweep, 2026-08-18. Full citation list in the session record; the
load-bearing items only, here.

### (i) THERE IS NO BOUND. Four groups beat "Q costs length", with numbers.
- **Watts, Johnson, Haus, Joannopoulos, Opt. Lett. 27, 1785 (2002)** —
  *"neither a complete photonic bandgap NOR a trade-off in mode localization for
  Q is required … our V is roughly independent of Q."* Their system is a
  **quarter-wave-shifted index-guided Bragg cavity — topologically OUR DEVICE.**
  Junctions are radiation-free when the core-minus-cladding permittivity contrast
  is preserved across the step. 3D: Q > 1e5 with only N = 10.
- ★**Lalanne & Hugonin, IEEE JQE 39, 1430 (2003)** (numbers from the open ECIO
  companion): a **1D Bragg cavity**, two parameters on the inner segments
  (−30 nm size, +65 nm outward displacement) → mode-matching took Q 200→750, and
  **radiation recycling added another >100x**, for a **~500x gain in Q/V at only
  +6% mode volume.** Closest published geometry to ours in either sweep.
- **Johnson, Fan, Mekis, Joannopoulos, APL 78, 3388 (2001)** — multipole
  cancellation, *"we do not sacrifice localization"*, 16x (2D) / 4.5x (3D).
  ★Design warning: the Q peak is a **sharp Lorentzian in parameter space**
  (R² = 0.9994) while near-fields look identical — **a coarse sweep steps over
  it.** Relevant to how we scan the comb and the inner teeth.
- **Dharanipathy/Minkov/Savona, APL 105, 101101 (2014)**: Q x7 for V x1.48 with V
  explicitly constrained in the optimizer — our exact situation. And in **SiN**:
  **Vij/Waks, arXiv:2509.16827 (2025)**, measured 3.9x Q improvement, hole
  positions only, footprint fixed.
- **[NOT FOUND]** any theorem bounding radiative Q at fixed V. The only ceilings
  named in the primary sources are material absorption and fabrication disorder.

### (ii) ★OUR ASPECT RATIO IS IN THE TM-HOSTILE REGIME
**Zhang/McCutcheon/Burgess/Loncar, Opt. Lett. 34, 2694 (2009)** measured, in ONE
structure, Q_TM collapsing **2.4e6 → 9,000 (~270x)** as thickness:width went
3:1 → 1:1, *"and a narrow bandgap also leads to large penetration depth of the
mode into the Bragg mirrors, thereby increasing the mode volume"* — **both of our
symptoms (low TM Q, long mode) from one variable.** Q_TE was unchanged.
**Our core is 350 nm tall x 800 nm wide = 1:2.3 — i.e. past 1:1, further into the
bad regime.** Independent confirmation: Johnson et al., PRB 60, 5751 (1999) finds
TM gaps want h ~ 2.3a and TE gaps h ~ 0.6a; **our h/a = 350/517 = 0.68 is
TE-optimal.** And Barclay's SiN group fixed exactly this by going **350 → 610 nm
thick** to pull the mode off the light line (arXiv:1905.03341, Q ~ 1e6).
If height is frozen by fab (single-litho 350 nm, per memory), this is a
**standing explanation for the TM penalty, not an action** — but it should be
stated in any writeup, and it argues the TM device is being asked to do something
the geometry disfavours.

### (iii) ★★THE APODIZATION TAPER IS TOO SHORT — the cheapest actionable finding
The Quan & Loncar recipe (Opt. Express 19, 18529 (2011)) is **constant pitch +
quadratic taper of scatterer size => LINEAR ramp of mirror strength => Gaussian
envelope**. Our builder already does the right thing: `bragg_device.py:1039-1050`
ramps corrugation depth **linearly** over `n_apod` teeth, and we have **kappa ∝
corrugation MEASURED**, so a linear depth ramp IS a linear mirror-strength ramp.
**But the length is wrong.** apod-20 = 20 periods = **10.3 um**, against a
Gaussian 1/e half-width of **~17 um** for a 20 um FWHM mode — the taper covers
only **0.61x the mode**, so beyond tooth 20 the envelope reverts to exponential
and the Lorentzian tail returns. Numerical light-cone model, FWHM held at 20 um
and kappa_max re-solved per taper length:

| taper N_t | length | leakage reduction vs no taper |
|---|---|---|
| 10 | 5.2 um | 1.3x |
| **20 (current)** | 10.3 um | **4.4x** |
| 30 | 15.5 um | 25x |
| **40** | **20.7 um** | **~180x (plateau)** |
| 50-80 | 26-41 um | no further gain |

**RECOMMENDATION: extend apodization from 20 to ~40 periods/side** (~1.2x the
mode's 1/e half-width), re-trimming full depth to hold 20 um. DERIVED from a 1D
model, not measured — but it is a cheap, well-posed FDTD test.

### (iv) THERE IS A CROSSOVER, AND WE ARE JUST PAST IT
Same model, holding intensity FWHM fixed (our actual constraint), comparing a
Gaussian envelope against the plain exponential:

| mode FWHM | dk*L | Gaussian / exponential leakage |
|---|---|---|
| 12 um | 2.6 | **0.56x — apodization HURTS** |
| **20 um (our spec)** | **4.4** | **~105x** |
| 24 um | 5.3 | ~5e3 |

Crossover at dk*L ~ 3 (FWHM ~ 14 um for our TM device). **Below it, apodization
is actively harmful** — the Gaussian's broader k-space core beats the
Lorentzian's narrow core only once dk*L is large enough. We sit just past it,
which is why apodization helps but not spectacularly.
★CAVEAT the agent flagged and I repeat: this gain is violently sensitive to
n_eff — 11x at 1.4995, 105x at 1.5084, 3.5e3 at 1.52. A 1.4% n_eff change moves
it ~300x. Treat as a trend, never as a Q prediction.

### (v-RESULT) ★★★DIAGNOSTIC RUN 2026-08-18 — THE LOSS **IS** ENVELOPE-LIMITED
MEASURED from stored .mat (no GPU), two INDEPENDENT parameter axes:

**Axis 1 — N-ladder at corr 325, identical box** (`results_from_igum/tm_nladder_c325/`):

| N | mode L (um) | T | Q_i |
|---|---|---|---|
| 60 | 16.804 | 0.9674 | 24,054 |
| 70 | 17.738 | 0.9624 | 30,488 |
| 80 | 18.394 | 0.9524 | 35,081 |
| 100 | 19.245 | 0.9104 | 38,409 |
| 120 | 19.661 | 0.8441 | 43,747 |

**=> Q_i ~ L^3.60** (5-point log-log fit).

**Axis 2 — corrugation at fixed N** (325 vs 400, `tm_nladder_c400/`):
- N=60: L 16.804→14.567, Q_i 24,054→16,961 => **Q_i ~ L^2.45**
- N=70: L 17.738→15.151, Q_i 30,488→20,297 => **Q_i ~ L^2.58**

**VERDICT: the measured exponent is 2.5-3.6, bracketing the predicted L^3 for a
radiation/envelope-limited mode.** There is **no dominant distributed loss
floor**. Envelope engineering is therefore the correct axis, the taper-length
finding in (iii) is worth testing, and the Gaussian-vs-exponential shape gain in
(iv) is physically available.
★CAVEATS: Q_i = Q_L/(1-sqrt(T)) is stiff near T->1 — at T~0.91 a 0.01 error in T
moves Q_i ~11%, so each exponent carries roughly +-0.3-0.5. The two axes differ
(3.60 vs ~2.5) because they are different paths: N changes the mirror length,
corr changes kappa. On the corr axis alone Q_i ~ corr^-1.8, notably flatter than
the -3 that kappa ∝ corr would give — i.e. **kappa ∝ corr is NOT holding well
between 325 and 400 nm** (L moved only 13% for a 23% corr change). Any surrogate
built on kappa ∝ corr over that range should be re-checked.
★BONUS, and it matters for the spec: **the mode length SATURATES.** N=100→120
grows L only 2.2% (19.245→19.661) while T falls hard (0.9104→0.8441) as coupling
weakens. So corr-325's mirror-limited asymptote is **~19.7-20 um** — the ~20 um
spec is essentially this family's NATURAL mode length, which is presumably why
corr-325 was chosen. Consequence: **L is not available as a lever anyway**; at
N>=100 you are already at the asymptote, so the only remaining route to Q is
changing the envelope SHAPE at fixed L — exactly (iv) and the see-saw (§6e).

### (v) ★FREE DIAGNOSTIC — the test that produced the result above
**Is our loss envelope-limited at all?** For an exponential (un-apodized)
envelope the light-cone integral gives **Q_rad ∝ L³ ∝ kappa⁻³ ∝ corr⁻³**
(DERIVED and numerically verified: successive doublings gave 7.53/7.88/7.93 vs 8).
We already have kappa ∝ corr MEASURED and Q_i at several corrugations in stored
.mat files. **Fit log Q_i vs log(corrugation):**
- slope ≈ **−3** => loss IS envelope-limited => apodization/taper work is the
  right axis and (iii) should pay.
- slope **flatter** => a distributed loss floor dominates (per-period scattering,
  roughness, mode conversion) that apodization **cannot touch**, and the whole
  envelope-engineering program is capped.
This costs nothing and it decides which half of the program is worth funding.

### (vi) Q ~ L² WAS WRONG
Un-apodized (exponential envelope): **Q ∝ L³**. Apodized (Gaussian):
**Q ∝ exp(dk²L²/2)** — the papers' "Q exponential in N, V linear in N" and
"Q exponential in mode size" are the same law in different variables. Measured
corroboration in SiN: **Zhan et al., APL Photonics 5, 066101 (2020)** — stored
energy **cubic in cavity length**. Every earlier Q_i decomposition in this
programme that assumed L² should be redone.

### (vii) CONTEXT: WHERE OUR DEVICE CLASS ACTUALLY SITS
Measured Q for phase-shifted *sidewall-corrugated* grating cavities in the
literature: X. Wang 2013 (SOI slot, corrugated, phase-shifted) **3e4**;
J. Biophotonics 2013 **1.5e4**; Velha 2007 (etched grooves, tapered) **5.8e4**
vs ~9,000 untapered (**tapering bought 6.4x**); Md Zain 2008 **1.49e5**.
**Our Q_i ~ 46,500 is already at or above that class.** Hole-based nanobeams
reach 1e6-1e9, but on a different physics (large index contrast, real bandgap).
★**Get Husko, Ducharme, Fahrenkopf, Guest, OSA Continuum 4, 933 (2021)** —
foundry **SiN, quarter-wave-shifted, square-wave sidewall corrugations, Λ=520 nm,
ΔW=250/350/450 nm, N=100/200/300**. That is a near-exact match to our device and
neither sweep could open it. **Institutional access needed; read it before any
writeup.**

### (viii) OPEN IN THE LITERATURE (i.e. our results may be novel)
**[NOT FOUND]** in either sweep: any application of the gentle-confinement /
Gaussian-mirror recipe to a hole-free **sidewall-corrugated** cavity with a **Q**
result (all corrugation-apodization papers target spectral sidelobes); any
Q-vs-taper-length curve for such a cavity; any published kappa_TE/kappa_TM ratio
for a given corrugation depth in SiN; anyone redoing Englund's Gaussian-envelope
k-space asymptotics for a **TM** mode using E_z (Englund's Eqs. 6-7 carry the E_z
term from the start — the TE simplification is an explicit later choice, so the
TM version is a small derivation nobody has published). Our side-comb geometry
was also absent from the first sweep. **Several genuinely publishable gaps.**

---

## 6e. ★★★THE "TM VERSION" ALREADY EXISTS AND WE MEASURED IT: THE INNER SEE-SAW

User's question, 2026-08-18: *"is there no way to do tooth shift that does for tm
decrease in loss and fwhm the same, like was in te"* — **yes, and it is recorded
in `memory/project_loss_exploration_chain.md` (job 117814, accurate mesh dx~35,
converged box).** It is not a longitudinal shift; it is a localized, zero-sum,
antisymmetric WIDTH perturbation on the innermost teeth:

> Family A INNER SEE-SAW (teeth ±1 = 1000+δ, ±2 = 1000−δ, **zero net area, even
> parity**): δ=+10 → 0.0814, **+20 → 0.0810**, +30 → 0.0810 (SATURATES);
> δ=−10 → 0.0834, −20 → 0.0851, −30 → 0.0871.
> **CHAMPION: rect-1050 cavity + see-saw δ=+20 → loss 0.0810 (−31% vs control),
> T 0.878 → 0.9179, fwhm +0.8%, λ_res UNMOVED.**

**−31% loss at +0.8% width with the resonance unmoved.** That is precisely the
TE-like behaviour the user is asking for, achieved in TM. The recorded reading:
*"Antisymmetric + saturating + linear-through-zero = genuine INTERFERENCE
cancellation of the residual cavity-local radiating moment (the multipole
prediction)."*

### ★THE DESIGN RULE (why this works and the tooth shift does not)

| property | inner see-saw | campaign tooth shift |
|---|---|---|
| **localized?** | YES — innermost 2 tooth pairs | NO — spread over 25 teeth |
| **zero net area / DC index?** | YES by construction | NO — shortens narrow gaps, raises n_avg |
| **resonance drift** | **λ unmoved** | **+1.6 nm per +374 nm of 2Σs (MEASURED)** |
| **transverse or longitudinal?** | transverse (width) | longitudinal (segment length) |
| **effect on mode length** | **+0.8%** | **+9.5%** |
| mechanism | cancels a radiating multipole | spreads the defect = lengthens the mode |

All four properties matter, and the tooth shift fails all four. Spreading the
phase over 25 teeth **is** lengthening the mode — that is what a distributed
phase shift does by definition (the whole DFB literature says so, §6c-(ii)). A
zero-net-area antisymmetric pair instead leaves the envelope alone and kills a
radiation *moment* by interference.

### Literature backing (found independently, §6d)
- **Johnson, Fan, Mekis, Joannopoulos, APL 78, 3388 (2001)** — multipole
  cancellation: *"unlike a previous, mode-delocalization mechanism, **we do not
  sacrifice localization**."* Exactly our see-saw's signature. Gains 16x (2D),
  4.5x (3D).
- **Lalanne & Hugonin, IEEE JQE 39, 1430 (2003)** — **~500x in Q/V at +6% mode
  volume**, in a 1D Bragg cavity, from **two localized inner-segment parameters**.
- ★**Johnson 2001's design warning applies to us:** the Q peak is a **sharp
  Lorentzian in parameter space** while near-fields look identical. **Our see-saw
  was scanned on a coarse 10 nm grid (δ = 10/20/30) and "saturated" — a
  saturation on a coarse grid is exactly what stepping over a sharp peak looks
  like.** A fine δ scan (2-5 nm steps, and δ > 30) is cheap and may not be
  saturated at all.

### ★THE SEE-SAW IS PURE CORRUGATION — IT IS ALREADY IN OUR PARAMETER BASIS
The engine's per-tooth basis is `corr_d = w_wide − w_narrow` and
`avg_d = (w_wide + w_narrow)/2`, both FREE for the inner 25 teeth. The see-saw
moves only the WIDE tooth by ±δ, so in campaign coordinates it is exactly:

    Δcorr_d = ±δ ,  Δavg_d = ±δ/2      (tooth 1 positive, tooth 2 negative)

Concretely on corr-325 / W800 (w_wide 962.5, w_narrow 637.5) with δ = 20 nm:

| tooth | corr (nm) | avg (nm) |
|---|---|---|
| 1 | **345** | **810** |
| 2 | **305** | **790** |
| 3..25 | 325 | 800 |

A stricter, **per-tooth area-neutral** variant is worth testing alongside it:
`Δcorr_d = ±δ, Δavg_d = 0` (w_wide and w_narrow move oppositely), i.e.
corr = [345, 305, 325, ...] with avg flat at 800.

### ★★SO WHY DIDN'T THE OPTIMIZER FIND IT? (it had the freedom the whole time)
BEST_T9635's corrugation profile is
`282.6, 289.2, 303.6, 311.7, 313.2, 316.5, 318.8, 319.7, 318.7, 320.1, 322.3, ...`
— a **smooth monotone taper** rising to ~322 with only ~±2 nm ripple. It is NOT
an alternating ±20 nm see-saw. Three reasons, and they are all fixable:
1. From a uniform seed the greedy gradient direction is "lower kappa near the
   cusp", which widens the mode — and with the width guard broken (§2) that read
   as **free transmission**. The optimizer took the cheap direction because we
   priced it wrong.
2. An alternating pattern is a high-spatial-frequency feature; L-BFGS-B from a
   smooth seed with a smooth gradient converges to smooth solutions.
3. Johnson 2001: the multipole-cancellation Q peak is a **sharp Lorentzian in
   parameter space** with visually identical near-fields — gradient descent
   steps straight over it unless it starts nearby.
**=> SEED the see-saw; do not expect to discover it.** This is the single
clearest lesson for the corrected campaign's initial conditions, and it costs
nothing to act on.

### ★WHAT HAS NEVER BEEN TRIED
The see-saw was measured on the **corr-400 / W800 / cavity-1050 / N=80** device
under the cavity-only scope. It has **never** been tried:
- in the **corr-325** family the current campaign uses,
- at **N=100/165**,
- **combined with the comb** (the two mechanisms are independent — the see-saw
  cancels a cavity-local moment, the comb cancels a grazing far-field lobe, so
  they may stack),
- with more than 2 tooth pairs, or with the finer δ grid above.
**This is the highest-value untested lever in the programme**, and unlike the
shifts it is already measured to be width-neutral.

---

## 6f. ★★★WHAT IS ACTUALLY LEFT TO TRY — checked against the CLOSED list

Source: `memory/project_loss_exploration_chain.md` (cavity loss program, Rounds
1-8, `results_from_athena/LOSS_EXPLORATION_FINDINGS.md`). **Read this before
proposing anything — a lot is already falsified.**

### ★THE FINDING THAT REPRIORITISES EVERYTHING (Round 7 k-space diagnostic)
> *"only ~30% of radiating weight is **cavity-local** and the champion already
> harvests ≈ that; remaining **~70% is distributed along the arms** → arm/envelope
> levers = phase 2, further cavity reshaping ≈ exhausted."*

So **cavity work is capped at ~30% of the radiation and the see-saw already took
most of it.** This DOWNGRADES §6e's expected value: the see-saw is still worth
porting to corr-325 and stacking with the comb, but do NOT expect another −31%.
**The remaining 70% lives in the ARMS — envelope/arm levers are the real target**,
which agrees with the independent finding that loss is envelope-limited (§6d-v).

### CLOSED — do not re-propose (all measured, evidence in FINDINGS.md)
distributed π-shift (+21..+39% loss) · step-envelope islands · **inner-tooth
shapes** (null-to-bad; note *"the cavity-side face of tooth 1 is load-bearing —
fab corner rounding there is a loss risk"*) · **wall-phase offset** ·
anti-radiator asym-DW · hourglass · external scatterers · cavity SHAPE on top of
rect-1050 (barrel/tri hurt; the cavity optimum is purely SCALAR = added area).

★**RETRACTION:** §6c recommended ramping `wall_phase_offset_deg` as a
length-neutral κ taper, on literature grounds. **It is on the closed list.** I
proposed it without checking the archive. Withdrawn — do not spend GPU on it
unless someone re-reads why it closed and finds the earlier test was scoped
differently (it was tested as a global uniform knob, not a ramp, so a per-tooth
ramp is *arguably* untested — but the burden of proof is on that argument).

### ★OPEN AND ALREADY MEASURED — the "OUT-OF-SCOPE parking list" was parked FOR US
The loss program explicitly deferred these to the inverse-design phase, i.e. now:

| parked route | measured | note |
|---|---|---|
| **TM whole-device SINUSOID corrugation** | **−10% loss @ +0.8% fwhm** | ★the non-rectangular-teeth answer, and it is nearly width-neutral |
| W1000 / C500 + cav1250 | **−60% loss @ +6% fwhm** | biggest effect on the list; costs width, but re-trim it (§6) and see what survives |
| tapered island (8 teeth) | −36% @ +4.8% | an arm/envelope lever |
| TE whole-device sinusoid | −29% @ +7.5% | TE only |
| TE barrel300 | −9% | TE only |

**Non-rectangular teeth: the useful version is the whole-device corrugation
PROFILE, not the inner-tooth shape.** `corrugation_profile` already exists as a
builder feature (listed under "Machinery — UNCOMMITTED builder features"), so the
sinusoid is a config change, not new code. Literature agrees the profile matters:
Lee & Streifer, JOSA **68**, 1071 (1978) computes radiation for rectangular vs
sinusoidal vs triangular corrugations; and *Polarization-Independent Complex
Bragg Grating Filters on Silicon Nitride*, Laser Photon. Rev. (2024),
doi:10.1002/lpor.202402114, equalises TE/TM specifically with a **triangular**
lateral corrugation.

### ★★★CLOSED 2026-08-19 — MOIRÉ (user: "moiré width too big"), x-ASYMMETRY, GAUSSIAN

**MOIRÉ — REFUTED, and the reason is a hard geometric conflict.** DERIVED with
κ₀=0.0353 µm⁻¹, Λ=516.83 nm, N=100 (device 103.4 µm):
- one beat node in the device requires Δk ≤ 0.061 µm⁻¹ ⇒ **mode FWHM ≥ 50.8 µm**
- a 20 µm mode requires Δk = 0.393 µm⁻¹ ⇒ node spacing 16 µm ⇒ **6.5 nodes in
  the device**, i.e. a coupled-cavity array, not one cavity.

| Δpitch nm | node spacing | nodes in device | mode FWHM |
|---|---|---|---|
| 2 | 133.6 µm | 0.77 | 57.8 µm |
| 5 | 53.4 | 1.93 | 36.5 |
| 10 | 26.7 | 3.87 | 25.8 |
| 16.7 | 16.0 | 6.46 | **20.0** |

**The single-node moiré floor is ~51 µm against a 20 µm spec — 2.5× too wide.**
The user called this from lab experience before the calculation; the calculation
agrees. **Do not propose moiré for this device.** (It would only make sense on a
device whose spec mode is ≳50 µm.)

**x-ASYMMETRY — PROVABLY HARMFUL, already answered in the archive.**
`memory/project_bic_kerker_batch1_dispatch.md`:
> *"SYMMETRY answered: **mirror-symmetric is PROVABLY optimal** (antisym
> perturbation → odd δA ⊥ even A₀ → strictly adds radiation; anti-moment study
> confirmed)."*
The argument: the device's radiating amplitude A₀ is **even**; an antisymmetric
perturbation produces an **odd** δA; odd ⊥ even, so the cross term in
|A₀+δA|² vanishes and the perturbation can only **add** |δA|² in quadrature — it
can never interfere destructively. Confirmed by the anti-moment study, and
`asym_inner_dw_delta_nm` ("anti-radiator asym-DW") is on the CLOSED list.
★**Why the comb is NOT a counter-example** (it is deliberately not x-mirrored):
the comb is a **separate radiator placed away from the guide**, whose emitted
field is phased to be anti-phase with the *total* far-field leak. That is
far-field interference between two sources. Perturbing the **grating itself**
antisymmetrically instead modifies the cavity's own amplitude by an odd amount,
which is the orthogonal case above. **Two different mechanisms — do not conflate
them.** So: keep the grating mirror-symmetric; keep the comb free to be
asymmetric.
Also from the same entry, a ceiling worth knowing: *"passive scatterer α is REAL
but optimal α is COMPLEX → phase-limited; even sign-correct real-α site models
only ~13% cancel and parasitics dominate"* — i.e. **comb-like passive cancellers
are phase-limited to ~13%**, consistent with the comb's measured +17.1% Q_i.

**GAUSSIAN ENVELOPE — already tried by the user's lab** (user, 2026-08-19). So
§6d-(iii)'s "extend the taper to ~40 periods" is *not* virgin territory. ★Before
spending GPU on it, ask the user what their lab measured: if a Gaussian envelope
was already built and did not deliver, the envelope-shape axis is much more
constrained than §6d suggests, and the ~180× model number is contradicted by
experiment.

### ★★★ARE WE ALREADY AT THE FAB CEILING? — check this BEFORE optimising further
Measured Q for **phase-shifted sidewall-corrugated** grating cavities in the
literature: X. Wang et al., Opt. Express 21, 19029 (2013) **3e4**; X. Wang et
al., J. Biophotonics 6, 821 (2013) **1.5e4**. (Groove/hole-type cousins reach
more: Velha 2007 **5.8e4**, Md Zain 2008 **1.49e5**.)
**Our SIMULATED Q_i is 46,500** — at or above every measured sidewall-corrugated
value found. Every simulation here is of a perfectly smooth, perfectly periodic
device; real ones are limited by sidewall roughness and phase/stitching error,
and Englund 2005 states the general ceiling plainly: *"Qs are bounded to
currently ~1e4 by material absorption and surface roughness."*
⇒ **It is entirely possible the fabricated device is already roughness-limited,
in which case further simulated Q gains buy nothing.** This is cheap to settle
and would save months: **compare a measured Q from a fabricated device of this
family against its simulated Q.** If they diverge badly, the productive work is
fab/process, not geometry. Nothing in this handoff is worth GPU until that is
known. (The user's lab has fabricated devices — ask.)

### OTHER AXES CONSIDERED 2026-08-19 — with why they do or do not help
- **Shorter operating wavelength.** δk = (n_eff−n_clad)·2π/λ, so δk ∝ **1/λ** at
  fixed indices. 1550 → 1310 nm is **+18% δk**, and the Gaussian suppression
  exponent goes as δk²L² ⇒ **~+40%** in the exponent, free, no geometry change.
  Only viable if the application's source wavelength is not fixed — probably it
  is, but it costs one question to ask.
- **Narrowing the waveguide to fix the TM aspect ratio: NO.** Zhang 2009 says TM
  wants tall:wide ≈ 3:1 and ours is 1:2.3, which tempts one to shrink W. But
  narrowing W **lowers n_eff toward n_clad**, collapsing the very light-cone
  margin (§6c-ii) that is TM's core problem. The two effects fight and the
  light-cone one dominates. **Only going TALLER helps**, which is the fab-stack
  lever below.
- **Relaxing the acoustic constraint itself.** The archive lists "relax a
  constraint" as one of only three escapes. The 20 µm spec comes from the
  acoustic transducer; if the transducer could be co-designed to a longer
  interaction length, the entire width-vs-Q fight changes character (Q_i rises
  steeply with L — measured L^2.5-3.6). Outside our scope to decide, but it is a
  legitimate engineering question for the user, not a physics dead end.
- **Higher-order defect modes / coupled cavities: no.** Both give wider or
  multi-peaked modes — the moiré refutation above is the same arithmetic.
- **Duty cycle: no** (§6d-vi) — no harmonic of a first-order grating reaches the
  light cone, so harmonic suppression buys nothing radiatively.

### ★THE LEVER THE ARCHIVE ITSELF POINTS AT (converges with §6c-ii)
Same entry: *"The genuine-novelty levers all relax something: **CLADDING INDEX /
light-cone (suspended air-clad membrane = the big lever)**, SWG metamaterial
cladding, or the width-cost Pareto."* This is exactly the δk finding of §6c-(ii)
arrived at independently: raising (n_eff − n_clad) enlarges the k-space margin,
and TM's margin is only half TE's. Barclay's SiN group did precisely this by
going 350 → 610 nm thick (arXiv:1905.03341, Q ~ 1e6). **It changes the fab stack,
so it is a user decision, not a study we can just run.**

### ★★MOIRÉ — the original write-up (superseded by the refutation above)
Nothing in the archive mentions it. **And the physics lands exactly on the
gentle-confinement prescription:**
Superimpose two pitches Λ₁, Λ₂ (or modulate the pitch) so the coupling beats:
`κ(x) = κ₀·|cos(Δk·x/2)|`, Δk = 2π(1/Λ₁ − 1/Λ₂).
- At a beat **node** κ→0 **and the grating phase flips by π** — so the moiré
  *generates its own π-shift*. There is no separate abrupt defect, hence **no
  cusp to smooth** — the thing we have been fighting all along.
- Near that node, `κ(x) ≈ κ₀·(Δk/2)·|x|` — i.e. **mirror strength increasing
  LINEARLY with distance**, which is precisely Quan & Lončar's rule that yields a
  **GAUSSIAN envelope** (Opt. Express 19, 18529 (2011)). Our apod ramp
  approximates this with 20-40 free teeth; a moiré gets it from **ONE global
  parameter**.
- Mode length is then set by σ = κ₀Δk/2 ⇒ **L ∝ 1/√(κ₀Δk)**, so the pitch
  detuning is a clean, single-knob **mode-length dial** — exactly what the
  re-trim scheme in §6 needs, and far better conditioned than 25 coupled
  corrugations.
- Moiré/phase-shifted gratings are standard in FBGs and have been done in
  integrated sidewall gratings for narrow-band filters, so it is fabricable; what
  is (per both sweeps) **unpublished is using it as a radiation-minimising
  envelope for a fixed-mode-length cavity**.
★**Caveats to check first, cheaply:** (a) at the node κ→0 locally, so the local
stop-band vanishes — confirm the mode is still confined by the surrounding strong
regions and no new leakage channel opens; (b) a moiré has TWO beat nodes per beat
period — the device must contain exactly one, or you build a coupled-cavity pair;
(c) the beat also modulates the DC index unless the two components are balanced —
that is the §6c-(iv) detuning trap again, so balance it by construction.
**RECOMMENDATION: this is the most promising untried idea on the list**, it is a
config-level change if `corrugation_profile`/per-tooth arrays can express a
pitch beat, and it attacks the 70% (arms/envelope), not the exhausted 30%.

---

## 7. PARKED — needs the user, do not do alone

- **scancel of 134032 (stage-4, Athena) and 55801 (bare, IGUM)** — both steer on
  void widths. Recommended, but stopping runs is confirm-first (CLAUDE.md §6);
  use the `stop-runs` skill and resolve the job ID first.
- **scancel of 134299** — its t0/t1 are void and superseded by 134334. Only t2
  is worth keeping (post-fix). Low value either way; it drains on its own.
- Dispatching `rho_neutral_shape.py`.
- Large uncommitted git inventory — never commit without permission.
- Ansys adjoint gradient-bug report; far-field needle readout; width-band Pareto.

Note: seedB2 (IGUM 56033) **finished cleanly** at 12:23 (exit 0,
best_fom 0.702857, `_best.json` fetched to `results_from_igum/lumopt2_logs/`).
It was not a failure. Its final design is width-suspect like the rest.

---

## 8. Commands

```bash
# status
ssh evyatarrubin@athena.technion.ac.il "squeue -r -u evyatarrubin -o '%.14i %.8T %.10M %R'"
ssh evyatarrubin@132.68.58.101 "squeue -u evyatarrubin -o '%.10i %.8T %.10M'"   # IGUM

# license seats — MANDATORY before any multi-task dispatch; probe from IGUM
ssh evyatarrubin@132.68.58.101 "\$HOME/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261/licensingclient/linx64/lmutil lmstat -c 1055@132.68.48.51 -f lum_fdtd_solve | grep 'Users of'"

# dispatch a lumopt2 study (verify flags against the parser first; unknown flags ABORT)
SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.<module> \
  --max-concurrent=1 [--after=<jobid>] [--array-tasks=<lo>-<hi>]

# code-only push (NOT --no-submit, which does not exist)
bash athena/deploy_athena.sh --upload-only
```

Offline width check on any stored .mat, no GPU:
```python
import scipy.io, numpy as np, sys; sys.path.insert(0,'.')
from runners.lumopt2_design import lumopt2_design as eng
d = scipy.io.loadmat(PATH, squeeze_me=True)
x = np.asarray(d["field_x"],float)*1e6; I = np.asarray(d["field_energy_density_1D"],float)
print(eng.fwhm_env_of_line(x, I), float(d["fwhm_m"])*1e6)   # must agree
```

---

## 9. Files

- Engine: `c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\lumopt2_design.py`
- Width recovery tool: `...\runners\lumopt2_design\fsp_width.py`
- Ready, undispatched study: `...\runners\lumopt2_design\rho_neutral_shape.py`
- Banked designs: `...\runners\lumopt2_design\best_designs.py`
- Registry: `...\runners\lumopt2_design\DESIGNS.md` (★its FWHM block is VOID)
- IGUM campaign logs: `...\results_from_igum\lumopt2_logs\`
- Runbook: `...\.claude\skills\lumopt2-design\SKILL.md`
- Memory: `C:\Users\evyat\.claude\projects\c--Users-evyat-Lumerical-phase-shift-grating-FTDT-codes\memory\project_lumopt2_campaign_state.md`

---

## 10. Lessons recorded today (skill items 25-27)

1. **Never let the CONTROLLED quantity differ from the SPECIFIED quantity
   without measuring both every evaluation.** The campaign controlled sigma for
   days while the spec is FWHM.
2. **A constraint stated in a surrogate's units is meaningless until you
   measure its conversion into the spec's units.** `RHO_DN = 0.95` was quietly
   permitting large width growth; nobody had ever expressed that band in microns.
3. **Never restructure a program on a number from a metric you wrote the same
   day and have not cross-checked.** Three estimators gave three answers; the
   new one was the wrong one.
4. **Reuse the project's own analysis functions rather than reimplementing
   them.** The fix was to call `sim_helpers.extract_envelope_peaks` and
   `calculate_fwhm_relative` directly — which then matched to 1e-15.
5. **Keep the raw profile.** All of today's GPU re-runs existed only because no
   field profile was ever saved. Every eval now writes a ~30 kB `.npz`.
