# HANDOFF — SiN post arrays that cancel the radiated leak (TE pi-shift gratings), 2026-10-23 evening

Read first: this file, then `docs/postarray_radiation_cancellation_2026-10-23.md` (method; its table covers production devices up to 269305/269020 only, later
results are in section 2 here), the skill
`.claude/skills/postarray-design/SKILL.md` (recipe + every validity limit), memory `project_radcancel_overnight_2026-10-23.md` (dated log).
Branch `claude-te`; commits 2ed1f68, e6e17f1, 5bd3971, d36452e (handoff), d533a67, abffc1a, 331213b, c7d6afd, f42ca25, e07229a (+ this fix). CLAUDE.md has UNCOMMITTED
rule edits (10-20..10-23) awaiting the user's OK to commit. Summary page for the user: https://claude.ai/artifact/FTLf6cia2xv9WodhGgEnuW

## 1. Goal and rules (user)
- Goal: **x2 Q_i** (spec = Q at -3 dB ~ 0.293 Q_i) on the **TE overshoot** grating Nt80x (SiN n 1.97, 350 nm, oxide 1.444, pitch 514 nm,
  96 periods/side, cavity 950.3 nm, lambda 1619.5 nm, apodized teeth up to 1800 nm wide). Plain TE (1050-nm best cavity) is the test bed.
- New planar concept only: full-height SiN posts in the same layer; nothing reachable by tooth widths/shifts/apodization/cavity size counts.
- **A narrower mode is fine/good**; only widening counts against a device.
- STALE HEADLINES elsewhere: the (uncommitted) CLAUDE.md banner and the skill may still say +63 %; the honest number is +33 %
  (converged). The skill's step 2 (calibrated model) predates the Fable review: do not pre-register with kappa*x1 until refit.
- Validate expensively (fine mesh, -3 dB device) only the BEST design, after a bug/geometry audit.
- Never state model ceilings/plateaus as physical limits; test one factor per FDTD run; model-first design, inverse design later.
- User concern: posts close to the cavity can act as an **effective-cavity enlargement** (mode perturbation = inverse-design territory),
  not cancellation. Always separate the two.

## 2. What is established (MEASURED unless marked; Q_i itself is DERIVED from Q_L (trace decay) and T x gamma^2, assuming symmetric
## port coupling, without the straight-guide port calibration of CLAUDE.md 2 - fine for ratios at matched numerics)
| Device (Q_i vs own control, same numerics) | Mesh | Gain | Notes |
|---|---|---|---|
| Overshoot 3-row array pa3 (238 posts/side, y 1.2/1.5/1.8, r 50-100, gaps >= 100, abs(x) <= 24) | production dy 57.7 | x1.63 | reproduced by 3 independent checks; no bug |
| same pa3 | **core-refined dy 28.8** | **x1.33** | best honest number so far (post/full-mesh convergence still pending); T 0.671->0.735, width held |
| 2-row pa2 / 4-row pa4 / 5-row pa5 | production | x1.40 / x1.55 / x1.42 | pa4/pa5 r<=120 + 60 nm gaps: model failed out of regime |
| comb pair (independent phases) / single comb | production | x1.17 / x1.075 (x1.083 dy29) | arms are NOT mirror images -> own phases |
| Plain TE 3-row array (162 posts/side, y 0.8/1.1/1.4) | production | x1.50, width -4 %, lambda +0.49 nm | see next row |
| Plain TE 3-row array, CONVERGED (269500/269501, dy 24.7) | core-refined | **x1.50** (2.14e4 -> 3.20e4), T 0.703->0.784, width -0.63 um | robust to mesh |
| ★**Overshoot c29: 3-row array DESIGNED on the converged control (175 posts; 269488)** | core-refined | **x1.42** (5.67e5 -> 8.04e5), T 0.671->0.748, width +0.015 um, lambda +0.06 nm, P/Pc 0.699 | BEST so far; beats plain Born x1.32 and Fable band x1.33-1.40; calibrated x1.50 too high; tiny lambda shift with larger gain = cancellation, NOT cavity enlargement |
| Overshoot 7-row pa7 (r<=120, 50-nm gaps, box 8.8; 269442) | core-refined | **x1.36** (5.70e5 -> 7.76e5), T 0.672->0.736 | best converged overshoot so far; calibrated pre-reg x1.81 FAILED |
- Stored controls: overshoot production `results_from_athena/comb_overshoot/control_nt80x_T340.mat`, dy37 `control_nt80x_dy37.mat` (for the harmonics
  mesh-stability check, step 7),
  dy29 `results_from_athena/budget_overshoot/dy29_box6p8.mat` (267605_0) and `dy29_box8p8.mat` (267605_1); plain TE production
  `results_from_athena/postarray_plain/control_plain_c1050.mat` (1050 cavity) and `control_plain_c800_roundA.mat`. Overshoot devices are in
  `results_from_athena/comb_overshoot/` (incl. the dy29 incumbent `pa3_r100_dy29.mat`), plain devices in `postarray_plain/`. Memory dir:
  `C:/Users/evyat/.claude/projects/c--Users-evyat-Lumerical-phase-shift-grating-FTDT-codes/memory/`.
- Loss appears radiation-dominated: Q_i gain tracks 1/(measured far-field leak), corr 0.983 over 13 devices (empirical; the leak is
  normalised by a core-field energy proxy, so this is supporting evidence, not a closed loss budget; treat 1/leak as conditional).
- **Why +63 % fell to +33 %:** pa3 was designed against the production-mesh leak, which contains staircase-mesh radiation at mid angles
  (abs(u_x) 0.5-0.8); the array "cancelled" that artefact. The real effect = the grazing leak (abs(u_x) > 0.95) halved on both meshes. Rule now
  in CLAUDE.md (uncommitted working copy): design and judge on the converged mesh; production gains are screening only.
- Cavity enlargement vs cancellation: pa2 (inner row 1.5 um) shifts lambda only +0.03 nm yet +40 % (prod); pa3 (inner row 1.2) +0.24 nm;
  coupling Q unchanged 1.22e5 -> mostly cancellation, but pa3's inner row may add a mode-perturbation part (UNTESTED).

## 3. Model state (numbers here are MODEL PREDICTIONS unless marked MEASURED)
- Born dipole model (`python_tools/radcancel/postarray_born.py`): drive = stored control's tube side-plane field (y 2.1 um) back-propagated
  through oxide (valid only outside the widest tooth, y > 0.9 um); short- and steady-run normalisation. `postarray_ceiling.py` (Gram + NNLS),
  `postarray_design.py` (discrete posts, all-pairs gap, per-row r cap, X_LIM binds positions).
- Calibrated form E = C(1 + kappa s x1) + s Bp (s 1.01, kappa 0.30; `postarray_work/calib_s_kappa.npy`; fit script
  `python_tools/radcancel/postarray_calibrate.py`), fit on 13 production devices; the kappa term is SUSPECT (Fable, section 7);
  on the CONVERGED control it predicts pa3 x1.37 vs measured x1.33 (good). Out of regime (big/dense posts) it fails (pa4/pa5).
- Converged-control work dir: `results_from_athena/comb_overshoot/postarray_work_dy29/` (Grams c3x30, c5x30 and arch14 = 14 candidate
  rows y 1.1-2.4 every 0.1 um, 100 nm x sites, |x| <= 35, all saved; designs). The arch14 greedy scan ran from a temporary
  `python_tools/radcancel/_arch_tmp.py` (self-deletes when it finishes; rerun with postarray_rowselect.py instead). Row choice / per-row length tool:
  `python python_tools/radcancel/postarray_rowselect.py <ceiling_G file> [max_rows]` (greedy best rows + 90 % x-extent per row).
- NOTE: `postarray_design.py` prints PLAIN Born leak at scale 0.70; the calibrated gains below come from `postarray_calibrate.py`'s form
  (x1 = Re<C,Bp>/|C|^2). The greedy-insertion experiment (scratch insert.py) is not in the repo.
- Converged-mesh predictions (calibrated, r<=100, gaps>=100, abs(x)<=30): best 1 row y 1.2 (36 posts) x1.30; 2 rows x1.46; 3 rows x1.50;
  greedy post insertion adds ~1 % -> this regime saturates near x1.5 IN THE MODEL (not a physical limit). Ceilings: 1/2/3 rows 25/37/40 %
  removable and abs(x) 24/30/45 um 37/40/45 % are PRODUCTION-control scans; CONVERGED ceilings: 3 rows 41.3 %, 5 rows 43.4 % (length
  is a model claim; TM evidence says otherwise).
- Can 3 rows get back near +64 % on the CONVERGED mesh? The +63 % itself was partly artefact (production control Q_i 5.35e5 vs 5.67e5
  converged), so the old array cannot be 'fixed' back to it. But the converged 3-row ceiling is 41 % removable = x1.70 in the model; the
  realistic design (r <= 100, gaps >= 100) predicts x1.50 (269488 tests it). The gap x1.50 -> x1.70 is constraints + optimizer, not physics:
  close it with single-factor tests (r <= 110 / gaps >= 80; free (x,y) placement incl. near-cavity pockets with a direct field;
  element-pattern correction; NOT coupled dipoles, see section 7). Converged 4/5-row designs (c5x30, r <= 100, gaps >= 100, |x| <= 30; plain Born x0.7 leak): 3 rows 0.756
  (175 posts), 4 rows 0.723 (230), 5 rows 0.763 (241, worse = the under-populating design step, not physics; its ceiling is 43 %).
  Files `postarray_work_dy29/design_c5x30_R100_rows0123_x30.npz` / `rows01234_x30.npz`. Fix the design step before trusting row trends.
- ★**X/Y CONVERGENCE (model ceilings, any size, converged control, `postarray_work_dy29/converge_xy.py`):** |x| <= 5/10/15/20/25/30/35 um ->
  26/31/37/42/44/47/48 % removable (still rising slowly at 35; beyond needs a longer far-field tube); outer y <= 1.1/1.2/1.4/1.6/1.8/2.0 ->
  29/40/43/45/46/47 % (converged by ~1.6 um). The binding limit is the INNER y = 1.1 um (model validity), not the outer one.
- **175-post 2-D design** (`twod/des_2D_x30_g100_N175.npz`, model 0.692 vs 0.666 for 504 posts): posts cluster within |x| <~ 17 um as an
  inner layer y 1.1-1.25 um + scattered outer posts -> the value is in the inner +-15 um. Figure
  `results_from_athena/comb_overshoot/devices_twod_x30_N175.png` (504-post: devices_twod_x30_y2000.png).
- ★**2-D PLACEMENT VERDICT (Fable agent, finished; files `postarray_work_dy29/twod/`: engine pa2d.py, direct_eval.py, all designs
  des_*.npz, FINAL_2D_x30_y2000_g100.npz = recommended FDTD candidate):** free 2-D beats straight rows at MATCHED post budgets
  (direct Born re-score, plain x0.7): 2-D 175 posts 0.692 vs stored rows 0.755; 3 rows fixed optimizer 0.720 (326 posts); 2-D |x| <= 24/
  30/35 = 0.677/0.661/0.647 (545/632/719); recommended tube-valid 2-D (y 1.1-2.0, |x| <= 30, 504 posts) 0.666 -> model x1.50 vs x1.32
  for 269488's design. Gaps 80/60 nm worth only 0.004-0.007 (not a lever); multi-start spread ~0.007. The optimum is a dense cloud that
  first fills a layer along the guide near the cavity (58 % of a 175-post budget within y 1.1-1.4, |x| < 12 um) -> also the user's
  cavity-enlargement risk zone (untested). Gates: y-interpolation -0.2 %, x-midpoint columns -11 % in |B|^2 (all designs re-scored
  with direct Born). Next FDTD candidate = FINAL_2D_x30_y2000_g100 at dy29 vs 267605_0 (needs a runner variant + user go).
- **Rows may be the wrong abstraction (user, critical):** 5 rows < 4 rows is impossible for a correct optimizer (it can leave a row
  empty). A free 2-D placement prototype (Fable agent, Gram `ceiling_G_arch14.npz`: rows every 0.1 um from 1.1 to 2.4, |x| <= 35, 100 nm
  x grid) and a GPT method review were launched at handoff time; if their results are not in this file, re-run that task: continuous
  (x, y, r) posts, all-pairs gaps, greedy packing from the NNLS density + insertion/removal + multi-start, monotone warm starts,
  compared with straight rows at matched post budgets.
  ROW SCAN (greedy, arch14, model ceilings, any size): best rows pack near the guide - 1 row 1.2 (30 %); 1.1+1.2 (40 %); +1.6 (43 %);
  1.1/1.2/1.3/1.6/1.9 (46 %); +2.4 (47 %); 90 % of each row's strength within |x| ~24-32 um. 47 % is a CEILING (~x1.9), not measured.
  2-D extra: |x| <= 30/35 = 0.654/0.638; tube-valid (y <= 2.0, |x| <= 30) 0.656 with 504 posts = the FDTD candidate; 2-D ceilings
  0.553/0.524/0.507 (|x| 24/30/35); rows 1.1/1.4/1.7/2.0 = 0.695 (408); multi-start spread ~0.01; gap 80 vs 100 only -0.006.
  INTERIM RESULTS (Fable 2-D agent, arch14 Gram, plain Born x0.7, gaps >= 100 nm, fixed design step = greedy packing + insert/remove +
  monotone warm starts): rows 3/4/5 at |x| <= 30 = 0.720 (326 posts) / 0.704 (435) / 0.698 (517) -> MONOTONE (the old 0.763 was the
  design step); free 2-D at |x| <= 24 = 0.673 (545 posts), better than any row design but with more material (matched-budget
  comparison was still running). Agent scratch: `scratchpad/twod/` of the old session (temporary) - its final npz must be copied to
  `results_from_athena/comb_overshoot/postarray_work_dy29/` before use.

## 4. Running on Athena at handoff (snapshot 2026-10-23 ~19:00; check `squeue` fresh before acting; the old chat's watchers die with it)
| Job | What | Score against | Pre-registered |
|---|---|---|---|
| 269442 DONE (x1.36) | 7-row pa7 (r<=120, 50 nm gaps), box 8.8, dy29 | stored 267605_1 (dy29 box 8.8, Q_i 5.70e5) | x1.81 (calibrated, suspect); box 8.8 puts the tube side face at y 3.104 um (outside all posts) -> P/Pc may be valid; check the far-field lambda vs the resonance |
| 269488 DONE | c29 3-row designed ON the converged control (175 posts) | 267605_0 | MEASURED x1.42 (calibrated x1.50 / Fable x1.33-1.40) |
| 269505 (afterok 269442; if 269442 dies, resubmit without --after) | c29_r1_x30 (1 row, 36 posts) / c29_r2_x30 (2 rows, 90) | 267605_0 | calibrated x1.30 / x1.46; Fable x1.19-1.26 / x1.30-1.38 |
| 269500 / 269501 DONE (plain x1.50 converged) | plain TE control / plain 3-row array, y_mesh_cells 26 (dy 24.7 nm on the plain device) | each other | plain +50 % at production; P/Pc likely invalid (lambda +0.49 nm leaves the 2D window) |
| 269448 (task 0 = N108 DONE, file on Athena `..._m3db_overshoot/results/`; task 1 = N116 running) | bare Nt80x at 108 / 116 periods (-3 dB length calibration) | Q_c(N) = 1/(1/Q_L - 1/Q_i) vs the PRODUCTION control control_nt80x_T340.mat | production mesh: an initial estimate of N for any winner; verify T ~ 0.5 at the winner's own numerics |
Scoring: `python python_tools/radcancel/solve_v4.py --short --energy <control.mat> <device.mat>` (plain: drop --short, keep --energy;
Q_i is flag-independent). Side effect: solve_v4 also fits combinations and OVERWRITES <control.mat>.alpha_v4.npy; ignore that alpha for
  single-device scoring. Decision rule for 269488: <= x1.40 -> the kappa*x1 calibration is falsified (use plain Born); >= x1.48 -> it holds. Results dirs on
Athena: `~/bragg_sim_athena/results/farfield_sph_15um_postarray_overshoot_<variant>/results/*_ff.mat`, `..._postarray_plain_dy29*`,
`..._m3db_overshoot`, `..._postarray_overshoot_r7`. Copy to `results_from_athena/comb_overshoot/` or `postarray_plain/`. pa4/pa5 far-field
P/Pc invalid (posts outside the tube / lambda outside the +-0.25 nm 2D window); Q_i, T, width valid.

## 5. Next steps (ordered; propose each run to the user first, CLAUDE.md 6)
0. **Primary plan = Fable R1-R6, run ORDER per section 12 (re-anchored on c29)** (measured-residual iteration, then length, then a wall-following inner layer with
   in-situ drive monitors). Suggested FDTD order (each ~3 h EXPECTED at dy29, 7 seats; measured anchors production 2:23 h / 34 GB,
   dy29 3:33 h / 43 GB): R1 (+ x-line field monitors) -> cavity-shift control (2b) -> R2 -> R3 -> R4 -> R5/R6 on the winner only.
   Every run is proposed to the user first (CLAUDE.md 6). The items below are the supporting checks.
1. Score 269488 / 269505 / 269442 / plain pair vs BOTH pre-registrations (Claude calibrated vs Fable plain-Born ranges); refit the
   calibration on CONVERGED devices only (`postarray_calibrate.py` reads the production dataset; build a converged dataset first).
2. Separate cancellation from cavity enlargement (discriminating checks, not a classification rule; compare mode profile, core-energy
   fraction, coupling and radiation, not only lambda): (a) model one-row phase/distance scan (oscillating vs monotonic Q_i); (b) one FDTD control: bare Nt80x with a small cavity change giving the same +0.24 nm shift, no posts.
3. Architecture (GPT 01a16951, literature): optimise posts over a 2-D allowed region with LOCAL tooth clearance instead of fixed straight
   rows; posts nearer the cavity need a DIRECT stored field -> one control run with a field monitor over the placement region (same
   numerics as 267605_0). Row distances/lengths: run `postarray_rowselect.py` on ceiling_G_c5x30.npz or ceiling_G_arch14.npz.
   Design recipe (absolute PA_WORK required): cd python_tools/radcancel; PA_WORK=c:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/
   results_from_athena/comb_overshoot/postarray_work_dy29 TAG=_c5x30 R_MAX=100 GAP_NM=100 X_LIM_UM=30 ROW_IDX=0,1,2,3 python postarray_design.py
4. Model upgrade: NOT coupled dipoles (Fable: tested, no gain); element-pattern correction (angular L 4-6, or 2D FDFD line dipole); keep r <= 100 / gaps >= 100 until measured otherwise
   (the user is NOT sure of a 100/120 nm threshold; TM data: nonlinearity grows with drive strength and post proximity).
5. Validation of the final winner only (variants fine/finectrl are HARDCODED to pa3_r100 with a fixed |y| 1.05-1.95 um band: edit NAMES
   and the band to cover the winner's full geometry): matched fine pair (variants fine/finectrl: dx 25.7 + 25 nm post band + dy29 core; 275 GB QOS cap,
   ~14 h/run EXPECTED, fits 275 GB) and a direct -3 dB device using 269448's Q_c(N).
6. Later: post-only adjoint (objective 1/Q_i, width upper bound only) starting from the best model design.
7. Harmonics (user 2026-10-23: "could they still help?") - not as the objective (Parseval), but: (a) **mesh-stability diagnostic**: compare the
   control's harmonic tables at production / dy37 / dy29 (stored: comb_overshoot/control_nt80x_T340.mat, comb_overshoot/control_nt80x_dy37.mat, budget_overshoot/dy29_box6p8.mat; tool python_tools/farfield_multipole.py) and use them to (i) choose the converged control as the target and (ii) correct
   the model per harmonic. Do NOT drop harmonics from the objective (that can just push radiation into the excluded ones); keep total
   radiated power as objective and acceptance; (b) per-harmonic model-vs-FDTD response to find where
   the Born model is wrong and weight/correct those components; (c) harmonic content of pa3's residual leak -> which transverse patterns
   (row distances) are still uncancelled.

## 6. Pitfalls already paid for (all in CLAUDE.md / skill)
Widest tooth 1800 nm (drive validity + clearance); >= 1.5 um post-edge-to-PML clearance; short-run rules (CLAUDE.md 2: never Q from a
short run's spectrum, never monitor apodization); file-tag collisions for scatterer lists; 4d_1g caps memory at 275 GB; far-field tube
y 2.104 / abs(x) <= 35 and the +-0.25 nm 2D window; same-phase pairs on asymmetric arms cancel; optimizer cancels mesh artefacts.

## 7. Fable deep review (max effort, finished 2026-10-23 ~19:30) — the most important input for the next chat
Scripts + data (copied, keep): `results_from_athena/comb_overshoot/postarray_work_dy29/fable_review/` (qi_check, design_audit,
dataset_models, ff_patterns, cda_test, mep_test, drivecheck, born29, gram29, incr29, pack29, arch29 .py; ff_C29/C37/PA3_29.npz;
gram_dy29_r3*.npz; work29/). Its findings (labels as given by the reviewer):
- **+33 % is real**, +-5 pts from T and ~+-5 from mesh (MEASURED re-derivation; raw-T floor x1.13). Leak-pattern overlap prod vs dy37 0.68,
  dy37 vs dy29 0.83, dy29 box 6.8 vs 8.8 0.991 (Fable's estimate of the numerical part, ~25 % prod / ~6 % dy29, is a model-based INFERENCE from overlaps, not a decomposition). The
  posts' RESPONSE is mesh-stable (pa3 prod vs dy29 overlap 0.985): the drop x1.63 -> x1.33 is most likely the TARGET, not the posts (overlap shows pattern similarity, not magnitude convergence).
- dx = pitch/10 lands tooth edges on cell edges (verified). pa4/pa5/pa7 had unresolved 60-nm gaps, posts outside the tube, PML 0.9-1.6 um,
  lambda outside the far-field window, and several factors changed at once; plain Born (x0.7-0.84) predicted x1.48-1.52 vs measured
  1.55/1.42 -> the failure was the kappa*x1 CALIBRATION term extrapolating, not Born.
- Model: response AMPLITUDE within ~+-10 % of Born on all 13 devices (power ratio 0.83-1.19, GPT recheck); coupled dipoles (Foldy-Lax) do NOT
  help (LOO worse); per-post complex weights barely help -> the leading suspect is the ELEMENT PATTERN (a free-oxide dipole beside a
  SiN strip); provisional, other model errors not excluded. A smooth angular correction (L = 4-6 harmonics) gives LOO rms log-gain error 0.057-0.062 (trained on production devices). Better element model: k_x-resolved 2D FDFD of a line
  dipole beside the local strip (zero GPU, ~1 day).
- The discrete design step UNDER-POPULATES rows (one post per NNLS cluster, no additions): a packed proxy (one r100 post per 300 nm) beats
  the script (3 rows 0.711 vs 0.756-0.761; |x| <= 45: 0.669 vs 0.723). Fixing the design step ~ as valuable as length.
- Architecture: drive decays x0.43 per 0.3 um and, under size/packing limits (packed proxy), 3 rows hold ~95 % of what more rows give (decay alone
  does not limit the unbounded ceiling); rows are partly redundant in pattern. The
  marginal value of a post at |x| < 4 um is 5-10x the arm value -> the FIRST layer should follow the local wall (clearance ~0.2 um; wall
  0.3 -> 0.9 um along x), not a straight y = 1.2 line. Needs in-situ drive: add x-line field monitors at y 0.75/0.9/1.05/1.2/1.5/1.8 (z 0).
- Pre-registered by Fable: **269488 x1.33-1.40** (Claude's calibrated x1.50 likely high); 269505 1-row x1.19-1.26, 2-row x1.30-1.38.
- Inverse design: right class, wrong moment (lumopt2 parametrises teeth; FD-through-mesher dEps infeasible for ~500 posts; ~6-7 GPU-h
  per iteration at Q_L 1e5). Do the measured-residual Gauss-Newton step first (same direction, 1 solve/iteration, existing code).
- **Fable's run plan (dy29, box 6.8, ~3 h each, every run adds the x-line monitors):** R1 pa3-DWBA-1: shrink/remove + packed adds designed
  on pa3's MEASURED dy29 residual, |x| <= 24 (model 0.647, half-step 0.680; accept >= x1.40) - NOTE: `fable_review/incr29.py` is a
  RELAXATION (no gap constraints, amplitudes below the 50-nm minimum allowed, no post list): R1 first needs discretisation, gap/wall
  checks, saved full + half designs and fresh predictions; the x-line field monitors are NOT implemented in the runner yet; R2 same at |x| <= 45 (model 0.609; the runner's span assert passes at
  |x| 45 although the tube is tapered beyond ~35 um: widen farfield_x_span_m to ~120 um + margins (15-um taper per end) or declare the run Q_i-only); R3 wall-following inner layer (|x| < 5 um, clearance 0.2 um, r <= 80) from R1's in-situ drive; R4
  DWBA-2 on the winner's residual; R5 matched fine pair for the winner (25-nm y band over the posts only); R6 -3 dB device of the winner
  (N from 269448). Expected endpoint x1.6-1.8 by R4 (INFERENCE); x2 needs R3 to deliver. If R1 misses its half-step prediction, build the
  element-pattern model first.

## Appendix — deep-review brief (re-usable for a Fable/GPT reviewer)
```
DEEP REVIEW + PATH TO x2 Q_i (verdict first; then a concrete ordered plan). Project: c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes
(read CLAUDE.md, docs/postarray_radiation_cancellation_2026-10-23.md, .claude/skills/postarray-design/SKILL.md, python_tools/radcancel/
postarray_{born,ceiling,design}.py, runners/sweeps/farfield_sph_15um_postarray_overshoot.py, bragg_device.py mesh code ~lines 770-870).
GOAL (user): Q_i x2 (+100 %) on the TE overshoot pi-shift grating (Nt80x: SiN n 1.97, 350 nm, oxide 1.444, pitch 514 nm, 96 periods/side,
lambda 1619.5 nm, apodized teeth up to 1800 nm wide) using planar full-height SiN posts beside the grating; a narrower mode is fine.
MEASURED (Q_i = Q_L/(1-sqrt(T gamma^2)), short runs, scored vs a stored control at identical numerics):
 - production mesh (dy 57.7 nm core, dx 51.4 = pitch/10, posts on the 50 nm background): control 5.35e5; 3-row array (238 posts/side,
   y 1.2/1.5/1.8 um, r 50-100, gaps >= 100 nm, |x| <= 24 um) 8.74e5 (+63 %); 4-row (319, r <= 120, cross-row gaps 60 nm, |x| <= 30)
   8.28e5 (+55 %); 5-row (379, to y 2.4) 7.59e5 (+42 %); 2-row 7.51e5 (+40 %).
 - CONVERGED core mesh dy 28.8 nm: control 5.67e5; the SAME 3-row array 7.56e5 = +33 %. Single comb +7.5 % (prod) vs +8.3 % (dy29).
 - Plain TE (1050-nm cavity) 3-row array: Q_i +50 %, mode 4 % narrower.
 - 13-device dataset: Q_i gain = 1/(measured far-field leak) (corr 0.983) -> loss ~all radiation. Born model (drive = control's tube field
   back-propagated to the post plane, point dipoles, free-oxide far field) ranks right (corr 0.976) but under-predicts ~1.6x; calibrated
   E = C(1 + kappa s x1) + s Bp (s 1.01, kappa 0.30) failed out of regime (pre-registered x1.81/x1.83 for 4/5 rows vs measured 1.55/1.42).
 - Model ceilings (production-mesh control): rows saturate after ~3 (1/2/3/5/10/15 rows: 25/37/40/42/45/47 % removable); length helps
   (|x| <= 24/30/45 um: 37/40/45 %); posts near the cavity (|x| < 4 um) matter most.
USER QUESTIONS / HYPOTHESES: (1) did we make mistakes in the 3/4/5-row designs? User recalls r <= 100 nm is ~linear, 120 nm is not. (2) Are
the mesh rules OK (dx lands on the period? posts on a coarse background? is dy 28.8 converged enough? should posts get their own fine
mesh?) and are the current results trustworthy? (3) How to improve the 3-row design and the MODELING (coupled dipoles / multiple
scattering, designing on the converged-mesh control, calibration, iterating around the measured pa3 field)? (4) Is it time for INVERSE
DESIGN (adjoint gradients of Q_i or radiated power w.r.t. post radii/positions in FDTD, starting from pa3 as the initial point; the repo
has a lumopt2 adjoint pipeline under runners/lumopt2_design)? (5) What would most plausibly reach x2 today, and what is the cheapest
discriminating next FDTD set (Athena A100, ~2.5-3.5 h per run, 7 license seats each, 4d_1g memory < 400 GB)?
Label MEASURED / DERIVED / INFERENCE / FROM-SOURCE; cite file:line or papers. Be adversarial about our reasoning.
```

## 11. Open user questions at handoff (2026-10-23, late)
- **Placement region shape.** The box (|x| <= 30, y 1.1-2.0 um) is a MODEL-validity box, not physics: inner y = drive back-propagation
  valid only outside the widest tooth (0.9 um); outer y = tube face 2.104 um minus clearance; |x| = untapered tube range. Proposed
  (zero GPU): inner boundary = fixed clearance from the LOCAL sidewall (0.48 um near the cavity -> 0.9 um at the widest teeth), which
  needs a DIRECT in-situ field near the cavity (x-line monitors) instead of back-propagation; let NNLS pick the shape inside
  {clearance from wall} AND {inside tube}, then read the amplitude map's contour (circle/ellipse emerges or not).
- **Beyond the point dipole.** k*r in oxide 0.28-0.56 -> quadrupole terms ~10-30 %. Options, ranked: (1) embedded element pattern by
  reciprocity (post radiation = overlap with the field a far-field plane wave makes on the BARE strip, not free oxide; Fable's leading
  model-error suspect); (2) cylinder T-matrix / multipole (dipole + quadrupole); (3) FDTD adjoint Jacobian on post radii/positions as
  the refinement step; also non-circular elements (x-bars/ellipses change the polarizability tensor).
- FSP of the best measured device (c29, x1.42, dy29, build-only): `results_from_athena/comb_overshoot/fsp/c29_pa3_r100_x30_dy29_best_x1p42.fsp`.
- **Clarification (user asked: is the box optimal?): NO.** The box is only where the model is valid; it is not an optimal limit. A
  circle/ellipse should not be imposed either: inside a given allowed region the model already chooses where posts go, so a smaller
  region (circle inside the box) can only do equal or worse. Right approach: make the allowed region as large as validity allows
  (inner edge = clearance from the LOCAL wall, needs direct near-cavity field; outer edge = tube face, 3.1 um with box 8.8), let the
  model place posts, and READ the shape from its amplitude map. A shape only matters under a post budget: then keep the contour that
  holds most of the removal per post (175-post design ~ 504-post design suggests most value sits in a compact core region).
- **Port normalization (CLAUDE.md 2) is NOT applied in post-array scoring.** solve_v4.py uses the raw result T (|S21|^2) in
  Q_i = Q_L/(1-sqrt(T)). DERIVED effect (bias +0.006 on both): control Q_i +2.0 %, c29 +2.6 % -> ratio x1.42 -> ~x1.41; absolute Q_i
  ~2-3 % high (partly offset by the short-run T ~0.006 low). Fix = ONE straight-guide calibration at the far-field-scene dy29 numerics
  (~25 min), reused for every device; stored portref_nt80 is a different scene/mesh (identity not checked). User 2026-10-23: NOT needed now (effects matter, not the last digit); do it only for the final best-result validation.

## 12. Late Fable planning review (2026-10-24) - SUPERSEDES the run ORDER of section 7 (same building blocks, re-anchored on c29)
- Estimate (DERIVED by Fable): measured/model removed fraction on c29 = 0.301/0.244 = 1.23; applied to the tube-valid 2-D design
  (model 0.666) -> P/Pc ~0.59 -> x1.7. x2 needs the inner wall-following layer near the cavity (|x| < 6 um, y 0.5-0.8 um: wall is
  620 nm at |x| < 5 vs 900 nm beyond, drive x2.5-3 that of y 1.2). Rows, x-extent and r <= 120 are exhausted levers (r 100 -> 120
  buys 0.5 pt; |x| > 15 adds little at realizable radii). Designs are inter-row cancellations: a 50 nm x-shift 0.69 -> 0.80 (fab risk).
- Order: zero-GPU O1 re-score all converged devices with plain Born x0.7 on the dy29 control (one multiplicative calibration, drop
  kappa*x1); O2 project c29's MEASURED residual onto arch14; O3 discretize (gaps/wall/tube asserts, full + half-step predictions).
  R1 bare Nt80x with the cavity trimmed to c29's lambda shift, no posts, + x-line field monitors y 0.55/0.7/0.85/1.0/1.2/1.5 (runner
  edit needed): cavity-enlargement null (expect Q_i within +-3 % of 5.67e5) AND the direct inner-layer drive. R2 DWBA-1 on c29's
  residual (pass: >= x1.55). R3 inner wall-following layer (0.2 um clearance, r <= 80) from R1's drive = the x2 test, with a matched
  lambda/width/core-energy check. R4 DWBA-2 if R2 confirmed. R5 matched fine pair + port calibration, R6 direct -3 dB, winner only.
- Fable also corrected a premise: in lossless FDTD the only non-port loss IS radiation; the doc's "share 0.8-0.9" mismatch is the
  energy proxy + T bias, not a missing loss channel. Reviewer scripts: scratchpad review_ceil/review_design/review_capped.py (not kept).
- **Fable FINAL version (same review, later message) - use this over the bullets above where they differ:** realisable endpoint with
  posts at y >= 1.1 is x1.5-1.65 (INFERENCE), not x1.7; x2 needs the inner layer. Binding limit = the >= 100 nm gap / discreteness
  (capped optimum is a quasi-continuous low-amplitude cloud over ~3700 of 3844 sites), not r or length; the "length helps" scans were
  uncapped. Drop: |x| <= 45, rows > 4, r <= 120 / 50-nm-gap designs. Single pa3 rows each RAISE the leak (1.01/1.27/1.08); only the
  sum cancels -> the fragile quantity is the model's inter-row PHASE (element pattern or near-field coupling).
  Offline: O1 capped (r <= 100) + gap-aware ceiling on the CONVERGED Gram for |x| <= 15/24/30, rows <= 4 (lsq_linear/bvls took 5-7 h
  at n 3844: use nnls with a variable split or projected gradient); O2 score 269505 vs Fable bands (fixes model scale, no run); O3
  discriminant table (dlam, Q_c, width, core-energy fraction) for all converged devices; O4 angular map of c29's measured vs model
  residual. Runs (all with x-line monitors y 0.75-1.8): R1 DWBA-1 on c29's measured residual, |x| <= 24 (expect x1.5-1.55); R2
  cavity-shift control (+0.1-0.25 nm, no posts), concurrent with R1; R3 inner layer from R1's in-situ drive, accept only if
  dlam <= 0.1 nm, Q_c within 3 %, width not wider, gain beats R2; R4 DWBA-2 if R3 >= x1.7; R5 fine pair; R6 -3 dB device.
- **Earlier Fable code review (production-mesh era) - items not yet fixed:** `solve_v4.py:36` uses t0 = 5 ps; a DFT from 0 ps is flat
  to 0.2 % and t0 = 5 biases T +1 % on both devices (Q_i +3-4 %, RATIO only +0.5 %). Coupling symmetry verified (port ring-down
  amplitudes 0.9935 / 1.0048 -> Q_i bias < 0.1 %). `postarray_born.py:26` default KCUT 9 vs 7.5 used for designs (pin one).
  `postarray_design.py:19-25` seeds one capped post per NNLS cluster with fixed N (no splitting): discretisation costs ~5 pt.
  `postarray_ceiling.py:31` "any size" ceiling allows ~4x pa3's polarizable volume (not physical). Positivity (no sign flip from a
  shift) costs ~10 pt. None of these change a decision now; fix t0 and KCUT before the final numbers.
