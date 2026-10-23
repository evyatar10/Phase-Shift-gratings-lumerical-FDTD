# HANDOFF — SiN post arrays that cancel the radiated leak (TE pi-shift gratings), 2026-10-23 evening

Read first: this file, then `docs/postarray_radiation_cancellation_2026-10-23.md` (method + full results table), the skill
`.claude/skills/postarray-design/SKILL.md` (recipe + every validity limit), memory `project_radcancel_overnight_2026-10-23.md` (dated log).
Branch `claude-te`; commits 2ed1f68, e6e17f1, 5bd3971, d36452e (handoff), d533a67, abffc1a (+ later handoff edits). CLAUDE.md has UNCOMMITTED
rule edits (10-20..10-23) awaiting the user's OK to commit. Summary page for the user: https://claude.ai/artifact/FTLf6cia2xv9WodhGgEnuW

## 1. Goal and rules (user)
- Goal: **x2 Q_i** (spec = Q at -3 dB ~ 0.293 Q_i) on the **TE overshoot** grating Nt80x (SiN n 1.97, 350 nm, oxide 1.444, pitch 514 nm,
  96 periods/side, cavity 950.3 nm, lambda 1619.5 nm, apodized teeth up to 1800 nm wide). Plain TE (1050-nm best cavity) is the test bed.
- New planar concept only: full-height SiN posts in the same layer; nothing reachable by tooth widths/shifts/apodization/cavity size counts.
- **A narrower mode is fine/good**; only widening counts against a device.
- Validate expensively (fine mesh, -3 dB device) only the BEST design, after a bug/geometry audit.
- Never state model ceilings/plateaus as physical limits; test one factor per FDTD run; model-first design, inverse design later.
- User concern: posts close to the cavity can act as an **effective-cavity enlargement** (mode perturbation = inverse-design territory),
  not cancellation. Always separate the two.

## 2. What is established (MEASURED unless marked)
| Device (Q_i vs own control, same numerics) | Mesh | Gain | Notes |
|---|---|---|---|
| Overshoot 3-row array pa3 (238 posts/side, y 1.2/1.5/1.8, r 50-100, gaps >= 100, abs(x) <= 24) | production dy 57.7 | x1.63 | reproduced by 3 independent checks; no bug |
| same pa3 | **converged dy 28.8** | **x1.33** | THE honest number; T 0.671->0.735, width held |
| 2-row pa2 / 4-row pa4 / 5-row pa5 | production | x1.40 / x1.55 / x1.42 | pa4/pa5 r<=120 + 60 nm gaps: model failed out of regime |
| comb pair (independent phases) / single comb | production | x1.17 / x1.075 (x1.083 dy29) | arms are NOT mirror images -> own phases |
| Plain TE 3-row array (162 posts/side, y 0.8/1.1/1.4) | production | x1.50, width -4 % | converged-mesh check running (269500/269501) |
- Stored controls: overshoot production `results_from_athena/comb_overshoot/control_nt80x_T340.mat`, dy37 `control_nt80x_dy37.mat`,
  dy29 `results_from_athena/budget_overshoot/dy29_box6p8.mat` (267605_0) and `dy29_box8p8.mat` (267605_1); plain TE production
  `results_from_athena/postarray_plain/control_plain_c1050.mat` (1050 cavity) and `control_plain_c800_roundA.mat`. Device results next to them.
- Loss is ~all radiation: Q_i gain = 1/(measured far-field leak), corr 0.983 over 13 devices.
- **Why +63 % fell to +33 %:** pa3 was designed against the production-mesh leak, which contains staircase-mesh radiation at mid angles
  (abs(u_x) 0.5-0.8); the array "cancelled" that artefact. The real effect = the grazing leak (abs(u_x) > 0.95) halved on both meshes. Rule now
  in CLAUDE.md: design and judge on the converged mesh; production gains are screening only.
- Cavity enlargement vs cancellation: pa2 (inner row 1.5 um) shifts lambda only +0.03 nm yet +40 % (prod); pa3 (inner row 1.2) +0.24 nm;
  coupling Q unchanged 1.22e5 -> mostly cancellation, but pa3's inner row may add a mode-perturbation part (UNTESTED).

## 3. Model state
- Born dipole model (`python_tools/radcancel/postarray_born.py`): drive = stored control's tube side-plane field (y 2.1 um) back-propagated
  through oxide (valid only outside the widest tooth, y > 0.9 um); short- and steady-run normalisation. `postarray_ceiling.py` (Gram + NNLS),
  `postarray_design.py` (discrete posts, all-pairs gap, per-row r cap, X_LIM binds positions).
- Calibrated form E = C(1 + kappa s x1) + s Bp (s 1.01, kappa 0.30; `postarray_work/calib_s_kappa.npy`), fit on 13 production devices;
  on the CONVERGED control it predicts pa3 x1.37 vs measured x1.33 (good). Out of regime (big/dense posts) it fails (pa4/pa5).
- Converged-control work dir: `results_from_athena/comb_overshoot/postarray_work_dy29/` (Grams c3x30 and c5x30 saved; arch14 = 14 candidate
  rows y 1.1-2.4 at 100 nm sites, may or may not have been saved when the old chat ended; designs). Row choice / per-row length tool:
  `python python_tools/radcancel/postarray_rowselect.py <ceiling_G file> [max_rows]` (greedy best rows + 90 % x-extent per row).
- Converged-mesh predictions (calibrated, r<=100, gaps>=100, abs(x)<=30): best 1 row y 1.2 (36 posts) x1.30; 2 rows x1.46; 3 rows x1.50;
  greedy post insertion adds ~1 % -> this regime saturates near x1.5 IN THE MODEL (not a physical limit). Ceilings: 1/2/3 rows 25/37/40 %
  removable; 5 rows (converged) 43.4 %; abs(x) 24/30/45 um 37/40/45 % (length is a model claim; TM evidence says otherwise).
- Can 3 rows get back near +64 % on the CONVERGED mesh? The +63 % itself was partly artefact (production control Q_i 5.35e5 vs 5.67e5
  converged), so the old array cannot be 'fixed' back to it. But the converged 3-row ceiling is 41 % removable = x1.70 in the model; the
  realistic design (r <= 100, gaps >= 100) predicts x1.50 (269488 tests it). The gap x1.50 -> x1.70 is constraints + optimizer, not physics:
  close it with single-factor tests (r <= 110 / gaps >= 80; free (x,y) placement incl. near-cavity pockets with a direct field;
  coupled-dipole model). Converged 4/5-row designs (c5x30, r <= 100, gaps >= 100, |x| <= 30; plain Born x0.7 leak): 3 rows 0.756
  (175 posts), 4 rows 0.723 (230), 5 rows 0.763 (241, worse = the under-populating design step, not physics; its ceiling is 43 %).
  Files `postarray_work_dy29/design_c5x30_R100_rows0123_x30.npz` / `rows01234_x30.npz`. Fix the design step before trusting row trends.

## 4. Running on Athena at handoff (the old chat's watchers die with it: re-arm a queue watcher; seats ~42/50)
| Job | What | Score against | Pre-registered |
|---|---|---|---|
| 269442 | 7-row pa7 (r<=120, 50 nm gaps), box 8.8, dy29 | stored 267605_1 (dy29 box 8.8, Q_i 5.70e5) | x1.81 (production calibration; likely lower) |
| 269488 | c29 3-row designed ON the converged control (175 posts) | 267605_0 (`budget_overshoot/dy29_box6p8.mat`) | **x1.50** |
| 269505 (after 269442) | c29_r1_x30 (1 row, 36 posts) / c29_r2_x30 (2 rows, 90) | 267605_0 | x1.30 / x1.46 |
| 269500 / 269501 | plain TE dy29 control / plain 3-row array at dy29 | each other | plain +50 % at production |
| 269448 (2 tasks) | bare Nt80x at 108 / 116 periods (-3 dB length calibration) | Q_c(N) = 1/(1/Q_L - 1/Q_i) | reusable for any winner |
Scoring: `python python_tools/radcancel/solve_v4.py --short --energy <control.mat> <device.mat>` (plain: no --short). Results dirs on
Athena: `~/bragg_sim_athena/results/farfield_sph_15um_postarray_overshoot_<variant>/results/*_ff.mat`, `..._postarray_plain_dy29*`,
`..._m3db_overshoot`, `..._postarray_overshoot_r7`. Copy to `results_from_athena/comb_overshoot/` or `postarray_plain/`. pa4/pa5/pa7 far-field
P/Pc invalid (posts outside the tube / lambda outside the +-0.25 nm 2D window); Q_i, T, width valid.

## 5. Next steps (ordered; propose each run to the user first, CLAUDE.md 6)
0. **Primary plan = Fable's R1-R6 in section 7** (measured-residual iteration, then length, then a wall-following inner layer with
   in-situ drive monitors). The items below are the supporting checks.
1. Score 269488 / 269505 / 269442 / plain pair vs BOTH pre-registrations (Claude calibrated vs Fable plain-Born ranges); refit the
   calibration on CONVERGED devices only (the kappa*x1 term is suspect).
2. Separate cancellation from cavity enlargement: (a) model one-row phase/distance scan (oscillating Q_i = cancellation, monotonic =
   loading); (b) one FDTD control: bare Nt80x with a small cavity change giving the same +0.24 nm shift, no posts.
3. Architecture (GPT 01a16951, literature): optimise posts over a 2-D allowed region with LOCAL tooth clearance instead of fixed straight
   rows; posts nearer the cavity need a DIRECT stored field -> one control run with a field monitor over the placement region (same
   numerics as 267605_0). Row distances/lengths: run `postarray_rowselect.py` on ceiling_G_c5x30.npz (or arch14 if saved).
4. Model upgrade: NOT coupled dipoles (Fable: tested, no gain); element-pattern correction (angular L 4-6, or 2D FDFD line dipole); keep r <= 100 / gaps >= 100 until measured otherwise
   (the user is NOT sure of a 100/120 nm threshold; TM data: nonlinearity grows with drive strength and post proximity).
5. Validation of the final winner only: matched fine pair (variants fine/finectrl: dx 25.7 + 25 nm post band + dy29 core; 275 GB QOS cap,
   ~14 h/run) and a direct -3 dB device using 269448's Q_c(N).
6. Later: post-only adjoint (objective 1/Q_i, width upper bound only) starting from the best model design.
7. Harmonics (user 2026-10-23: "could they still help?") - not as the objective (Parseval), but: (a) **mesh-stable target**: compare the
   control's harmonic tables at production / dy37 / dy29 (stored: comb_overshoot/control_nt80x_T340.mat, comb_overshoot/control_nt80x_dy37.mat, budget_overshoot/dy29_box6p8.mat; tool python_tools/farfield_multipole.py) and design only against mesh-stable harmonics so the
   optimizer cannot chase staircase-mesh radiation (the cause of +63 % -> +33 %); (b) per-harmonic model-vs-FDTD response to find where
   the Born model is wrong and weight/correct those components; (c) harmonic content of pa3's residual leak -> which transverse patterns
   (row distances) are still uncancelled.

## 6. Pitfalls already paid for (all in CLAUDE.md / skill)
Widest tooth 1800 nm (drive validity + clearance); file-tag collisions for scatterer lists; 4d_1g caps memory at 275 GB; far-field tube
y 2.104 / abs(x) <= 35 and the +-0.25 nm 2D window; same-phase pairs on asymmetric arms cancel; optimizer cancels mesh artefacts.

## 7. Fable deep review (max effort, finished 2026-10-23 ~19:30) — the most important input for the next chat
Scripts + data (copied, keep): `results_from_athena/comb_overshoot/postarray_work_dy29/fable_review/` (qi_check, design_audit,
dataset_models, ff_patterns, cda_test, mep_test, drivecheck, born29, gram29, incr29, pack29, arch29 .py; ff_C29/C37/PA3_29.npz;
gram_dy29_r3*.npz; work29/). Its findings (labels as given by the reviewer):
- **+33 % is real**, +-5 pts from T and ~+-5 from mesh (MEASURED re-derivation; raw-T floor x1.13). Leak-pattern overlap prod vs dy37 0.68,
  dy37 vs dy29 0.83, dy29 box 6.8 vs 8.8 0.991 -> numerical part of the leak ~25 % (prod), ~10 % (dy37), ~6 % (dy29) (INFERENCE). The
  posts' RESPONSE is mesh-stable (pa3 prod vs dy29 overlap 0.985): the drop x1.63 -> x1.33 is the TARGET, not the posts.
- dx = pitch/10 lands tooth edges on cell edges (verified). pa4/pa5/pa7 had unresolved 60-nm gaps, posts outside the tube, PML 0.9-1.6 um,
  lambda outside the far-field window, and several factors changed at once; plain Born (x0.7-0.84) predicted x1.48-1.52 vs measured
  1.55/1.42 -> the failure was the kappa*x1 CALIBRATION term extrapolating, not Born.
- Model: amplitude-linear to +-10 % up to r 100 on all 13 devices (|B_meas|^2/|B_Born|^2 0.94-1.07); coupled dipoles (Foldy-Lax) do NOT
  help (LOO worse); per-post complex weights barely help -> the missing physics is the ELEMENT PATTERN (a free-oxide dipole beside a
  SiN strip). A smooth angular correction (L = 4-6 harmonics) gives LOO 0.057-0.062. Better element model: k_x-resolved 2D FDFD of a line
  dipole beside the local strip (zero GPU, ~1 day).
- The discrete design step UNDER-POPULATES rows (one post per NNLS cluster, no additions): a packed proxy (one r100 post per 300 nm) beats
  the script (3 rows 0.711 vs 0.756-0.761; |x| <= 45: 0.669 vs 0.723). Fixing the design step ~ as valuable as length.
- Architecture: drive decays x0.43 per 0.3 um -> 3 rows hold ~95 % of what more rows give; rows are partly redundant in pattern. The
  marginal value of a post at |x| < 4 um is 5-10x the arm value -> the FIRST layer should follow the local wall (clearance ~0.2 um; wall
  0.3 -> 0.9 um along x), not a straight y = 1.2 line. Needs in-situ drive: add x-line field monitors at y 0.75/0.9/1.05/1.2/1.5/1.8 (z 0).
- Pre-registered by Fable: **269488 x1.33-1.40** (Claude's calibrated x1.50 likely high); 269505 1-row x1.19-1.26, 2-row x1.30-1.38.
- Inverse design: right class, wrong moment (lumopt2 parametrises teeth; FD-through-mesher dEps infeasible for ~500 posts; ~6-7 GPU-h
  per iteration at Q_L 1e5). Do the measured-residual Gauss-Newton step first (same direction, 1 solve/iteration, existing code).
- **Fable's run plan (dy29, box 6.8, ~3 h each, every run adds the x-line monitors):** R1 pa3-DWBA-1: shrink/remove + packed adds designed
  on pa3's MEASURED dy29 residual, |x| <= 24 (model 0.647, half-step 0.680; accept >= x1.40); R2 same at |x| <= 45 (model 0.609; needs tube
  x-span >= 110 um or a Q_i-only read); R3 wall-following inner layer (|x| < 5 um, clearance 0.2 um, r <= 80) from R1's in-situ drive; R4
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
