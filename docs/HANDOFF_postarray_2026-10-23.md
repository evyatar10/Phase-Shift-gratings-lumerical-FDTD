# HANDOFF — SiN post arrays that cancel the radiated leak (TE pi-shift gratings), 2026-10-23 evening

Read first: this file, then `docs/postarray_radiation_cancellation_2026-10-23.md` (method + full results table), the skill
`.claude/skills/postarray-design/SKILL.md` (recipe + every validity limit), memory `project_radcancel_overnight_2026-10-23.md` (dated log).
Branch `claude-te`; commits 2ed1f68, e6e17f1, 5bd3971 (+ this handoff). Summary page for the user: https://claude.ai/artifact/FTLf6cia2xv9WodhGgEnuW

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
- Converged-control work dir: `results_from_athena/comb_overshoot/postarray_work_dy29/` (Grams c3x30, c5x30, arch14 pending; designs).
- Converged-mesh predictions (calibrated, r<=100, gaps>=100, abs(x)<=30): best 1 row y 1.2 (36 posts) x1.30; 2 rows x1.46; 3 rows x1.50;
  greedy post insertion adds ~1 % -> this regime saturates near x1.5 IN THE MODEL (not a physical limit). Ceilings: 1/2/3 rows 25/37/40 %
  removable; 5 rows (converged) 43.4 %; abs(x) 24/30/45 um 37/40/45 % (length is a model claim; TM evidence says otherwise).

## 4. Running on Athena at handoff (watcher key file: scratchpad watch19.sh; seats ~42/50)
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
1. Score 269488 / 269505 / 269442 / plain pair vs the pre-registered numbers; refit the calibration on CONVERGED devices only.
2. Separate cancellation from cavity enlargement: (a) model one-row phase/distance scan (oscillating Q_i = cancellation, monotonic =
   loading); (b) one FDTD control: bare Nt80x with a small cavity change giving the same +0.24 nm shift, no posts.
3. Architecture (GPT 01a16951, literature): optimise posts over a 2-D allowed region with LOCAL tooth clearance instead of fixed straight
   rows; posts nearer the cavity need a DIRECT stored field -> one control run with a field monitor over the placement region (same
   numerics as 267605_0). Read `postarray_work_dy29` arch14 greedy row selection (scratchpad arch.txt) when it lands.
4. Model upgrade: self-consistent coupled dipoles (post-post), rank strong posts; keep r <= 100 / gaps >= 100 until measured otherwise
   (the user is NOT sure of a 100/120 nm threshold; TM data: nonlinearity grows with drive strength and post proximity).
5. Validation of the final winner only: matched fine pair (variants fine/finectrl: dx 25.7 + 25 nm post band + dy29 core; 275 GB QOS cap,
   ~14 h/run) and a direct -3 dB device using 269448's Q_c(N).
6. Later: post-only adjoint (objective 1/Q_i, width upper bound only) starting from the best model design.

## 6. Pitfalls already paid for (all in CLAUDE.md / skill)
Widest tooth 1800 nm (drive validity + clearance); file-tag collisions for scatterer lists; 4d_1g caps memory at 275 GB; far-field tube
y 2.104 / abs(x) <= 35 and the +-0.25 nm 2D window; same-phase pairs on asymmetric arms cancel; optimizer cancels mesh artefacts.

## 7. Pending reviews at handoff
Fable deep review (max effort; mistakes, mesh, modeling, inverse design, architecture) — agent still running in the old chat; its
verdict was not yet received. Re-run if needed with the brief in the appendix below.
GPT sessions logged in `docs/ASK_GPT_BRIEF.md` (01a165e7, 01a16603, 01a16842, 01a16881, 01a1688a, 01a1691c, 01a16951).

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
