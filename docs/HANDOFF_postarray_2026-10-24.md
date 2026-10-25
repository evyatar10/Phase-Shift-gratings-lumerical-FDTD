# HANDOFF — SiN post arrays on the TE overshoot pi-shift grating, 2026-10-24 (evening)

> ★★★ PRIORITY (user 2026-10-24): the chat that takes this handoff is the MOST IMPORTANT chat — it has priority over every other
> chat running on other subjects (e.g. the twin-cavity chat, worktree `..._twin`, server base ~/bragg_sim_twin). It owns
> ~/bragg_sim_athena and the shared working tree; other chats must not deploy into them and must yield seats to it.


Supersedes `docs/HANDOFF_postarray_2026-10-23.md` for state; that file still holds the method background. Branch `claude-te`
(user commit b8025ed holds the engine edits). Dated log: memory `project_postarray_night_2026-10-23.md`.

## ★★ UPDATE 2026-10-25 evening — CURRENT STATE (supersedes the morning update below for state)
- BEST MEASURED: v21_seedfy Q_i x2.290 (1.272e6; T 0.824; Q_L 117390; W 16.114; 245 posts/side, 2-D, diameter 100-200 nm, gaps >= 100,
  |x| <= 42 um), Athena 271566, file results_from_athena/comb_overshoot/v21_seedfy_dy29.mat, layouts layout_v21_seedfy{,_stretched,_true_aspect,_phone}.png.
  Row reference ext42 x2.022. All gains use the CORRECTED solve_v4 (T truncation fix 4ced19e: older docs ~3 % high).
- MODEL v2.1 (coupled dipoles + strip re-radiation + mode reshaping, fitted on Q_i): code python_tools/radcancel/model_v2/{model_v2,design_v2}.py,
  weights/data/results in results_from_athena/comb_overshoot/model_v2_work/ (RESULTS_model_v2.md, PREREG_*.json, design_*.json).
  Blind record: ext42_hx 1.708 (meas 1.711), uni 0.805 (0.798), c2d_full 1.475 (1.361), v21_seedfy 2.44 (2.290), v21_scratch 2.14 (1.922)
  -> over-predicts its own optima by 6-10 %: pre-register designs at 0.94 x v2.1.
- RUNNING: Athena 272017 (variant v21b, 4 tasks, ~3.5 h): v21b_rows (342 posts, v2.1 x2.63), v21b_y1500 (264, x2.68), v21b_y1700 (352, x3.00),
  v21b_acons (CONSERVATIVE fab diameter >= 150 / gaps >= 150, 366, x3.14). Pre-reg 0.94 x v2.1 (band 0.90-0.98).
- MEASURED today: x-shift knobs -> ext42 at its x-optimum; uniform rows x0.80; fine 25 nm mesh keeps the gain (x1.98 vs 1.85, resonance -4.2 nm:
  re-centre windows for ANY mesh change); short-run method validated (Q_L exact, coupling symmetric); single post beside strip ~ Born (probe 271246).
- USER RULES (10-25): fix the cloud model first; fab limits diameter >= 100 / gaps >= 100 (nothing tighter for now); posts |x| <= 42 um;
  y range is not a fab limit (include only if >= ~5 % gain); chase only >= ~5 % gains; NO adjoint for now (design note docs/ADJOINT_POSTS_DESIGN_2026-10-25.md);
  Fable for big decisions only, GPT as main worker (ask-gpt --scratch), Opus subagents execute. Experimental Nt80x chip data (N94/96/98) PARKED (theory first).
- NEXT: score 272017; refit v2.1 with all new devices; next lineage step from the best measured field; GPT ideas/lines studies in scratch (gptIDEAS, gptLINES).

## ★ UPDATE 2026-10-25 morning (Claude, autonomous block) — read this first
- BEST MEASURED (dy29, Q_i vs control 5.67e5): x42_r0910 x2.096 (352 posts/side) ~= x42_j3 x2.098 (392) > nw_r2pack_ext42 x2.076 (309, 270593) >
  nw_r2pack x1.887. Files results_from_athena/comb_overshoot/{x42_r0910,x42_j3,nw_r2pack_ext42}_dy29.mat; layout fig layout_nw_r2pack_ext42.png.
  ext42 beat its model (+10 % vs +6.1 %); the x42 add-ons gave 1/4-1/3 of their model gain. Repeat 270650 is BIT-IDENTICAL (run noise 0).
- 2-D diagnosis (Fable + GPT, scratch .../scratchpad/fable2d/DIAG.md): shadowing refuted; loading not the cause; y >= 1.75 posts weak (element pattern);
  at MATCHED length cloud == rows; under one plain-Born yardstick the measured/model factor is clouds 0.97-1.26, rows 1.04-1.21, nw-family 1.32-1.39,
  ext42 1.49. GPT: for w4_rs50 tube power fell 3 % while 1/Q_i rose 4 % -> the radiation objective and Q_i disagree at the increment level.
- RUNNING: 270952 (uni_sub500 uniform control, model x0.95; ext42_hx = ext42 shifted +25.7 nm in x, NOT a pure noise test: also moves posts vs grating),
  270981 (c2d_full = fresh full-footprint free 2-D cloud, 199 posts, PRE-REG x1.6-1.9, kill < 1.65 = the 2-D gate test), 270754/755 fine2 (refined mesh).
- MODEL v2 in progress (Opus agent, scratch .../scratchpad/modelv2/RESULTS.md): step 0 reconcile tube P vs Q_i; then mesh-strength per radius,
  finite height (sinc(k u_z h/2)), per-angle element pattern, own in-situ drive; held-out-FAMILY gate; predict(posts, h, pol). Small-scene post-beside-strip
  probe being built (runners/single/post_strip_probe.py, NOT dispatched — needs user "run").
- USER RULES (2026-10-25): fix the cloud model first (fab limits r>=50 / gap>=100 kept while fixing; smaller posts later); model gate on stored data
  before ANY model-based dispatch; physics on small scenes; full device = batched confirmation (skill postarray-design WORKFLOW RULE). Adjoint still gated
  (asked user to lift it; no answer yet). Fable quota ~70 % used -> GPT for reviews, Opus for execution, Fable for decisions.

## 0. THE GOAL — go well beyond x2, toward x3 (user, emphatic)
x1.89 is a waypoint, not the result. The user wants > x2 and is aiming at x3 Q_i on the fabricated 96-period device; do not stop at
"good enough", do not declare plateaus as limits (model ceilings are conditional). Where the remaining leak is (DERIVED, nw_r2pack
angular split, fraction of the control's total): mid-angle in-plane |u_x| 0.5-0.9 = 0.24 (~48 % of what is left, the biggest),
grazing |u_x| > 0.95 = 0.12 (~25 %), broadside |u_x| < 0.5 = 0.09, steep vertical |u_z| > 0.6 = 0.10 (overlaps; barely reduced so far).
Paths, in order: (1) the refined-mesh re-baseline (sec. 6) and redesign on it; (2) designs that explicitly target the mid-angle and
vertical residual (planar posts CAN radiate vertically: p_y dipoles radiate along z); (3) extension along x (270593 tests |x| <= 42) since
the outer posts carry ~1/3 of the gain; (4) fix the model's element-pattern error so free 2-D designs beat rows; (5) post-only inverse
design once the 2-D gate passes; (6) finer textures (tier C: posts < 100 nm, gaps < 100 nm) on a converged post mesh - the model's
continuum ceiling is x3.2, blocked mainly by discreteness at r >= 50 / gaps >= 100.

Vertical / mid-angle physics (for path 2; user asked, 2026-10-25): every post is a planar, full-height (350 nm) dipole driven mostly
along y (in-plane, transverse). A y-dipole radiates in all directions except along y - strongly up/down (+-z) and along x - so planar posts
CAN cancel the vertical leak; no height change is needed. The phase each post contributes is set by its x position (cavity-field phase
varies along x) and its strength by its radius. Each post radiates its fixed pattern into all directions at once, so cancelling one band
trades against others; the designs so far mostly cancelled the grazing band (largest in the control). Next designs should weight the
mid-angle and vertical bands in the objective (e.g. band-weighted far-field cost) and check in the model, before any run, how much of
them a single planar layer can remove without giving back the grazing gain (INFERENCE: not yet quantified).

## 1. Goal and standing user rules (2026-10-24)
- Maximise Q_i of the FABRICATED 96-period TE overshoot device (Nt80x, SiN 350 nm in oxide, 1619.4 nm) with full-height SiN posts in the
  same layer. Only far-field cancellation counts; cavity loading (lambda shift, narrower mode) is neutral. x2 then toward x3.
- Report at N = 96: peak T and Q_i; the -3 dB Q is quoted as 0.293 Q_i (validated, sec. 2).
- No new long validation runs (> ~6 h) unless the user asks; a long run well under way is let finish.
- GATE: no inverse design / FDTD adjoint until a FREE 2-D ("cloud") design is MEASURED >= the best row-based device.
- Fab baseline: post diameter >= 100 nm, gaps >= 100 nm is FINE (Technion EBPG 5200; report results_from_athena/comb_overshoot/
  fab_limits_sin_posts.md). Smaller (tier C) only after the mesh probe (sec. 4).
- Model routing (budget): Opus manages, ONE live Fable agent for model/design science (continue via SendMessage), GPT-6-Astra reviews.
- Every dispatch: assert designs in the runner (`FF15_PA_SET` variants), check seats from IGUM (<= ~42-45/50), log the job ID.

## 2. Measured state (dy29 core mesh, box 6.8, 340 ps short runs, Q_i from ring-down + T x gamma^2; control Q_i 5.67e5)
- Best: **nw_r2pack = 2 packed rows (y 1.2/1.5 um, 211 posts/side) + 9 near-cavity r50 posts: Q_i 1.07e6 = x1.89, peak T 0.671 -> 0.801
  (-1.73 -> -0.96 dB), Q_L 1.03e5 -> 1.12e5, width unchanged (16.13 -> 16.12 um), dlam +0.14 nm.** Ellipse variant x1.900 (difference unresolved, within noise).
- -3 dB version (N 117, Athena 269640): T 0.513 (approximately -3 dB), **Q3dB 3.07e5** (predicted 3.13e5), Q_i 1.08e6 -> the 0.293 Q_i formula holds to 2 %.
- Measured designs ranked (registry regenerated 2026-10-25): `results_from_athena/comb_overshoot/postarray_design_registry.md` (post lists in
  `runners/sweeps/postarray_overshoot_designs.json`, result files `results_from_athena/comb_overshoot/<name>_dy29.mat`).
- What failed: model increments on nw_r2pack (resize x1.813/1.790, + small posts x1.87, + row 1.8 x1.85; all ~0.9 of prediction);
  free 2-D clouds x1.46-1.60 (model said better); 3rd packed row +0.03.
- Knob wave (270198): row radii -10 nm -7 %, +<=10 nm -11 %, rows +50 nm -9 %, near-cavity r60 0 % -> no tested global knob improved nw_r2pack (finite knob set: consistent with, not proof of, a local optimum).

## 3. Model state (DERIVED)
- Born dipole model, drive = control tube field back-propagated (y >= 1.1) or the DIRECT in-situ XY map (`python_tools/radcancel/
  postarray_nearfield.py`; gates passed 0.98-1.00). Tools: postarray_dwba.py, postarray_dwba_design.py, postarray_nearwall.py,
  postarray_place2d.py.
- Fable recalibration (`results_from_athena/comb_overshoot/postarray_work_dy29/fable_recal/RESULTS_recal.md`): ranks all devices
  (Spearman 0.97, LOO ~5 % in log P) but misses +-5 % increments. In-situ test: using each device's own measured field at the posts does
  NOT reduce the ~50 % unexplained response -> error = element pattern (post beside the SiN strip) / core-radiation change, not drive.
- GPT reviews (log in `docs/ASK_GPT_BRIEF.md`): fragility vs loading confounded; the class-weighted model is non-PSD (don't optimise it);
  recipe for 2-D = bounded correction on nw_r2pack with loading caps (dlam <= 0.18 nm, |dW| <= 0.02).
- Model ceilings: continuum SiN x3.2, but discreteness at r >= 50 / gaps >= 100 binds (~x1.9-2.2). Literature: partly novel
  (`postarray_work_dy29/lit2.md`; closest Acharyya & Kozyreff PR Appl 2019, Mekis US6853789, Rakich US8625939).

## 4. Since the first version (MEASURED, dy29)
- Compactness 270206: nw_r2pack cut to |x| <= 20 um (137 posts) x1.55 -> the posts at 20-30 um carry ~1/3 of the gain.
- Near-cavity size 270427: r 40 x1.80, r 50 x1.89, r 60 x1.89 -> optimum r 50-60 (mesh caveat below).
- Mesh probe 270533 (single post, sigma_sca ratio to a 12.5 nm mesh, Ey/Ez): 50 nm mesh r100 0.91/0.86, r75 0.87/0.70, r50 0.65/0.68,
  r35 0.51/0.59 (r25 broken); 25 nm mesh r >= 35 within 0.85-0.99 (12.5 nm = REFERENCE, not proven converged). => at the production mesh posts act ~5-30 % weaker in amplitude than
  drawn (sim r50 ~ converged r45). Designs are self-consistent at that mesh; the absolute x1.89 and any small-post work need a finer post mesh.
- Built-scene audit of nw_r2pack (script results_from_athena/comb_overshoot/postarray_work_dy29/audit_nw_r2pack_scene.py; TE numbers from nw_r2pack_dy29.mat field_xy/trace): 440 posts exactly at design positions/radii, mirrored exactly, min edge clearance to a built tooth 213 nm,
  BCs y anti-symmetric / z symmetric, dy29, 340 ps. TE confirmed: cavity trace 100 % Ey; core field 97.4 % Ey, 2.6 % Ex, 0 % Ez.
- Fab (results_from_athena/comb_overshoot/fab_limits_sin_posts.md): user accepts 100 nm diameter / 100 nm gap as the baseline; Technion
  EBPG 5200; 350 nm SiN stretch ~100/70 nm; oxide gap-fill voids are a risk for narrow deep gaps.

- Figures (2026-10-25, user prefers LINEAR scales for far-field comparisons on top of an already-good device):
  results_from_athena/comb_overshoot/xymap_ctrl_vs_nw_r2pack.png (z=0 |E|^2/U maps + ratio; plain TE has no map - user: skip it),
  farfield_ctrl_vs_nw_r2pack.png (log) and farfield_ctrl_vs_nw_r2pack_linear.png (linear: grazing peaks at |u_x|~0.97 drop ~4.5 -> ~1.4;
  cumulative 1.00 -> 0.49). Scripts matlab_plotting/studies/plot_postarray_{xymap,farfield,farfield_linear}.m (fold the linear copy into
  the main script as an option - CLAUDE.md sec. 10). Data: comb_overshoot/{xymap,farfield}_ctrl_vs_nw_r2pack.mat.

## 5. Running at handoff (Athena; check `squeue` fresh)
| Job | What | Read against |
|---|---|---|
| 270538 DONE | w8_c2d: FRESH compact free 2-D, 42 posts, abs(x) <= 19 um (Fable) | MEASURED x1.49 (Q_i 8.46e5, T 0.759, dlam +0.07): MISSED pre-reg 1.87 (1.73-2.01); ~95 % of rows-cut-to-20 um (x1.55, 137 posts) with 1/3 of the posts; 2-D GATE NOT PASSED |
| 270593 | nw_r2pack_ext42: + 89 posts on rows 1.2/1.5 at 30 < abs(x) <= 42 um (DWBA add-on on nw_r2pack's measured field) | pre-reg ~x2.00 (model x1.061 over nw_r2pack); posts reach |x| 42 but the tube is untapered only to 35 -> P/Pc approximate, Q_i valid |
| 270650 | nw_r2pack_rep: IDENTICAL repeat on A100 (L40S 270553 never started, cancelled) | noise floor: compare with nw_r2pack x1.887 |
| 270651 DONE | (twin chat, base ~/bragg_sim_twin) Nt80x antisymmetric twin | twin study CLOSED NEGATIVE (reported by that chat 2026-10-25); no twin jobs left - all seats are ours |
Score: `python python_tools/radcancel/solve_v4.py --short --energy <COPY of results_from_athena/budget_overshoot/dy29_box6p8.mat> <dev>`
(Q_i is the arbiter; solve_v4 P/Pc lacks the short-run frequency factor g = |D_fp|/|D_ff| - apply it as in
postarray_work_dy29/fable_refit/s11_features.py:14-19,43 before comparing radiated power; `--energy` is a core-field energy proxy).

## 6. Overnight refined-mesh re-baseline - DISPATCHED 2026-10-25 ~02:00: Athena 270754 (fine2ctrl, 1 task) + 270755 (fine2, 2 tasks)
Variants in `runners/sweeps/farfield_sph_15um_postarray_overshoot.py` (asserts pass): `fine2ctrl` (bare control) and `fine2` (2 tasks:
`nw_r2pack` as drawn + `nw_r2pack_map` = radii mapped to converged-equivalent strength, r_eff = r (sigma50/sigma12.5)^(1/4), near-cavity
posts midway; r 45-97, min gap 106 nm). The map is a HYPOTHESIS (assumes sigma ~ r^4 and an unchanged pattern; the probe refined all
three axes uniformly, fine2 is 25.7 / 25 / core z) - the mapped run tests it. Both fine2 tasks and fine2ctrl also record the in-situ XY map
(needed to design on the new control). Mesh: dx = pitch/20 = 25.7 nm everywhere (`simulation_mode fine_x10`), dy 25 nm over
|y| 0.70-1.70 um, core y_mesh_cells 26, box 6.8, 340 ps, ff window +-0.75 nm. EXPECTED ~10-14 h each, ~250 GB (4d_1g cap 275 GB).
Dispatch (Athena, A100): `SBATCH_MEM=250G bash athena/deploy_athena.sh --spec=runners.sweeps.farfield_sph_15um_postarray_overshoot_fine2ctrl
--max-concurrent=1 --qos=4d_1g --gpu=a100` and the same with `_fine2` (`--max-concurrent=2`). Read: Q_i of both vs the fine control;
expected drop for the as-drawn design ~5-10 % (EXPECTED, from the knob slope), mapped design should recover most.

## 7. Rules for the continuing chat
- The user AUTHORIZED the overnight refined-mesh set (sec. 6) on 2026-10-24/25; any other new long run needs the user's go.
- Compare only within identical numerics (matched control per mesh); report Q_i, peak T, width and dlam (loading) for every device.
- Fab numbers in fab_limits_sin_posts.md are estimates, not a qualified process.

## 8. Next decisions
1. w8 MEASURED x1.49 -> 2-D gate NOT passed (model over-predicted by ~20 %; per-post efficiency of free 2-D is high). Next: fix the
   element-pattern error in the model (Fable), then a full-length (abs(x)<=30) free 2-D design must reach >= x1.89. w8 lower -> fix the element-pattern error in the model (Fable) before more 2-D designs.
2. Extension (270593) >= x1.95 -> new best; carry it into the refined-mesh family.
3. Refined-mesh results -> switch the program to that mesh (design on its control, like the dy29 switch), then tier C (smaller posts + gaps).
4. Noise repeat -> run-to-run reproducibility only; mesh-registration bias needs the half-cell-offset test (still outstanding).
