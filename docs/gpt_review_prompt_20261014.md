GPT review request, prepared 2026-10-12 by the claude-te lane, to run once the
ChatGPT quota resets (2026-10-14 06:40). Read-only review.

Repo: this directory, branch claude-te. Rules: CLAUDE.md (§5 verification, §9 labels,
§11 style). Method: runners/lumopt2_design/THEORY.md §13, §13a, §13b, §13c. Live state:
the newest blocks of the memory file project_te_inverse_design_lane.md are summarised below.

What changed (review `git diff f5c049f~1..HEAD -- runners/lumopt2_design/`):
1. Far-field objectives: rad_fom (|E|^2 on a cladding plane, trapz-weighted adjoint source),
   rad_farfield (angular-spectrum radiated power, propagating waves, kz/k0 weight, exact
   adjoint through a y-resampling matrix), rad_norm_T (−log(V / T(lambda_rad)), detuning-free).
2. Opt-in width-row knobs: per-tooth shift width profile, cw noise floor.
3. Lane defaults (shift metric h375, ±25 nm box, λ projection, 0.1 nm λ trust, fixed sim time
   S1 50 ps / S2 150 ps) merged into every lane of campaign_te_n98.py.
4. Forward-row runners that edit design vectors (te_shift_sweep_fixt, s2_simtime_ladder,
   s1_narrow_widen retrim).

Measured results to sanity-check against the code (fixed sim time, PVA, N=98):
- Auto-shutoff biased peak T low by 0.003 (S1) to 0.012 (S2); 150 vs 250 ps agree to 7e-5.
- S2: seed 0.98069, corrugation best 0.98605, shift-lane best 0.98422; shifts on top of the
  corrugation best: +0.0002 at most (teeth 2–6 × 4 nm), −0.0014 at 10 nm. Two optimizer lanes
  from the corrugation best repeated the same failed step (pred +0.0022, measured −0.0009).
- S1: 0.97434 → 0.97766 with shifts (13 accepted, 3 rejected; later steps rejected on width).
- Far field on S1 with shifts ended 0.0018 behind the peak-T lane; S2 far field (rad_norm_T)
  +0.0002 after one rejected step (running).

Questions (verdict first, ≤ 30 lines):
1. Any bug in items 1–4 (wrong gradient, stale state such as spec._rad_lam_nm, lane knobs not
   what the comments say, resume/requeue hazards, design-vector edits)?
2. Why would the v3 QP keep proposing the same failing S2 step from the corrugation best
   (pred +0.0022, measured −0.0009) even with a corrected width row? What should change?
3. Is peak T at fixed width the right objective for the user's real figure, Q3dB ≈ 0.29·Q_i?
4. The single best next experiment to raise S2's Q_i, within or beyond tooth space.
