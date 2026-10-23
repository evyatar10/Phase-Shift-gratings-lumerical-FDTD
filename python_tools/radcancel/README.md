# radcancel — analysis tools of the radiation-cancellation study (2026-10-20/21)

Copied from the session scratchpad so the next session can continue; one-off study tools, archive with the study (CLAUDE.md 10).
Handoff: docs/HANDOFF_radiation_cancellation_2026-10-21.md.

- solve_v4.py — measured-response solve. `python solve_v4.py [--short] [--pair i j] control.mat row1.mat ...`. Normalises each far field per
  stored energy (--short: DFT of the row's own cavity time trace at the far-field frequency; steady state: Lorentzian, s = -1), prints per
  row dlam/T/Q_L/Q_i/dW, |b|^2/|c|^2, P/Pc, overlap, and a free + bounded linear solve. Long result names break Windows MAX_PATH: copy to
  short names first.
- parity_readout.py — x-even/x-odd split about the leak's own plane x0 = +0.195 um, overlap per sector, 4-knob arms-independent solve.
- junction_fit.py — vector-current source fit of a far field at chosen x positions (held-out test); where the leak is born.
- line_source_model.py — per-ring TE-class fit + band-limited source image along x.
- reach_map.py / reach_opt.py — analytic reach map: which point-dipole pattern (position, signs) reaches which (l, m); single-harmonic
  and null-space designs.
- harm_matrix_analyze.py / harm_matrix_build_cache.py — per-harmonic response matrix of all stored Nt80x rows (selectivity, K-harmonic solves).

## 2026-10-23: designed post arrays (doc docs/postarray_radiation_cancellation_2026-10-23.md)
- postarray_born.py — Born dipole model of full-height SiN side posts on the stored Nt80x control field (tube side-plane field back-propagated
  to the post plane; valid for y > 0.9 um); `posts_ff(x, y, r)` = normalised far-field change; `__main__` = gate vs the 8 measured comb rows.
- postarray_ceiling.py — response (Gram) matrix on a 50 nm site grid per row + NNLS ceiling (env ROWS_UM, X_MAX_UM, KCUT_PER_UM, TAG).
- postarray_design.py — discrete post design (free x, r bounds, spacing penalty, L-BFGS-B; env TAG, R_MAX, ROW_IDX). Work files (control C,
  matrices, designs) live in results_from_athena/comb_overshoot/postarray_work/ (not in git).
