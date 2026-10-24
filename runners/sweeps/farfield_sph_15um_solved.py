"""Nt80x (15.5-um lab device): VALIDATION of the bounded 15-knob solve designed on the converged l<=10 part of the leak.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Design (DERIVED, scratch solve_v4.py --short --lcut 10 --pair 1 8 --exclude 9 on the 265111 basis + 265046 control): bounded real
least squares |alpha| <= 1 with lambda (0.02 nm) and W (0.1 um) rows, scored on harmonics l <= 10 (the part whose pattern is
mesh-converged: overlap 0.94 dy37 vs dx50). Predicted cusp power 0.548 of control (full), 0.741 (half). Acceptance = TOTAL far
power per stored energy and Q_i, T, lambda, W measured at dx50 here (and at dy 37 later), never the l<=10 score alone.
Knobs: mean-width groups {0,1} {2,3} {4,5} {6,7} {8,9} {10,11} {12,13,14} {15,16,17} (5 nm * alpha on both teeth) and contrast
groups {0,1} {2,3} {4,5} {6,7} {8,9} {10,11} {12,13,14} (+-2.5 nm * alpha wide/narrow). ALPHA pasted from the printed solve and
asserted (incident 2026-10-20: a stale .npy once produced a zero profile).
Dispatch (Athena, submit HELD; scheduler releases when the pool is <= 26 in use):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_15um_solved --max-concurrent=2 --qos=4d_1g --gpu=a100
"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um_bumps import BASE, NAR0, WID0, N_SIDE  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

MEAN_GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13, 14), (15, 16, 17)]
DIFF_GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13, 14)]
ALPHA_FILE = os.environ.get("NT80X_ALPHA_NPY", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nt80x_solved15_alpha.npy"))   # 15 entries (8 mean, 7 diff), copied from the solve of 2026-10-20 12:45
ALPHA = np.load(ALPHA_FILE) if ALPHA_FILE else None
SCALES = [1.0, 0.5]


def solved_teeth(scale):
    assert ALPHA is not None and ALPHA.shape == (15,) and np.abs(ALPHA).max() > 0.5, ALPHA
    nar, wid = list(NAR0), list(WID0)
    for a, g in zip(ALPHA[:8], MEAN_GROUPS):
        for d in g:
            nar[d] += scale * 5.0 * a
            wid[d] += scale * 5.0 * a
    for a, g in zip(ALPHA[8:], DIFF_GROUPS):
        for d in g:
            nar[d] -= scale * 2.5 * a
            wid[d] += scale * 2.5 * a
    return nar, wid


CARDS = ([ExperimentCard(width_narrow_per_tooth_nm=solved_teeth(s)[0], width_wide_per_tooth_nm=solved_teeth(s)[1], label=f"solved15_x{s:g}")
          for s in SCALES] if ALPHA is not None else [])

if __name__ == "__main__":
    print("alpha:", np.round(ALPHA, 3))
    for card, s in zip(CARDS, SCALES):
        cfg = card.to_config(base=BASE)
        n = [v * 1e9 for v in cfg.grating.width_narrow_per_tooth_m]; w = [v * 1e9 for v in cfg.grating.width_wide_per_tooth_m]
        print(f"{card.label}: " + " ".join(f"{d}:{n[d]-NAR0[d]:+.1f}/{w[d]-WID0[d]:+.1f}" for d in range(18)) + f" | max|dn| {max(abs(a-b) for a,b in zip(n,NAR0)):.1f} nm")
