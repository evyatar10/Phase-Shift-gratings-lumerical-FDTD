"""Auto-shutoff threshold convergence vs device Q (settles bragg_device 1e-7).

Study dir: runners/sweeps/   |   Created 2026-08-04   |   Job(s): TBD
Purpose: production `auto shutoff min` is 1e-7 (bragg_device.py) — 100x stricter
than the Lumerical 1e-5 default; the 1e-5->1e-7 ringdown tail is ~half of every
solve at Q~19k. The coded convergence study was never run, and the safe
threshold may depend on Q (user 2026-08-04: longer/less-corrugated devices ring
longer). Grid: shutoff {1e-4, 1e-5, 1e-6, 1e-8} x three devices spanning
Q ~2.5k..27k, all at corr 325 / h350 / p516.83 / W800 with numerics
byte-identical to trench_q3db_20um round 2 — so the existing 1e-7 results for
N=165 ctrl (Q 13930) and N=185 trench (Q 26714) are two free grid points; only
the N=110 ctrl needs its own 1e-7 row (task 12). 13 tasks.

Verdict rule (per device): coarsest shutoff whose Q is within 2% of the 1e-8
row (mesh-convergence convention), T within 0.015. Check every task ended via
"Early termination" in its solver log, NOT the 2000 ps cap — a timed-out row is
invalid, not converged. If the coarsest-safe value drifts with Q -> per-Q rule.

CONVERGENCE-STUDY DATA IS KEEP-FOREVER (CLAUDE.md section 7).

Dispatch (IGUM):
    SBATCH_MEM=160G bash igum/deploy_igum.sh \
        --option3 --spec=runners.sweeps.autoshutoff_qspan --max-concurrent=4
Output -> results/autoshutoff_qspan/results/ (download to results_from_igum/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_NM   = 325.0
SHUTOFFS  = [1e-4, 1e-5, 1e-6, 1e-8]   # 1e-7 rows exist in trench_q3db_20um
# (N, trench_on): Q at 1e-7 is ~2.5k (est) / 13930 / 26714 (measured).
DEVICES   = [(110, False), (165, False), (185, True)]

TRENCH_W_NM, TRENCH_D_NM, TRENCH_H_NM = 800.0, 1800.0, 12000.0
TRENCH_LEN_UM = {110: 116.0, 165: 173.0, 185: 194.0}

BOX_Y_UM, SCAN_CENTER_NM, SCAN_WIDTH_NM, N_WL_POINTS = 8.0, 1559.5, 20.0, 4001

# Row = (shutoff, N, trench_on, pol, pitch_nm, corr_nm, n_apod, center_nm).
_TM = ("TM", 516.83, CORR_NM, 0, SCAN_CENTER_NM)
ROWS = [(s, n, on, *_TM) for s in SHUTOFFS for n, on in DEVICES]
ROWS += [(1e-7, 110, False, *_TM)]     # the one missing 1e-7 reference

# ── Family panel (tasks 13-18, appended 2026-08-04; indices 0-12 UNCHANGED so
# the license-race resubmission --array-tasks=1,3,5 stays valid).
# User: goal is a GENERAL shutoff rule + one-decade spare, via examples, not a
# full grid per family. TE baseline tests the "1e-6 was enough for TE" memory
# and TE-vs-TM; apod-20 is a different ring-down class (tapered mirrors, high
# T). Tooth-shift deliberately skipped (same cavity family as the plain
# pi-shift Q-span) unless these panels show family dependence.
# 1e-8 is UNREACHABLE (measured live 2026-08-04: total-field energy plateaus
# at ~5-7e-8, so the 1e-8 criterion never fires and the rows march toward the
# 2000 ps cap / 23:30 SLURM kill — tasks 9-11 cancelled mid-run). 1e-7, which
# every production run reaches, is the finest usable anchor; the TM devices'
# 1e-7 anchors exist already (trench_q3db_20um + task 12).
PANEL_SHUTOFFS = [1e-5, 1e-6, 1e-7]    # 1e-7 = per-family truth row
ROWS += [(s, 80, False, "TE", 500.0, 300.0, 0, 1560.0) for s in PANEL_SHUTOFFS]
ROWS += [(s, 80, False, "TM", 516.83, 400.0, 20, 1559.6) for s in PANEL_SHUTOFFS]

BASE = _common.build_ports_base()
BASE.geometry.corrugation_depth_m = CORR_NM * 1e-9
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

SPEC = SweepSpec(
    auto_shutoff_min     = [r[0] for r in ROWS],
    n_periods_each_side  = [r[1] for r in ROWS],
    scatterer_shape      = ["rect"] * len(ROWS),
    scatterer_x_span_um  = [TRENCH_LEN_UM[r[1]] if r[2] else 0.0 for r in ROWS],
    scatterer_y_span_nm  = [TRENCH_W_NM if r[2] else 0.0 for r in ROWS],
    scatterer_y_nm       = [TRENCH_D_NM] * len(ROWS),
    scatterer_index      = [1.0] * len(ROWS),
    scatterer_height_nm  = [TRENCH_H_NM if r[2] else 350.0 for r in ROWS],
    polarization         = [r[3] for r in ROWS],
    pitch_nm             = [r[4] for r in ROWS],
    corrugation_depth_nm = [r[5] for r in ROWS],
    apod_method          = ["linear"] * len(ROWS),
    n_apod_periods_each_side = [r[6] for r in ROWS],
    center_wavelength_nm = [r[7] for r in ROWS],
    mode  = "zipped",
    label = "autoshutoff_qspan",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (s, n, on, pol, pitch, corr, napod, _c) in enumerate(ROWS):
        fam = "trench" if on else ("apod" if napod else "plain")
        print(f"  task {i:2d}: shutoff={s:.0e}  {pol}  N={n:3d}  corr={corr:3.0f}  {fam}")
