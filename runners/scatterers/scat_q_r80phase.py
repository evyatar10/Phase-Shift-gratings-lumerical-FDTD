"""Stage Q — r-scaling probe at the cancelling phase (the smooth-structure gate).

Study dir: runners/scatterers/   |   Created 2026-08-09   |   Job(s): TBD (IGUM)
Purpose: stage P (job 129989, Athena) confirmed the anti-needle interference but
every row lost T: post parasitic scattering scales r^4 (0.0039@r80 -> 0.0145@r110)
vs coherent anti-needle amplitude r^2. The smooth sinusoidal structure attacks
exactly that ratio, but per-site radii are unsupported (builder extension) and
equal-size post approximations cannot carry the Lambda-fundamental efficiently
(bunching analysis, docs/antineedle_comb_design context). GATE instead: the r=80
comb at the measured best phase (dx=409 = 270 deg) — parasitic /3.7, coherent
/1.9; sinusoid fit through stage-P's 4 phase points predicts T ~ 0.887 = break-
even with ctrl 0.8851. PASS (>= ctrl - floor) => smooth route promising, build
per-site-radius extension; FAIL low => smooth ceiling shrinks too. Row 2 at
dx=477 (315 deg) brackets the user's "between 270 and 360 deg?" question
(first-harmonic fit says 270 deg is the max; 315 tests harmonic content).

Control row (r=0) IS justified here: solver CLUSTER changes (IGUM native vs
Athena container) — on the CLAUDE.md section-2 identical-numerics list; the
stored 0.8851 baselines are all Athena.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800,
N = 80/side; target resonance 1558.6 nm, window 1548.5-1568.5 nm (20 nm/1501).

Dispatch (IGUM; license seats SHARED with Athena — check both queues):
    ARRAY_TIME=02:00:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.scatterers.scat_q_r80phase --max-concurrent=1
Output -> results/scat_q_r80phase/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0
N_HALF    = 15                # 31 posts, ~16.4 um — stage-P geometry exactly
LAM_NM    = 545.0
HEIGHT_NM = 350.0

# rows: (dx_nm, r_nm); row 0 = control (r=0)
ROWS = [(0.0, 0.0), (409.0, 80.0), (477.0, 80.0)]   # ctrl | 270 deg | 315 deg

BOX_Y_UM      = 16.0          # stage-H/P numerics exactly
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

def comb_x(dx_nm):
    return [round(k * LAM_NM + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

for dx, r in ROWS[1:]:
    assert D_NM + r + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to y PML"
    assert D_NM - r >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"

SPEC = SweepSpec(
    scatterer_radius_nm = [r for _, r in ROWS],
    scatterer_x_list_nm = [[0.0] if r == 0 else comb_x(dx) for dx, r in ROWS],
    scatterer_y_list_nm = [[D_NM] * (1 if r == 0 else 2 * N_HALF + 1) for _, r in ROWS],
    scatterer_height_nm = [HEIGHT_NM] * len(ROWS),
    mode  = "zipped",
    label = "scat_q_r80phase",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for dx, r in ROWS:
        tag = "ctrl" if r == 0 else f"dx={dx} ({dx/LAM_NM*360:.0f} deg), r={r}"
        print(" ", tag)
    print("PASS gate: r80@270deg T >= 0.8833 (ctrl - floor) at IGUM-own ctrl")
