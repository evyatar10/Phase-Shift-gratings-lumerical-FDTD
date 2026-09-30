"""Stage X3 — in-core oxide comb: period scan at 0 and 270 deg + two re-runs (Athena, a100).

Study dir: runners/scatterers/   |   Created 2026-09-14   |   Job(s): TBD (Athena)
Purpose (user): the SiN-outside comb has stored PERIOD scans at two phases — 0 deg
(stage P: Lambda 539/542/545/548/551; stage R: 536) and 270 deg (stages R/S:
530/531/532/534/536/540, stage P: 545). Run the in-core SiO2 comb at the SAME
periods and phases so the two families plot together. Stage X2 (job 148812)
already holds in-core 531/536/545 at both phases — NOT re-run. Rows 8-9 redo the
two X2 tasks (9, 11) that stalled on node n317 (cancelled after 2.9 h idle).
Same geometry as stage X2: r 80 (minimum renderable), oxide n 1.444, h 350
through-core, y +/- 250, 31 holes.
Controls: NOT re-run — Athena identical-numerics ctrl MEASURED T 0.8851.

Rows (Lambda, n_half, dx_nm):
  270 deg, 31 holes: Lambda 530 / 532 / 534 / 540
    0 deg, 31 holes: Lambda 539 / 542 / 548 / 551
  re-runs: Lambda 545, 31 holes, 180 deg (dx 273); Lambda 531, 9 holes, 0 deg

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.

Dispatch (Athena, pinned to a100-public — rtx6k node n317 stalled two X2 tasks):
    bash athena/deploy_athena.sh --option3 --gpu=a100 \
        --spec=runners.scatterers.scat_x3_incore_lamscan --max-concurrent=4
Output -> results/scat_x3_incore_lamscan/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

Y_NM, R_NM = 250.0, 80.0
N_OXIDE = 1.444

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(lam_nm, n_half, dx_nm):
    return [round(k * lam_nm + dx_nm, 1) for k in range(-n_half, n_half + 1)]

# rows: (Lambda_nm, n_half, dx_nm)   phase = 360 * dx / Lambda; 270 deg dx = 0.75 * Lambda
ROWS = ([(lam, 15, round(0.75 * lam, 1)) for lam in (530.0, 532.0, 534.0, 540.0)] +
        [(lam, 15, 0.0) for lam in (539.0, 542.0, 548.0, 551.0)] +
        [(545.0, 15, 273.0), (531.0, 4, 0.0)])

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(ROWS),
    scatterer_x_list_nm = [comb_x(lam, n, dx) for lam, n, dx in ROWS],
    scatterer_y_list_nm = [[Y_NM] * (2 * n + 1) for _, n, _ in ROWS],
    scatterer_height_nm = [350.0] * len(ROWS),
    scatterer_index     = [N_OXIDE] * len(ROWS),
    mode  = "zipped",
    label = "scat_x3_incore_lamscan",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for lam, n, dx in ROWS:
        print(f"Lambda {lam:5.1f}  holes {2*n+1:2d}  dx {dx:6.1f}  phase {360*dx/lam:5.1f} deg")
