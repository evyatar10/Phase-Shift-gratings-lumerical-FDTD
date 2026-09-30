"""Stage X2 — in-core oxide comb: the full period x phase x count grid (Athena).

Study dir: runners/scatterers/   |   Created 2026-09-14   |   Job(s): TBD (Athena)
Purpose (user): repeat the SiN-outside comb phase circles (stages P/R: Lambda 545
and 536, 0/90/180/270 deg, 31 posts) with SiO2 holes INSIDE the core, on the
short device, and plot both families together. Stage X already measured three
in-core rows at Lambda 531 (31 holes @270; 9 holes @270 and @90) — NOT re-run.
Hole size: r=80 is the minimum renderable. MEASURED from the stored bare-device
field plane, |E|^2 at y=250 is ~400x the value at the outside comb site y=1800,
so the amplitude-matched in-core hole would be r ~ 25 nm (< dx/2, unbuildable);
r=80 is ~10x overdrive. Amplitude is therefore scanned by HOLE COUNT (31 / 9 / 3;
3 holes ~ matched amplitude), never by radius.
Controls: NOT re-run — Athena identical-numerics ctrl MEASURED T 0.8851 (stage H
123563; trench_h350 125276 task 0); IGUM ctrl 0.8864 (51285_0) for the stage-X rows.

Rows (Lambda, n_half, dx_nm; r=80, oxide n=1.444, h350 through-core, y=+/-250):
  Lambda 531, 31 holes: 0 / 90 / 180 deg     (270 stored: stage X)
  Lambda 536, 31 holes: 0 / 90 / 180 / 270 deg  (= stage R phases, dx 0/134/268/402)
  Lambda 545, 31 holes: 0 / 90 / 180 / 270 deg  (= stage P phases, dx 0/136/273/409)
  Lambda 531,  9 holes: 0 / 180 deg          (90, 270 stored: stage X; 90 was dx=132)
  Lambda 531,  3 holes: 0 / 90 / 180 / 270 deg  (matched-amplitude row)

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.

Dispatch (Athena):
    bash athena/deploy_athena.sh --option3 \
        --spec=runners.scatterers.scat_x2_incore_circle --max-concurrent=3
Output -> results/scat_x2_incore_circle/results/.
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

# rows: (Lambda_nm, n_half, dx_nm)   phase = 360 * dx / Lambda
ROWS = ([(531.0, 15, 0.0), (531.0, 15, 133.0), (531.0, 15, 265.5)] +
        [(536.0, 15, dx) for dx in (0.0, 134.0, 268.0, 402.0)] +
        [(545.0, 15, dx) for dx in (0.0, 136.0, 273.0, 409.0)] +
        [(531.0, 4, 0.0), (531.0, 4, 265.5)] +
        [(531.0, 1, dx) for dx in (0.0, 133.0, 265.5, 398.0)])

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(ROWS),
    scatterer_x_list_nm = [comb_x(lam, n, dx) for lam, n, dx in ROWS],
    scatterer_y_list_nm = [[Y_NM] * (2 * n + 1) for _, n, _ in ROWS],
    scatterer_height_nm = [350.0] * len(ROWS),
    scatterer_index     = [N_OXIDE] * len(ROWS),
    mode  = "zipped",
    label = "scat_x2_incore_circle",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for lam, n, dx in ROWS:
        print(f"Lambda {lam:5.1f}  holes {2*n+1:2d}  dx {dx:6.1f}  phase {360*dx/lam:5.1f} deg")
