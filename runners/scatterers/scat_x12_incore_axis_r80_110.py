"""Stage x12 — on-axis (y 0) single-hole in-core comb, Lambda 524 / 270 deg, radius series [80.0, 110.0] (IGUM).

Study dir: runners/scatterers/   |   Created 2026-09-16   |   Job(s): IGUM 90593 (r 80, task 0); IGUM 90599 (r 110) DIED at run() — bare "in run:" (same-node license-daemon race) — resubmitted as Athena 151353 (task 1)
Purpose (user): radius series of the ON-AXIS single hole per period (stage X10, job 151320 = r 50,
running). Same rows as the +/-250 pair series (X4/X5/X6/X7: r 30/40/50/80/110) plus r 60, so the two
transverse placements can be compared point by point on the T-vs-width line. Split across clusters
for parallelism: 30/40/60 on Athena (with X10 = 4 running, the QOS cap), 80/110 on IGUM (staggered
starts — the ansyscl startup race, memory project_igum_ansyscl_startup_race).
Controls: NOT re-run — Athena identical-numerics ctrl MEASURED T 0.8851 / 15.53 um (scat_h_retrocomb);
cross-cluster reproducibility proven (CLAUDE.md section 6), same engine R1.3 on both.
Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance ~1558, window 1548.5-1568.5 (20 nm / 1501), box y=16 — stage X exactly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM_NM, DX_NM, N_HALF = 524.0, 393.0, 15      # 270 deg, 31 holes — the X4/X6/X10 row exactly
RADII_NM = [80.0, 110.0]
Y_NM, N_OXIDE = 0.0, 1.444                     # y 0 = one cylinder on the axis per site

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

X_LIST = [round(k * LAM_NM + DX_NM, 1) for k in range(-N_HALF, N_HALF + 1)]

SPEC = SweepSpec(
    scatterer_radius_nm = RADII_NM,
    scatterer_x_list_nm = [X_LIST] * len(RADII_NM),
    scatterer_y_list_nm = [[Y_NM] * len(X_LIST)] * len(RADII_NM),
    scatterer_height_nm = [350.0] * len(RADII_NM),
    scatterer_index     = [N_OXIDE] * len(RADII_NM),
    mode  = "zipped",
    label = "scat_x12_incore_axis_r80_110",
)

if __name__ == "__main__":
    print(SPEC.describe())
