"""Air comb — the opposite-material version of the confirmed anti-needle comb (IGUM).

Study dir: runners/scatterers/   |   Created 2026-08-12   |   Job(s): TBD (IGUM)
Purpose (user): can the comb be made of AIR instead of SiN, and does an air comb
INSIDE the core do anything the oxide one didn't?

PHYSICS (TM => E along the post axis => amplitude ~ d(eps) * volume, no
depolarization factor):
    SiN post in oxide cladding : d(eps) = 1.97^2 - 1.444^2 = +1.796
    air  hole in oxide cladding: d(eps) = 1      - 1.444^2 = -1.085
=> (1) SIGN FLIP = pi phase shift => the optimum comb offset moves by LAM/2 =
       265.5 nm: the confirmed SiN optimum dx=398 (270 deg) becomes dx=133 for air
       (the slot that LOSES for SiN — measured circle 0/90/180/270 deg =
       0.8664/0.8407/0.8657/0.8928 at LAM=536).
   (2) matched radius r_air = r_SiN * sqrt(1.796/1.085) = 1.29x => 141 nm for r110.
   (3) the net-gain ceiling 2*eta*sqrt(Pc*Pn) - Pc depends on AMPLITUDE only, so the
       material picks the radius, not the optimum => air should TIE the SiN comb,
       plus a small bonus from the low-index/TIR (trench) share of each hole.

IN-CORE (row 3): d(eps) for air in SiN core = 1 - 3.881 = -2.881 vs oxide's -1.796
=> air drives 1.60x HARDER at equal radius, i.e. deeper into the measured overdrive
regime. Prior in-core measurements (all negative, both phases already covered):
oxide N=31 @270deg T 0.8460 / N=9 @270deg 0.8654 / N=9 @90deg 0.5438, and the mode
DELOCALIZES (15.53 -> 17.85 / 25.64 um), which alone breaks the fixed-width spec.
Row 3 is therefore a user-requested FALSIFIER, run with a stated expectation of
T ~ 0.83-0.86; only T >= 0.8882 (ctrl + floor) would reopen the in-core axis.

Controls: NOT re-run — IGUM own-cluster ctrl MEASURED T 0.8864 at these exact
numerics (job 51285 task 0, box16 / 1501 pts / 20 nm). Floor 0.0018.
SiN reference at the same numerics (Athena 130135): T 0.8966 = +0.0115 vs its own
Athena ctrl 0.8851 — compare dT to dT, not T to T (cross-cluster offset +0.0013).

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501 pts), box y=16, opt mesh.

Dispatch (IGUM; Athena is running job 131348 in parallel -> license peak 2+3 <= 6):
    ARRAY_TIME=02:00:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.scatterers.scat_air_comb --max-concurrent=2
Output -> results/scat_air_comb/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM      = 531.0     # comb period (confirmed winner)
DX_AIR   = 133.0     # 90.2 deg = the SiN optimum 398 shifted by LAM/2 (sign flip).
                     # 133 not 132.5: round() in the file tag would collide with the
                     # stored in-core oxide 90 deg row (X-1992to2256). 0.34 deg cost.
DX_SIN   = 398.0     # 270 deg — the SiN optimum, i.e. the WRONG phase for air
D_NM     = 1800.0    # cladding standoff (comb optimum, d axis closed 1.5-1.8)
Y_CORE   = 250.0     # in-core row: same site as the stored oxide steelman

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM


def comb_x(n_half, dx_nm):
    return [round(k * LAM + dx_nm, 1) for k in range(-n_half, n_half + 1)]


# rows: (r_nm, n_half, dx_nm, y_nm, label)
ROWS = [
    (141.0, 15, DX_AIR, D_NM,   "air cladding comb, mirrored optimum"),
    (141.0, 15, DX_SIN, D_NM,   "falsifier: SiN's winning phase must now lose"),
    (115.0, 15, DX_AIR, D_NM,   "amplitude bracket (ka~0.8 at r=141)"),
    ( 80.0,  4, DX_AIR, Y_CORE, "in-core air falsifier, 9 holes"),
]

SPEC = SweepSpec(
    scatterer_radius_nm = [r for r, _, _, _, _ in ROWS],
    scatterer_x_list_nm = [comb_x(n, dx) for _, n, dx, _, _ in ROWS],
    scatterer_y_list_nm = [[y] * (2 * n + 1) for _, n, _, y, _ in ROWS],
    scatterer_height_nm = [350.0] * len(ROWS),
    scatterer_index     = [1.0] * len(ROWS),
    mode  = "zipped",
    label = "scat_air_comb",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("anchors: IGUM ctrl 0.8864 (51285_0) | floor 0.0018 | "
          "SiN comb dT +0.0115 (130135) | in-core oxide 0.8460/0.8654/0.5438")
