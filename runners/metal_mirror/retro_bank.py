"""Retro-reflector test bank — autonomous follow-up of the PEC d-scan (job 124253).

Study dir: runners/metal_mirror/   |   Created 2026-07-19   |   Job(s): TBD
Purpose: the flat-PEC d-scan CONFIRMED mirror-leak interference (cos @2.68 um,
amp ~0.0015) but specular geometry is now bounded by that measurement. This bank
tests the remaining open channels in ONE array (user away; single deploy so all
jobs live on the server up front):

  row 0  control (identical numerics)
  row 1  flat PEC wall d=2.80 um  — cos-fit constructive peak (completes d-scan)
  row 2  flat Al  wall d=3.00 um  — real-metal penalty vs PEC's +0.0019
  row 3  PEC Littrow comb r=110, Lambda=551 nm, d=3.0 — stage-H retro geometry
         with metal amplitude (dielectric version measured NULL at ~1%/row)
  row 4  PEC Littrow comb r=150 — per-post amplitude scaling
  row 5  PEC Littrow comb 2-row (3.0 + 5.68 um) — does row-stacking turn on
         with metal (dielectric 2-row measured = 1-row)
  row 6  PEC corner-array wall (90 deg V-teeth, faces 3 um at +/-45 deg),
         envelope d=3.00 — angle-independent geometric retroreflector
  row 7  same corner wall at d=3.054 — HALF a retro-phase cycle away: the
         retro path runs along the ray (~5x the perpendicular lever), so the
         return phase cycles every ~107 nm of d; rows 6/7 sample both phases
         (also a fab-fragility measurement).

Numerics = stage H / metal_mirror_dscan (box y=16 um, 1548.5-1568.5 nm /
1501 pts, opt mesh, stage-H lambda sidecar, ~85 GB per task). Registered
expectations: comb rows — per-row amplitude should jump ~10-100x vs dielectric;
stacking gate = row5 vs row3; corner rows — if the retro channel converts,
|dT| can exceed the specular bound 0.0019; rows 6 vs 7 out of phase.
Verdicts vs the 0.0018 jitter floor. Side FF monitor shadowed for wall rows
(instrument note as before); ports are the metric.

Dispatch (queue EMPTY first — CLAUDE.md section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.retro_bank --max-concurrent=3
Output -> results/retro_bank/results/.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

PEC = "PEC (Perfect Electrical Conductor)"
AL  = "Al (Aluminium) - Palik"

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0
LOCKED_LAMBDA_FILE = "/work/results/scat_h_retrocomb_lambda_res.json"

# — Littrow comb (stage-H geometry, metal) —
LAMBDA_X_NM = 551.0
N_HALF      = 75
COMB_X = [round(k * LAMBDA_X_NM, 1) for k in range(-N_HALF, N_HALF + 1)]
ROW_CYCLE_NM = 2680.0

# — Corner-array wall: Λ-teeth of two 3.0 um faces at +/-45 deg, 90 deg apex —
FACE_UM   = 3.0
FACE_T_NM = 200.0
_P_NM  = 2.0 * FACE_UM * 1000.0 * math.cos(math.radians(45.0))   # 4243 nm period
_D_NM  = FACE_UM * 1000.0 * math.sin(math.radians(45.0))         # 2121 nm depth
_N_PER = int(82600.0 // _P_NM)                                   # 19 periods

def corner_sites(d_nm):
    """Face centers + rotations for the Λ-tooth zigzag with inner envelope at d_nm.
    Rising (+45) face then falling (-45) face per period; apex away from guide."""
    x0 = -0.5 * _N_PER * _P_NM
    xs, ys, rots = [], [], []
    for i in range(_N_PER):
        base = x0 + i * _P_NM
        y_c = round(d_nm + _D_NM / 2.0, 1)
        xs += [round(base + 0.25 * _P_NM, 1), round(base + 0.75 * _P_NM, 1)]
        ys += [y_c, y_c]
        rots += [45.0, -45.0]
    return xs, ys, rots

_C6 = corner_sites(3000.0)
_C7 = corner_sites(3054.0)   # half a ~107 nm retro-phase cycle from row 6

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

# rows: (shape, xspan_um, yspan_nm, r_nm, x_nm, y_nm, xlist, ylist, rotlist, mat)
ROWS = [
    ("rect",     0.0,  0.0,   0.0, 0.0, 3000.0, None,        None,        None,    None),  # 0 ctrl
    ("rect",     82.6, 200.0, 0.0, 0.0, 2800.0, None,        None,        None,    PEC),   # 1 flat PEC d=2.8
    ("rect",     82.6, 200.0, 0.0, 0.0, 3000.0, None,        None,        None,    AL),    # 2 flat Al d=3.0
    ("cylinder", 0.0,  0.0, 110.0, 0.0, 3000.0, COMB_X,      [3000.0]*len(COMB_X), None, PEC),  # 3 comb r110
    ("cylinder", 0.0,  0.0, 150.0, 0.0, 3000.0, COMB_X,      [3000.0]*len(COMB_X), None, PEC),  # 4 comb r150
    ("cylinder", 0.0,  0.0, 110.0, 0.0, 3000.0, COMB_X * 2,
     [3000.0]*len(COMB_X) + [3000.0 + ROW_CYCLE_NM]*len(COMB_X), None, PEC),                    # 5 comb 2-row
    ("rect",     FACE_UM, FACE_T_NM, 0.0, 0.0, 3000.0, _C6[0], _C6[1], _C6[2], PEC),  # 6 corner d=3.000
    ("rect",     FACE_UM, FACE_T_NM, 0.0, 0.0, 3054.0, _C7[0], _C7[1], _C7[2], PEC),  # 7 corner d=3.054
]

_PML_CLEAR_NM = 1200.0
for shape, xs_um, ysp, r, _x, y1, xl, yl, rl, _m in ROWS:
    ys_all = (yl if yl else [y1])
    half = (0.5 * math.hypot(xs_um * 1000.0, ysp) if rl
            else (r if shape == "cylinder" else 0.5 * ysp))
    assert max(ys_all) + half + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0, \
        f"row too close to y PML: max y {max(ys_all)} + {half:.0f}"
    if xl:  # standoff floor applies to the structure's inner edge
        assert min(ys_all) - half >= 2400.0, f"standoff below floor: {min(ys_all)}"

SPEC = SweepSpec(
    scatterer_shape       = [r[0] for r in ROWS],
    scatterer_x_span_um   = [r[1] for r in ROWS],
    scatterer_y_span_nm   = [r[2] for r in ROWS],
    scatterer_radius_nm   = [r[3] for r in ROWS],
    scatterer_x_nm        = [r[4] for r in ROWS],
    scatterer_y_nm        = [r[5] for r in ROWS],
    scatterer_x_list_nm   = [r[6] for r in ROWS],
    scatterer_y_list_nm   = [r[7] for r in ROWS],
    scatterer_rot_list_deg = [r[8] for r in ROWS],
    scatterer_material    = [r[9] for r in ROWS],
    mode  = "zipped",
    label = "retro_bank",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"corner wall: {_N_PER} periods x 2 faces of {FACE_UM} um at +/-45 deg, "
          f"period {_P_NM:.0f} nm, depth {_D_NM:.0f} nm; comb {len(COMB_X)} sites")
