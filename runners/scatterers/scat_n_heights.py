"""Stage N — tall-pillar diagnostic: the validated [0,270] pair at h = 4 / 6 um.

Study dir: runners/scatterers/   |   Created 2026-07-23   |   Job(s): TBD
Purpose: every scatterer so far sat at the core height 350 nm (single-litho
rule; tall variants are explicit diagnostics only — user request 2026-07-23).
Stage K showed captured power escapes through the VERTICAL channel; a tall
pillar samples more of the radiation cone's z-spread. Measure the validated
[0, 270] r=80 y=700 pair at h = 350 nm (must reproduce dT +0.0227) vs
h = 4000 and 6000 nm, same device, identical stage-E numerics, with the r=0
in-study control (canary: T = 0.8862).

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83 nm, corr 400 nm,
W800, N = 80/side; target resonance ~1558.5 nm, scan window 30 nm
(1543.5-1573.5 nm), 3001 pts, opt mesh, box y 6.8 / z ~8.8 um.

Dispatch (queue must be EMPTY of other --option3 arrays — section 6):
    bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_n_heights
Output -> results/scat_n_heights/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR

PAIR_X_NM  = [0.0, 270.0]         # the validated stage-E combination
PAIR_Y_NM  = [700.0, 700.0]       # row-1 offset (each site mirrors to ±y)
HEIGHTS_NM = [350.0, 4000.0, 6000.0]   # core-height reproduce + tall diagnostics

# rows: control (r=0, h inert) + one row per height
ROWS_R_NM = [0.0] + [_common.RADIUS_NM] * len(HEIGHTS_NM)
ROWS_H_NM = [350.0] + HEIGHTS_NM

assert len(set(HEIGHTS_NM)) == len(HEIGHTS_NM), \
    "heights must be unique (the _H file tag is the only difference between rows)"
# z-clearance: pillar top must stay > lambda/n_clad (~1.08 um) below the z PML.
_z_half_um = 0.5 * (0.35 + _common.BOX_Z_MULT * 1.5585)
for _h in HEIGHTS_NM:
    assert _h / 2000.0 + 1.2 <= _z_half_um, \
        f"pillar h={_h} nm too close to the z PML (half-span {_z_half_um:.2f} um)"

BASE = _common.build_ff_base()

SPEC = SweepSpec(
    cavity_width_nm     = [800.0] * len(ROWS_R_NM),
    scatterer_radius_nm = ROWS_R_NM,
    scatterer_x_list_nm = [PAIR_X_NM] * len(ROWS_R_NM),
    scatterer_y_list_nm = [PAIR_Y_NM] * len(ROWS_R_NM),
    scatterer_height_nm = ROWS_H_NM,
    mode  = "zipped",
    label = "scat_n_heights",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"[0,270] r={_common.RADIUS_NM:.0f} y=700 pair, heights {HEIGHTS_NM} nm "
          f"+ r=0 control (W800, N=80/side)")
