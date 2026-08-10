"""Stage U — flush-deep comb: posts from the BOX up to the SiN top (fab question).

Study dir: runners/scatterers/   |   Created 2026-08-10   |   Job(s): TBD
Purpose (user): if fab etches the comb posts down to the BOX (trench-flush
convention: top flush with core top +0.175 um, bottom at the BOX interface
-3.975 um), does the confirmed comb (Lambda 531, 270 deg, r110, d1.8, T +0.0115)
survive? Priors all negative but at WRONG design points (z-centered tall pillars
-0.077 stage N; full-z comb = green catastrophe at Lambda 551 / uncontrolled
phase). Deep posts intercept ~2-3x more carrier tail => row 2 compensates with
r=70 (amplitude ~ r^2 x depth-overlap). REGISTERED: r110 flush overshoots +
vertical-escape drain => T well below +0.0115, possibly < ctrl; r70 flush =
the honest fab answer (may recover part of the gain).

Controls: z-symmetry must be OFF for z-asymmetric rows => own z-asym control
row (section-2 numerics list: symmetry BCs). Expect ctrl ~ 0.8851 (z-sym is
exact for the symmetric device; also a z-mesh sanity check).

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501). Box y=16, z-sym OFF.

Dispatch (queue EMPTY of other --option3 arrays; z-sym off => ~2x runtime):
    SBATCH_MEM=180G ARRAY_TIME=03:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_u_flushcomb --max-concurrent=3
Output -> results/scat_u_flushcomb/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM, DX, D_NM = 531.0, 398.0, 1800.0     # confirmed winner geometry
N_HALF = 15
Z_MIN_UM = -3.975                        # BOX interface (core bottom -0.175 - 3.8)

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

def comb_x(dx_nm):
    return [round(k * LAM + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.symmetry.use_z_symmetry = False     # required by scatterer_z_min (all rows)

# rows: (r_nm, z_min_um)  — row 0 = z-asym control
ROWS = [(0.0, None), (110.0, Z_MIN_UM), (70.0, Z_MIN_UM)]
# row 3 (appended): flush TRENCH (rect, winner geometry L84/w800/d1.8) BOX->SiN
# top — the fabricable half-trench, never measured on this N=80 device; fills
# the comb-vs-flush-trench comparison. Uses the SAME z-asym ctrl (row 0).

SPEC = SweepSpec(
    scatterer_radius_nm = [r for r, _ in ROWS] + [0.0],
    scatterer_shape     = ["cylinder"] * len(ROWS) + ["rect"],
    scatterer_x_span_um = [0.0] * len(ROWS) + [84.0],
    scatterer_y_span_nm = [0.0] * len(ROWS) + [800.0],
    scatterer_index     = [None] * len(ROWS) + [1.0],
    scatterer_x_list_nm = [[0.0] if r == 0 else comb_x(DX) for r, _ in ROWS] + [[0.0]],
    scatterer_y_list_nm = [[D_NM] * (1 if r == 0 else 2 * N_HALF + 1) for r, _ in ROWS] + [[D_NM]],
    scatterer_height_nm = [350.0] * len(ROWS) + [350.0],
    scatterer_z_min_um  = [z for _, z in ROWS] + [Z_MIN_UM],
    mode  = "zipped",
    label = "scat_u_flushcomb",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("anchors: core-height comb +0.0115 (130117/130135) | ctrl 0.8851")
