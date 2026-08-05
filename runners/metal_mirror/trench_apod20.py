"""Apod-20 + full-z air trench: the last untested combo of the two loss levers.

Study dir: runners/metal_mirror/   |   Created 2026-08-01   |   Job(s): TBD (IGUM)
Purpose: stack the strongest apodization (n=20: accurate-mesh T 0.9829 /
loss 0.0169, tm_pareto_stack_vs_apod) with the straight full-z air trench
(d=1800, cuts 15-20% of the REMAINING leak on every TM device measured:
W800 -13.6% / W1050 -19.9% / apod10 -17%). Registered prediction:
dT +0.002..+0.004, dloss ~ -0.003 (near the 0.0018 floor -> single-point
CANDIDATE either way; section-2 two-step before any confirmed claim).

Control justification (add-study step 0): no same-numerics apod-20 control
exists — the stored 0.9829 is accurate mesh on Athena (different mesh, box,
window, cluster); this study is opt mesh / box y=8 / IGUM, so both rows are new.

Rows (zipped, 2 tasks), TM apod-20 N=80/side:
0 apod20 ctrl (no trench) | 1 apod20 + straight air trench d_center=1800 nm,
w 800 nm, h 12000 nm (full-z through z-PML = stage-M best geometry), L 84 um.

Physics line (section 4): TM h350, n 1.97/1.444, pitch 516.83, corr 400
(apod linear n=20, center depth 40 nm), N=80/side; target resonance
~1559 +/- 2 nm (apod10 box-8 measured 1559.2, job 124531); window
1543.5-1573.5 nm (30 nm / 1501 pts); own-resonance reads, no sidecar
(apod shifts lambda_res — stage-G rule).

Dispatch (IGUM; queue must be EMPTY of other --option3 arrays — section 6):
    bash igum/deploy_igum.sh --option3 --spec=runners.metal_mirror.trench_apod20
Output -> results/trench_apod20/results/ (download -> results_from_igum/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

TRENCH_W_NM = 800.0
TRENCH_H_NM = 12000.0           # full-z: through the ~8.8 um z domain (stage M)
TRENCH_D_NM = 1800.0            # confirmed optimum (124379 scan + 124898 refine)
TRENCH_LEN_UM = 84.0

BOX_Y_UM      = 8.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 30.0

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = TRENCH_H_NM * 1e-9

assert TRENCH_D_NM + 0.5 * TRENCH_W_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0  # PML
assert TRENCH_D_NM - 0.5 * TRENCH_W_NM >= 500.0                # clear of tooth tips

TRENCH_ON = [False, True]

SPEC = SweepSpec(
    polarization             = ["TM"] * 2,
    pitch_nm                 = [516.83] * 2,
    corrugation_depth_nm     = [400.0] * 2,
    apod_method              = ["linear"] * 2,
    n_apod_periods_each_side = [20] * 2,
    scatterer_shape          = ["rect"] * 2,
    scatterer_x_span_um      = [TRENCH_LEN_UM if on else 0.0 for on in TRENCH_ON],
    scatterer_y_span_nm      = [TRENCH_W_NM if on else 0.0 for on in TRENCH_ON],
    scatterer_y_nm           = [TRENCH_D_NM] * 2,
    scatterer_index          = [1.0] * 2,
    mode  = "zipped",
    label = "trench_apod20",
)

if __name__ == "__main__":
    print(SPEC.describe())
