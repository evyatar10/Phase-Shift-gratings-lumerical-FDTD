"""Full-size W800 (N = 150/side) with the SINGLE-LITHO trench (h = 350 nm).

Study dir: runners/metal_mirror/   |   Created 2026-07-24   |   Job(s): TBD
Purpose (user): the N=80 ladder measured the core-height trench at half the
h=2um gain (job 125276: +0.0079 vs +0.0159). ONE run tests it on the full
N=150 W800 device at the EXACT trench_n150_full numerics — read against the
stage-M rows already measured there (job 124551 W800 ctrl T 0.1841; job
124590 full-z trench T 0.2322). No re-runs of controls (user: lean on
existing results).

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800,
N = 150/side; expected resonance ~1558.3 nm (ctrl 1558.60, full-z trench
1557.84, N=80 h350 drag -0.31 nm), window 10 nm centered 1557.91 covers all.

Dispatch (queue must be EMPTY of other --option3 arrays — section 6):
    SBATCH_MEM=180G ARRAY_TIME=08:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.trench_n150_h350 --max-concurrent=3
Output -> results/trench_n150_h350/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

N_PERIODS     = 150
TRENCH_LEN_UM = 156.0        # full-grating span (stage-M geometry)
TRENCH_W_NM   = 800.0
TRENCH_H_NM   = 350.0        # THE knob: exactly the core litho layer
TRENCH_D_NM   = 1800.0

BOX_Y_UM         = 8.0       # stage-M numerics exactly
SCAN_WIDTH_NM    = 10.0
N_WL_POINTS      = 2001
N_2D_FREQ_POINTS = 41
SCAN_CENTER_NM   = 1557.91   # stage-M W800 +trench row center (covers ctrl too)
MON2D_CENTER_NM  = 1558.30   # EXPECTED h350 resonance (between ctrl and full-z)
MON2D_SPAN_NM    = 2.5

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.spectral.n_2d_monitor_points = N_2D_FREQ_POINTS
BASE.monitors.record_2d_fields = True

assert TRENCH_D_NM + 0.5 * TRENCH_W_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0
assert TRENCH_D_NM - 0.5 * TRENCH_W_NM >= 500.0

SPEC = SweepSpec(
    n_periods_each_side  = [N_PERIODS],
    center_wavelength_nm = [SCAN_CENTER_NM],
    monitor_2d_center_nm = [MON2D_CENTER_NM],
    monitor_2d_span_nm   = [MON2D_SPAN_NM],
    scatterer_shape      = ["rect"],
    scatterer_x_span_um  = [TRENCH_LEN_UM],
    scatterer_y_span_nm  = [TRENCH_W_NM],
    scatterer_y_nm       = [TRENCH_D_NM],
    scatterer_index      = [1.0],
    scatterer_height_nm  = [TRENCH_H_NM],
    mode  = "zipped",
    label = "trench_n150_h350",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"W800 N=150 + air trench L={TRENCH_LEN_UM} um x w={TRENCH_W_NM} x "
          f"h={TRENCH_H_NM:.0f} nm, d={TRENCH_D_NM}; vs 124551 ctrl 0.1841 / "
          f"124590 full-z 0.2322")
