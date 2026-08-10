"""Stage V — confirmed comb on the apodized (apod-10) device: does it transfer?

Study dir: runners/scatterers/   |   Created 2026-08-10   |   Job(s): TBD
Purpose (user Q4): the pillar-pair FAILED on apod devices (stage G, -0.008 =
cavity-dressing artifact); the comb is a radiation-channel lever (W1050 transfer
+0.0092 measured). The trench worked on apod10 (+0.0039, job 124531). One row:
apod-10 + confirmed comb (Lambda 531, 270 deg, r110, d1.8, h350).
REGISTERED: leak-scaled gain ~ +0.002-0.003 (apod10 leak 0.023 = ~5x smaller
budget) if the apodized residual leak is still needle-like; ~0/negative if not.

Controls: NOT re-run — apod-10 ctrl MEASURED at T 0.9770 (job 124531 task 2,
IDENTICAL numerics: ports base, box y=8, 1501 pts / 30 nm centered 1558.5).

Physics line (section 4): TM h350, pitch 516.83, corr 400, linear apod n=10
default depth, N=80/side; own-resonance read (apod shifts lambda; stage-G rule).

Dispatch (queue EMPTY of other --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_v_apodcomb --max-concurrent=3
Output -> results/scat_v_apodcomb/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM, DX, D_NM, R_NM = 531.0, 398.0, 1800.0, 110.0   # confirmed winner geometry
N_HALF = 15

BASE = _common.build_ports_base()
BASE.y_span_override_m = 8.0e-6          # trench_te_apod numerics exactly
BASE.spectral.n_wl_points = 1501
BASE.spectral.scan_width_nm = 30.0

COMB_X = [round(k * LAM + DX, 1) for k in range(-N_HALF, N_HALF + 1)]

assert D_NM + R_NM + 1200.0 <= 4000.0, "comb too close to y PML at box 8"

SPEC = SweepSpec(
    apod_method              = ["linear"],
    n_apod_periods_each_side = [10],
    scatterer_radius_nm      = [R_NM],
    scatterer_x_list_nm      = [COMB_X],
    scatterer_y_list_nm      = [[D_NM] * (2 * N_HALF + 1)],
    scatterer_height_nm      = [350.0],
    mode  = "zipped",
    label = "scat_v_apodcomb",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("anchor: apod-10 ctrl T 0.9770 (job 124531 task 2, identical numerics)")
