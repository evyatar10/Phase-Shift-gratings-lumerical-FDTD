"""Engine-version canary — reused at every Lumerical version bump, both clusters.

Study dir: runners/metal_mirror/   |   Created 2026-08-11
Purpose: ONE task re-running the comb_q3db wave-1 ctrl row (no comb; corr 325,
N 165, q3db numerics). CLAUDE.md section-6 no-rerun exemption: an engine version
bump IS the named numerics change that permits re-measuring a stored point.
PASS = repro of the stored anchor, job 130458 row 0
(results_from_athena/comb_q3db/result_N165_TM_avg_C325_Ybox8p0_Zbox8p8.mat):
T 0.490578 (-3.09 dB), Q 13930.5, lambda 1559.0010 nm — expected EXACT.
Any mismatch -> roll the runtime back to the previous version and stop.

Version-bump log:
  2026-08-11  R1.1 -> R1.2  Athena container.  Job 131009: PASS, exact
              (lambda delta 0.000000 nm, T delta -3.2e-7, Q delta -0.0001 %).
  2026-08-12  R1.2 -> R1.3  BOTH clusters (Athena container rebuilt from the
              user's LINX64 RPM; IGUM native tree at
              ~/research/lumerical/Lumerical-2026-R1.3). Engine 8.35.4572.
              Athena job 131295: PASS, EXACT in every printed digit
              (lambda 1559.0010, T 0.490579, spectral_fwhm -0.111913,
              Q 13930.5, mode 19.9702 um). IGUM job 52223: PASS, T 0.490578
              (delta 1e-6 = 0.0002 %), all other fields identical.
              -> cross-cluster lockstep re-proven at R1.3. Solve times
              9470 s / 9488 s (real solves, not license no-ops).

Dispatch (after the runtime swap; queue empty of other --option3 arrays):
    SBATCH_MEM=160G ARRAY_TIME=08:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.engine_canary --max-concurrent=1
    SBATCH_MEM=160G ARRAY_TIME=08:00:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.metal_mirror.engine_canary --max-concurrent=1
Output -> results/engine_canary/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

# q3db family numerics — copied verbatim from comb_q3db.py (job 130458).
CORR_LOCKED_NM = 325.0
N_CTRL         = 165
BOX_Y_UM       = 8.0
SCAN_CENTER_NM = 1559.5
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001        # 5 pm

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
assert BASE.symmetry.use_z_symmetry, "ctrl device is z-symmetric — keep the 2x z saving"

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_LOCKED_NM],
    n_periods_each_side  = [N_CTRL],
    center_wavelength_nm = [SCAN_CENTER_NM],
    # radius 0 = no-scatterer control. REQUIRED: build_ports_base() hands back a
    # config with the scatterer ENABLED at its defaults (r150/x0/y1000), so
    # omitting this silently runs a pillar device (job 130913, 1.75 A100-h wasted).
    scatterer_radius_nm  = [0.0],
    mode  = "zipped",
    label = "engine_canary",
)

# The built device must reproduce the anchor's filename exactly — that tag IS the
# device+numerics fingerprint (verified locally before dispatch; a mismatch here is
# what made job 130913 run a pillar device instead of the control).
ANCHOR_TAG = "N165_TM_avg_C325_Ybox8p0_Zbox8p8"
assert SPEC.scatterer_radius_nm == [0.0], "canary must be the NO-scatterer control"

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"  task 0: ctrl no-comb  corr={CORR_LOCKED_NM:.0f} N={N_CTRL}  (engine canary)")
    print(f"  expected file tag: {ANCHOR_TAG}")
    print("anchor (130458 row 0): T 0.4906 / -3.09 dB, Q 13930, lambda 1558.3-1559.0")
