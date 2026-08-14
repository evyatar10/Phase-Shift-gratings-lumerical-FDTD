"""Anti-needle comb at the q3db operating point (-3 dB / 20 um mode): wave 1.

Study dir: runners/metal_mirror/   |   Created 2026-08-10   |   Job(s): 130458 (wave 1), wave 2 below
Purpose (user): transfer the final comb design (Lambda 531 / 270 deg / r80-class,
h350, program-best +0.0150 at N=80/corr-400) to the corr-325 q3db family and
lock N for -3 dB, like trench_flush_q3db. Wave 1 = transfer + in-study anchor
at N=165; wave 2 (separate dispatch) rides the stored corr-325 dB(N) curves to
the -3 dB crossing (expected N ~ 168-169, Q ~ 16.5-17.5k).

Rows (zipped, all N=165 / corr 325):
  0  ctrl no-comb      family canary — MUST reproduce Q ~ 13930, T ~ 0.5,
                       lambda 1558.3-1559.0 (stored ctrls are IGUM/old-numerics)
  1  adapted comb      Lambda 531, 57 posts (+-14.9 um), r 80, d 1.9 um,
                       dx 401 (270 deg + transit) — the transfer row
  2  90-deg sign check dx 136 — phase flip MUST lose (mechanism ported?)
  3  aim hedge         Lambda 536, dx 404
Adaptation (recorded derivation): pitch shared with corr-400 family -> n_eff
and Lambda carry over; kappa ~ corr -> needle narrower x0.81 -> matched length
25 -> 31 um -> 57 posts; amplitude ~ kappa -> keep r 80, d 1.8 -> 1.9 um
(e^{-gamma*dd} rule, gamma 1.95/um). REGISTERED: comb dB-gain retention
f ~ 0.8 of full-trench (from measured N=80 head-to-head +0.0150 vs +0.0174).

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 325, W800,
N=165/side; expected resonance ~1558.3-1559.0 nm, window 20 nm centered 1559.5
(1549.5-1569.5, 4001 pts) — q3db study numerics. z-sym ON (comb is z-symmetric,
unlike the flush trench). Auto-shutoff builder default 1e-7.

Dispatch (queue must be EMPTY of other --option3 arrays — CLAUDE.md section 6):
    SBATCH_MEM=160G ARRAY_TIME=08:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.comb_q3db --max-concurrent=4
Output -> results/comb_q3db/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_LOCKED_NM = 325.0
R_NM           = 80.0
D_NM           = 1900.0
H_NM           = 350.0
N_HALF         = 28          # 57 posts, +-14.9 um

# WAVE 2 (2026-08-10, after wave-1 job 130458 passed all gates): N ladder at the
# winning comb (Lambda 531, dx 401 = 270 deg). Wave-1 MEASURED: ctrl N165
# T 0.4906 (-3.09 dB) Q 13930 (exact IGUM-anchor repro) | comb531 T 0.5361
# (-2.71 dB) Q 14584 | 90deg 0.4371 (loses, sign check) | 536 hedge 0.5283.
# NO control row (CLAUDE.md section-6 no-rerun rule): anchor = 130458 ctrl.
# -3 dB target T = 0.5*0.4906/0.5 ... = 0.5012 at the ctrl lambda; registered
# crossing N ~ 167-168 riding the stored corr-325 dB(N) slope.
LAM_NM, DX_NM  = 531.0, 401.0
N_LADDER       = [167, 168, 169]

# (wave-1 rows, dispatched as job 130458, kept for the record:
#  ROWS = [None, (531, 401), (531, 136), (536, 404)] all at N=165)

BOX_Y_UM       = 8.0         # q3db family numerics exactly (trench_flush_q3db)
SCAN_CENTER_NM = 1559.5
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001        # 5 pm (>=18 pts across the narrowest expected lw)

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
assert BASE.symmetry.use_z_symmetry, "comb is z-symmetric — keep the 2x z saving"

def comb_x(lam_nm, dx_nm):
    return [round(k * lam_nm + dx_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

X_COMB = comb_x(LAM_NM, DX_NM)

assert D_NM + R_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0   # y-PML clearance
assert D_NM - R_NM >= _common.TOOTH_EDGE_NM + _common.GAP_MIN_NM
# Comb well inside the grating; 20-um mode containment (half-device >= 2x FWHM).
assert N_HALF * LAM_NM + DX_NM + R_NM + 2000.0 <= min(N_LADDER) * 516.83
assert min(N_LADDER) * 0.51683 >= 2.0 * 20.0
assert len(set(N_LADDER)) == len(N_LADDER), "rows must be tag-unique (N is the tag)"

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_LOCKED_NM] * len(N_LADDER),
    n_periods_each_side  = list(N_LADDER),
    center_wavelength_nm = [SCAN_CENTER_NM] * len(N_LADDER),
    scatterer_radius_nm  = [R_NM] * len(N_LADDER),
    scatterer_x_list_nm  = [X_COMB] * len(N_LADDER),
    scatterer_y_list_nm  = [[D_NM] * (2 * N_HALF + 1)] * len(N_LADDER),
    scatterer_height_nm  = [H_NM] * len(N_LADDER),
    mode  = "zipped",
    label = "comb_q3db",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, n in enumerate(N_LADDER):
        print(f"  task {i}: corr={CORR_LOCKED_NM:.0f} N={n}  comb Lam={LAM_NM:.0f} dx={DX_NM:.0f} (57 posts, r={R_NM:.0f}, d={D_NM:.0f})")
    print("anchors (130458): ctrl N165 0.4906/-3.09dB Q13930 | comb N165 0.5361/-2.71dB Q14584")
    print("target: -3 dB crossing (T ~ 0.501 family-relative), expect N ~ 167-168")
