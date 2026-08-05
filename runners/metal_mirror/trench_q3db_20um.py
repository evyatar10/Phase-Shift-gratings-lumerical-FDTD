"""Max loaded Q at -3 dB peak T for a 20 um TM mode: trench vs no-trench (IGUM).

Study dir: runners/metal_mirror/   |   Created 2026-08-02   |   Job(s): TBD
Purpose: fixing peak T = 0.5 fixes the coupling (Q_L = 0.293*Q_i), so the max
loaded Q at -3 dB is set by the intrinsic Q of the 20-um-FWHM mode; N periods is
only the knob that lands T = 0.5.
ROUND 1 (array 47910, done): corr ladder {400 ctrl-pair..266} x both arms at
N=150 + hedged N ladder {110..165} at corr 276. All 18 sane; controls
reproduced the hscan anchors (0.183/16.22 ctrl, 0.230/16.04 trench).
ROUND 2 (this SPEC): N ladders at the locked corr 325 -> ln(T) vs N per arm,
interpolate to T = 0.5; accept T = 0.5 +/- 0.03, fwhm 20 +/- 1 um.

Physics anchor (= trench_n150_hscan / stage-M exactly): TM h350, pitch 516.83,
W800, n 1.97/1.444, Ybox 8.0; trench W 800 nm, d 1800 nm, full-z H12000.
Window: corr change moves lambda_res a few nm off 1558.3 -> 30 nm centered
1560 (1545-1575), 4001 pts = 7.5 pm (>=16 pts across the narrowest expected
~120 pm linewidth; ~2x the hscan port-DFT RAM -> SBATCH_MEM=160G).

Filename note: corr rows at W800 need the _C{corr} tag added to
sim_helpers.generate_file_tag (2026-08-02) -- without it same-N rows clobber.

Dispatch (license ceiling: MAX 6 concurrent solves across BOTH clusters):
    SBATCH_MEM=160G bash igum/deploy_igum.sh \
        --option3 --spec=runners.metal_mirror.trench_q3db_20um --max-concurrent=4
Output -> results/trench_q3db_20um/results/ (download to results_from_igum/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

# ── ROUND 2 (2026-08-03; round 1 = IGUM array 47910, 18 tasks, all sane).
# Round-1 fits (fit_round1, results_from_igum/trench_q3db_20um): corr(20um) =
# 324.7 nm on the no-trench arm (lock 325); measured dlnT/dN at corr 276 is
# shallow while overcoupled (-0.0030 ctrl / -0.0024 trench), so the T=0.5
# crossings sit ABOVE N=150: est ~160-185 ctrl, ~190-260 trench. Ladders below
# bracket both; every row also confirms fwhm(corr 325) ~= 20 um.
CORR_LOCKED_NM = 325.0
# ROUND 3 (final confirm): round-2 crossings = ctrl N=165 MEASURED IN BAND
# (T 0.491, Q 13930, fwhm 19.97) -- no ctrl rerun needed; trench interpolates
# to N=169.2 -> single confirm row below.
N_LADDER_CTRL   = []
N_LADDER_TRENCH = [170]    # N=169 MEASURED T=0.513; 170 brackets the crossing

TRENCH_W_NM = 800.0
TRENCH_D_NM = 1800.0
TRENCH_H_NM = 12000.0        # full-z (through both z-PMLs), stage-M winner
# Trench must span the whole grating: ~2*N*0.51683 um + margin. 150 -> 156
# kept byte-identical to the hscan anchors.
TRENCH_LEN_UM = {110: 116.0, 125: 132.0, 140: 147.0, 150: 156.0, 165: 173.0,
                 169: 177.0, 170: 178.0, 180: 188.0, 185: 194.0, 195: 204.0,
                 205: 214.0, 225: 235.0}

BOX_Y_UM       = 8.0         # stage-M numerics exactly
SCAN_CENTER_NM = 1559.5      # round-1 lambda_res at corr~325: ~1559.8/1559.1
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001        # 5 pm (>=17 pts across the narrowest expected lw)

# Rows: (corr_nm, n_side, trench_on).
ROWS = [(CORR_LOCKED_NM, n, False) for n in N_LADDER_CTRL]
ROWS += [(CORR_LOCKED_NM, n, True) for n in N_LADDER_TRENCH]

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

assert TRENCH_D_NM + 0.5 * TRENCH_W_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0
assert TRENCH_D_NM - 0.5 * TRENCH_W_NM >= 500.0
# 20-um mode containment: half-device >= ~2x target FWHM even at the shortest N.
assert min(N_LADDER_CTRL + N_LADDER_TRENCH) * 0.51683 >= 2.0 * 20.0

SPEC = SweepSpec(
    corrugation_depth_nm = [c for c, n, on in ROWS],
    n_periods_each_side  = [n for c, n, on in ROWS],
    center_wavelength_nm = [SCAN_CENTER_NM] * len(ROWS),
    scatterer_shape      = ["rect"] * len(ROWS),
    scatterer_x_span_um  = [TRENCH_LEN_UM[n] if on else 0.0 for c, n, on in ROWS],
    scatterer_y_span_nm  = [TRENCH_W_NM if on else 0.0 for c, n, on in ROWS],
    scatterer_y_nm       = [TRENCH_D_NM] * len(ROWS),
    scatterer_index      = [1.0] * len(ROWS),
    scatterer_height_nm  = [TRENCH_H_NM if on else 350.0 for c, n, on in ROWS],
    mode  = "zipped",
    label = "trench_q3db_20um",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (c, n, on) in enumerate(ROWS):
        arm = "trench" if on else "ctrl  "
        print(f"  task {i:2d}: corr={c:5.0f} nm  N={n}  {arm}")
