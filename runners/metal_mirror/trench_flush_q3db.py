"""Max loaded Q at -3 dB / 20 um mode with the FLUSH-TOP trench: N ladder.

Study dir: runners/metal_mirror/   |   Created 2026-08-06   |   Job(s): TBD
Purpose (user): redo the closed trench_q3db_20um verdict (ctrl N=165 Q 13930 /
full-z trench N=170 Q 18777, corr 325) with the flush-top trench (top face =
SiN top, floor 3.8 um below the SiN bottom, oxide beneath — job 128918 geometry).
Zero re-scans: corr stays 325 (mode width trench-insensitive: 19.97/19.57 um
stored; flush fwhm sits between its ctrl and full-z at N80/corr400) and the N
ladder is placed by the measured flush retention f = 0.77 of the full-trench dB
gain applied to the stored corr-325 dB(N) curves -> -3 dB crossing predicted
N ~ 168.6 (band 167-170 for f in 0.65-0.90). 4 tasks bracket it; NO ctrl /
full-z reruns (10 corr-325 points stored; z-sym OFF==ON measured exact,
si_substrate_check row 0).

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 325, W800
trench d 1800, N = 167..170/side; expected resonance ~1558.3-1559.0 nm,
window 20 nm centered 1559.5 (1549.5-1569.5, 4001 pts) — study numerics.
z-sym OFF (flush trench breaks the z=0 mirror) -> ~2x z cost per sim.

Dispatch (queue must be EMPTY of other --option3 arrays — CLAUDE.md section 6):
    SBATCH_MEM=160G ARRAY_TIME=08:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.trench_flush_q3db --max-concurrent=4
Output -> results/trench_flush_q3db/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_LOCKED_NM = 325.0
N_LADDER       = [167, 168, 169, 170]

TRENCH_W_NM    = 800.0
TRENCH_D_NM    = 1800.0
TRENCH_ZMIN_UM = -3.975      # floor = -(0.175 + 3.8); top face stays +175 nm (SiN top)
# Trench spans the whole grating (~2*N*0.51683 um + margin, q3db convention:
# 169 -> 177, 170 -> 178).
TRENCH_LEN_UM  = {167: 175.0, 168: 176.0, 169: 177.0, 170: 178.0}

BOX_Y_UM       = 8.0         # stage-M / q3db numerics exactly
SCAN_CENTER_NM = 1559.5
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001        # 5 pm (>=17 pts across the narrowest expected lw)

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.symmetry.use_z_symmetry = False   # flush trench is z-asymmetric (guard raises if on)

assert TRENCH_D_NM + 0.5 * TRENCH_W_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0
assert TRENCH_D_NM - 0.5 * TRENCH_W_NM >= 500.0
# 20-um mode containment: half-device >= ~2x target FWHM.
assert min(N_LADDER) * 0.51683 >= 2.0 * 20.0
# Trench floor inside the z mesh (oxide continues below before the PML).
assert -TRENCH_ZMIN_UM < 0.5 * (0.35 + _common.BOX_Z_MULT * 1.5585)

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_LOCKED_NM] * len(N_LADDER),
    n_periods_each_side  = list(N_LADDER),
    center_wavelength_nm = [SCAN_CENTER_NM] * len(N_LADDER),
    scatterer_shape      = ["rect"] * len(N_LADDER),
    scatterer_x_span_um  = [TRENCH_LEN_UM[n] for n in N_LADDER],
    scatterer_y_span_nm  = [TRENCH_W_NM] * len(N_LADDER),
    scatterer_y_nm       = [TRENCH_D_NM] * len(N_LADDER),
    scatterer_index      = [1.0] * len(N_LADDER),
    scatterer_z_min_um   = [TRENCH_ZMIN_UM] * len(N_LADDER),
    mode  = "zipped",
    label = "trench_flush_q3db",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, n in enumerate(N_LADDER):
        print(f"  task {i}: corr={CORR_LOCKED_NM:.0f} nm  N={n}  flush trench L={TRENCH_LEN_UM[n]:.0f} um")
