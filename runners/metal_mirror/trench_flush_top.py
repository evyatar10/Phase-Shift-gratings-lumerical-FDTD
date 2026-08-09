"""Air trench flush with the SiN TOP face, 3.8 um deep (oxide below, no Si).

Study dir: runners/metal_mirror/   |   Created 2026-08-06   |   Job(s): TBD
Purpose (user): every previous trench was z-symmetric about the core mid-plane
(or full-z through both PMLs). This ONE run makes the trench top coincide with
the SiN top (z max = +175 nm) and the floor 3.8 um below the SiN bottom
(z min = -3.975 um), oxide continuing underneath — the z=0 mirror is broken,
so use_z_symmetry=False (~2x z cost). 1 task, NO control row: read against the
si_substrate_check rows already measured z-sym-OFF at these EXACT numerics
(IGUM 43459_0 ctrl T 0.8862 / lambda 1558.616; full-z trench 0.9037; row 5
floor-only trench, Athena 126104_5, T 0.9035).

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, N=80/side,
trench L 84 um x w 800 nm, d 1800 nm (d-scan optimum); target resonance between
ctrl 1558.62 and full-z 1557.85; window 20 nm centered 1558.5 covers both.

Dispatch (queue must be EMPTY of other --option3 arrays — CLAUDE.md section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.trench_flush_top
Output -> results/trench_flush_top/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

TRENCH_LEN_UM   = 84.0        # full arm span (job 124379 geometry)
TRENCH_W_NM     = 800.0
TRENCH_D_NM     = 1800.0      # measured d-optimum
TRENCH_ZMIN_UM  = -3.975      # floor = -(0.175 + 3.8): 3.8 um below the SiN bottom face
# Trench height stays at the default core height (350 nm) -> top face +175 nm
# = the SiN top; scatterer_z_min then pulls the bottom face down to -3.975 um.

BOX_Y_UM      = 8.0           # si_substrate_check numerics exactly
N_WL_POINTS   = 2001          # 10 pm over the 20 nm window
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.symmetry.use_z_symmetry = False   # trench is z-asymmetric (guard raises if on)

_PML_CLEAR_NM = 1200.0        # > lambda/n_clad = 1080
assert TRENCH_D_NM + 0.5 * TRENCH_W_NM + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0
assert TRENCH_D_NM - 0.5 * TRENCH_W_NM >= _common.TOOTH_EDGE_NM
# Trench floor must sit inside the z mesh (oxide continues below it before the
# PML): half-span 0.5*(core + BOX_Z_MULT*lambda).
_z_half_um = 0.5 * (0.35 + _common.BOX_Z_MULT * 1.5585)
assert -TRENCH_ZMIN_UM < _z_half_um, "trench floor below the z domain"

SPEC = SweepSpec(
    scatterer_shape     = ["rect"],
    scatterer_x_span_um = [TRENCH_LEN_UM],
    scatterer_y_span_nm = [TRENCH_W_NM],
    scatterer_y_nm      = [TRENCH_D_NM],
    scatterer_index     = [1.0],
    scatterer_z_min_um  = [TRENCH_ZMIN_UM],
    mode  = "zipped",
    label = "trench_flush_top",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"air trench L={TRENCH_LEN_UM} um x w={TRENCH_W_NM:.0f} nm, d={TRENCH_D_NM:.0f} nm, "
          f"z from {TRENCH_ZMIN_UM} um to +0.175 um (flush with SiN top); "
          f"vs 43459_0 ctrl 0.8862 / full-z 0.9037 / floor-only 0.9035")
