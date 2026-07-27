"""Si handle wafer under the 3.8 um BOX — does the fab stack change anything?

Study dir: runners/metal_mirror/   |   Created 2026-07-27   |   Job(s): TBD
Purpose (user): our model is all-oxide below the core; the fab stack is
Si / 3.8 um BOX / SiN / oxide. Two comparisons, both at identical numerics:
  (1) plain W800: does adding the Si below change T/lambda at all?
  (2) full-height air trench: fab trenches are etched down TO the Si — does
      the trench+Si combination differ from our through-domain trench?
If both are null the Si feature gets dropped from the code (user decision).

Design:
- ALL rows use_z_symmetry=False (Si breaks the z=0 mirror; controls must match).
  The z-sym-OFF control is a NEW numerics point — the stored z-sym-ON benchmark
  (T 0.8862 / lambda 1558.611, jobs 120797..124379) is NOT re-run; row 0 vs that
  stored value doubles as the "z-sym off changes nothing" cross-check.
- Si slab: top face at z = -(0.175 + BOX) um, down through the bottom z-PML;
  ports auto-clip above the Si (see bragg_device). z half-span 4.4 um keeps the
  interface (-3.975 um) inside the mesh with 0.43 um of Si before the PML.
- Row 2 (BOX 3.665 um = 3.8 - lambda/(8*n_clad)) guards the null against
  sitting on a reflection-interference node: quarter-fringe offset moves any
  Si-echo phase by ~90 deg. Same-null at both -> robust null.
- Trench: measured best W800 geometry (L 84 um, w 800 nm, d 1800 nm, job
  124379) at FULL height 12 um (through both z-PMLs, stage-M convention);
  with Si present the builder terminates it on the Si top face (fab geometry).
- Numerics: box y = 8 um (trench-validated, job 124531), z-mult 5.42, opt mesh,
  ports-only, window 20 nm / 2001 pts (10 pm) centered 1558.5 — covers the
  ctrl resonance 1558.61 and the trench-dragged 1557.91 (both N=80 MEASURED).
- REGISTERED PREDICTION (stage-J PEC bound + Fresnel 17% at oxide/Si): all
  Si deltas below the 0.0018 jitter floor; trench rows keep dT ~ +0.016.
Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800,
N = 80/side; target resonance 1558.6 nm, window 1548.5-1568.5 nm.

Dispatch (queue must be EMPTY of other --option3 arrays — CLAUDE.md section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.si_substrate_check --max-concurrent=3
Output -> results/si_substrate_check/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

BOX_FAB_UM   = 3.8            # fab BOX (core bottom -> Si top)
BOX_NODE_UM  = 3.665          # quarter-fringe offset: 3.8 - lambda/(8*n_clad)

TRENCH_LEN_UM    = 84.0       # full arm span (job 124379 geometry)
TRENCH_W_NM      = 800.0
TRENCH_D_NM      = 1800.0     # measured d-optimum
TRENCH_H_NM      = 12000.0    # full-z (stage-M convention); Si rows clip at Si top

BOX_Y_UM      = 8.0           # trench-validated transverse box (job 124531)
N_WL_POINTS   = 2001          # 10 pm over the 20 nm window
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = TRENCH_H_NM * 1e-9

# rows: (label, si_box_um, trench?)
ROWS = [
    ("ctrl (no Si, no trench)", None,        False),
    ("Si, fab BOX 3.8",         BOX_FAB_UM,  False),
    ("Si, BOX 3.665 node-guard", BOX_NODE_UM, False),
    ("trench full-z (no Si)",   None,        True),
    ("trench + Si (fab stack)", BOX_FAB_UM,  True),
]

_PML_CLEAR_NM = 1200.0        # > lambda/n_clad = 1080
assert TRENCH_D_NM + 0.5 * TRENCH_W_NM + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0
assert TRENCH_D_NM - 0.5 * TRENCH_W_NM >= _common.TOOTH_EDGE_NM
# Si interface must sit inside the z mesh: half-span 5.42*1.5585/2 + core/2.
_z_half_um = 0.5 * (0.35 + 5.42 * 1.5585)
assert 0.175 + BOX_FAB_UM < _z_half_um, "Si top below the z domain"

SPEC = SweepSpec(
    use_z_symmetry      = [False] * len(ROWS),
    si_box_um           = [r[1] for r in ROWS],
    scatterer_shape     = ["rect"] * len(ROWS),
    scatterer_x_span_um = [TRENCH_LEN_UM if r[2] else 0.0 for r in ROWS],
    scatterer_y_span_nm = [TRENCH_W_NM if r[2] else 0.0 for r in ROWS],
    scatterer_y_nm      = [TRENCH_D_NM] * len(ROWS),
    scatterer_index     = [1.0] * len(ROWS),
    mode  = "zipped",
    label = "si_substrate_check",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (name, box, trench) in enumerate(ROWS):
        print(f"  task {i}: {name}")
