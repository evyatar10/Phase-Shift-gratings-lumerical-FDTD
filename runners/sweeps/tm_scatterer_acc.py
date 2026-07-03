"""
Accurate-mesh spot-check of the tm_scatterer_scan candidate winners.

The optimization-mesh (dx=50 nm) scan found best-case gains of dT = +0.0020
(r=100 nm @ x=0.810 um) — but the measured mesh-jitter noise floor is
dT = 0.0018, so the net improvement is AT the noise floor, not above it.
Per the study plan, candidate winners are re-run at simulation_mode="accurate"
(dx ~ 35 nm) before being believed. Six tasks:

  idx 0: r=0 control                       (accurate-mesh baseline)
  idx 1: r=100 @ x=540  nm                 (candidate winner #2)
  idx 2: r=100 @ x=810  nm                 (candidate winner #1)
  idx 3: r=100 @ x=835  nm                 (+25 nm jitter partner of #1 —
                                            the noise floor AT the accurate mesh)
  idx 4: r=100 @ x=4050 nm                 (candidate winner #3)
  idx 5: r=100 @ x=1620 nm                 (the r=100 worst case, for scale)

Same anchored TM device / window / domain as tm_scatterer_scan; only the mesh
changes. Accurate tasks run ~2-3x longer (~15-25 min each).

Read-out: if dT(x=810) at dx=35 nm stays ~ +0.002 AND the accurate jitter pair
(idx 2 vs 3) differs by much less than that, the improvement is real; if the
gain shrinks into the new jitter spread, it was a mesh artifact.

Dispatch: bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_scatterer_acc
Output  -> results/tm_scatterer_acc/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BASE = build_base()                      # identical device/window/domain as the scan
BASE.mesh.simulation_mode = "accurate"   # dx ~ 35 nm (the only change)


SPEC = SweepSpec(
    scatterer_radius_nm = [0.0,    100.0, 100.0, 100.0, 100.0,  100.0],
    scatterer_x_nm      = [0.0,    540.0, 810.0, 835.0, 4050.0, 1620.0],
    scatterer_y_nm      = [1000.0] * 6,
    mode  = "zipped",
    label = "tm_scatterer_acc",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
