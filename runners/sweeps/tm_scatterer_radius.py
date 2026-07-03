"""
Radius ladder at the WINNING pillar positions — find the optimal scatterer size.

Theory: the recycled contribution to T is linear in the pillar polarizability
(interference cross-term) while the parasitic scattered-away power is quadratic,
so dT(alpha) = c1*alpha - c2*alpha^2 has an OPTIMUM at finite size — smaller is
not monotonically better. Data brackets it: r=200 always loses, r=150 breaks
even, r=100 wins (+0.0021 confirmed) -> optimum near/below 100 nm. This ladder
measures r = {80, 100, 125} nm at the three confirmed winner positions.
r < 80 nm is not honestly resolvable at dx~35 nm (radius < 2.3 cells).

Numerics: ACCURATE mesh (dx~35 nm — the jitter floor there is ~1e-4, well below
the ~2e-3 effects being compared), converged transverse box (set BOX_Y_UM from
the tm_span_convergence result before dispatching!). Includes its own control
and a +25 nm jitter pair at the best position.

Grid (zipped): control + 3 radii x 3 positions + 2 jitter partners = 12 tasks,
~20 min each at accurate mesh.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_scatterer_radius
Output -> results/tm_scatterer_radius/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


# Converged transverse box (tm_span_convergence + round 2, jobs 116854/116870):
# y converges at 6.8 um; z at 8.8 um (mult 5.42: z = 0.35 + 5.42*1.5585).
BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.mesh.simulation_mode = "accurate"           # dx ~ 35 nm
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT

_WINNERS_NM = [540.0, 810.0, 4050.0]             # confirmed single-pair winner positions

_rs = [0.0]
_xs = [0.0]
for r in (80.0, 100.0, 125.0):
    _rs += [r] * len(_WINNERS_NM)
    _xs += list(_WINNERS_NM)
# jitter partners (+25 nm = half a coarse cell) at the best position, small radii
_rs += [80.0, 100.0]
_xs += [835.0, 835.0]

assert len(_rs) == len(_xs) == 12
assert len(set(zip(_rs, _xs))) == len(_rs), "duplicate (r, x) row -> tag collision"

SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_nm      = _xs,
    scatterer_y_nm      = [1000.0] * len(_rs),
    mode  = "zipped",
    label = "tm_scatterer_radius",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
