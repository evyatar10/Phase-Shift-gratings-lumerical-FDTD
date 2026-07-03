"""
TM corr-400 domain convergence, ROUND 2 — the Z axis is the big one.

Round 1 (tm_span_convergence, job 116854) found: y converges at ~6.8 um
(T 0.8278 -> 0.7925 going 3.8 -> 6.8+ um), but Z moves T by +9 POINTS
(z 3.2 -> 5.8 um: T 0.7995 -> 0.8905, loss 0.189 -> 0.106) and had NOT
converged at 5.8 um. The TM mode's evanescent tail above/below the slab is
strong (E along z); the 1.8-lambda z-boundary absorbs stored near-field energy
and massively overstates the loss. This round:

  idx 0-4 : Z-LADDER at converged y=6.8 um: z = 4.2, 5.8, 6.8, 8.8, 10.8 um
  idx 5-6 : Y-RECHECK at large z=8.8 um:    y = 4.8, 8.8 um
            (y=6.8 @ z=8.8 is already idx 3; cross-term check on both sides)
  => 7 tasks. Big boxes run longer (~15-40 min each); ports-only RAM stays low.

Same anchored TM device / window / optimization mesh as round 1.
NOTE (§7): convergence results are KEEP-FOREVER data.

Dispatch (queue must be EMPTY of other --option3 arrays):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_span_convergence2
Output -> results/tm_span_convergence2/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BASE = build_base()
BASE.scatterer.enabled = False
BASE.y_span_override_m = None            # per-row

# z is driven via span_mult (z_span = core + mult*lambda): mult = (z - 0.35)/1.5585
#   z: 4.2, 5.8, 6.8, 8.8, 10.8 um  ->  mult: 2.47, 3.50, 4.14, 5.42, 6.70
_y_um  = [6.8,  6.8,  6.8,  6.8,  6.8,   4.8,  8.8]
_zmult = [2.47, 3.50, 4.14, 5.42, 6.70,  5.42, 5.42]

assert len(_y_um) == len(_zmult) == 7
assert len(set(zip(_y_um, _zmult))) == 7, "duplicate (y, z) row -> tag collision"

SPEC = SweepSpec(
    y_span_um = _y_um,
    span_mult = _zmult,
    mode  = "zipped",
    label = "tm_span_convergence2",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
