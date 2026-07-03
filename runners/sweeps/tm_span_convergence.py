"""
TM corr-400 transverse-domain CONVERGENCE study — how big a box does this device need?

Motivation (2026-07-03): the 1.8*lambda transverse box was validated as sufficient for
TE and for TM at corrugation 300 nm (few-% radiation loss). The anchored TM device at
corrugation 400 nm radiates 16-19% at resonance, mostly IN-plane and nearly along the
axis (grazing incidence on the y PML), and its absolute T moved 0.828 -> 0.799 when the
y box went 3.8 -> 4.8 um. This study finds the minimum transverse size where the
absolute observables (T, lambda_res, Q, loss) flatten.

Device: anchored TM baseline, NO scatterer (pitch 516.83 nm, corr 400 nm, h 350 nm,
N=80, n 1.97/1.444, optimization mesh dx=50 nm, window 1558.5/30 nm/3001 pts).

Grid (mode='zipped', one task per row):
  idx 0-6 : Y-LADDER — absolute y box = 3.805 (the 1.8-lambda default), 4.3, 4.8,
            5.8, 6.8, 8.8, 10.8 um; z stays at the 1.8-lambda default (3.15 um).
  idx 7-8 : Z-CHECK — y fixed at 4.8 um, span_mult = 2.5 / 3.5 (z = 4.25 / 5.80 um;
            with y_span_um set, span_mult affects Z only).
  => 9 tasks, ~6-15 min each (cells grow with the box; ports-only, RAM stays low).

Read-out: T / lambda_res / Q / loss vs y-span. Converged = successive steps change T
by less than the accurate-mesh jitter scale (~1e-3). The minimum adequate box is the
smallest y with T within that tolerance of the 10.8 um value; the z-check rows tell
whether z=1.8-lambda also needs revisiting.

NOTE (§7): convergence-study results are KEEP-FOREVER data.

Dispatch (queue must be free of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_span_convergence
Output -> results/tm_span_convergence/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BASE = build_base()                      # anchored TM device + proven window
BASE.scatterer.enabled = False           # bare device — this is a domain study
BASE.y_span_override_m = None            # per-row via y_span_um


# Row layout: y-ladder at default z, then z-check at fixed y=4.8 um.
_y_um  = [3.805, 4.3, 4.8, 5.8, 6.8, 8.8, 10.8, 4.8, 4.8]
_zmult = [None,  None, None, None, None, None, None, 2.5, 3.5]

assert len(_y_um) == len(_zmult) == 9

SPEC = SweepSpec(
    y_span_um = _y_um,
    span_mult = _zmult,
    mode  = "zipped",
    label = "tm_span_convergence",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
