"""TE transfer check — do the TM program's winning levers do anything for TE?

Study dir: runners/sweeps/   |   Created 2026-07-18   |   Job(s): TBD
Purpose (user, small check): TE sits farther from the light line (n_eff 1.559 vs
TM 1.508) and radiates far less sideways, so every TM loss lever should be ~null
on TE. Measure it instead of asserting it: TE control vs the two strongest TM
levers — the pillar pair [0,270]@y=700 (r=80, TM dT +0.0227) and the widened
W1050 cavity (TM dT +0.0354).

TE anchors: pitch 500.0 nm, corr 300 nm, N=80/side, h=350, n 1.97/1.444 —
co-resonant lambda_TE = 1558.74 (job 113163). Window 1558.5 +/- 15 nm covers it.
EXPECTED: TE resonant loss is a few % at most -> both levers within the jitter
floor or negative. Any dT > +0.002 would be a genuine surprise worth chasing.

Dispatch (ONLY after the queue is empty — serialize rule, section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh --option3 \
        --spec=runners.sweeps.te_transfer_check --max-concurrent=3
Output -> results/te_transfer_check/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.sweeps._tm_base import build_base

BASE = build_base()                       # then re-anchor to TE
BASE.source.polarization = "TE"
BASE.grating.pitch_m = 500.0e-9
BASE.geometry.corrugation_depth_m = 300e-9
# scatterer stays enabled (build_base default); r=0 rows are the controls.
# y-span stays the 4.8 um scatterer default (pillar rows at |y|<=780 incl. r).

# rows: (cavity_width_nm None=avg, radius, x list, y list)
ROWS = [
    (None,   0.0,  [0.0],        [700.0]),          # 0 TE control
    (None,   80.0, [0.0, 270.0], [700.0, 700.0]),   # 1 TE + TM pillar pair
    (1050.0, 0.0,  [0.0],        [700.0]),          # 2 TE + TM width optimum
]

SPEC = SweepSpec(
    cavity_width_nm     = [w for w, _, _, _ in ROWS],
    scatterer_radius_nm = [r for _, r, _, _ in ROWS],
    scatterer_x_list_nm = [xs for _, _, xs, _ in ROWS],
    scatterer_y_list_nm = [ys for _, _, _, ys in ROWS],
    mode  = "zipped",
    label = "te_transfer_check",
)

if __name__ == "__main__":
    print(SPEC.describe())
