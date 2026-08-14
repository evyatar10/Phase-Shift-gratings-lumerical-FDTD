"""TM surrogate-N ladder, corr-400 family: can the proven N=80 platform go shorter?

Study dir: runners/sweeps/   |   Created 2026-08-11   |   Job(s): (fill at dispatch)
Purpose (user): the whole corr-400 loss/scatterer program ran at N=80/side
(Q ~ 1320, T 0.886 — converged baseline, jobs 116854/116870) and its findings
transferred to production. This ladder asks whether an even shorter surrogate
(N = 50/60/70) still has a well-formed resonance, at the scatterer-program
numerics exactly. Verdict criterion as in tm_nladder_c325.

Rows (zipped, bare device, radius 0): N = 50, 60, 70 per side.
NO control row and NO N=80 row (CLAUDE.md section 6 no-rerun): anchor = stored
converged baseline N=80 (jobs 116854/116870, T 0.886 / Q ~1320 / lambda ~1558.5)
at these EXACT numerics (build_ports_base: box y 6.8 um / z-mult 5.42, window
30 nm / 3001 pts, mesh optimization). Nothing on the section-2 numerics list
changed -> stored anchor serves as control.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800;
target resonance ~1558.5 nm, window 30 nm centered 1558.5 (3001 pts, ~10 pm)
— anchored-TM window from _tm_base, unchanged.

Dispatch (SECOND, only after tm_nladder_c325's array has fully drained —
shared sweep_list.txt, CLAUDE.md section 6 serialize rule):
    ARRAY_TIME=04:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.sweeps.tm_nladder_c400 --max-concurrent=4
Output -> results/tm_nladder_c400/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_NM  = 400.0
N_LADDER = [50, 60, 70]

BASE = _common.build_ports_base()   # scatterer-program numerics verbatim (y 6.8 / z 8.8 um)
assert BASE.symmetry.use_z_symmetry, "bare device is z-symmetric — keep the 2x z saving"

assert len(set(N_LADDER)) == len(N_LADDER), "rows must be tag-unique (N is the tag)"
assert 80 not in N_LADDER, "N=80 exists (jobs 116854/116870) — never re-run it"

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM] * len(N_LADDER),
    n_periods_each_side  = list(N_LADDER),
    scatterer_radius_nm  = [0.0] * len(N_LADDER),
    mode  = "zipped",
    label = "tm_nladder_c400",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, n in enumerate(N_LADDER):
        print(f"  task {i}: corr={CORR_NM:.0f} N={n}  bare device")
    print("anchor (116854/116870, same numerics): N80 T 0.886 Q ~1320 lambda ~1558.5")
    print("verdict wanted: does any N<80 stay well-formed (in-window, shoulder-separated, T_peak<=0.9)?")
