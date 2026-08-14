"""Does the innermost-tooth shift change anything on the ANCHORED corr-400 TM device?

Study dir: runners/sweeps/   |   Created 2026-08-12   |   Job(s): (fill at dispatch)
Purpose (user): the tooth-shift lever was mapped on OTHER devices — corr-300 at
pitch 516.14 (runners/sweeps/tm_shift_p518, T 0.958 -> 0.972 but the mode widened
19.0 -> 20.3 um) and small doses on a W1050 cavity at accurate mesh (archived
tm_shift_frontier). Neither is the regular TM device. Four short N=80 rows ask
whether the shift moves T / Q / mode width on the plain anchored corr-400 W800
device at all.

Rows: innermost-tooth shift = 20/40/60/80 % of the half-pitch (258.4 nm) ->
51.7 / 103.4 / 155.1 / 206.7 nm — the SAME relative grid as the corr-300 study,
so the two corrugations are directly comparable. lengthen_cavity stays default
True (cavity absorbs 2*shift, total grating length preserved).

NO control row (CLAUDE.md section 6 no-rerun): anchor = stored
results_from_athena/asym_dw_study/results/result_N80_TM_avg_Ybox6p8_Zbox8p8.mat
(jobs 116854/116870) at these EXACT numerics — build_ports_base: box y 6.8 um /
z-mult 5.42, window 30 nm / 3001 pts, mesh optimization, ports-only. T 0.8864,
lambda 1558.617, |FWHM| 1.1885 nm (Q 1311), mode width 15.53 um. Nothing on the
section-2 numerics list changed -> the stored anchor serves as the control.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800;
target resonance ~1558.6 nm, window 30 nm centered 1558.5 (3001 pts, ~10 pm).

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    ARRAY_TIME=04:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.sweeps.tm_shift_c400 --max-concurrent=4
Output -> results/tm_shift_c400/results/  (tags _S52 / _S103 / _S155 / _S207).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

HALF_PITCH_NM = 516.83 / 2.0
SHIFT_PCT     = [20, 40, 60, 80]          # of the half-pitch; 100 % = teeth touch
SHIFTS_NM     = [round(p / 100.0 * HALF_PITCH_NM, 2) for p in SHIFT_PCT]

BASE = _common.build_ports_base()   # scatterer-program numerics verbatim (y 6.8 / z 8.8 um)

assert max(SHIFTS_NM) < HALF_PITCH_NM, "shift must stay below half-pitch (teeth would touch)"
assert len({round(s) for s in SHIFTS_NM}) == len(SHIFTS_NM), "rows must be tag-unique (_S{round(nm)})"

SPEC = SweepSpec(
    innermost_tooth_shift_nm = SHIFTS_NM,
    scatterer_radius_nm      = [0.0] * len(SHIFTS_NM),   # build_ports_base leaves the scatterer ON
    mode  = "zipped",
    label = "tm_shift_c400",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for pct, s in zip(SHIFT_PCT, SHIFTS_NM):
        print(f"  shift {s:6.2f} nm  ({pct} % of half-pitch)")
    print("anchor (116854/116870, same numerics): T 0.8864  lambda 1558.617  Q 1311  mode 15.53 um")
    print("question: does any row move T / Q / mode width off the anchor by more than the dx=50 nm jitter floor?")
