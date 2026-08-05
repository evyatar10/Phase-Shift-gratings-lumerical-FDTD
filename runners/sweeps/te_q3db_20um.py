"""Max loaded Q at -3 dB peak T for a 20 um TE mode, no trench (Athena).

Study dir: runners/sweeps/   |   Created 2026-08-04   |   Job(s): TBD
Purpose: TE sibling of trench_q3db_20um. Trench arm deliberately ABSENT: the
trench is a measured NULL on TE (trench_te_apod: ctrl 0.8756 vs trench 0.8747).
Single arm, two targets: fwhm_m = 20 um (corr knob) and peak T = 0.5 (N knob);
deliverable = loaded Q at the crossing. Auto-shutoff left at the production
default 1e-7 (user instruction; the shutoff study is still open).

Round 1 (this SPEC, 9 tasks):
  tasks 0-3: corr ladder {300, 260, 233, 210} at N=110 (half-device 55 um >=
             2.5x target fwhm -> no truncation bias). Seed 233 = through-origin
             scaling of the LEGACY anchor TE corr 300 -> 15.54 um (seed only;
             the lock uses in-study points, lesson of the TM corr-276 hedge).
  task  4:   control row: N=80 corr 300 = the legacy TE baseline (expect
             T ~ 0.83, Q ~ 1712 at these default-box numerics).
  tasks 5-8: hedged N ladder at corr 233: N {140, 170, 200, 230} -> ln(T) vs N.
Round 2 (later edit): corr20 lock + T=0.5 interpolation + 1 integer confirm.

Physics line (section 4): TE h350, pitch 500, W800, n 1.97/1.444, default box
(no trench -> no widened Ybox needed; all comparisons in-study). Target
resonance ~1559-1561 nm; window 30 nm centered 1560 (1545-1575), 4001 pts
= 7.5 pm (TE linewidths >= 0.25 nm -> >= 33 pts/linewidth).

Dispatch (Athena; license: sum of throttles across clusters <= 6, IGUM holds 4):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.te_q3db_20um \
        --max-concurrent=2
Output -> results/te_q3db_20um/results/ (download to results_from_athena/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from simulation_config import SimulationConfig

# ROUND 2 (2026-08-05). Round 1 ran LEAN off the original 9-row list: only
# canary task 4 (corr300 N80: 15.56 um, T 0.870) + tasks 2/6 (corr233:
# N110 21.60 um T 0.921 / N170 22.54 um T 0.652) were dispatched — jobs
# 128580/128581. 2-point width line -> corr(20um) ~= 250; ln T line ->
# crossing N ~= 190-210 at corr 250. Rows below = the bracket pair that
# confirms both at once.
# ROUND 3 (confirm pair). Round 2 (128593): corr250 width CONFIRMED
# (20.51/20.54 um at N190/215) but T(190)=0.255 — crossing sits BELOW the
# bracket; measured deep slope -0.043/period vs flatter near-crossing slope
# puts N* at 166-175. Pair {168,176} brackets it under both hypotheses.
# ROUND 4 (polish): round 3 measured N168 T 0.472 (IN BAND, fallback) and
# N176 T 0.390 -> local slope -0.024/period -> crossing N*=165.6; N=166
# predicts T~0.495 (TM-endgame precedent: take the closest integer).
CORR_LADDER_NM = []
CORR_GUESS_NM  = 250.0
N_LADDER       = [166]
N_WIDTH        = 110                    # (round-1 corr-ladder N; unused now)

SCAN_CENTER_NM = 1560.0
SCAN_WIDTH_NM  = 30.0
N_WL_POINTS    = 4001                   # 7.5 pm

# Rows: (corr_nm, n_side)
ROWS = [(c, N_WIDTH) for c in CORR_LADDER_NM]
ROWS += [(CORR_GUESS_NM, n) for n in N_LADDER]

BASE = SimulationConfig()               # TE / pitch 500 / W800 defaults
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

# Containment: half-device >= ~2x target fwhm for every WIDTH-measuring row
# (the N=80 legacy-control row is exempt — its fwhm is not used for the lock).
assert min([N_WIDTH] + N_LADDER) * 0.500 >= 2.0 * 20.0

SPEC = SweepSpec(
    corrugation_depth_nm = [c for c, n in ROWS],
    n_periods_each_side  = [n for c, n in ROWS],
    center_wavelength_nm = [SCAN_CENTER_NM] * len(ROWS),
    mode  = "zipped",
    label = "te_q3db_20um",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (c, n) in enumerate(ROWS):
        print(f"  task {i}: corr={c:5.0f} nm  N={n}")
