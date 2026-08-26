"""Itai's HH device EXACTLY AS DRAWN -- the control we never ran (IGUM).

Study dir: runners/sweeps/   |   Created 2026-08-26   |   Job(s): TBD
Purpose: runners/sweeps/itai_hh_apod.py only ever ran MODIFIED versions of his
design (profile amplitude scaled 0.72/0.78, pitch retuned to our 1559 nm band).
This runner is his device untouched: his 61-value profile at FULL depth, HIS
pitch 514 nm, 98 periods/side, avg 1.0 um, cavity pitch/2 at 950.3 nm. It answers
"what does HIS design actually do", independent of anything we changed.

Separate file (not a row in itai_hh_apod.py) for one reason: that study's sweep
list is being read by in-flight array 63438, and rewriting it mid-array is the
CLAUDE.md 6 clobber that killed tasks 13-97 on 2026-07-02. Its own label -> its
own data/sweep_list_itai_hh_asdrawn.txt.

Box: y 9.0 um / span_mult 5.4 (z 8.77 um) -- the top rung of the 63438 ladder,
NOT the derived default, because the derived box is what produced T+R > 1
(y sized from the scalar avg+corr/2 = 1184 nm while his real teeth reach 1915 nm
at full depth -> ratio 2.09, worse than any rung of the stored convergence
series). At full depth the widest tooth is 1915.1 nm, so y 9.0 um = ratio 4.70,
exactly the rung where the stored series converged.

This is a LOCATOR round. lambda_res is uncertain: our calibrated FDE chain puts
his as-drawn device at ~1632 nm in OUR stack, his own measured LUT says ~1605,
his design target was 1600 -- a 32 nm spread, so the window is wide and the
resolution coarse. A high-Q line may come back under-resolved; round 2 then
re-runs it in a narrow window. Do NOT quote Q from this round without checking
points-per-linewidth.

Row 1 = TE (his polarization). Row 2 = the same drawn geometry in TM, which our
FDE puts near 1562 nm -- cheap, and it says what his geometry does in the
polarization we care about, with no retuning at all.

Dispatch (check the queue first; 63438 must not be disturbed -- different label,
different sweep list, so they cannot collide):
    SBATCH_MEM=256G bash igum/deploy_igum.sh \
        --option3 --spec=runners.sweeps.itai_hh_asdrawn --max-concurrent=2
Output -> results/itai_hh_asdrawn/results/ (download to results_from_igum/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.sweeps.itai_hh_apod import (APOD_NARROW_NM, APOD_WIDE_NM, BULK_NARROW_NM,
                                         BULK_WIDE_NM, N_APOD, N_SIDE, AVG_WIDTH_NM,
                                         CAVITY_W_NM)

PITCH_NM = 514.0          # HIS pitch, not retuned
Y_SPAN_UM, SPAN_MULT = 9.0, 5.4

# His drawn widths at FULL depth: the .npy profile straight through his own
# dw-LUT + index correction, no scaling and no re-solve.
NARROW = list(APOD_NARROW_NM) + [BULK_NARROW_NM] * (N_SIDE - N_APOD - 1)
WIDE   = list(APOD_WIDE_NM) + [BULK_WIDE_NM] * (N_SIDE - N_APOD - 1)
assert len(NARROW) == len(WIDE) == N_SIDE
assert abs(max(WIDE) - 1915.1) < 0.2, max(WIDE)      # his overshoot peak

# (polarization, centre_nm, window_nm) -- wide, because lambda_res is uncertain
ROWS = [("TE", 1620.0, 70.0),
        ("TM", 1565.0, 40.0)]

SPEC = SweepSpec(
    n_periods_each_side       = [N_SIDE] * len(ROWS),
    pitch_nm                  = [PITCH_NM] * len(ROWS),
    avg_width_nm              = [AVG_WIDTH_NM] * len(ROWS),
    cavity_width_nm           = [CAVITY_W_NM] * len(ROWS),
    corrugation_depth_nm      = [round(BULK_WIDE_NM - BULK_NARROW_NM, 1)] * len(ROWS),
    width_narrow_per_tooth_nm = [NARROW] * len(ROWS),
    width_wide_per_tooth_nm   = [WIDE] * len(ROWS),
    polarization              = [p for p, _, _ in ROWS],
    center_wavelength_nm      = [c for _, c, _ in ROWS],
    scan_width_nm             = [w for _, _, w in ROWS],
    y_span_um                 = [Y_SPAN_UM] * len(ROWS),
    span_mult                 = [SPAN_MULT] * len(ROWS),
    mode  = "zipped",
    label = "itai_hh_asdrawn",
)

if __name__ == "__main__":
    print(SPEC.describe().split("width_narrow")[0])
    for i, (pol, c, w) in enumerate(ROWS):
        print(f"  task {i}: {pol} pitch {PITCH_NM} N={N_SIDE} FULL-depth profile "
              f"| widest tooth {max(WIDE):.1f} nm | y {Y_SPAN_UM} um (ratio "
              f"{Y_SPAN_UM*1e3/max(WIDE):.2f}) | window {c:.0f}+-{w/2:.0f} nm")
