"""HH apodization in TM: the genuine ln(T)-vs-N crossing to the -3 dB point (IGUM).

Study dir: runners/sweeps/   |   Created 2026-08-26   |   Job(s): TBD
Purpose: get Q at peak T = 0.5 for Itai's apodization in TM by THE SAME METHOD
that produced our own anchors (te_q3db_20um / trench_q3db_20um): an N ladder at
fixed geometry, ln(T) vs N, interpolate to T = 0.5. That makes the comparison
against our stored TM Q3dB (Q 13930, T 0.4906, mode 19.97 um) method-identical,
not a model conversion.

Geometry = his profile at scale 0.72, which MEASURED 20.08-20.10 um mode width
(exactly our 20 um spec) and is box-independent to 0.02 um. pitch 510.94 nm from
the calibrated-FDE retune (its scale-0.72 row landed lam 1558.583, -0.42 nm from
the 1559.00 target).

Box 6.8 x 6.81 um = the inverse-design programme's own numerics
(campaign_c325_seed*.py box_y_um=6.8, box_z_mult=4.14), so every number here is
directly comparable to that work too. Our round-1a ladder converged EARLIER than
this (T+R 0.9576 at y6.0/z5.03 -> 0.9582 at y7.5/z6.9), so 6.8 is safe.

WHY TM ONLY: the same crossing in TE needs N ~ 177/side with a 1760 ps ring-down
(Q_L ~ 132k) -> ~40 h per ladder point, ~6 days for a crossing. TM needs N ~ 124
with a 207 ps ring-down -> ~3.4 h per point. TE is measured instead via a
well-conditioned Q_i (Q_i is constant to 4% across our own N=166-215 ladder, so
Q(-3dB) = 0.293*Q_i is a MEASURED relation in our data, not an assumption).

N ladder: 110 / 124 / 140. DERIVED crossing estimate N ~ 124 from
ln(Q_c) proportional to kappa*N, anchored on the measured scale-0.72 row
(N=98 -> Q_L 2681) and Q_i ~ 53000 from the as-drawn TM row. The ladder brackets
it either side so ln(T) vs N interpolates rather than extrapolates.
Containment: min N * pitch = 110 * 0.51094 = 56.2 um >= 2 x 20 um (asserted).

Window: 10 nm centred 1558.6 (the measured scale-0.72 resonance; the box moved
TE's lambda by only 0.03 nm, so 5 nm of margin is ample) at 3001 pts = 3.3 pm,
giving ~30 points across the ~100 pm linewidth expected at the crossing.

Dispatch:
    SBATCH_MEM=256G bash igum/deploy_igum.sh \
        --option3 --spec=runners.sweeps.itai_hh_tm_cross --max-concurrent=2
Output -> results/itai_hh_tm_cross/results/ (download to results_from_igum/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.sweeps.itai_hh_apod import (teeth, pitch_for, AVG_WIDTH_NM, CAVITY_W_NM,
                                         BULK_NARROW_NM, BULK_WIDE_NM)

SCALE = 0.72                     # measured 20.08 um mode -- our Q3dB spec
# N=189 added 2026-08-26 after the 3-point fit: Q_c grows 3.19%/period and Q_i is
# N-independent (111029/117649/121007, 9%), putting the genuine T=0.5 crossing at
# N=189. Tasks 0-2 are ALREADY MEASURED (job 63451) -- resubmit ONLY task 3 with
# --array-tasks=3 (CLAUDE.md 6: never re-measure a stored result).
N_LADDER = [110, 124, 140, 189]
PITCH_NM = pitch_for("TM", SCALE)
BOX_Y_UM, BOX_Z_MULT = 6.8, 4.14
SCAN_CENTER_NM, SCAN_WIDTH_NM = 1558.6, 10.0

_T = [teeth(n, SCALE) for n in N_LADDER]
assert min(N_LADDER) * PITCH_NM >= 2.0 * 20.0 * 1e3      # containment

SPEC = SweepSpec(
    n_periods_each_side       = list(N_LADDER),
    pitch_nm                  = [PITCH_NM] * len(N_LADDER),
    avg_width_nm              = [AVG_WIDTH_NM] * len(N_LADDER),
    cavity_width_nm           = [CAVITY_W_NM] * len(N_LADDER),
    corrugation_depth_nm      = [round((BULK_WIDE_NM - BULK_NARROW_NM) * SCALE, 1)] * len(N_LADDER),
    width_narrow_per_tooth_nm = [t[0] for t in _T],
    width_wide_per_tooth_nm   = [t[1] for t in _T],
    polarization              = ["TM"] * len(N_LADDER),
    center_wavelength_nm      = [SCAN_CENTER_NM] * len(N_LADDER),
    scan_width_nm             = [SCAN_WIDTH_NM] * len(N_LADDER),
    y_span_um                 = [BOX_Y_UM] * len(N_LADDER),
    span_mult                 = [BOX_Z_MULT] * len(N_LADDER),
    mode  = "zipped",
    label = "itai_hh_tm_cross",
)

if __name__ == "__main__":
    print(SPEC.describe().split("width_narrow")[0])
    for i, n in enumerate(N_LADDER):
        tn, tw = _T[i]
        print(f"  task {i}: TM N={n:3d} scale {SCALE} pitch {PITCH_NM:.2f} "
              f"| half-device {n*PITCH_NM/1000:.1f} um | teeth {len(tn)} "
              f"| widest {max(tw):.1f} nm | window {SCAN_CENTER_NM}+-{SCAN_WIDTH_NM/2:.0f} nm")
