"""HH apodization in TE at ~19.7 um: walk N out to a WELL-CONDITIONED Q_i (IGUM).

Study dir: runners/sweeps/   |   Created 2026-08-26   |   Job(s): TBD
Purpose: every TE row so far sits at N=98, where the scaled profile gives T~0.98
and Q_i = Q_L/(1-sqrt(T)) carries a 60-85% error per 1% error in T. The mode
width and lambda_res are solid there; the LOSS is not. This ladder walks N out at
FIXED geometry (scale 0.58, the 19.688 um device) until T drops into the range
where Q_i is actually measurable.

ANCHOR (MEASURED, job 63454 task 1): N=98, scale 0.58, box 6.8x6.81 ->
lam 1559.7757, Q_L 3452, T 0.98393, T+R 0.98432, Q_i 427874, mode 19.688 um.
=> Q_c = 3480.

WHY A LADDER AND NOT ONE POINT: Q_c growth is only bracketed to 3.2-4.7%/period
  (a) his own TE scale-dependence at N=98, 3 points -> 4.81%/period
  (b) his TM ladder, MEASURED over 4 N            -> 3.19%/period
  (c) our TE corr250 ladder N=166..215            -> 3.48%/period
so the N that reaches a given T is uncertain by ~+-25. Guessing one large N is
how the TM ladder undershot (sized on Q_i=53k when the real value was 114k) and
how the other chat's invdesign_q3db_20um had to append N=240 today. Stage 1 is
therefore two rows: N=140 pins the growth rate against the N=98 anchor, N=175
should land T~0.8 (5% conditioning). Stage 2 places the next N from the fit.

WHY NOT GO STRAIGHT TO T=0.5: that needs N~184-223 with a 1673 ps ring-down,
estimated ~31 h -> exceeds the 23:30 walltime. T~0.8 is measurable at ~9 h and
Q(-3dB) = 0.293*Q_i then follows; that relation was cross-validated on TM to
0.4% (direct crossing 2.44x vs Q_i route 2.45x).

NUMERICS, learned from the TM ladder (job 63451/63491):
  * lambda_res does NOT move with N -- TM sat at 1558.6073 at N=110/124/140/189,
    identical to 4 dp. So a TIGHT window is safe: 6 nm centred on the measured
    1559.776, at 4001 pts = 1.5 pm -> ~23 points across the 34.5 pm line expected
    at T~0.8 (the TM ladder ran 3.3 pm and was fine at 45.8 pm).
  * box 6.8 x 6.81 = the inverse-design numerics; our own ladder converged
    EARLIER (T+R 0.9576 at y6.0/z5.03 -> 0.9582 at y7.5/z6.9).
  * T+R > 1 is a LOW-T-loss artefact and eases as T falls: TM at N=189 gave
    T+R = 0.603. Every row here is still gated on T+R <= 1 before use.
  * SBATCH_MEM=256G (the N=189 TM row ran fine at that).

Containment: min N * pitch = 140 * 0.48955 = 68.5 um >= 2 x 20 um (asserted).

Dispatch (seat-probe first; hold at >=35/50):
    SBATCH_MEM=256G bash igum/deploy_igum.sh \
        --option3 --spec=runners.sweeps.itai_hh_te_cross --max-concurrent=2
Output -> results/itai_hh_te_cross/results/ (download to results_from_igum/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.sweeps.itai_hh_apod import (teeth, pitch_for, AVG_WIDTH_NM, CAVITY_W_NM,
                                         BULK_NARROW_NM, BULK_WIDE_NM)

SCALE = 0.58                     # the 19.688 um device
# EXTENDED 2026-08-26: full ladder. Task 0 stays N=140 (the row already running as
# 63511_0) so a preemption+requeue re-reads the SAME row -- appending is safe,
# reordering would not be. 155/175/195 span T ~0.85 down to ~0.6-0.7, giving a
# proper Q_c(N) fit instead of a two-point extrapolation.
N_LADDER = [140, 155, 175, 195]
PITCH_NM = pitch_for("TE", SCALE)
BOX_Y_UM, BOX_Z_MULT = 6.8, 4.14
SCAN_CENTER_NM, SCAN_WIDTH_NM = 1559.776, 3.0   # narrowed from 6.0: the guard
#   failed at N=195 (5.9 samples on the worst-case 8.9 pm line). lambda_res is
#   N-STABLE (TM sat at 1558.6073 across N=110..189), so a tight window is safe.
N_WL_POINTS = 4001               # 0.75 pm

_T = [teeth(n, SCALE) for n in N_LADDER]
assert min(N_LADDER) * PITCH_NM >= 2.0 * 20.0 * 1e3

# ── SAMPLING GUARD (adopted from the other chat's invdesign_q3db_20um, which
# found the failure mode the hard way): UNDER-SAMPLING BIASES PEAK T LOW, and a
# low T drags the apparent crossing DOWN while looking perfectly self-consistent.
# So assert >=10 samples across the narrowest line this ladder could produce,
# i.e. under the FASTEST Q_c growth branch (4.7%/period), not the expected one.
_DLAM_PM = SCAN_WIDTH_NM * 1000.0 / (N_WL_POINTS - 1)
_QC0, _QI, _N0 = 3480.0, 427874.0, 98
for _n in N_LADDER:
    _qc = _QC0 * (1.047 ** (_n - _N0))              # fastest plausible growth
    _ql = 1.0 / (1.0 / _qc + 1.0 / _QI)             # narrowest plausible line
    _pts = (SCAN_CENTER_NM * 1000.0 / _ql) / _DLAM_PM
    assert _pts >= 10, (f"N={_n}: only {_pts:.1f} samples across the worst-case "
                        f"{SCAN_CENTER_NM*1000/_ql:.1f} pm line at {_DLAM_PM:.2f} pm "
                        f"-- narrow SCAN_WIDTH_NM or raise N_WL_POINTS")

BASE = None
SPEC = SweepSpec(
    n_periods_each_side       = list(N_LADDER),
    pitch_nm                  = [PITCH_NM] * len(N_LADDER),
    avg_width_nm              = [AVG_WIDTH_NM] * len(N_LADDER),
    cavity_width_nm           = [CAVITY_W_NM] * len(N_LADDER),
    corrugation_depth_nm      = [round((BULK_WIDE_NM - BULK_NARROW_NM) * SCALE, 1)] * len(N_LADDER),
    width_narrow_per_tooth_nm = [t[0] for t in _T],
    width_wide_per_tooth_nm   = [t[1] for t in _T],
    polarization              = ["TE"] * len(N_LADDER),
    center_wavelength_nm      = [SCAN_CENTER_NM] * len(N_LADDER),
    scan_width_nm             = [SCAN_WIDTH_NM] * len(N_LADDER),
    y_span_um                 = [BOX_Y_UM] * len(N_LADDER),
    span_mult                 = [BOX_Z_MULT] * len(N_LADDER),
    mode  = "zipped",
    label = "itai_hh_te_cross",
)

if __name__ == "__main__":
    print(SPEC.describe().split("width_narrow")[0])
    for i, n in enumerate(N_LADDER):
        tn, tw = _T[i]
        print(f"  task {i}: TE N={n:3d} scale {SCALE} pitch {PITCH_NM:.2f} "
              f"| half-device {n*PITCH_NM/1000:.1f} um | widest {max(tw):.1f} nm "
              f"| window {SCAN_CENTER_NM}+-{SCAN_WIDTH_NM/2:.1f} nm @ {N_WL_POINTS} pts "
              f"= {SCAN_WIDTH_NM*1000/(N_WL_POINTS-1):.2f} pm")
