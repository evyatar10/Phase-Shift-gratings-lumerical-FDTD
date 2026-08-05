"""Flared-trench test of the d(x) shape formula on the apodized device.

Study dir: runners/metal_mirror/   |   Created 2026-07-29   |
Job(s): Athena 126913 (rows 0-3, 2026-07-29) | IGUM TBD (rows 0-4, 2026-08-01)
Purpose: the local variational model (OneDrive Meetings/air gaps/
air_trench_shape_formulation.pdf) predicts the optimal trench wall is straight
on uniform devices but FLARES over the apod ramp: gap D*(x) = D0 +
min(CAP, DELTA*ln(k0/kappa(x))), kappa from the designed apod-10 depth ramp
(40 -> 400 nm over teeth 1..10, verified in bragg_device.get_mod_depth).
Registered prediction: flare - straight = +0.001..+0.002 in T (at the 0.0018
floor); null within floor closes the shape axis on apod too.
RESULT (126913, MEASURED): flare -0.052 vs straight (25x floor) — REFUTED.

Round 2 (2026-08-01, user: "I don't buy one result"): the 126913 flare was a
COARSE STAIRCASE (2-um segments, facets up to ~0.4 um). Row 4 re-implements
the SAME smoothed envelope with mesh-scale segments (0.25 um pitch, facet
steps ~<=50 nm = 1 dx cell) to discriminate: LPG-envelope mechanism predicts
smooth ~= staircase (~-0.05); facet-scattering artifact predicts |smooth| <<
|staircase| (shape axis would REOPEN).

Rows (zipped, 5 tasks), all TM apod-10 N=80/side, trench h 2 um, w 800 nm:
0 ctrl (no trench) | 1 straight trench d=1800 (124531 geometry) |
2 flared trench (42 x 2-um staircase segments, y per segment from the formula)
| 3 flare jitter partner (segments shifted +25 nm = half a mesh cell; the
row-2/row-3 spread measures the numeric floor per section 2) |
4 SMOOTH flare (336 x 0.25-um segments, same Gaussian-smoothed formula).

Re-run justification (add-study step 0): round 2 runs on IGUM — solver
CLUSTER changed vs every stored baseline (126913 = Athena; igum/README.md
forbids mixing clusters inside one comparison), so rows 0-3 re-run as the
in-study controls/floor pair at identical numerics on IGUM. Round-1 note:
the 124531 rows (ctrl 0.9770 / straight 0.9809, box y=8) were checked and
REJECTED as comparators (flare peaks d_center ~2.9 um -> BOX y=10 here).

Physics line (section 4): TM h350, n 1.97/1.444, pitch 516.83, corr 400
(apod linear n=10, center depth 40 nm), N=80/side; expected resonances
1558.5-1559.2 nm; window 1543.5-1573.5 nm (30 nm / 1501 pts).

Dispatch — IGUM round 2 (queue must be EMPTY of other --option3 arrays, §6):
    bash igum/deploy_igum.sh --option3 --spec=runners.metal_mirror.trench_flare_apod
(Athena round 1 used: ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
    --option3 --spec=runners.metal_mirror.trench_flare_apod --max-concurrent=3)
Output -> results/trench_flare_apod/results/ (IGUM -> results_from_igum/).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

# ---- trench (matches the measured winner except the flared wall) ----
TRENCH_W_NM  = 800.0
TRENCH_H_NM  = 2000.0
STRAIGHT_D_NM = 1800.0          # straight-row center distance (job-124379 optimum)

# ---- formula constants (air_trench_shape_formulation.pdf, DERIVED) ----
GAP0_UM   = 0.9                 # optimal oxide gap at full corrugation
DELTA_UM  = 1.2                 # 1/(gamma-q), central value
CAP_UM    = 1.2                 # max offset (gap 2.1 um = measured zero-gain)
SIGMA_UM  = 1.5                 # Gaussian smoothing ~ needle non-locality
Y_REF_UM  = 0.9                 # gap -> d_center: tooth tip 0.5 + w/2 0.4

# ---- staircase (row 2/3, the 126913 implementation) ----
SEG_PITCH_UM = 2.0              # segment center spacing
SEG_LEN_UM   = 2.05             # segment x-span (25 nm overlap, no oxide slivers)
N_SEG_HALF   = 21               # per side -> centers +/-1 .. +/-41 um (L 84 um)

# ---- smooth (row 4): same envelope at mesh-scale steps ----
SEG_PITCH_SM_UM = 0.25          # facet height ~<=50 nm = 1 dx cell (vs ~0.4 um)
SEG_LEN_SM_UM   = 0.275         # 25 nm overlap, same convention
N_SEG_HALF_SM   = 168           # centers +/-0.125 .. +/-41.875 um (same L 84 um)

# ---- apod-10 designed ramp (bragg_device.get_mod_depth, verified) ----
PITCH_UM   = 0.51683
CAV_HALF_UM = PITCH_UM / 4.0    # cavity = pitch/2 wide, teeth start at pitch/4
N_APOD     = 10
DEPTH_CENTER_NM, DEPTH_FULL_NM = 40.0, 400.0


def _kappa_rel(x_um):
    """Local corrugation strength / full, from the designed linear ramp."""
    ax = abs(x_um)
    if ax < CAV_HALF_UM:
        return 0.0                                   # cavity: no teeth
    d = int((ax - CAV_HALF_UM) // PITCH_UM) + 1      # tooth index from cavity
    if d > N_APOD:
        return 1.0
    depth = DEPTH_CENTER_NM + (DEPTH_FULL_NM - DEPTH_CENTER_NM) * (d - 1) / N_APOD
    return depth / DEPTH_FULL_NM


def _gap_raw(x_um):
    k = _kappa_rel(x_um)
    off = CAP_UM if k <= 0.0 else min(CAP_UM, DELTA_UM * math.log(1.0 / k))
    return GAP0_UM + off


def flare_profile(seg_pitch_um=SEG_PITCH_UM, n_seg_half=N_SEG_HALF):
    """(x_centers_nm, d_centers_nm): Gaussian-smoothed formula on the staircase."""
    xs, ys = [], []
    for i in range(-n_seg_half, n_seg_half):
        xc = (i + 0.5) * seg_pitch_um
        num = den = 0.0
        s = -4.0 * SIGMA_UM
        while s <= 4.0 * SIGMA_UM:
            w = math.exp(-0.5 * (s / SIGMA_UM) ** 2)
            num += w * _gap_raw(xc + s)
            den += w
            s += 0.025
        xs.append(xc * 1000.0)
        ys.append((num / den + Y_REF_UM) * 1000.0)
    return xs, ys


FLARE_X_NM, FLARE_Y_NM = flare_profile()
JITTER_X_NM = [x + 25.0 for x in FLARE_X_NM]        # half a dx=50 nm mesh cell
SMOOTH_X_NM, SMOOTH_Y_NM = flare_profile(SEG_PITCH_SM_UM, N_SEG_HALF_SM)

BOX_Y_UM      = 10.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 30.0

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = TRENCH_H_NM * 1e-9

_PML_CLEAR_NM = 1200.0
for _ys in (FLARE_Y_NM, SMOOTH_Y_NM):
    assert max(_ys) + 0.5 * TRENCH_W_NM + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0
    assert min(_ys) - 0.5 * TRENCH_W_NM >= 500.0             # clear of tooth tips

# rows: (trench_on, x_list, y_list, y_scalar, seg_len_um)
ROWS = [
    (False, None,        None,       STRAIGHT_D_NM, 84.0),          # 0 ctrl
    (True,  None,        None,       STRAIGHT_D_NM, 84.0),          # 1 straight
    (True,  FLARE_X_NM,  FLARE_Y_NM, STRAIGHT_D_NM, SEG_LEN_UM),    # 2 flare
    (True,  JITTER_X_NM, FLARE_Y_NM, STRAIGHT_D_NM, SEG_LEN_UM),    # 3 +25 nm jit
    (True,  SMOOTH_X_NM, SMOOTH_Y_NM, STRAIGHT_D_NM, SEG_LEN_SM_UM),# 4 smooth
]

SPEC = SweepSpec(
    polarization             = ["TM"] * len(ROWS),
    pitch_nm                 = [PITCH_UM * 1000.0] * len(ROWS),
    corrugation_depth_nm     = [400.0] * len(ROWS),
    apod_method              = ["linear"] * len(ROWS),
    n_apod_periods_each_side = [10] * len(ROWS),
    scatterer_shape          = ["rect"] * len(ROWS),
    scatterer_x_span_um      = [r[4] if r[0] else 0.0 for r in ROWS],
    scatterer_y_span_nm      = [TRENCH_W_NM if r[0] else 0.0 for r in ROWS],
    scatterer_y_nm           = [r[3] for r in ROWS],
    scatterer_x_list_nm      = [r[1] for r in ROWS],
    scatterer_y_list_nm      = [r[2] for r in ROWS],
    scatterer_index          = [1.0] * len(ROWS),
    mode  = "zipped",
    label = "trench_flare_apod",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("\nflare profile (x_um, d_center_um):")
    for x, y in zip(FLARE_X_NM, FLARE_Y_NM):
        if x >= 0:
            print("  %6.2f  %.3f" % (x / 1000.0, y / 1000.0))
    _steps = [abs(SMOOTH_Y_NM[i + 1] - SMOOTH_Y_NM[i])
              for i in range(len(SMOOTH_Y_NM) - 1)]
    print("\nsmooth row: %d segments/side, max facet step %.1f nm "
          "(staircase had ~400 nm)" % (len(SMOOTH_X_NM), max(_steps)))
