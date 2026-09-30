"""Off-centre comb — the end-scattering phase knob (physics rethink 2026-09-11/12).

Study dir: runners/scatterers/   |   Created 2026-09-12   |   Job(s): Athena 146364 (9 tasks, 2026-09-12)
Purpose: the measured Lambda=536 phase circle (stage R, job 130091) is a sinusoid on a
PEDESTAL: in units of the radiative rate, mean +0.178, swing +/-0.25. The k-space model
(python_tools/comb_kspace_model.py, docs/comb_physics_rethink_2026-09-11.md section 6b)
attributes most of that pedestal to m=0 end-scattering of the counter-propagating carrier,
whose phase is set by the comb CENTRE x_c (rotates at beta-k_c = 0.257 rad/um, so a shift
of pi/(beta-k_c) = 12 um flips its sign), not by dx.  Every comb ever run was centred on the
cavity.  This run measures the SAME 4-point phase circle (Lambda 536, r 110, 31 posts,
d 1.8 um, dx 0/134/268/402) with the comb centre displaced by +12 um (narrow-tooth side)
and by -12 um (wide-tooth side), plus a control at the new monitor placement.

REGISTERED (all four model variants agree on the mechanism, not on the side):
  P1 the circle's PEDESTAL (mean of the four x = dgamma/gamma) drops well below the
     centred +0.178 on at least one side (model: -0.03..+0.09);
  P2 the best T on that side exceeds the centred best 0.8928 by >= 2x floor (0.8964);
  P3 the two sides DIFFER (the standing wave has arg(c-/c+) = 108 deg at x = 0).
  REFUTED if both sides reproduce the centred circle within the floor (0.0018).
Second-comb step: if P1-P3 pass, the pair (centred winner + a second comb on the winning
side at its best phase) is dispatched as a follow-up (1-2 tasks).

Control row (row 8): re-run ONLY because the far-field monitors moved (farfield_dist_wls
0.8 -> 2.0: side monitor 6.75 -> 4.88 um, top 3.15 -> 1.28 um, captures grazing rays to
u_x ~ 0.995 instead of ~0.98).  Its T MUST reproduce the stored 0.8851 (job 123563):
monitors do not touch the mesh or the converged box (y 16 / z 8.8 um).  All dT vs 0.8851.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800, N = 80/side;
target resonance 1558.6 nm, window 1548.5-1568.5 nm (20 nm / 1501 pts), opt mesh.

Dispatch (queue must be EMPTY of other Athena --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_offcentre --max-concurrent=3
Output -> results/scat_offcentre/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0
N_HALF    = 15                # 31 posts — stage-P/R geometry exactly
LAM_NM    = 536.0             # the stored phase circle's period (stage R)
R_NM      = 110.0
HEIGHT_NM = 350.0
DX_LIST   = [0.0, 134.0, 268.0, 402.0]      # 0/90/180/270 deg of Lambda=536
XC_LIST   = [12000.0, -12000.0]             # comb centre, nm (pi/(beta-k_c) = 12.2 um)

BOX_Y_UM       = 16.0         # stage-H/P/R numerics exactly
N_WL_POINTS    = 1501
SCAN_WIDTH_NM  = 20.0
FF_DIST_WLS    = 2.0          # closer far-field monitors (was 0.8): side 4.88 um, top 1.28 um

def comb_x(dx_nm, xc_nm):
    return [round(k * LAM_NM + dx_nm + xc_nm, 1) for k in range(-N_HALF, N_HALF + 1)]

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.farfield.farfield_dist_wls = FF_DIST_WLS

assert D_NM + R_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0, "comb too close to y PML"
assert D_NM - R_NM >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"
assert max(abs(x) for xc in XC_LIST for dx in DX_LIST for x in comb_x(dx, xc)) + R_NM <= _common.FF_X_SPAN_UM * 1000.0 / 2.0, \
    "displaced comb reaches past the far-field monitor"

X_ROWS = [comb_x(dx, xc) for xc in XC_LIST for dx in DX_LIST]
N_ROWS = len(X_ROWS) + 1     # + control

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(X_ROWS) + [0.0],
    scatterer_x_list_nm = X_ROWS + [[0.0]],
    scatterer_y_list_nm = [[D_NM] * (2 * N_HALF + 1)] * len(X_ROWS) + [[D_NM]],
    scatterer_height_nm = [HEIGHT_NM] * N_ROWS,
    mode  = "zipped",
    label = "scat_offcentre",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (xc, dx) in enumerate([(xc, dx) for xc in XC_LIST for dx in DX_LIST]):
        xs = comb_x(dx, xc)
        print(f"  row {i}: xc={xc/1000:+.0f} um dx={dx} nm ({dx/LAM_NM*360:.0f} deg), x=[{xs[0]}..{xs[-1]}]")
    print(f"  row {N_ROWS-1}: control (no comb), monitors at dist {FF_DIST_WLS} wls")
    print("dT vs MEASURED Athena ctrl 0.8851 (job 123563); centred circle: 0.8664/0.8407/0.8657/0.8928")
