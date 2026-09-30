"""One long centred comb at full radius — the 'ends, not centre' test (2026-09-12, round 3).

Study dir: runners/scatterers/   |   Created 2026-09-12   |   Job(s): Athena 146564 (1 task, 2026-09-12)
Purpose: the k-space model (refit on 16 rows; scratch round3b_model -> results_from_athena/
comb_physics_rethink/data/round3b_model.json) says the comb's phase-independent cost is set by
where its ENDS sit (m=0 end-scattering, sign period 12.2 um), so a centred comb long enough to put
its ends in the good zone behaves like the displaced pair: 61 posts, Lambda 531, r 110, 270 deg,
ends at +-16 um -> model +0.0327 ('top') / +0.0320 ('lorentz'), the same as the pair (+0.0285 /
+0.0254 model, +0.0189 MEASURED, job 146419) and as any two-halves-with-a-phase-slip variant; the
family saturates there (71 posts +0.0331).  The stored 61-post row (scat_y_polish, N-scan) was run
at r 78 to hold sum r^2 fixed (an amplitude rule now known to be pedestal-confounded) and measured
+0.0137 vs the IGUM ctrl 0.8864; the model puts that row at +0.0182 / +0.0173 (its usual +0.004
optimism).  No 61-post comb at r 110 exists.
  REGISTERED: dT >= +0.0153 (pair 0.9040 - 2x floor, vs ctrl 0.8851): the long comb matches the
  pair -> the end-position rule holds and one comb replaces two.  REFUTED if dT <= +0.0137 + floor
  (no better than the r-78 row: the long comb is amplitude-limited after all).
Numerics = scat_offcentre2 exactly (far-field base, box y 16 / z 8.8 um, 1501 pts / 20 nm, opt
mesh, monitors at 2.0 wls; ctrl 0.8851 job 123563 / 0.8864 with these monitors, job 146364 row 8).
Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800, N = 80/side.

Dispatch:  bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_longcomb --max-concurrent=3
Output -> results/scat_longcomb/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0
LAM_NM    = 531.0
DX_NM     = 398.0             # 270 deg of 531 — the measured single-comb optimum (stage R/S)
R_NM      = 110.0
HEIGHT_NM = 350.0
N_POSTS   = 61                # ends at +-15.9 um (+ dx)
FF_DIST_WLS = 2.0

XS = [round(k * LAM_NM + DX_NM, 1) for k in range(-(N_POSTS - 1) // 2, (N_POSTS - 1) // 2 + 1)]

BASE = _common.build_ff_base()
BASE.y_span_override_m = 16.0e-6
BASE.spectral.n_wl_points = 1501
BASE.spectral.scan_width_nm = 20.0
BASE.farfield.farfield_dist_wls = FF_DIST_WLS

assert D_NM - R_NM >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"
assert max(abs(x) for x in XS) + R_NM <= _common.FF_X_SPAN_UM * 1000.0 / 2.0, "comb reaches past the far-field monitor"

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM],
    scatterer_x_list_nm = [XS],
    scatterer_y_list_nm = [[D_NM] * len(XS)],
    scatterer_height_nm = [HEIGHT_NM],
    mode  = "zipped",
    label = "scat_longcomb",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"  {N_POSTS} posts, Lambda {LAM_NM}, dx {DX_NM} nm, x=[{XS[0]}..{XS[-1]}]; ctrl 0.8851; pair 0.9040; stored r-78 N61 +0.0137")
