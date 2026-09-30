"""The displaced comb PAIR on the apodized device (apod-10, corr-400) — does the pair stack where the
single comb did not?

Study dir: runners/scatterers/   |   Created 2026-09-12   |   Job(s): Athena 146557 (1 task, 2026-09-12)
Purpose: the centred single comb LOST on apod-10 (scat_v_apodcomb, job 130171: 0.9723 vs ctrl
0.9770, dT -0.0047) because its phase-independent cost (end-scattering, set by the comb CENTRE)
outweighed what little needle the softened cusp leaves.  The displaced pair (scat_offcentre2,
job 146419: +12 um / 90 deg and -12 um / 90 deg, T 0.9040 on the uniform device) carries no such
cost (measured pedestal -0.009 / -0.002).  ONE row: the pair on apod-10.
  REGISTERED: dT vs ctrl 0.9770 in (-0.002, +0.010) EXPECTED (no k-space planes exist for apod-10,
  so no model number); the single comb's -0.0047 is the comparison.  If dT <= -0.0047 the pair
  inherits the single's cost after all; if dT >= +0.0036 (2x floor) the comb stacks with
  apodization once it is displaced.
Numerics = scat_v_apodcomb / trench_te_apod EXACTLY (ports base, box y 8.0 / z 8.8 um, 1501 pts /
30 nm, opt mesh, no far-field): control = results_from_athena/trench_te_apod/results/
result_N80_A10_TM_avg_Ybox8p0_Zbox8p8.mat, T 0.977024, lam 1559.196 (job 124531) — NOT re-run.
Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800, N = 80/side, linear
apodization over 10 periods each side of the cavity (center_mod_depth default 100 nm).

Dispatch:  bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_pair_apod --max-concurrent=3
Output -> results/scat_pair_apod/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0
LAM_NM    = 536.0
R_NM      = 110.0
HEIGHT_NM = 350.0
COMBS = [(31, 12.0, 134.0), (31, -12.0, 134.0)]     # (n_posts, centre um, site shift nm) — the measured pair

def comb_x(n, xc_um, dx_nm):
    return [round(k * LAM_NM + dx_nm + xc_um * 1e3, 1) for k in range(-(n - 1) // 2, (n - 1) // 2 + 1)]

XS = sum((comb_x(*c) for c in COMBS), [])

BASE = _common.build_ports_base()
BASE.y_span_override_m = 8.0e-6          # trench_te_apod numerics exactly
BASE.spectral.n_wl_points = 1501
BASE.spectral.scan_width_nm = 30.0

assert D_NM + R_NM + 1200.0 <= 4000.0, "comb too close to y PML at box 8"
assert min(b - a for a, b in zip(sorted(XS)[:-1], sorted(XS)[1:])) > 2 * R_NM, "posts overlap"

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM],
    scatterer_x_list_nm = [XS],
    scatterer_y_list_nm = [[D_NM] * len(XS)],
    scatterer_height_nm = [HEIGHT_NM],
    apod_method = ["linear"],
    n_apod_periods_each_side = [10],
    mode  = "zipped",
    label = "scat_pair_apod",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"  pair on apod-10: {len(XS)} posts, x=[{min(XS)}..{max(XS)}]; ctrl 0.9770 (job 124531), single comb 0.9723 (job 130171)")
