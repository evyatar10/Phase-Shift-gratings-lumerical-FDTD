"""The displaced pair lengthened to 61 posts per comb (user request, 2026-09-12, round 4).

Study dir: runners/scatterers/   |   Created 2026-09-12   |   Job(s): Athena 146614 (2 tasks, 2026-09-12) — MEASURED 0.9057 / 0.9040 = TIE with the 31-pair (0.9070)
Purpose: the best measured pair is two 31-post combs at +-12 um, Lambda 531, r 110, d 1.8 (T 0.9070,
job 146553 row 2); one centred 61-post comb ties it (0.9064, job 146564).  Two 61-post combs cannot
sit at +-12 um (they would overlap across the cavity), so they are pushed out until their inner ends
meet at the cavity: centres +-XC um, 122 posts spanning +-32 um.  Phases re-optimised in the k-space
model for these end positions (scratch round4_model -> results_from_athena/comb_physics_rethink/
data/round4_model.log); row 1 is the model's second choice as a phase/centre hedge (the 31-pair
phases 60/120 deg would score -0.035 here and are NOT used).
  REGISTERED: the model predicts +0.039 / +0.038 for row 0 against +0.032 for the 31-pair and the
  single 61-post comb (both measured ~0.907), i.e. the outer stretch 20-33 um still pays +0.007 in
  model units; with the model's usual +0.004 optimism the expectation is T ~ 0.910-0.912.
  CONFIRMED (outer stretch pays) if T >= 0.9106 (0.9070 + 2x floor); TIE/saturation if within
  +-0.0036 of 0.9070; REFUTED (amplitude past optimum) if T <= 0.9034.
Numerics = scat_offcentre2 exactly (far-field base, box y 16 / z 8.8 um, 1501 pts / 20 nm, opt mesh,
monitors at 2.0 wls) EXCEPT the far-field monitor x-span raised 60 -> 70 um so the posts stay inside
it (a DFT monitor: T untouched, cf. control 0.8864 vs 0.8851 with moved monitors).  dT vs ctrl 0.8851.
Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800, N = 80/side.

Dispatch:  bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_longpair --max-concurrent=3
Output -> results/scat_longpair/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

D_NM      = 1800.0
LAM_NM    = 531.0
R_NM      = 110.0
HEIGHT_NM = 350.0
N_POSTS   = 61
FF_X_SPAN_UM = 70.0

# rows: (centre um, dx right nm, dx left nm) — filled from round4_model
ROWS = [
    (17.0, 398.2, 398.2),   # 0: model best in both calibrations: +-17 um, 270 / 270 deg -> +0.0394 ('top') / +0.0384 ('lorentz')
    (17.5, 442.5, 398.2),   # 1: hedge: +-17.5 um, 300 / 270 deg -> +0.0389 ('lorentz'), ~+0.038 ('top')
]

def comb_x(xc_um, dx_nm):
    return [round(k * LAM_NM + dx_nm + xc_um * 1e3, 1) for k in range(-(N_POSTS - 1) // 2, (N_POSTS - 1) // 2 + 1)]

X_ROWS = [comb_x(xc, dr) + comb_x(-xc, dl) for xc, dr, dl in ROWS]

BASE = _common.build_ff_base()
BASE.y_span_override_m = 16.0e-6
BASE.spectral.n_wl_points = 1501
BASE.spectral.scan_width_nm = 20.0
BASE.farfield.farfield_dist_wls = 2.0
BASE.farfield.farfield_x_span_m = FF_X_SPAN_UM * 1e-6

assert D_NM - R_NM >= _common.TOOTH_EDGE_NM, "comb overlaps the teeth"
for xs in X_ROWS:
    assert max(abs(x) for x in xs) + R_NM <= FF_X_SPAN_UM * 1000.0 / 2.0, "comb reaches past the far-field monitor"
    assert min(b - a for a, b in zip(sorted(xs)[:-1], sorted(xs)[1:])) > 2 * R_NM, "posts overlap"

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(ROWS),
    scatterer_x_list_nm = X_ROWS,
    scatterer_y_list_nm = [[D_NM] * len(xs) for xs in X_ROWS],
    scatterer_height_nm = [HEIGHT_NM] * len(ROWS),
    mode  = "zipped",
    label = "scat_longpair",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, ((xc, dr, dl), xs) in enumerate(zip(ROWS, X_ROWS)):
        print(f"  row {i}: 2 x {N_POSTS} posts at +-{xc} um, dx {dr}/{dl} nm ({dr/LAM_NM*360:.0f}/{dl/LAM_NM*360:.0f} deg), x=[{min(xs)}..{max(xs)}], central gap {min(x for x in xs if x > 0) - max(x for x in xs if x < 0):.0f} nm")
