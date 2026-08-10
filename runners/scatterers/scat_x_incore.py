"""Stage X — in-core oxide-hole comb: the falsification test (IGUM).

Study dir: runners/scatterers/   |   Created 2026-08-10   |   Job(s): TBD (IGUM)
Purpose (user, explicit): can the anti-needle mechanism work with SiO2 holes
INSIDE the core (phase-adjusted), instead of SiN posts outside? THEORY: no —
in-core drive is ~20-30x the evanescent tail; even the steelman (r=80 minimum
renderable, y=+/-250 near the core edge ~0.7x field, N=10) is ~3-4x over the
matched amplitude = the regime where stage P MEASURED all phases losing; the
matched-amplitude variant (N~4) has eta ~ 0.15 => ceiling +0.0004 << floor.
Prior art: in-core hole lattice job 123303 (T 0.032-0.41 incl. the +pitch/4
phase-shifted row = "phase no help", measured). REGISTERED: row 0 severe loss;
rows 1-2 both < ctrl with a visible phase DIFFERENCE between them. Any row >
ctrl + floor (0.8882) FALSIFIES the amplitude model => refine.

Controls: NOT re-run — IGUM own-cluster ctrl MEASURED T 0.8864 (job 51285
task 0, identical numerics box16/1501/20nm).

Rows (Lambda 531, oxide holes n=1.444, r=80, h350 through-core, y=+/-250):
  0: N=31 (span +/-8 um), dx=398 (270 deg) — the comb transplanted in-core
  1: N=10 (span +/-2.4 um), dx=398 (270 deg) — steelman, cancelling phase
  2: N=10, dx=132 (90 deg) — steelman, opposite phase (the flip readout)

Physics line (section 4): TM h350, pitch 516.83, corr 400, W800, N=80/side;
resonance 1558.6, window 1548.5-1568.5 (20 nm / 1501), box y=16.

Dispatch (IGUM; Athena stage-W runs in parallel — license peak 2+1 <= 6):
    ARRAY_TIME=02:00:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.scatterers.scat_x_incore --max-concurrent=1
Output -> results/scat_x_incore/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LAM, Y_NM, R_NM = 531.0, 250.0, 80.0
N_OXIDE = 1.444

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

def comb_x(n_half, dx_nm):
    return [round(k * LAM + dx_nm, 1) for k in range(-n_half, n_half + 1)]

# rows: (n_half, dx_nm)
ROWS = [(15, 398.0), (4, 398.0), (4, 132.0)]   # N=31 | N=9 steelman x2 phases

SPEC = SweepSpec(
    scatterer_radius_nm = [R_NM] * len(ROWS),
    scatterer_x_list_nm = [comb_x(n, dx) for n, dx in ROWS],
    scatterer_y_list_nm = [[Y_NM] * (2 * n + 1) for n, _ in ROWS],
    scatterer_height_nm = [350.0] * len(ROWS),
    scatterer_index     = [N_OXIDE] * len(ROWS),
    mode  = "zipped",
    label = "scat_x_incore",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("anchors: IGUM ctrl 0.8864 (51285_0) | falsify-gate 0.8882 | "
          "lattice prior T 0.032-0.41 (123303)")
