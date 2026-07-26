"""Stage I — real-space 2D field-profile images (user request 2026-07-18).

Study dir: runners/scatterers/   |   Created 2026-07-18   |   Job(s): TBD
Purpose: E-field cross-section IMAGES (not far-field) for three devices, to SEE
how the mode + leak behave: (0) plain W800 control, (1) W800 + validated pillar
pair [0,270]@y=700 r=80 (stage-E winner, +0.0227), (2) W800 + retro comb 1-row
d=3.0 (stage-H null — visual autopsy: is there any standing pattern at the comb).

All three at IDENTICAL numerics (box y=16 um, window 1548.5-1568.5 / 1501 pts,
opt mesh) so the images are directly comparable; memory proven at this box in
stage H (~85 GB vs 128G request; 2D monitors add ~2 GB at 51 freq points).
2D profile monitors ON (XY "Side view", XZ "Top view", YZ cross) — post picks
the recorded point nearest the resonance (band centered by the lambda sidecar).
Server .mat will be LARGE: extract the resonance-lambda planes on Athena and
download slices only (CLAUDE.md section 6).

Dispatch (queue empty; no prelim — reuses the stage-H sidecar, valid at this
box, lambda difference across the three devices is ~pm):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.scatterers.scat_i_fieldmaps --max-concurrent=3
Output -> results/scat_i_fieldmaps/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

BOX_Y_UM      = 16.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0

LOCKED_LAMBDA_FILE = "/work/results/scat_h_retrocomb_lambda_res.json"   # written by job 123561

_LAM, _NH = 551.0, 75
COMB_X = [round(k * _LAM, 1) for k in range(-_NH, _NH + 1)]

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.monitors.record_2d_fields = True

# rows: (radius, x list, y list)
ROWS = [
    (0.0,   [0.0],        [700.0]),                 # 0 control
    (80.0,  [0.0, 270.0], [700.0, 700.0]),          # 1 pillar pair (stage-E winner)
    (110.0, COMB_X,       [3000.0] * len(COMB_X)),  # 2 retro comb d=3.0 (stage-H)
]

_keys = {(r, len(xs)) for r, xs, _ in ROWS}
assert len(_keys) == len(ROWS), "stage-I rows must be unique in (r, N)"

SPEC = SweepSpec(
    scatterer_radius_nm = [r for r, _, _ in ROWS],
    scatterer_x_list_nm = [xs for _, xs, _ in ROWS],
    scatterer_y_list_nm = [ys for _, _, ys in ROWS],
    mode  = "zipped",
    label = "scat_i_fieldmaps",
)

if __name__ == "__main__":
    print(SPEC.describe())
