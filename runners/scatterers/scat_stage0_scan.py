"""Stage 0 — single-scatterer effectiveness scan (position x line, per y row, per radius).

Study dir: runners/scatterers/   |   Created 2026-07-12   |   Job(s): TBD
Purpose: quick "where does one scatterer do anything" map to pick RADIUS_NM / Y0_NM
in _common.py before the response-matrix stages. Fresh parameterized re-home of the
closed 2026-07 scan pattern (runners/archive/sweeps/scatterers/tm_scatterer_scan.py —
archive untouched); kept at that study's numerics (opt mesh, 4.8 um y-box) so points
are directly comparable with its FINDINGS.md tables.

Dispatch (queue empty of other --option3 arrays):
    bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_stage0_scan
Smoke: --array-tasks=0-2. Output -> results/scat_stage0_scan/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.sweeps._tm_base import build_base
from runners.scatterers import _common

# ── Scan knobs (edit here) ─────────────────────────────────────────────────────
X_STEP_NM  = 270.0                    # x-line step (pitch/2 ~ full phase flip per step)
X_MAX_NM   = 6480.0                   # x-line extent (0 .. X_MAX, +x arm; device symmetric)
Y_LIST_NM  = [700.0, 1000.0]          # lateral rows to compare (closest clean row vs 2026-07 line)
R_LIST_NM  = [_common.RADIUS_NM]      # radii to bracket (add e.g. 80.0, 125.0 if re-checking)

BASE = build_base()                   # archived-scan numerics: opt mesh, 4.8 um y-box
BASE.mesh.simulation_mode = _common.MESH_MODE

_x_line = [round(i * X_STEP_NM, 1) for i in range(int(round(X_MAX_NM / X_STEP_NM)) + 1)]

_rs, _xs, _ys = [0.0], [0.0], [Y_LIST_NM[0]]          # idx 0 = no-scatterer control
for r in R_LIST_NM:
    for y in Y_LIST_NM:
        for x in _x_line:
            _rs.append(r); _xs.append(x); _ys.append(y)

_keys = {(r, x, y) for r, x, y in zip(_rs, _xs, _ys)}
assert len(_keys) == len(_rs), "stage-0 rows must be unique in (r, x, y)"

SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_nm      = _xs,
    scatterer_y_nm      = _ys,
    mode  = "zipped",
    label = "scat_stage0_scan",
)

if __name__ == "__main__":
    print(SPEC.describe())
