"""Air trench on TE and on the apodized device — the two untested ports.

Study dir: runners/metal_mirror/   |   Created 2026-07-19   |   Job(s): TBD
Purpose: the air trench is measured working on TM W800 (+0.0159, job 124379)
and TM W1050 (+0.0157, job 124400) via TIR + light-cone lowering. Two user
questions remain: (1) does it work for TE (ctrl loss 0.1217, job 124194 —
leak also in-plane, but needle angle unmeasured); (2) does it survive
apodization (apod n=10 leak only ~0.023, and stage G measured overlays
FLIPPING sign on apod devices)? Each question = its own in-study control at
identical numerics.

Rows (zipped, 4 tasks): 0 TE ctrl | 1 TE + trench | 2 TM apod-10 ctrl |
3 TM apod-10 + trench. Trench = the measured winner geometry: air (n=1.0)
rect, L 84 um x w 800 nm x h 2 um, mirrored +/-y, center d = 1800 nm.
TE anchors: pitch 500, corr 300 (tips 475 nm). TM apod: pitch 516.83,
corr 400, linear apod n=10, default depth (stage-G device).
Registered predictions: P1 TE transfers (dT ~ +0.01, EXPECTED — same
in-plane grazing physics); P2 apod: genuinely open — leak-scaled ~ +0.003
if the mechanism survives, ~0/negative if the apodized envelope's residual
leak is no longer grazing-dominated.

Numerics: ports base, box y = 8 um (trench outer 2.2 + 1.8 clearance; TE
converged at 4.8 so 8 is conservative), 1501 pts / 30 nm window centered
1558.5 (TE resonates 1558.74; apod resonance shifts — NO lambda sidecar,
each row's T read at its own found resonance, stage-G rule). Physics line
(section 4): h350, n 1.97/1.444, N = 80/side; windows 1543.5-1573.5 nm.

Dispatch (queue must be EMPTY of other --option3 arrays — section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.trench_te_apod --max-concurrent=3
Output -> results/trench_te_apod/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

TRENCH = dict(L_um=84.0, w_nm=800.0, d_nm=1800.0, h_nm=2000.0, index=1.0)

BOX_Y_UM      = 8.0
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 30.0

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = TRENCH["h_nm"] * 1e-9

# rows: (pol, pitch_nm, corr_nm, apod_method, trench_on)
ROWS = [
    ("TE", 500.0,  300.0, "none",   False),   # 0 TE control
    ("TE", 500.0,  300.0, "none",   True),    # 1 TE + trench
    ("TM", 516.83, 400.0, "linear", False),   # 2 apod-10 control (stage-G device)
    ("TM", 516.83, 400.0, "linear", True),    # 3 apod-10 + trench
]

_PML_CLEAR_NM = 1200.0
assert TRENCH["d_nm"] + 0.5 * TRENCH["w_nm"] + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0
assert TRENCH["d_nm"] - 0.5 * TRENCH["w_nm"] >= 500.0   # clear of tooth tips (<=500 both devices)

SPEC = SweepSpec(
    polarization             = [r[0] for r in ROWS],
    pitch_nm                 = [r[1] for r in ROWS],
    corrugation_depth_nm     = [r[2] for r in ROWS],
    apod_method              = [r[3] for r in ROWS],
    n_apod_periods_each_side = [10] * len(ROWS),          # inert on 'none' rows
    scatterer_shape          = ["rect"] * len(ROWS),
    scatterer_x_span_um      = [TRENCH["L_um"] if r[4] else 0.0 for r in ROWS],
    scatterer_y_span_nm      = [TRENCH["w_nm"] if r[4] else 0.0 for r in ROWS],
    scatterer_y_nm           = [TRENCH["d_nm"]] * len(ROWS),
    scatterer_index          = [TRENCH["index"]] * len(ROWS),
    mode  = "zipped",
    label = "trench_te_apod",
)

if __name__ == "__main__":
    print(SPEC.describe())
