"""Air-trench d-scan on the W800 corr-400 base — the TIR reflector for the 11deg needle.

Study dir: runners/metal_mirror/   |   Created 2026-07-19   |   Job(s): TBD
Purpose: the W800 TM corr-400 residual leak is a grazing needle at ux = 0.980
(theta ~ 11.5 deg off-axis => ~78.5 deg incidence on a guide-parallel wall,
FAR beyond the oxide->air critical angle 43.8 deg). A lateral AIR trench
(n = 1.0) is therefore a LOSSLESS TIR mirror for exactly this needle — and at
small standoff it also shrinks the in-plane light cone (modifies emission at
the source, the regime where the reflector family ever won). Measured
precedent on the STACK device (job 118893, archived tm_air_trench.py):
d-optimum 1.8 um, loss 0.0545 -> 0.0423 (-22%), SiN control catastrophic =
low-index/TIR mechanism load-bearing. UNTESTED on this device (leak 0.110,
2x the stack's) — that transfer is what this scan measures.

Design: air trench (rect, index 1.0), L = 84 um (full arm span — short trench
measured harmful), width 800 nm (measured better than 400 on the stack),
height 2 um (archived winner; covers core + in-plane leak z-extent ~1-1.5 um),
mirrored +/-y, x = 0. d = trench CENTER |y|; scan d = 0.9..3.0 um: inner edge
from touching the wide-tooth tips (y = 500 nm, d = 0.9 row — near-field /
light-cone regime) out to the far-field regime bounded by the PEC result
(+0.002 ceiling at d >= 2.5 um). Registered predictions: P1 near rows (d <=
1.8) beat the +0.002 far ceiling if the light-cone mechanism transfers
(stack scaling ~ +0.01-0.02 in T); P2 a d-optimum exists (stack: 1.2 worse <
1.8 best > 2.4 fading); P3 lambda drag grows toward d = 0.9 (evanescent
contact) — drag with NO T gain = drain, report it.

Numerics: stage-H/J instrumentation exactly (box y = 16 um, 1501 pts / 20 nm
window, opt mesh, ff base) so the control reproduces 0.8851 and rows compare
to the PEC scan at identical numerics. Trench outer edge <= 3.4 um leaves the
side FF monitor (6.75 um) CLEAR — needle-bin |ux| > 0.95 is a live mechanism
instrument here (unlike the shadowed PEC rows). Physics line (CLAUDE.md
section 4): TM h350, pitch 516.83, corr 400, W800, N = 80/side; target
resonance 1558.6 nm, window 1548.5-1568.5 nm.

Dispatch (queue must be EMPTY of other --option3 arrays — CLAUDE.md section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.air_trench_dscan --max-concurrent=3
Output -> results/air_trench_dscan/results/.

To reuse/extend (edit the knobs, nothing else): TRENCH_W_NM, D_SCAN_NM,
TRENCH_HEIGHT_NM (e.g. full-cladding wall), TRENCH_LEN_UM.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

TRENCH_INDEX     = 1.0            # air; the low-index/TIR mechanism is the point
TRENCH_LEN_UM    = 84.0           # full arm span (short 20 um trench measured harmful)
TRENCH_W_NM      = 800.0          # stack winner width
TRENCH_HEIGHT_NM = 2000.0         # stack winner height (in-plane leak z-extent ~1-1.5 um)
D_SCAN_NM        = [900.0, 1200.0, 1500.0, 1800.0, 2100.0, 2400.0, 3000.0]

BOX_Y_UM      = 16.0              # stage-H/J numerics (memory-proven ~85 GB at 1501 pts)
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0              # window 1548.5-1568.5 nm around the 1558.6 resonance

# Reuse the stage-H sidecar (job 123561) — identical box/window/device numerics.
LOCKED_LAMBDA_FILE = "/work/results/scat_h_retrocomb_lambda_res.json"

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.scatterer.height_m = TRENCH_HEIGHT_NM * 1e-9

# rows: (x_span_um, y_span_nm, d_nm) — x_span 0 = in-study control
ROWS = [(0.0, 0.0, 900.0)]
ROWS += [(TRENCH_LEN_UM, TRENCH_W_NM, d) for d in D_SCAN_NM]

_PML_CLEAR_NM = 1200.0            # > lambda/n_clad = 1080
for _, w, d in ROWS:
    assert d + 0.5 * w + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0, \
        f"trench too close to the y PML: d {d}"
    assert d - 0.5 * w >= _common.TOOTH_EDGE_NM, \
        f"trench overlaps the teeth (inner edge < 500 nm): d {d}"
assert len({d for _, _, d in ROWS[1:]}) == len(ROWS) - 1, \
    "d-scan rows must have unique distances (file-tag uniqueness)"

SPEC = SweepSpec(
    scatterer_shape     = ["rect"] * len(ROWS),
    scatterer_x_span_um = [r[0] for r in ROWS],
    scatterer_y_span_nm = [r[1] for r in ROWS],
    scatterer_y_nm      = [r[2] for r in ROWS],
    scatterer_index     = [TRENCH_INDEX] * len(ROWS),
    mode  = "zipped",
    label = "air_trench_dscan",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"air trench (n={TRENCH_INDEX}) L={TRENCH_LEN_UM} um x w={TRENCH_W_NM} nm x "
          f"h={TRENCH_HEIGHT_NM:.0f} nm, mirrored +/-y; "
          f"d-scan {[d / 1000 for d in D_SCAN_NM]} um (inner edge d - 0.4 um)")
