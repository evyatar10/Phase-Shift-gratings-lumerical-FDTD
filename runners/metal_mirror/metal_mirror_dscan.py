"""Metal-mirror d-scan — "option B" from the stage-H FINDINGS (scat_h_retrocomb).

Study dir: runners/metal_mirror/   |   Created 2026-07-18   |   Job(s): TBD
Purpose: the W800 TM corr-400 leak is ~62% in-plane, peaked at grazing needles
ux = 0.980 (theta ~ 11.5 deg from the axis, MEASURED sub-pixel, stage E). Every
DIELECTRIC reflector failed with a measured mechanism (SiN DBR/PhC = drain, job
119163; retro comb = transparent, stage H). A near-ideal mirror (PEC film) at
standoff d is the discriminator: mirror-image interference gives a T(d)
oscillation of period lambda/(2*n_clad*cos(78.5deg)) = lambda_y/2 ~ 2.68 um —
BOTH mechanisms (recoupling AND Drexhage suppression of the emission channel)
share that period, and a wrong d can INCREASE loss. No oscillation = the
reflect-and-recouple concept is dead for this film geometry.

Design: PEC rect film (ideal mirror: no material fit, no skin-depth meshing,
grid-snapped surface), length 82.6 um (+/-41.3, same as the stage-H comb, no
PML contact), thickness 200 nm, core height 350 nm (user choice: fab-shaped thin
film, NOT a tall wall — a null closes only THIS geometry, not "any mirror"),
mirrored +/-y, d-scan 3.0..5.7 um in 5 steps = one full 2.68 um cycle.
Numerics identical to stage H (box y=16 um, 1501 pts / 20 nm window, opt mesh);
reuses the stage-H lambda sidecar (same numerics; no prelim job needed).
Registered predictions + decision rule: see the plan / FINDINGS-to-be.

Instrument notes: the side FF monitor (auto at 6.75 um) sits BEHIND the mirror
for all d — side-P / needle-bin metrics are shadowed and NOT meaningful in
mirror rows; ports (T/loss/lambda/Q) + top monitor are the instruments.
lambda drift vs control is the evanescent-drag canary (stage H measured ZERO
at d >= 3 um).

Dispatch (queue must be EMPTY of other --option3 arrays — CLAUDE.md section 6):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.metal_mirror_dscan --max-concurrent=3
Output -> results/metal_mirror_dscan/results/.

To reuse/extend (edit the knobs below, nothing else):
  - different distances: edit D_SCAN_NM (keep 2500 nm <= d, PML clearance holds)
  - real metal instead of PEC: MIRROR_MATERIAL = "Al (Aluminium) - Palik"
    (check the material fit over the window in Material Explorer first; a
    5-10 nm mesh override over the film is the accurate-phase option)
  - taller wall: FILM_HEIGHT_NM = e.g. 6000.0 (the "any mirror" instrument)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

MIRROR_MATERIAL = "PEC (Perfect Electrical Conductor)"
FILM_LEN_UM     = 82.6            # +/-41.3 um along the guide (= stage-H comb span)
FILM_THICK_NM   = 200.0           # >> any real-metal skin depth; PEC value is moot
FILM_HEIGHT_NM  = None            # None -> core height 350 nm (user decision)
D_SCAN_NM       = [3000.0, 3675.0, 4350.0, 5025.0, 5700.0]   # one full 2.68 um cycle

BOX_Y_UM      = 16.0              # stage-H numerics (memory-proven ~85 GB at 1501 pts)
N_WL_POINTS   = 1501
SCAN_WIDTH_NM = 20.0              # window 1548.5-1568.5 nm around the 1558.6 resonance

# Reuse the stage-H sidecar (job 123561) — identical box/window/device numerics.
LOCKED_LAMBDA_FILE = "/work/results/scat_h_retrocomb_lambda_res.json"

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
if FILM_HEIGHT_NM:
    BASE.scatterer.height_m = FILM_HEIGHT_NM * 1e-9

# rows: (x_span_um, y_span_nm, d_nm, material) — x_span 0 = in-study control
ROWS = [(0.0, 0.0, 3000.0, None)]
ROWS += [(FILM_LEN_UM, FILM_THICK_NM, d, MIRROR_MATERIAL) for d in D_SCAN_NM]

_PML_CLEAR_NM = 1200.0            # > lambda/n_clad = 1080
for _, w, d, _m in ROWS:
    assert d + 0.5 * w + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0, \
        f"mirror too close to the y PML: d {d}"
    assert d - 0.5 * w >= 2500.0, "standoff below the 2.5 um near-field floor"
assert len({d for _, _, d, _m in ROWS[1:]}) == len(ROWS) - 1, \
    "d-scan rows must have unique distances (file-tag uniqueness)"

SPEC = SweepSpec(
    scatterer_shape     = ["rect"] * len(ROWS),
    scatterer_x_span_um = [r[0] for r in ROWS],
    scatterer_y_span_nm = [r[1] for r in ROWS],
    scatterer_y_nm      = [r[2] for r in ROWS],
    scatterer_material  = [r[3] for r in ROWS],
    mode  = "zipped",
    label = "metal_mirror_dscan",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"PEC film L={FILM_LEN_UM} um x t={FILM_THICK_NM} nm x "
          f"h={FILM_HEIGHT_NM or 350.0:.0f} nm, mirrored +/-y; "
          f"d-scan {[d/1000 for d in D_SCAN_NM]} um (cycle 2.68 um)")
