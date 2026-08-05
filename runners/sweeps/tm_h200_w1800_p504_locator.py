"""
Resonance locator + convergence set for the NEW TM cross-section:
h = 200 nm, avg width 1800 nm, corrugation 400 nm (wide 2000 / narrow 1600),
pitch 504 nm, n_core 1.97 / n_clad 1.444.

Study dir: results/tm_h200_w1800_p504_locator/     Job ID(s): 127302 (v1, invalid), 127309 (v2, COMPLETE)
Date: 2026-07-31
RESULTS (127309): resonance 1492.122 nm at N=150/300 and both boxes (stable to
1 pm); Q(N300)=1903/1883 (span 3.8/5.0); stopband (T<0.75) 1488.4-1495.9 nm;
span 3.8 leaves T_res=1.0155 (residual box artifact), span 5.0 is physical
(T_res=0.9868, loss 1.32%) => Q_rad ~ 2.8e5. N=1300 continuation:
runners/sweeps/tm_h200_w1800_p504_n1300.py
Purpose: the target device is N=1300/side (EME predicts lambda_res 1538.48 nm,
Q ~ 2e5) — far too long for a single 24 h FDTD job, so FDTD's role is to
(a) locate the actual resonance with OUR constant indices (EME index unknown;
lambda may shift by tens of nm) and (b) establish mesh + domain convergence at
this cross-section. All 4 rows are new points — no stored baseline exists at
h=200/w=1800/p504 (every prior study is h=350, w 800-1050), so no reuse applies.

v1 (job 127302) was INVALID and its 4 rows are legitimately re-run here (rule §0):
window 1520-1570 (from FDE lambda_B=1547.8 / EME 1538.5) missed the resonance,
and the default 1.8-lambda box gave T=1.5-1.9 (boundary artifact). Port-mode
diagnosis on the smoke .fsp: the port DOES select the true TM fundamental
(95% Ez, single lobe) but the FDTD mesh solves it at n_eff=1.476 (fine-mesh FDE:
1.535 — 200 nm core = only 7 dz cells, thin-core TM discretization red-shift),
so the FDTD-grid Bragg sits near 2*1.476*504 = 1488 nm; and the mode's ~0.8 um
vertical decay length puts ~2.5% of its power on the PML at the 1.8-lambda box
(known h200-TM behavior: SPAN_MULT 4-5 required).

Scan window: 1455-1525 nm (70 nm, center 1490) — covers the mesh-consistent
prediction 1488 with margin on both sides.

Rows (zipped) — all with enlarged box (span_mult, activates the domain file-tag):
  0  N=300, accurate mesh, span 3.8        — primary locator
  1  N=150, accurate mesh, span 3.8        — N-trend + cheap early sanity
  2  N=300, accurate mesh, span 5.0        — box-convergence check vs row 0
An optimization-vs-accurate MESH pair is deferred to a follow-up at the located
resonance: mesh mode is not in generate_file_tag(), so a same-N same-box mesh
pair inside one array would race on filenames (needs its own module/study dir).
NOTE for any future h=200 TM work: the ~50 nm FDTD-vs-FDE resonance offset here
is a dz-discretization artifact of the thin core — comparing to fab/EME needs a
dz-convergence check (bragg_device hardcodes dz = core_height/7 in the core box).

Dispatch:
  bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_h200_w1800_p504_locator
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


BASE = SimulationConfig()
BASE.geometry.core_height_m = 200e-9
BASE.geometry.avg_corrugation_width_m = 1800e-9
BASE.geometry.corrugation_depth_m = 400e-9
BASE.geometry.width_port_m = 1800e-9        # match the grating average — no port-junction step
BASE.grating.pitch_m = 504e-9
BASE.source.polarization = "TM"
BASE.spectral.center_wavelength_m = 1.490e-6
BASE.spectral.scan_width_nm = 70.0
BASE.monitors.record_2d_fields = False      # T/resonance only
BASE.farfield.enabled = False


SPEC = SweepSpec(
    n_periods_each_side = [300, 150, 300],
    simulation_mode     = ["accurate", "accurate", "accurate"],
    span_mult           = [3.8, 3.8, 5.0],
    mode = "zipped",
    label = "tm_h200_w1800_p504_locator",
)


if __name__ == "__main__":
    run_sweep_spec(SPEC, target="local", base=BASE)
