"""
TM RADIATION POLARIMETRY — the Phase-1 gating diagnostic of the loss program
(docs/tm_loss_program_phase0_2026-07-05.md).

Physics question (never DIRECTLY measured before): where does the ~8-11%
resonant radiation of the TM pi-shift device go, and in what polarization?
The in-plane picture so far rests on inference (side-by-side coupling decay +
paper_8); this study measures it with arm-length side (Y-normal) and top
(Z-normal) monitors near the converged-box edges, reduced SERVER-SIDE by
post_processing/extract_monitor_polarimetry to scalars + 1D x-profiles
(full nearfield maps pinned OFF — they are ~100s of MB/row).

Registered predictions (Phase-0 theory, filed before dispatch):
  P1. In-plane dominance: 2*P_side >> 2*P_top (expect >=70% of 1-T-R in-plane).
  P2. Energy audit: 2*(side+top) flux accounts for >=~60% of 1-T-R; if not,
      the monitors miss the radiation and NOTHING downstream is interpreted.
  P3. Lobe structure: arm contribution near-axial (~10 deg from x-axis);
      cavity contribution broadside-ish (the two-edge reading of the cavity
      ladder). Row 2 (champion 1050) should show LESS broadside side-flux near
      x~0 than row 1 (W800); row 5 (W1400, past the ladder reversal) should
      show it back.
  P4. Polarization split f_TE = P_te/(P_te+P_tm) on the side monitor is the
      REFLECTOR DESIGN GATE: f_TE >= 0.6 -> s-pol strip design; <= 0.3 ->
      p-pol, ceiling halved. No prediction registered — genuinely unknown.

Device: anchored TM (height 350 nm, pitch 516.83 nm, corr 400 nm, N=80/side,
n 1.97/1.444), converged box y=6.8 um / z-mult 5.42. Target resonance
~1558.6 nm (optimization mesh) / ~1556.0 nm (accurate); window 30 nm / 3001
pts, per-row CENTERED ON THAT ROW'S RESONANCE because the far-field monitors
record ONE frequency at the source band center.

Rows (zipped, 7 tasks):
   0 control W800, farfield OFF          (loss reference, identical numerics)
   1 W800 baseline, farfield ON          (the radiation map)
   2 cavity 1050 champion, farfield ON   (what did the -30% remove?)
   3 W800, center +0.6 nm off-peak       (band-center sensitivity control)
   4 W800, y_span 8.0 um                 (reactive-contamination discriminator:
                                          radiated flux ~unchanged, near-field
                                          tail drops e^(-gamma*dy))
   5 cavity 1400 (ladder reversal), ON   (two-edge lobe test)
   6 W800 accurate mesh, center 1555.95  (mesh robustness of the pattern)

Tags: farfield/mesh/window-center are NOT tag-encoded, so same-geometry rows
would clobber each other's .fsp/.h5/.mat (real incident class). Fix, keeping
the physics identical: row 1 sets cavity_width=800.0 explicitly (identical
geometry to option "avg", tag _W800 vs row 0's _avg); rows 3/6 carry cavity
detunings 0.02/0.04 nm (tags _D0p02/_D0p04; geometry change ~3000x below the
mesh cell). Row 4 is distinct via its _Ybox8p0 domain tag. All 7 unique.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    smoke: bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_radiation_polarimetry --array-tasks=0
    full:  bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_radiation_polarimetry
Output -> results/tm_radiation_polarimetry/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.scan_width_nm = 30.0
BASE.spectral.n_wl_points = 3001

# Far-field geometry: arm-length monitors (grating ~83 um + cavity), placed by
# the config layer near the box edges spanning ~the full cross-section. The
# guided-mode tail at the side monitor (y~2.15 um) is e^(-1.75*2.15) ~ 0.02 in
# amplitude -> ~5e-4 in power: negligible contamination of the flux integrals.
BASE.farfield.farfield_x_span_m = 84e-6
BASE.farfield.save_nearfield = False  # polarimetry reduction only (~MB, not 100s MB)

_n = 7
SPEC = SweepSpec(
    cavity_width_nm        = [None,  800.0, 1050.0, None,   None,  1400.0, None],
    cavity_neg_detuning_nm = [0.0,   0.0,   0.0,    0.02,   0.0,   0.0,    0.04],
    farfield               = [False, True,  True,   True,   True,  True,   True],
    center_wavelength_nm   = [1558.6, 1558.6, 1558.6, 1559.2, 1558.6, 1558.6, 1555.95],
    y_span_um              = [6.8,   6.8,   6.8,    6.8,    8.0,   6.8,    6.8],
    simulation_mode        = ["optimization"] * 6 + ["accurate"],
    mode  = "zipped",
    label = "tm_radiation_polarimetry",
)
assert all(len(getattr(SPEC, f)) == _n for f in (
    "cavity_width_nm", "cavity_neg_detuning_nm", "farfield",
    "center_wavelength_nm", "y_span_um", "simulation_mode"))


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
