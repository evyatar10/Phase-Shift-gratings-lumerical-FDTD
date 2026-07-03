"""
TE z-box sanity ladder — was 1.8*lambda actually enough for TE, or just assumed?

The TM corr-400 device showed the vertical (z) boundary at 1.8*lambda sits inside
the mode's reactive near field and inflates the measured loss massively (T moved
+9 points by z=5.8 um). Theory says TE should be far less sensitive — its E lies
PARALLEL to the top/bottom faces (no normal-E discontinuity boost, weak vertical
tail) and its vertical radiation is cleanly propagating (PML absorbs it at any
distance). This ladder MEASURES that instead of assuming it, on the project's
anchored TE baseline (pitch 500 nm, corr 300 nm, h 350 nm, N=80, n 1.97/1.444,
resonance ~1556 nm at these indices).

Grid (zipped): y fixed at the TE default (3.76 um); z = 3.15 (the 1.8-lambda
default), 4.2, 5.8, 8.8 um via span_mult (y_span_um pins y, so span_mult moves
z alone) => 4 tasks. Converged = successive-step change in T at the 1e-3 scale.

NOTE (§7): convergence results are KEEP-FOREVER data.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt;
runs AFTER tm_span_convergence2 in the chain):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.te_span_z_check
Output -> results/te_span_z_check/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


def build_base() -> SimulationConfig:
    """Anchored TE baseline — pinned explicitly (no env reads)."""
    cfg = SimulationConfig()
    cfg.grating.pitch_m = 500e-9                      # TE pitch
    cfg.grating.n_periods_each_side = 80
    cfg.grating.cavity_neg_detuning_nm = 0.0
    cfg.apodization.enabled = False

    cfg.geometry.corrugation_depth_m = 300e-9         # TE baseline corrugation
    cfg.geometry.core_height_m = 350e-9

    cfg.material.use_constant_materials = True
    cfg.material.n_core_const = 1.97
    cfg.material.n_clad_const = 1.444

    cfg.mesh.simulation_mode = "optimization"
    cfg.source.polarization = "TE"

    # Same proven narrow window; TE resonance ~1556 nm at n_core=1.97.
    cfg.spectral.center_wavelength_m = 1.5585e-6
    cfg.spectral.scan_width_nm = 30.0
    cfg.spectral.n_wl_points = 3001

    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False
    return cfg


BASE = build_base()

# y pinned at the TE default (width_wide 0.95 um + 1.8*lambda = 3.76 um);
# span_mult then drives z alone: z = 0.35 + mult*1.5585 um.
_y_um  = [3.76, 3.76, 3.76, 3.76]
_zmult = [1.8,  2.47, 3.50, 5.42]        # z = 3.16, 4.20, 5.80, 8.80 um

assert len(set(zip(_y_um, _zmult))) == 4

SPEC = SweepSpec(
    y_span_um = _y_um,
    span_mult = _zmult,
    mode  = "zipped",
    label = "te_span_z_check",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
