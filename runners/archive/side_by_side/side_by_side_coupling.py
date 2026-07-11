"""
Study: two side-by-side (parallel) pi-shift Bragg cavities — radiative coupling.

Two parallel SiN pi-shift waveguides in one FDTD scene. Light is injected into
device 1 only (Port_1/Port_2); device 2 is passive and its ports (Port_3/Port_4)
read the power that couples across the gap. The question: does the cavity
radiation of one device couple/recycle into the other, and is there an optimal
SEPARATION (lateral gap g and/or longitudinal stagger Δx) that switches it on?

Theory context (paper_8): a side-by-side pair sits in the in-plane SIDEWAYS NULL,
so at Δx=0 the radiative coupling should be ~zero. Staggering device 2 along the
guide (Δx > 0) slides it into device 1's forward/backward radiation lobe (the
near-axial TM lobe, ~10° half-angle → reaching it needs Δx ≳ 5.7·g). BIC framing:
a periodic coupling/Q in the gap g indicates Fabry–Pérot-BIC physics (period
λ0/2n_c ≈ 0.544 µm); an isolated peak vs corrugation detuning indicates
Friedrich–Wintgen (see side_by_side_detune.py for the detuning knob).

This is the HEADLINE sweep: a 2D grid of lateral gap × longitudinal stagger,
both devices at the 500 nm baseline corrugation, for TE and TM.

  device_gap_nm      : {1000, 1500, 2000}            (lateral edge-to-edge, nm)
  device_stagger_nm  : {0, 500, 1000, 2000, 3000,    (longitudinal Δx, nm)
                        4000, 6000, 8000}
  polarization       : {TE, TM}
  => 3 × 8 × 2 = 48 cartesian points (one SLURM array task each).

Baseline (build_base): pitch 500 nm, N=80/side, constant n_core=1.97 / n_clad=1.444
(PROJECT DEFAULT as of 2026-06-27), corrugation 500 nm (BOTH devices — stronger
than the fabricated 300 nm so the shed radiation, and hence any coupling, rises
above numerical noise). y-symmetry OFF (single-guide drive). Ports-only monitors
(record_2d/3d/farfield = False) → ~3.5 GB per task. Scan is a wide 80 nm window at
1.54 µm to capture the (index-shifted) resonance(s); refine to a narrow window for
accurate Q at interesting points afterwards.

Saved per task (.mat): T, R, loss (device-1), coupling_left/right/total and
loss_4port (device-2 capture + device-1 pure radiation), plus the geometry
(device_gap_m, device_stagger_m, corrugation_depth_2_m, device_separation_m).

PREREQUISITE CHECK (do once before the production grid): run a single-device
config per polarization and confirm the resonance lands inside the 80 nm window;
re-center build_base().spectral.center_wavelength_m if not. Also run the
PML-standoff convergence (span_multiplier_override 1.8 → 2.7) at one small gap —
the subradiant tail is long and PML clipping biases the coupling/Q.

Run locally (sequential):
    python -m runners.side_by_side.side_by_side_coupling

Run on Athena as a parallel SLURM array (default partition):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_coupling
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


def build_base() -> SimulationConfig:
    """Baseline for the side-by-side coupling study — pinned explicitly so the
    Athena dispatcher's global tweaks are NOT inherited. 500 nm corrugation on
    device 1, constant indices 1.97/1.444, y-symmetry off, ports-only monitors."""
    cfg = SimulationConfig()
    cfg.grating.pitch_m = 500e-9
    cfg.grating.n_periods_each_side = 80
    cfg.grating.cavity_neg_detuning_nm = 0.0
    cfg.apodization.enabled = False

    # Stronger grating so radiation (and any inter-cavity coupling) is measurable.
    cfg.geometry.corrugation_depth_m = 500e-9        # device-1 baseline

    # Project-wide default constant indices (user, 2026-06-28: n_core 1.977 everywhere).
    cfg.material.use_constant_materials = True
    cfg.material.n_core_const = 1.977
    cfg.material.n_clad_const = 1.444

    cfg.mesh.simulation_mode = "optimization"        # dx=50 nm; accurate=35.7 nm for a convergence check

    # The side-by-side pair breaks the y-symmetry plane (only device 1 is driven).
    # The device forces this off internally for n_devices==2; set it here too for clarity.
    cfg.symmetry.use_y_symmetry = False

    # Wide window to capture the index-shifted resonance(s) of BOTH polarizations.
    # (n_core 1.97 vs the old 1.9963 red-shifts ~12 nm: TE ≈ 1558, TM ≈ 1512 nm.)
    cfg.spectral.center_wavelength_m = 1.540e-6
    cfg.spectral.scan_width_nm = 80.0
    cfg.spectral.n_wl_points = 8001                  # ~10 pm spacing (Q up to ~1.7e4)

    # Ports-only (cheap, ~3.5 GB/task). Turn fields/far-field on only for hero runs.
    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False
    return cfg


BASE = build_base()


SPEC = SweepSpec(
    n_devices              = [2],
    corrugation_depth_2_nm = [500],                  # device 2 = device 1 (identical) for the distance grid
    device_gap_nm          = [1000, 1500, 2000],
    device_stagger_nm      = [0, 500, 1000, 2000, 3000, 4000, 6000, 8000],
    polarization           = ["TE", "TM"],
    mode  = "cartesian",
    label = "side_by_side_coupling",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
