"""
TE-only redo of the side-by-side coupling sweep with CLEAN resonance detection.

Why this file exists
--------------------
The original side_by_side_coupling.py used a strong 500 nm corrugation and a wide
80 nm scan window. That smeared the Bragg stopband and made the specialized
resonance finder (find_bragg_resonance, sharpness x dip-depth) lock onto a
confusing / wrong peak — hence the "many confusing results". This runner keeps the
SAME geometry, the SAME open-device-2 ports, and the SAME resonance-finding
algorithm (NOT changed), but cleans up the measurement:

  - both devices back to the fabricated 300 nm corrugation (sharper, cleaner
    cavity resonance than 500 nm),
  - N = 80 periods/side (the standard device),
  - a NARROW 20 nm scan centered at 1555.5 nm (the TE resonance sits ~1555-1556 nm
    at n_core=1.977 with 300 nm corrugation), 2001 points (~10 pm spacing),
  - TE only.

Figure of merit
---------------
Device-1 input->output PEAK TRANSMISSION T (=|S21|^2) at the cavity resonance
(resonance_transmission, located by find_bragg_resonance). The deliverable is a 2D
map of that peak T over (Δy gap, Δx stagger). The Q factor and spectral FWHM are
also recorded per task (for the map / records), but transmission is the headline.

Grid (per user: more steps in Δy, fewer in Δx, capped at Δx=6 µm because the
results were flat from ~5 to 8 µm before):

  device_gap_nm      : {1000, 1200, 1400, 1600, 1800, 2000}    (6, lateral edge-to-edge)
  device_stagger_nm  : {0, 1000, 2000, 3000, 4000, 5000, 6000} (7, longitudinal Δx)
  polarization       : {TE}
  => 6 × 7 × 1 = 42 cartesian points (one SLURM array task each).

Output
------
RUN_NAME = module short name → results land in
results/side_by_side_te_300nm/results/ on Athena and download to
results_from_athena/side_by_side_te_300nm/. Per task .mat: T, R, loss (device-1),
resonance_transmission / resonance_wavelength_nm / spectral_fwhm_nm / Q,
coupling_left/right/total + loss_4port, plus geometry.

Run on Athena as a parallel SLURM array (default partition — independent tasks):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_te_300nm
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


def build_base() -> SimulationConfig:
    """Baseline for the clean TE side-by-side scan — pinned explicitly so the
    Athena dispatcher's global tweaks are NOT inherited. 300 nm corrugation
    (device 1), constant indices 1.977/1.444, y-symmetry off, ports-only
    monitors, narrow 20 nm window at 1555.5 nm."""
    cfg = SimulationConfig()
    cfg.grating.pitch_m = 500e-9
    cfg.grating.n_periods_each_side = 80
    cfg.grating.cavity_neg_detuning_nm = 0.0
    cfg.apodization.enabled = False

    # Fabricated-strength grating (300 nm) — sharper, cleaner cavity resonance
    # than the 500 nm baseline that produced the confusing results.
    cfg.geometry.corrugation_depth_m = 300e-9         # device-1 baseline

    # Project-wide default constant indices (n_core 1.97 as of 2026-06-28).
    cfg.material.use_constant_materials = True
    cfg.material.n_core_const = 1.97
    cfg.material.n_clad_const = 1.444

    cfg.mesh.simulation_mode = "optimization"         # dx=50 nm

    # Side-by-side pair breaks the y-symmetry plane (only device 1 is driven).
    # The device forces this off internally for n_devices==2; set here for clarity.
    cfg.symmetry.use_y_symmetry = False

    # NARROW window centered on the TE DEFECT resonance. A single-device-like
    # smoke (gap2000/Δx6000, task 41) at n_core=1.97 located the pi-shift defect
    # transmission peak at 1558.88 nm (T=0.716) — a clean single peak punching
    # through the stopband (prominence 0.70). (At the older n_core=1.977 it was
    # 1562.82 nm; the -0.007 index drop blue-shifts it ~3.9 nm.) NOTE: the Bragg
    # stopband CENTER (max reflection) is ~6 nm to the RED of the defect — the
    # window is centered on the DEFECT (1558.9 nm), not the stopband center.
    # 20 nm span -> 1548.9-1568.9 nm brackets the defect with ±10 nm margin;
    # 2001 points (~10 pm) resolves it for find_bragg_resonance (algorithm unchanged).
    cfg.spectral.center_wavelength_m = 1.5589e-6
    cfg.spectral.scan_width_nm = 20.0
    cfg.spectral.n_wl_points = 2001

    # Ports-only (cheap, ~3.5 GB/task). Fields/far-field off.
    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False
    return cfg


BASE = build_base()


SPEC = SweepSpec(
    n_devices              = [2],
    corrugation_depth_2_nm = [300],                   # device 2 = device 1 (identical, 300 nm)
    device_gap_nm          = [1000, 1200, 1400, 1600, 1800, 2000],
    device_stagger_nm      = [0, 1000, 2000, 3000, 4000, 5000, 6000],
    polarization           = ["TE"],
    mode  = "cartesian",
    label = "side_by_side_te_300nm",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
