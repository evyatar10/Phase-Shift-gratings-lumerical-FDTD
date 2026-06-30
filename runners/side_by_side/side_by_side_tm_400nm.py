"""
TM side-by-side coupling sweep — gap (Δy) × stagger (Δx), 400 nm corrugation.

Why this file exists
--------------------
This is the TM counterpart of side_by_side_te_300nm.py. We finished the TE
side-by-side coupled-cavity study (two parallel pi-shift Bragg cavities in one FDTD
scene; light driven into device 1 only; device 2 passive, its ports read the
coupled power). Now we replicate the SAME study for TM, using the current TM
device, and keeping the SAME open-device-2 ports, the SAME metric extraction, and
the SAME resonance-finding algorithm (find_bragg_resonance, sharpness x dip-depth) —
nothing in the shared pipeline changes; only the device + scan window are TM-tuned.

TM device
---------
  - pitch 516.83 nm, corrugation 400 nm (BOTH devices), N = 80 periods/side,
  - constant indices n_core 1.97 / n_clad 1.444,
  - core height left at the project default (350 nm) — NOT pinned here,
  - a NARROW 30 nm scan centered at 1558.5 nm. The single-device TM defect resonance
    at this geometry is 1558.46 nm (T=0.827, |FWHM|~1.31 nm, Q~1194), co-resonant
    with the TE line by design. The 30 nm window [1543.5, 1573.5] nm brackets it with
    +-15 nm margin for any two-cavity supermode splitting, and EXCLUDES the band-edge
    passband ripple at ~1577 nm (which otherwise lures find_bragg_resonance onto the
    wrong peak — the same reason the TE redo used a narrow window). 3001 points
    (~10 pm) resolves the ~1.3 nm peak.

Figure of merit
---------------
Device-1 input->output PEAK TRANSMISSION T (=|S21|^2) at the cavity resonance
(resonance_transmission, located by find_bragg_resonance). The deliverable is a 2D
map of that peak T over (Δy gap, Δx stagger). Q / spectral FWHM, the device-2
coupling (coupling_left/right/total, loss_4port), and the full T(λ)/coupling spectra
(for supermode splitting) are also recorded per task.

Grid (mirrors side_by_side_te_300nm.py for a direct TE/TM comparison):

  device_gap_nm      : {1000, 1200, 1400, 1600, 1800, 2000}    (6, lateral edge-to-edge)
  device_stagger_nm  : {0, 1000, 2000, 3000, 4000, 5000, 6000} (7, longitudinal Δx)
  polarization       : {TM}
  => 6 × 7 × 1 = 42 cartesian points (one SLURM array task each).

Output
------
RUN_NAME = module short name → results land in
results/side_by_side_tm_400nm/results/ on Athena and download to
results_from_athena/side_by_side_tm_400nm/. Per task .mat: T, R, loss (device-1),
resonance_transmission / resonance_wavelength_nm / spectral_fwhm_nm / Q,
coupling_left/right/total + loss_4port, plus geometry.

Run on Athena as a parallel SLURM array (default partition — independent tasks):
    bash athena/deploy_athena.sh --option3 --spec=runners.side_by_side.side_by_side_tm_400nm
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from simulation_config import SimulationConfig


def build_base() -> SimulationConfig:
    """Baseline for the TM side-by-side scan — pinned explicitly so the Athena
    dispatcher's global tweaks are NOT inherited. 400 nm corrugation (device 1),
    pitch 516.83 nm, constant indices 1.97/1.444, y-symmetry off, ports-only
    monitors, narrow 30 nm window at 1558.5 nm. Core height is left at the project
    default (350 nm) — intentionally not overridden here."""
    cfg = SimulationConfig()
    cfg.grating.pitch_m = 516.83e-9                   # TM pitch (co-resonant with TE)
    cfg.grating.n_periods_each_side = 80
    cfg.grating.cavity_neg_detuning_nm = 0.0
    cfg.apodization.enabled = False

    # 400 nm corrugation (device-1 baseline; device 2 set identical via SPEC).
    cfg.geometry.corrugation_depth_m = 400e-9

    # Project-wide default constant indices (n_core 1.97 as of 2026-06-28).
    cfg.material.use_constant_materials = True
    cfg.material.n_core_const = 1.97
    cfg.material.n_clad_const = 1.444

    cfg.mesh.simulation_mode = "optimization"         # dx=50 nm

    # Side-by-side pair breaks the y-symmetry plane (only device 1 is driven).
    # The device forces this off internally for n_devices==2; set here for clarity.
    cfg.symmetry.use_y_symmetry = False

    # NARROW window centered on the TM DEFECT resonance. The single-device TM device
    # at pitch 516.83 / corr 400 / height 350 resonates at 1558.46 nm (T=0.827,
    # |FWHM|~1.31 nm). 30 nm span -> [1543.5, 1573.5] nm brackets the defect with
    # +-15 nm margin (room for supermode splitting) and excludes the ~1577 nm band-
    # edge ripple. 3001 points (~10 pm) for find_bragg_resonance (algorithm unchanged).
    cfg.spectral.center_wavelength_m = 1.5585e-6
    cfg.spectral.scan_width_nm = 30.0
    cfg.spectral.n_wl_points = 3001

    # Ports-only (cheap, ~3.5 GB/task). Fields/far-field off.
    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False
    return cfg


BASE = build_base()


SPEC = SweepSpec(
    n_devices              = [2],
    corrugation_depth_2_nm = [400],                   # device 2 = device 1 (identical, 400 nm)
    device_gap_nm          = [1000, 1200, 1400, 1600, 1800, 2000],
    device_stagger_nm      = [0, 1000, 2000, 3000, 4000, 5000, 6000],
    polarization           = ["TM"],
    mode  = "cartesian",
    label = "side_by_side_tm_400nm",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
