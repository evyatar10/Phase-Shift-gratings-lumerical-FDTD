"""Shared anchored-TM base config for the TM loss-program sweeps.

Extracted VERBATIM from tm_scatterer_scan.build_base on 2026-07-11 — that study
file had become the de-facto shared helper of ~36 sweeps (every consumer imported
build_base from runners.sweeps.tm_scatterer_scan), which blocked archiving the
closed scatterer study. This module is the honest home for it.

NOTE (historical quirk, kept for behavior compatibility): the returned config
has the scatterer pair ENABLED and y_span_override_m = 4.8 um — consumers that
are not scatterer studies must flip `cfg.scatterer.enabled = False` (and reset
the y-span) exactly as they always have. Do not "fix" this default: dozens of
dispatched studies depend on the current behavior.

Helper module only — no sweep is defined here (keep it that way so the deploy
menu grep never picks this file up).
"""

from simulation_config import SimulationConfig


def build_base() -> SimulationConfig:
    """Anchored TM device + proven narrow window — pinned explicitly (no env reads)
    so the Athena dispatcher's global tweaks are NOT inherited and the local and
    cluster expansions are identical. Mirrors side_by_side_tm_400nm.build_base,
    single-device, y-symmetry ON (the mirrored pair preserves it)."""
    cfg = SimulationConfig()
    cfg.grating.pitch_m = 516.83e-9                   # TM pitch (co-resonant with TE)
    cfg.grating.n_periods_each_side = 80
    cfg.grating.cavity_neg_detuning_nm = 0.0
    cfg.apodization.enabled = False

    cfg.geometry.corrugation_depth_m = 400e-9         # TE-mode-width-matched value
    cfg.geometry.core_height_m = 350e-9

    cfg.material.use_constant_materials = True        # const_material_mode default "object"
    cfg.material.n_core_const = 1.97
    cfg.material.n_clad_const = 1.444

    cfg.mesh.simulation_mode = "optimization"         # dx = 50 nm
    cfg.source.polarization = "TM"

    # NARROW window centered on the anchored TM defect resonance (1558.46 nm,
    # T=0.827, |FWHM|~1.31 nm). Excludes the ~1577 nm band-edge ripple. 3001
    # points (~10 pm) resolve the ~1.3 nm peak (>100 samples across the FWHM).
    cfg.spectral.center_wavelength_m = 1.5585e-6
    cfg.spectral.scan_width_nm = 30.0
    cfg.spectral.n_wl_points = 3001

    # Ports-only (cheap, ~3.5 GB/task). Fields/far-field OFF (MonitorConfig's
    # record_2d_fields defaults True — must be pinned off for the array).
    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled = False

    # Scatterer pair: enabled study-wide; per-task radius 0 = the control row.
    cfg.scatterer.enabled = True
    cfg.scatterer.mirrored_y = True                   # keeps the y=0 symmetry plane
    # Y-only domain widening: outermost cylinder edge 1.0+0.2 um -> >=1.2 um
    # clearance to the y PML (> lambda/n_clad = 1.08 um). z span stays default.
    cfg.y_span_override_m = 4.8e-6
    return cfg
