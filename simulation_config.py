"""
Centralized simulation configuration for Phase-Shift Bragg Grating FDTD.

All user-editable simulation parameters are defined here as dataclasses.
Machine-specific paths remain in config.py.

Usage:
    from simulation_config import SimulationConfig
    cfg = SimulationConfig()
    cfg.grating.n_periods_each_side = 120
    cfg.apodization.enabled = False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeometryConfig:
    """Waveguide cross-section and corrugation geometry."""
    avg_corrugation_width_m: float = 800e-9     # Average width of corrugated section
    corrugation_depth_m: float = 300e-9         # Full width difference (wide - narrow)
    core_height_m: float = 350e-9               # Si3N4 core thickness
    width_port_m: float = 1000e-9               # Access waveguide width at ports
    substrate_thickness_m: float = 10e-6        # SiO2 substrate thickness

    @property
    def width_wide_m(self) -> float:
        """Wide corrugation width (derived)."""
        return self.avg_corrugation_width_m + self.corrugation_depth_m / 2

    @property
    def width_narrow_m(self) -> float:
        """Narrow corrugation width (derived)."""
        return self.avg_corrugation_width_m - self.corrugation_depth_m / 2


# ═══════════════════════════════════════════════════════════════════════════════
# Grating
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GratingConfig:
    """Bragg grating periodicity and cavity parameters."""
    pitch_m: float = 500e-9                     # Grating period
    n_periods_each_side: int = 80              # Number of periods on each side of the pi-shift cavity
    override_cavity_length_nm: Optional[float] = None  # None or False = default (pitch/2)


# ═══════════════════════════════════════════════════════════════════════════════
# Apodization
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ApodizationConfig:
    """
    Apodization settings (width modulation tapering).

    Three profile methods are supported:
    - 'none': no apodization — uniform grating with constant corrugation depth.
    - 'linear': modulation depth varies linearly from center_mod_depth_nm
      at the cavity to the full corrugation_depth at the grating edges.
    - 'tanh': modulation depth follows tanh(a * 2 * frac) / tanh(2 * a),
      where 'a' is tanh_steepness. This concentrates the transition
      near the grating edge, giving a flatter center region.

    The transition occurs over n_apod_periods_each_side periods.
    """
    enabled: bool = True                        # Enable/disable apodization
    n_apod_periods_each_side: int = 10          # Number of tapered periods on each side
    center_mod_depth_nm: float = 4.0           # Modulation depth at the cavity center (nm)
    method: str = 'linear'                       # Apodization profile: 'linear' or 'tanh'
    tanh_steepness: float = 2.0                  # Steepness parameter for tanh profile


# ═══════════════════════════════════════════════════════════════════════════════
# Spectral
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralConfig:
    """Wavelength scan range and frequency resolution."""
    center_wavelength_m: float = 1.5625e-6      # Center of the wavelength scan
    # pervious was 16.0 nm, at 1.5625e-6  
    scan_width_nm: float = 200.0                 # Total scan bandwidth (nm)
    n_wl_points: int = 3001                     # Number of wavelength points (S-parameters)
    n_2d_monitor_points: int = 51               # Frequency points for 2D/3D monitors


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh & Domain
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MeshConfig:
    """FDTD simulation domain sizing and mesh settings."""
    n_periods_dist_to_port: int = 20            # Distance from grating edge to port (in periods)
    n_wls_dist_port_to_pml: float = 5.0         # Distance from port to PML (in wavelengths)
    use_cavity_mesh_override: bool = True        # Extra mesh refinement at the cavity


# ═══════════════════════════════════════════════════════════════════════════════
# Materials
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialConfig:
    """Refractive index and material model settings."""
    use_constant_materials: bool = True         # Use frequency-independent n
    n_core_const: float = 1.977                 # Constant core index (Si3N4)
    n_clad_const: float = 1.44                  # Constant cladding index (SiO2)
    n_eff_guess: float = 1.55                   # Estimated effective index for Bragg wavelength calc


# ═══════════════════════════════════════════════════════════════════════════════
# Symmetry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymmetryConfig:
    """Boundary condition symmetry settings."""
    use_y_symmetry: bool = True                 # Anti-symmetric BC in Y (TE mode)
    use_z_symmetry: bool = True                 # Symmetric BC in Z


# ═══════════════════════════════════════════════════════════════════════════════
# Monitors
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MonitorConfig:
    """2D and 3D field monitor settings."""
    record_2d_fields: bool = True               # Record XY (top) and YZ (cross) field profiles
    field_2d_x_span_m: Optional[float] = None   # X span for 2D monitors (None = full device)
    downsample_yz: int = 1                      # Spatial downsampling factor for monitors
    record_3d_fields: bool = False              # Record full 3D field volume
    field_3d_span_m: Optional[float] = None     # X span for 3D monitor (None = full device)


# ═══════════════════════════════════════════════════════════════════════════════
# Far-Field
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FarFieldConfig:
    """Far-field radiation monitor settings."""
    enabled: bool = True                        # Enable side and top far-field monitors
    farfield_x_span_m: float = 30e-6            # X extent of far-field monitors
    farfield_dist_wls: float = 0.8              # Monitor distance from PML edge (in wavelengths)
    ff_resolution: int = 201                    # Far-field ux/uy grid resolution


# ═══════════════════════════════════════════════════════════════════════════════
# Phase Correction
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseCorrectionConfig:
    """
    S-parameter phase correction settings.

    Three correction stages (see analysis.py):
      A. Feed-line de-embedding  (do_length_correction)
         Removes propagation phase in straight waveguide feed sections.
      B. Bragg carrier phase removal  (do_envelope_correction)
         Removes the expected beta_0 = pi/pitch phase slope.
      C. Phase alignment to -90 deg  (do_envelope_correction)
         Rotates S21 phase so resonance sits at -pi/2.
    """
    do_length_correction: bool = True           # Stage A: de-embed feed waveguides
    do_envelope_correction: bool = True         # Stages B+C: slope removal + phase alignment


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SweepConfig:
    """
    Parameter sweep settings.

    The sweep runs the core simulation pipeline once per value, overriding
    the parameter specified by 'parameter' on each iteration.

    Parameter uses dot notation to target nested config fields:
      "grating.n_periods_each_side"
      "apodization.center_mod_depth_nm"
      "apodization.n_apod_periods_each_side"
    """
    parameter: str = "grating.n_periods_each_side"  # Dot-path to the parameter to sweep
    values: list = field(default_factory=lambda: [100])  # Values to iterate over


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunConfig:
    """Runtime behavior flags."""
    cleanup_lumerical_data: bool = False         # Delete .h5 files after run to save disk
    export_interconnect: bool = False             # Write INTERCONNECT .txt alongside the .mat


# ═══════════════════════════════════════════════════════════════════════════════
# Simple Bragg Grating (no cavity, no apodization)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimpleGratingConfig:
    """
    Additional config for SimpleBraggFDTD (uniform grating, no pi-shift cavity).

    Shares geometry, spectral, material, symmetry, and phase correction settings
    with the main SimulationConfig. Only grating-specific differences go here.
    """
    n_periods_total: int = 40                   # Total number of periods (not per-side)
    n_periods_dist_to_port: int = 30            # Distance from grating to port (in periods)
    n_wls_dist_port_to_pml: float = 5.0         # Distance from port to PML (in wavelengths)


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """
    Top-level simulation configuration.

    Groups all parameter categories. Create an instance and modify fields
    before passing to the simulation pipeline:

        cfg = SimulationConfig()
        cfg.grating.n_periods_each_side = 120
        cfg.apodization.enabled = False
        run_single_sim(cfg)
    """
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    grating: GratingConfig = field(default_factory=GratingConfig)
    apodization: ApodizationConfig = field(default_factory=ApodizationConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    symmetry: SymmetryConfig = field(default_factory=SymmetryConfig)
    monitors: MonitorConfig = field(default_factory=MonitorConfig)
    farfield: FarFieldConfig = field(default_factory=FarFieldConfig)
    phase_correction: PhaseCorrectionConfig = field(default_factory=PhaseCorrectionConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # --- Simple Bragg (used by run_simple_bragg.py only) ---
    simple_grating: SimpleGratingConfig = field(default_factory=SimpleGratingConfig)

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def _span_multiplier(self) -> float:
        """Span multiplier for y/z domain sizing: extended when farfield monitors are on."""
        return 5.0 if self.farfield.enabled else 1.8

    @property
    def y_span(self) -> float:
        """Simulation domain Y extent (derived from geometry + mesh + spectral)."""
        return self.geometry.width_wide_m + self._span_multiplier * self.spectral.center_wavelength_m

    @property
    def z_span(self) -> float:
        """Simulation domain Z extent (derived from geometry + mesh + spectral)."""
        return self.geometry.core_height_m + self._span_multiplier * self.spectral.center_wavelength_m

    # ── Mapping to PiShiftBraggFDTD constructor ───────────────────────────

    def to_device_kwargs(self) -> dict:
        """
        Build the keyword arguments dict for PiShiftBraggFDTD.__init__().

        This is the bridge between the config and the device class.
        bragg_device.py needs no constructor changes.
        """
        import config

        g = self.geometry
        gr = self.grating
        ap = self.apodization
        sp = self.spectral
        me = self.mesh
        ma = self.material
        sy = self.symmetry
        mo = self.monitors
        ff = self.farfield

        lam = sp.center_wavelength_m

        # Far-field monitor placement: offset from simulation boundary
        calc_farfield_y = (self.y_span / 2.0) - (ff.farfield_dist_wls * lam)
        calc_farfield_z = (self.z_span / 2.0) - (ff.farfield_dist_wls * lam)

        # Monitor spans: slightly smaller than simulation domain
        monitor_y_span = self.y_span - 0.5 * lam
        monitor_z_span = self.z_span - 0.5 * lam

        # Cavity length override: convert None to the value bragg_device expects
        cavity_override = gr.override_cavity_length_nm
        if cavity_override is None:
            cavity_override = False  # bragg_device uses False for "use default pitch/2"

        return dict(
            pitch=gr.pitch_m,
            n_periods_each_side=gr.n_periods_each_side,
            n_apod_periods_each_side=ap.n_apod_periods_each_side if ap.enabled else None,
            width_narrow=g.width_narrow_m,
            width_wide=g.width_wide_m,
            width_port=g.width_port_m,
            core_height=g.core_height_m,
            substrate_thickness=g.substrate_thickness_m,
            override_cavity_length_nm=cavity_override,
            y_span=self.y_span,
            z_span=self.z_span,
            material_db_path=config.MATERIAL_DB_PATH,
            n_periods_dist_to_port=me.n_periods_dist_to_port,
            n_wls_dist_port_to_pml=me.n_wls_dist_port_to_pml,
            n_eff_guess=ma.n_eff_guess,
            n_wl_points=sp.n_wl_points,
            use_apodization=ap.enabled,
            center_mod_depth_nm=ap.center_mod_depth_nm,
            apod_method=ap.method if ap.enabled else 'linear',
            tanh_steepness=ap.tanh_steepness,
            use_cavity_mesh_override=me.use_cavity_mesh_override,
            use_symmetry=sy.use_y_symmetry,
            use_z_symmetry=sy.use_z_symmetry,
            use_constant_materials=ma.use_constant_materials,
            n_core_const=ma.n_core_const,
            n_clad_const=ma.n_clad_const,
            # 2D monitors
            record_2d_fields_top_and_cross=mo.record_2d_fields,
            field_2d_x_span_m=mo.field_2d_x_span_m,
            monitor_y_span_m=monitor_y_span,
            monitor_z_span_m=monitor_z_span,
            downsample_yz=mo.downsample_yz,
            # 3D monitors
            record_3d_fields=mo.record_3d_fields,
            field_3d_span_m=mo.field_3d_span_m,
            # Far-field
            record_farfield=ff.enabled,
            farfield_x_span_m=ff.farfield_x_span_m,
            farfield_y_dist_m=calc_farfield_y,
            farfield_z_dist_m=calc_farfield_z,
        )

    def to_simple_device_kwargs(self) -> dict:
        """
        Build the keyword arguments dict for SimpleBraggFDTD.__init__().
        Used by run_simple_bragg.py.
        """
        import config

        g = self.geometry
        sg = self.simple_grating
        sp = self.spectral
        ma = self.material
        sy = self.symmetry

        return dict(
            pitch=self.grating.pitch_m,
            n_periods=sg.n_periods_total,
            width_narrow=g.width_narrow_m,
            width_wide=g.width_wide_m,
            width_port=g.width_port_m,
            core_height=g.core_height_m,
            substrate_thickness=g.substrate_thickness_m,
            y_span=self.y_span,
            z_span=self.z_span,
            material_db_path=config.MATERIAL_DB_PATH,
            n_periods_dist_to_port=sg.n_periods_dist_to_port,
            n_wls_dist_port_to_pml=sg.n_wls_dist_port_to_pml,
            n_eff_guess=ma.n_eff_guess,
            n_wl_points=sp.n_wl_points,
            use_symmetry=sy.use_y_symmetry,
            use_z_symmetry=sy.use_z_symmetry,
            use_constant_materials=ma.use_constant_materials,
            n_core_const=ma.n_core_const,
            n_clad_const=ma.n_clad_const,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def set_nested_attr(cfg: SimulationConfig, dot_path: str, value) -> None:
    """
    Set a nested config attribute using dot notation.

    Example:
        set_nested_attr(cfg, "grating.n_periods_each_side", 120)
    """
    parts = dot_path.split(".")
    obj = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
