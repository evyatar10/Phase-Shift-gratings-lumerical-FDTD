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
    # Fab-stack check: silicon handle wafer under the BOX. None (default) = the
    # historical all-oxide model — scene unchanged. Set to the BOX thickness in
    # meters (core BOTTOM face -> Si top face; fab: 3.8e-6) to draw a
    # semi-infinite Si slab below. Requires symmetry.use_z_symmetry = False
    # (enforced in bragg_device — Si breaks the z=0 mirror plane).
    si_box_thickness_m: Optional[float] = None

    # ── Two side-by-side coupled devices (radiative-coupling study) ────────────
    # n_devices=2 builds a SECOND parallel pi-shift waveguide, offset laterally by
    # device_gap_m (edge-to-edge between the wide teeth) and longitudinally by
    # device_stagger_m (Δx of device 2's defect along the guide axis). Only the
    # first device is driven; the second is passive (ports 3/4 read the coupled
    # power). corrugation_depth_2_m gives the second device an independent grating
    # strength (None → equal to device 1). See bragg_device.PiShiftBraggFDTD.
    n_devices: int = 1                          # 1 (default) or 2 (side-by-side pair)
    device_gap_m: float = 1.0e-6                # Lateral edge-to-edge gap (wide-tooth edges) between the two guides
    device_stagger_m: float = 0.0               # Longitudinal Δx offset of device 2's defect along the guide
    corrugation_depth_2_m: Optional[float] = None  # Device-2 corrugation depth (None → equals device 1)
    # Device-2 AVERAGE corrugation width (None → equals device 1's avg_corrugation_width_m).
    # This is the Friedrich–Wintgen DETUNING knob: a different device-2 average width
    # shifts its n_eff → its π-shift defect resonance detunes from device 1's, so the
    # two side-coupled cavities can interfere at a controlled offset D (single-resonance
    # BIC route 4.1b). Default None keeps every existing two-device run bit-identical.
    avg_corrugation_width_2_m: Optional[float] = None
    device2_closed: bool = False                # Device 2 = CLOSED recycler (no feed waveguides / drain ports);
                                                # caught radiation re-radiates back toward device 1 instead of draining

    @property
    def width_wide_m(self) -> float:
        """Wide corrugation width (derived)."""
        return self.avg_corrugation_width_m + self.corrugation_depth_m / 2

    @property
    def width_narrow_m(self) -> float:
        """Narrow corrugation width (derived)."""
        return self.avg_corrugation_width_m - self.corrugation_depth_m / 2

    @property
    def _corrugation_depth_2(self) -> float:
        """Device-2 corrugation depth, falling back to device 1 when unset."""
        return self.corrugation_depth_2_m if self.corrugation_depth_2_m is not None else self.corrugation_depth_m

    @property
    def _avg_width_2(self) -> float:
        """Device-2 average corrugation width, falling back to device 1 when unset."""
        return (self.avg_corrugation_width_2_m if self.avg_corrugation_width_2_m is not None
                else self.avg_corrugation_width_m)

    @property
    def width_wide_2_m(self) -> float:
        """Wide corrugation width of device 2 (derived)."""
        return self._avg_width_2 + self._corrugation_depth_2 / 2

    @property
    def width_narrow_2_m(self) -> float:
        """Narrow corrugation width of device 2 (derived)."""
        return self._avg_width_2 - self._corrugation_depth_2 / 2


# ═══════════════════════════════════════════════════════════════════════════════
# Grating
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GratingConfig:
    """Bragg grating periodicity and cavity parameters."""
    pitch_m: float = 500e-9                     # Grating period
    n_periods_each_side: int = 80              # Number of periods on each side of the pi-shift cavity
    cavity_neg_detuning_nm: float = 0.0          # shortening from pitch/2 reference; 0 = no correction
    cavity_width_option: str = "avg"          # "narrow", "avg", or "avg_ext" (avg + extends into R_narrow_1)
    cavity_width_m: Optional[float] = None       # Numeric override for cavity-segment width; if None, falls back to cavity_width_option
    innermost_tooth_shift_m: float = 0.0         # X-shift of the tooth nearest the cavity (0 = no shift). Legacy single-tooth path; superseded by inner_shift_nm when that is set.
    lengthen_cavity: bool = True                  # When shift>0, lengthen cavity by 2*shift; else fix cavity length

    # ── Inverse-design freed inner teeth (mirror-symmetric per-side) ───────────
    # When inner_dw_nm or inner_shift_nm are set, the d ∈ [1, n_free_inner_teeth]
    # innermost teeth on each side use these per-tooth values instead of the
    # apodization envelope / scalar shift. Total grating length is preserved
    # by absorbing 2 * Σ(shifts) into the cavity (lengthen_cavity must be True).
    n_free_inner_teeth: int = 1                  # Number of innermost teeth per side with independent DW / shift
    inner_dw_nm: Optional[list] = None           # Per-tooth full corrugation depth in nm (W_wide - W_narrow). Length must equal n_free_inner_teeth. None → use apodization envelope.
    inner_shift_nm: Optional[list] = None        # Per-tooth gap shift in nm. Length must equal n_free_inner_teeth. None → use innermost_tooth_shift_m for tooth d=1, zero for d>=2.
    enforce_mirror_symmetry: bool = True         # Inverse-design optimizer: mirror DW and shift across left/right (always True at the bragg_device level — flagged here for future asymmetric runs).

    # ── Sidewall-corrugation phase offset (radiation null-steering study) ──────
    # Shifts the BOTTOM (-y) wall's teeth along x by (deg/360)*pitch relative to
    # the top wall. 0 = standard aligned width modulation (all existing devices).
    # The two walls' radiated fields then interfere with an extra phase φ at the
    # in-plane lobe angle: φ chosen so the walls cancel AT the measured lobe
    # (~14° for TM corr-400) while the guided-mode κ only drops as cos(φ/2)
    # (recoverable by deepening the corrugation). Breaks the y-mirror symmetry →
    # requires symmetry.use_y_symmetry = False (enforced in bragg_device).
    # Incompatible with apodization / per-tooth arrays / inner-tooth DW-shift /
    # n_devices=2 (guarded).
    wall_phase_offset_deg: float = 0.0

    # ── Corrugation profile shape (tooth-shape study) ───────────────────────────
    # "rect" (default — all existing devices) draws the historical rectangular
    # teeth. "sin"/"tri" draw a smooth sampled-polygon outline with the SAME
    # pitch and peak-to-peak depth semantics (corrugation_depth_m = W_wide −
    # W_narrow). Note the Bragg κ of a smooth profile is its FUNDAMENTAL Fourier
    # coefficient: to match the rect grating's κ (i.e. keep the spatial mode
    # width), scale depth by 4/π (sin) or π²/8 (tri). Symmetric in y (symmetry
    # BC stays usable). Uniform single-device gratings only (guarded).
    corrugation_profile: str = "rect"

    # ── Inner-tooth SHAPE (center-shape study) ──────────────────────────────────
    # Replaces the innermost n_shaped_inner_teeth teeth (per side, x-mirrored,
    # both walls, y-symmetric) with a shaped bump of the same footprint and
    # height: "rect" (default — unchanged device) | "ellipse" (smooth dome) |
    # "tri" (pointy) | "wedge_cav" (slanted face toward the cavity) |
    # "wedge_out" (slanted face away). Only the defect-adjacent teeth change —
    # the mirror body stays rectangular, so kappa / spatial mode width are
    # essentially untouched. Uniform single-device gratings only (guarded).
    inner_tooth_shape: str = "rect"
    n_shaped_inner_teeth: int = 1

    # ── Cavity-segment SHAPE (center-shape study) ───────────────────────────────
    # The pi-shift segment itself is the defect antenna; its sidewall profile is
    # an impedance-transition knob nobody has touched (always a plain rect).
    # "rect" (default — unchanged) | "barrel" (half-sine bulge, +depth at the
    # middle) | "hourglass" (half-sine pinch, -depth). Ends stay at the nominal
    # cavity width; length unchanged; y-symmetric (symmetry BCs stay usable).
    cavity_shape: str = "rect"
    cavity_shape_depth_nm: float = 150.0

    # ── Antisymmetric inner-tooth depth detuning (anti-radiator study) ──────────
    # Per-tooth delta (nm) applied ANTISYMMETRICALLY about the defect: left-arm
    # tooth d gets corrugation depth corr+delta[d-1], right-arm tooth d gets
    # corr-delta[d-1] (innermost first). An odd perturbation on the even cavity
    # mode radiates with a tunable phase (tooth index) and amplitude (delta) but
    # to FIRST ORDER does not shift the resonance or kappa (left/right cancel) —
    # a built-in destructive interferer aimed at the TM near-axial lobes.
    # None = standard symmetric device (all existing results unchanged).
    asym_inner_dw_delta_nm: Optional[list] = None

    # ── Explicit per-tooth width arrays (mirror-symmetric per-side) ────────────
    # When both are set, W_narrow[d]/W_wide[d] are taken verbatim for the innermost
    # d ∈ [1, len] teeth (index 0 = innermost). Teeth beyond the array length fall
    # back to the apodization envelope / uniform corrugation. Unlike inner_dw_nm
    # (depth only, around a fixed avg width), these let the average width vary
    # per tooth. Both arrays must have equal length ≤ n_periods_each_side.
    width_narrow_per_tooth_m: Optional[list] = None  # Per-tooth narrow width in m, innermost first.
    width_wide_per_tooth_m: Optional[list] = None    # Per-tooth wide width in m, innermost first.


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
    enabled: bool = False                        # Enable/disable apodization
    n_apod_periods_each_side: int = 5          # Number of tapered periods on each side was 10
    center_mod_depth_nm: float = 100.0           # Modulation depth at the cavity center (nm) was 4
    method: str = 'linear'                       # Apodization profile: 'linear' or 'tanh'
    tanh_steepness: float = 2.0                  # Steepness parameter for tanh profile


# ═══════════════════════════════════════════════════════════════════════════════
# Spectral
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralConfig:
    """Wavelength scan range and frequency resolution."""
    center_wavelength_m: float =  1.5601e-6      # Center of the wavelength scan
    # pervious was 16.0 nm, at 1.5625e-6
    scan_width_nm: float = 20.0                 # Total scan bandwidth (nm)
    n_wl_points: int = 3001                     # Number of wavelength points (S-parameters)
    n_2d_monitor_points: int = 51               # Frequency points for 2D/3D monitors


# ═══════════════════════════════════════════════════════════════════════════════
# Mesh & Domain
# ═══════════════════════════════════════════════════════════════════════════════

_MESH_MODE_CELLS = {
    "accurate":     7,   # pitch/(2*7) ≈ 35.7 nm dx (at pitch=500 nm) — final/accurate
    "optimization": 5,   # pitch/(2*5) = 50 nm dx   (at pitch=500 nm) — sweeps/optimization
}


@dataclass
class MeshConfig:
    """FDTD simulation domain sizing and mesh settings."""
    n_periods_dist_to_port: int = 20            # Distance from grating edge to port (in periods)
    n_wls_dist_port_to_pml: float = 5.0         # Distance from port to PML (in wavelengths)
    simulation_mode: str = "optimization"            # "accurate" (dx≈35nm) or "optimization" (dx=50nm)

    @property
    def cells_per_half_period(self) -> int:
        """Number of mesh cells across one half-period in X — controls dx
        as dx = pitch / (2 * cells_per_half_period). Period-based, so dx is
        independent of cavity length, tooth shift, or detuning."""
        if self.simulation_mode not in _MESH_MODE_CELLS:
            raise ValueError(
                f"Unknown simulation_mode '{self.simulation_mode}'. "
                f"Choose from: {list(_MESH_MODE_CELLS.keys())}"
            )
        return _MESH_MODE_CELLS[self.simulation_mode]


# ═══════════════════════════════════════════════════════════════════════════════
# Materials
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MaterialConfig:
    """Refractive index and material model settings."""
    use_constant_materials: bool = True         # Use frequency-independent n
    n_core_const: float = 1.97                  # Constant core index (Si3N4) — PROJECT-WIDE DEFAULT (user, 2026-06-28: 1.977 → 1.97)
    n_clad_const: float = 1.444                 # Constant cladding index (SiO2) — 2026-06-27 (was 1.44)
    n_eff_guess: float = 1.55                   # Estimated effective index for Bragg wavelength calc
    # Backend for constant-index mode (only used when use_constant_materials=True):
    #   "object"  — direct "<Object defined dielectric>" + set("index", n) per object
    #               (bypasses Lumerical's material fitting; a cleaner constant)
    #   "sampled" — flat 2-point "Sampled data" DB material (legacy)
    # Default is "object": validated equivalent to "sampled" on Athena (TM-study
    # geometry, TE+TM) to sub-fm resonance λ and ~1e-6 max|ΔT|/|ΔR| (2026-06-15).
    const_material_mode: str = "object"


# ═══════════════════════════════════════════════════════════════════════════════
# Source
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SourceConfig:
    """Injected mode settings.

    polarization selects the fundamental mode at the ports AND the matching
    symmetry-plane parity (see bragg_device._add_fdtd_region):
      "TE": E dominantly along y  -> y min Anti-Symmetric, z min Symmetric
      "TM": E dominantly along z  -> y min Symmetric,      z min Anti-Symmetric
    """
    polarization: str = "TE"                    # "TE" or "TM"


# ═══════════════════════════════════════════════════════════════════════════════
# Symmetry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymmetryConfig:
    """Boundary condition symmetry settings.

    The symmetric/anti-symmetric parity of each plane is chosen by
    SourceConfig.polarization; these flags only enable/disable the planes.
    """
    use_y_symmetry: bool = True                 # Use a symmetry plane at y=0
    use_z_symmetry: bool = True                 # Use a symmetry plane at z=0


# ═══════════════════════════════════════════════════════════════════════════════
# Scatterer (radiation-recycling study)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScattererConfig:
    """Small dielectric scatterer(s) placed beside (or inside) the guide in the
    horizontal plane — the lateral radiation-recycling study (paper_8 §7).

    Disabled unless enabled=True AND radius_m > 0 (radius 0 = clean in-study
    control with identical numerics). shape 'cylinder' is a vertical cylinder,
    z-centered on the core, full core height (same litho layer as the teeth).
    index None → n_core_const (SiN post in the oxide cladding); set n_clad_const
    for an oxide hole inside the core (mesh order lets the hole win overlaps).
    mirrored_y=True draws the pair at (x_m, ±y_m) — REQUIRED while the y=0
    symmetry plane is on: a single off-axis scatterer breaks that plane, and the
    TM 'Symmetric' BC would silently simulate the mirrored pair anyway."""
    enabled: bool = False
    shape: str = "cylinder"                     # 'cylinder' (addcircle) | 'rect' (addrect strip)
    radius_m: float = 150e-9                    # cylinder only
    # Rect strip only (lateral-reflector study): x_span = strip LENGTH along the
    # guide, y_span = strip WIDTH (the Fresnel/quarter-wave dimension). Rect is
    # active when enabled AND both spans > 0 (spans None/0 = in-study control).
    x_span_m: Optional[float] = None
    y_span_m: Optional[float] = None
    x_m: float = 0.0                            # center x (cavity defect = x 0)
    y_m: float = 1.0e-6                         # center y (+y side; pair mirrors to -y)
    index: Optional[float] = None               # None → material.n_core_const
    # Named Lumerical-database material for the scatterer object(s) ONLY — e.g.
    # "PEC (Perfect Electrical Conductor)" or "Al (Aluminium) - Palik" (metal-
    # mirror study). When set it overrides index/core material for the scatterer;
    # the rest of the scene keeps the configured material mode. Constant `index`
    # cannot express a metal (real scalar only) — this is the metal path.
    material: Optional[str] = None
    mirrored_y: bool = True
    height_m: Optional[float] = None            # None → core height (z-centered)
    # Multi-scatterer array (coherent-recycling study): when set, draws one
    # scatterer (or ±y pair) at EACH x in the list — same radius/index for all.
    # Overrides x_m. Positions in meters, e.g. the measured 540-nm winner comb.
    x_list_m: Optional[list] = None
    # Optional per-site y (same length as x_list_m) for arc / diagonal-ray
    # placements; None → all sites at y_m. Each site still mirrors to ±y_i.
    y_list_m: Optional[list] = None
    # Optional per-site in-plane rotation (deg about z; rect strips only) —
    # corner-array (sawtooth retro) study. -y mirror copies auto-negate.
    rot_list_deg: Optional[list] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Monitors
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MonitorConfig:
    """2D and 3D field monitor settings."""
    record_2d_fields: bool = True               # Record XY (top) and YZ (cross) field profiles
    field_2d_x_span_m: Optional[float] = None   # X span for 2D monitors (None = full device)
    monitor_2d_center_nm: Optional[float] = None  # 2D-monitor own wavelength center (None = use source limits)
    monitor_2d_span_nm: Optional[float] = None    # 2D-monitor own wavelength span (with center above)
    downsample_yz: int = 1                      # Spatial downsampling factor for monitors
    record_3d_fields: bool = False              # Record full 3D field volume
    field_3d_span_m: Optional[float] = None     # X span for 3D monitor (None = full device)


# ═══════════════════════════════════════════════════════════════════════════════
# Far-Field
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FarFieldConfig:
    """Far-field radiation monitor settings."""
    enabled: bool = False                        # Enable side and top far-field monitors
    farfield_x_span_m: float = 30e-6            # X extent of far-field monitors
    farfield_dist_wls: float = 0.8              # Monitor distance from PML edge (in wavelengths)
    ff_resolution: int = 201                    # Far-field ux/uy grid resolution
    save_nearfield: bool = True                 # Store full complex E on the monitor surfaces.
                                                # Pin False for arm-length monitors (~100s of MB
                                                # per row); the polarimetry reduction (scalars +
                                                # 1D x-profiles) is always saved regardless.
    save_complex: bool = False                  # Also store the complex far-field vectors
                                                # Ex_c/Ey_c/Ez_c per monitor (~1.9 MB each) —
                                                # needed by the response-matrix (Green's) study,
                                                # where phase carries the cancellation.


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
    source: SourceConfig = field(default_factory=SourceConfig)
    symmetry: SymmetryConfig = field(default_factory=SymmetryConfig)
    monitors: MonitorConfig = field(default_factory=MonitorConfig)
    farfield: FarFieldConfig = field(default_factory=FarFieldConfig)
    scatterer: ScattererConfig = field(default_factory=ScattererConfig)
    phase_correction: PhaseCorrectionConfig = field(default_factory=PhaseCorrectionConfig)
    run: RunConfig = field(default_factory=RunConfig)

    # --- Simple Bragg (used by run_simple_bragg.py only) ---
    simple_grating: SimpleGratingConfig = field(default_factory=SimpleGratingConfig)

    # Manual y/z domain-size override (× λ_center). None → default below (inert).
    # Used only for one-off domain-size checks via the SPAN_MULT env in the TM runners.
    span_multiplier_override: Optional[float] = None

    # Absolute Y-domain override in meters (single-device only; None → inert).
    # Unlike span_multiplier_override this widens Y ALONE (z stays at its default),
    # so off-axis scatterers can be given PML clearance without inflating the z
    # span/memory. Ignored for the two-device (n_devices==2) geometry.
    y_span_override_m: Optional[float] = None

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def _span_multiplier(self) -> float:
        """Span multiplier for y/z domain sizing: extended when farfield monitors are on."""
        if self.span_multiplier_override is not None:
            return self.span_multiplier_override
        return 5.0 if self.farfield.enabled else 1.8

    @property
    def y_span(self) -> float:
        """Simulation domain Y extent (derived from geometry + mesh + spectral).

        For the two-device side-by-side pair the domain must enclose BOTH guides
        (separated by the center-to-center distance s) plus their outer wide
        teeth, with a generous PML standoff on each side (the subradiant coupling
        tail is long — see span_multiplier_override for convergence runs).
        """
        g = self.geometry
        pad = self._span_multiplier * self.spectral.center_wavelength_m
        if g.n_devices == 2:
            s = g.device_gap_m + 0.5 * g.width_wide_m + 0.5 * g.width_wide_2_m
            # Absolute y-box override (FW-BIC study): decouples the transverse
            # radiation standoff from the z-standoff (span_multiplier), which for
            # TM must stay large (≈5.4λ) or vertical loss reads unphysically —
            # applying that multiplier to the y-pad too would inflate the pair
            # domain to ~20 µm. When set, it is used as the absolute enclosing y,
            # guarded so it cannot clip either guide's outer wide tooth + a 1.2 µm
            # PML clearance (> λ/n_clad). Unset → the historical pad-based box
            # (all existing two-device runs unchanged).
            if self.y_span_override_m is not None:
                min_enclose = s + 0.5 * g.width_wide_m + 0.5 * g.width_wide_2_m + 2.0 * 1.2e-6
                if self.y_span_override_m < min_enclose:
                    raise ValueError(
                        f"y_span_override_m={self.y_span_override_m*1e6:.2f} µm too small for the "
                        f"two-device pair: needs ≥ {min_enclose*1e6:.2f} µm to enclose both guides "
                        f"(center-to-center {s*1e6:.2f} µm) with 1.2 µm PML clearance."
                    )
                return self.y_span_override_m
            return s + 0.5 * g.width_wide_m + 0.5 * g.width_wide_2_m + 2.0 * pad
        if self.y_span_override_m is not None:
            return self.y_span_override_m
        return g.width_wide_m + pad

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

        # Convert detuning to the absolute override bragg_device expects
        pitch_half_nm = gr.pitch_m * 0.5e9
        cavity_override = (pitch_half_nm - gr.cavity_neg_detuning_nm) if gr.cavity_neg_detuning_nm != 0.0 else False

        return dict(
            pitch=gr.pitch_m,
            n_periods_each_side=gr.n_periods_each_side,
            n_apod_periods_each_side=ap.n_apod_periods_each_side if ap.enabled else None,
            width_narrow=g.width_narrow_m,
            width_wide=g.width_wide_m,
            width_port=g.width_port_m,
            core_height=g.core_height_m,
            substrate_thickness=g.substrate_thickness_m,
            si_box_thickness=g.si_box_thickness_m,
            # Two side-by-side coupled devices (n_devices=2)
            n_devices=g.n_devices,
            device_gap_m=g.device_gap_m,
            device_stagger_m=g.device_stagger_m,
            device2_closed=g.device2_closed,
            width_narrow_2=g.width_narrow_2_m,
            width_wide_2=g.width_wide_2_m,
            override_cavity_length_nm=cavity_override,
            cavity_width_option=gr.cavity_width_option,
            cavity_width_m=gr.cavity_width_m,
            innermost_tooth_shift_m=gr.innermost_tooth_shift_m,
            lengthen_cavity=gr.lengthen_cavity,
            n_free_inner_teeth=gr.n_free_inner_teeth,
            inner_dw_nm=gr.inner_dw_nm,
            inner_shift_nm=gr.inner_shift_nm,
            width_narrow_per_tooth_m=gr.width_narrow_per_tooth_m,
            width_wide_per_tooth_m=gr.width_wide_per_tooth_m,
            wall_phase_offset_deg=gr.wall_phase_offset_deg,
            corrugation_profile=gr.corrugation_profile,
            inner_tooth_shape=gr.inner_tooth_shape,
            n_shaped_inner_teeth=gr.n_shaped_inner_teeth,
            cavity_shape=gr.cavity_shape,
            cavity_shape_depth_m=gr.cavity_shape_depth_nm * 1e-9,
            asym_inner_dw_delta_nm=gr.asym_inner_dw_delta_nm,
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
            cells_per_half_period=me.cells_per_half_period,
            simulation_mode=me.simulation_mode,
            use_symmetry=sy.use_y_symmetry,
            use_z_symmetry=sy.use_z_symmetry,
            polarization=self.source.polarization,
            use_constant_materials=ma.use_constant_materials,
            n_core_const=ma.n_core_const,
            n_clad_const=ma.n_clad_const,
            const_material_mode=ma.const_material_mode,
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
            # Domain-size overrides active → file tag carries the box size
            # (keeps default-box filenames unchanged; prevents same-tag clobber
            # when sweeping the domain itself)
            domain_tag_active=(self.y_span_override_m is not None
                               or self.span_multiplier_override is not None),
            # Scatterer(s) (radiation-recycling study)
            scatterer_enabled=self.scatterer.enabled,
            scatterer_shape=self.scatterer.shape,
            scatterer_x_span_m=self.scatterer.x_span_m,
            scatterer_y_span_m=self.scatterer.y_span_m,
            scatterer_radius_m=self.scatterer.radius_m,
            scatterer_x_m=self.scatterer.x_m,
            scatterer_y_m=self.scatterer.y_m,
            scatterer_index=self.scatterer.index,
            scatterer_material=self.scatterer.material,
            scatterer_mirrored_y=self.scatterer.mirrored_y,
            scatterer_height_m=self.scatterer.height_m,
            scatterer_x_list_m=self.scatterer.x_list_m,
            scatterer_y_list_m=self.scatterer.y_list_m,
            scatterer_rot_list_deg=self.scatterer.rot_list_deg,
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
            cells_per_half_period=self.mesh.cells_per_half_period,
            simulation_mode=self.mesh.simulation_mode,
            use_symmetry=sy.use_y_symmetry,
            use_z_symmetry=sy.use_z_symmetry,
            use_constant_materials=ma.use_constant_materials,
            n_core_const=ma.n_core_const,
            n_clad_const=ma.n_clad_const,
            const_material_mode=ma.const_material_mode,
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
