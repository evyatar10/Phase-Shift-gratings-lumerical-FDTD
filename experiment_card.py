"""
Experiment card: simplified parameter interface for comparing against fabricated devices.

Specify only the parameters that vary between experiments; everything else
uses SimulationConfig defaults (or a custom base config you provide).

Usage:
    from experiment_card import ExperimentCard, run_card, run_cards

    card = ExperimentCard(n_periods_each_side=50, center_mod_depth_nm=15, label="Sample A")
    run_card(card)

    # Or compare multiple devices:
    run_cards([
        ExperimentCard(n_periods_each_side=40, center_mod_depth_nm=10, label="Dev1"),
        ExperimentCard(n_periods_each_side=60, center_mod_depth_nm=20, label="Dev2"),
    ])
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

from simulation_config import SimulationConfig


# Card field name → (dot-path on SimulationConfig, value transform or None).
# Single source of truth used by both ExperimentCard and SweepSpec.
_CARD_FIELD_MAP = {
    "n_periods_each_side":      ("grating.n_periods_each_side",          None),
    "center_mod_depth_nm":      ("apodization.center_mod_depth_nm",      None),
    "apod_method":              ("apodization.method",                   None),
    "tanh_steepness":           ("apodization.tanh_steepness",           None),
    "corrugation_depth_nm":     ("geometry.corrugation_depth_m",         lambda v: v * 1e-9),
    "avg_width_nm":             ("geometry.avg_corrugation_width_m",     lambda v: v * 1e-9),
    "pitch_nm":                 ("grating.pitch_m",                      lambda v: v * 1e-9),
    "n_apod_periods_each_side": ("apodization.n_apod_periods_each_side", None),
    "cavity_length_nm":         ("grating.override_cavity_length_nm",    None),
    "cavity_neg_detuning_nm":   ("grating.cavity_neg_detuning_nm",       None),
    "cavity_width_nm":          ("grating.cavity_width_m",               lambda v: None if v is None else v * 1e-9),  # None = use cavity_width_option
    "cavity_width_option":      ("grating.cavity_width_option",          None),
    "innermost_tooth_shift_nm": ("grating.innermost_tooth_shift_m",      lambda v: v * 1e-9),
    "wall_phase_deg":           ("grating.wall_phase_offset_deg",        None),  # bottom-wall tooth shift in deg of pitch (null-steering)
    "corr_profile":             ("grating.corrugation_profile",          None),  # "rect" | "sin" | "tri" tooth shape
    "inner_shape":              ("grating.inner_tooth_shape",            None),  # innermost-tooth shape (center-shape study)
    "n_shaped_teeth":           ("grating.n_shaped_inner_teeth",         None),  # how many innermost teeth per side get the shape
    "cavity_shape":             ("grating.cavity_shape",                 None),  # "rect" | "barrel" | "hourglass" pi-shift segment
    "cavity_shape_depth_nm":    ("grating.cavity_shape_depth_nm",        None),  # bulge/pinch depth (nm)
    "asym_dw_delta_nm":         ("grating.asym_inner_dw_delta_nm",       None),  # per-tooth ±delta list (anti-radiator); None = symmetric
    "simulation_mode":          ("mesh.simulation_mode",                 None),  # "optimization" (dx=50) | "accurate" (dx~35) per row
    # Explicit per-tooth width arrays (innermost first), nm — the width-envelope
    # ("graded island") study. None = uniform grating.
    "width_narrow_per_tooth_nm": ("grating.width_narrow_per_tooth_m",    lambda v: None if v is None else [float(x) * 1e-9 for x in v]),
    "width_wide_per_tooth_nm":   ("grating.width_wide_per_tooth_m",      lambda v: None if v is None else [float(x) * 1e-9 for x in v]),
    # Distributed pi-shift (theory study): per-tooth gap shifts in nm (innermost
    # first; negative = widen the gap, cavity shrinks via lengthen_cavity).
    "inner_shift_list_nm":       ("grating.inner_shift_nm",              None),
    "n_free_inner_teeth":        ("grating.n_free_inner_teeth",          None),
    "use_y_symmetry":           ("symmetry.use_y_symmetry",              None),  # False required for wall_phase rows (+ their controls)
    "lengthen_cavity":          ("grating.lengthen_cavity",              None),
    "center_wavelength_nm":     ("spectral.center_wavelength_m",         lambda v: v * 1e-9),
    "scan_width_nm":            ("spectral.scan_width_nm",               None),
    "farfield":                 ("farfield.enabled",                     None),
    "polarization":             ("source.polarization",                  None),  # "TE" | "TM"
    # Two side-by-side coupled devices (radiative-coupling study)
    "n_devices":                ("geometry.n_devices",                   None),  # 1 | 2
    "device_gap_nm":            ("geometry.device_gap_m",                lambda v: v * 1e-9),
    "device_stagger_nm":        ("geometry.device_stagger_m",            lambda v: v * 1e-9),
    "corrugation_depth_2_nm":   ("geometry.corrugation_depth_2_m",       lambda v: v * 1e-9),
    "avg_width_2_nm":           ("geometry.avg_corrugation_width_2_m",   lambda v: None if v is None else v * 1e-9),  # FW-BIC detuning knob (None → equals device 1)
    # Small dielectric scatterer(s) (radiation-recycling study). Drawn only when
    # scatterer.enabled=True in the base config AND radius > 0 (radius 0 = control).
    # Rect strips (lateral-reflector study): shape='rect' + both spans > 0.
    "scatterer_shape":          ("scatterer.shape",                      None),  # 'cylinder' | 'rect'
    "scatterer_x_span_um":      ("scatterer.x_span_m",                   lambda v: None if v is None else v * 1e-6),  # rect strip LENGTH along x
    "scatterer_y_span_nm":      ("scatterer.y_span_m",                   lambda v: None if v is None else v * 1e-9),  # rect strip WIDTH along y
    "scatterer_radius_nm":      ("scatterer.radius_m",                   lambda v: v * 1e-9),
    "scatterer_x_nm":           ("scatterer.x_m",                        lambda v: v * 1e-9),
    "scatterer_y_nm":           ("scatterer.y_m",                        lambda v: v * 1e-9),
    "scatterer_mirrored_y":     ("scatterer.mirrored_y",                 None),
    "scatterer_index":          ("scatterer.index",                      None),  # 1.97 pillar | 1.444 hole
    "scatterer_material":       ("scatterer.material",                   None),  # named DB material (PEC / metal); overrides index
    "scatterer_x_list_nm":      ("scatterer.x_list_m",                   lambda v: None if v is None else [float(x) * 1e-9 for x in v]),  # array of x centers
    "scatterer_y_list_nm":      ("scatterer.y_list_m",                   lambda v: None if v is None else [float(x) * 1e-9 for x in v]),  # per-site y (arc/diagonal)
    "scatterer_rot_list_deg":   ("scatterer.rot_list_deg",               None),  # per-site z-rotation (rect only; corner-array retro)
    "scatterer_height_nm":      ("scatterer.height_m",                   lambda v: None if v is None else v * 1e-9),  # z-height (None = core height); tall-variant diagnostics
    "auto_shutoff_min":         ("mesh.auto_shutoff_min",                None),  # solver early-termination threshold (None = builder 1e-7); convergence studies
    # Domain-size knobs (convergence studies). y_span_um sets the ABSOLUTE Y box
    # (single-device only); span_mult scales the default y/z multiplier — with
    # y_span_um also set, span_mult effectively controls Z alone.
    "monitor_2d_center_nm":     ("monitors.monitor_2d_center_nm",        None),
    "monitor_2d_span_nm":       ("monitors.monitor_2d_span_nm",          None),
    "y_span_um":                ("y_span_override_m",                    lambda v: v * 1e-6),
    "span_mult":                ("span_multiplier_override",             None),
}


@dataclass
class ExperimentCard:
    """
    Simplified experiment specification — only the knobs that vary between devices.

    All fields default to None, meaning "use SimulationConfig default".
    Only explicitly set fields override the base config.
    """

    n_periods_each_side: Optional[int] = None
    center_mod_depth_nm: Optional[float] = None
    apod_method: Optional[str] = None              # 'none', 'linear', 'tanh', or None (use config default)
    tanh_steepness: Optional[float] = None
    corrugation_depth_nm: Optional[float] = None
    pitch_nm: Optional[float] = None
    cavity_length_nm: Optional[float] = None       # None = default (pitch / 2). NOTE: maps to an attr to_device_kwargs ignores; use cavity_neg_detuning_nm to set cavity length.
    cavity_neg_detuning_nm: Optional[float] = None  # Cavity shortening from pitch/2 (nm). Working cavity-length control: cavity_length = pitch/2 − this.
    cavity_width_option: Optional[str] = None      # 'narrow' | 'avg' | 'avg_ext'
    cavity_width_nm: Optional[float] = None         # Numeric cavity-segment width override (nm); takes precedence over cavity_width_option.
    innermost_tooth_shift_nm: Optional[float] = None
    lengthen_cavity: Optional[bool] = None
    n_apod_periods_each_side: Optional[int] = None
    center_wavelength_nm: Optional[float] = None
    scan_width_nm: Optional[float] = None
    farfield: Optional[bool] = None
    # Explicit per-tooth width arrays in nm, ordered innermost → outermost (index 0 = tooth nearest cavity).
    width_narrow_per_tooth_nm: Optional[list] = None
    width_wide_per_tooth_nm: Optional[list] = None
    # Two side-by-side coupled devices (radiative-coupling study)
    n_devices: Optional[int] = None                # 1 (default) | 2 (side-by-side pair)
    device_gap_nm: Optional[float] = None          # lateral edge-to-edge gap between the two guides
    device_stagger_nm: Optional[float] = None      # longitudinal Δx offset of device 2
    corrugation_depth_2_nm: Optional[float] = None  # device-2 corrugation depth (None → equals device 1)
    avg_width_2_nm: Optional[float] = None          # device-2 average corrugation width (FW-BIC detuning; None → equals device 1)
    label: str = ''

    def to_config(self, base: Optional[SimulationConfig] = None) -> SimulationConfig:
        """
        Produce a full SimulationConfig by applying card values on top of defaults.

        Only fields explicitly set (non-None) override the base config.
        If *base* is provided it is deep-copied first; otherwise a fresh
        SimulationConfig() with all defaults is used.
        """
        cfg = copy.deepcopy(base) if base else SimulationConfig()

        if self.n_periods_each_side is not None:
            cfg.grating.n_periods_each_side = self.n_periods_each_side
        if self.pitch_nm is not None:
            cfg.grating.pitch_m = self.pitch_nm * 1e-9
        if self.corrugation_depth_nm is not None:
            cfg.geometry.corrugation_depth_m = self.corrugation_depth_nm * 1e-9
        if self.center_mod_depth_nm is not None:
            cfg.apodization.center_mod_depth_nm = self.center_mod_depth_nm
        if self.tanh_steepness is not None:
            cfg.apodization.tanh_steepness = self.tanh_steepness
        if self.cavity_length_nm is not None:
            cfg.grating.override_cavity_length_nm = self.cavity_length_nm
        if self.cavity_neg_detuning_nm is not None:
            cfg.grating.cavity_neg_detuning_nm = self.cavity_neg_detuning_nm
        if self.cavity_width_option is not None:
            cfg.grating.cavity_width_option = self.cavity_width_option
        if self.cavity_width_nm is not None:
            cfg.grating.cavity_width_m = self.cavity_width_nm * 1e-9
        if self.width_narrow_per_tooth_nm is not None:
            cfg.grating.width_narrow_per_tooth_m = [w * 1e-9 for w in self.width_narrow_per_tooth_nm]
        if self.width_wide_per_tooth_nm is not None:
            cfg.grating.width_wide_per_tooth_m = [w * 1e-9 for w in self.width_wide_per_tooth_nm]
        if self.innermost_tooth_shift_nm is not None:
            cfg.grating.innermost_tooth_shift_m = self.innermost_tooth_shift_nm * 1e-9
        if self.lengthen_cavity is not None:
            cfg.grating.lengthen_cavity = self.lengthen_cavity
        if self.n_apod_periods_each_side is not None:
            cfg.apodization.n_apod_periods_each_side = self.n_apod_periods_each_side
        if self.center_wavelength_nm is not None:
            cfg.spectral.center_wavelength_m = self.center_wavelength_nm * 1e-9
        if self.scan_width_nm is not None:
            cfg.spectral.scan_width_nm = self.scan_width_nm
        if self.farfield is not None:
            cfg.farfield.enabled = self.farfield
        if self.n_devices is not None:
            cfg.geometry.n_devices = self.n_devices
        if self.device_gap_nm is not None:
            cfg.geometry.device_gap_m = self.device_gap_nm * 1e-9
        if self.device_stagger_nm is not None:
            cfg.geometry.device_stagger_m = self.device_stagger_nm * 1e-9
        if self.corrugation_depth_2_nm is not None:
            cfg.geometry.corrugation_depth_2_m = self.corrugation_depth_2_nm * 1e-9
        if self.avg_width_2_nm is not None:
            cfg.geometry.avg_corrugation_width_2_m = self.avg_width_2_nm * 1e-9

        # Apodization enable/disable logic
        if self.apod_method is not None:
            if self.apod_method == 'none':
                cfg.apodization.enabled = False
            else:
                cfg.apodization.method = self.apod_method
                if (self.center_mod_depth_nm is not None
                        and self.center_mod_depth_nm < cfg.geometry.corrugation_depth_m * 1e9):
                    cfg.apodization.enabled = True

        # Mirror SweepSpec.expand invariant: n_apod_periods_each_side=0 disables apod.
        if cfg.apodization.n_apod_periods_each_side == 0:
            cfg.apodization.enabled = False

        return cfg

# ═══════════════════════════════════════════════════════════════════════════════
# Convenience runners
# ═══════════════════════════════════════════════════════════════════════════════

def _print_card_results(results: dict) -> None:
    """Print the key spectral results from a completed experiment card."""
    res_nm  = results.get('resonance_wavelength_nm')
    fwhm_nm = results.get('spectral_fwhm_nm')
    if res_nm is not None:
        print(f"  Resonance : {res_nm:.3f} nm")
    if fwhm_nm is not None:
        print(f"  Δλ (FWHM) : {fwhm_nm:.4f} nm")


def run_card(card: ExperimentCard, base: Optional[SimulationConfig] = None) -> dict:
    """Run a single simulation from an experiment card."""
    from runners.single.run_simulation import run_single_sim

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {card.label or card}")
    print(f"{'=' * 60}\n")
    results = run_single_sim(card.to_config(base))
    _print_card_results(results)
    return results


def run_cards(
    cards: list[ExperimentCard],
    base: Optional[SimulationConfig] = None,
) -> list[dict]:
    """Run one simulation per card, comparing multiple experiments."""
    import gc
    import matplotlib.pyplot as plt
    from runners.single.run_simulation import run_single_sim

    results = []
    for i, card in enumerate(cards):
        print(f"\n>>> CARD {i + 1}/{len(cards)}: {card.label or card} <<<\n")
        try:
            r = run_single_sim(card.to_config(base))
            _print_card_results(r)
            results.append(r)
        except Exception as e:
            print(f"ERROR in card {i + 1}: {e}")
            raise
        finally:
            plt.close("all")
            gc.collect()

    print(f"\n{'=' * 60}")
    print(f"ALL CARDS COMPLETE: {len(results)}/{len(cards)} succeeded.")
    print(f"{'=' * 60}")
    return results
