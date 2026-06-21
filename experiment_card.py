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
    "pitch_nm":                 ("grating.pitch_m",                      lambda v: v * 1e-9),
    "n_apod_periods_each_side": ("apodization.n_apod_periods_each_side", None),
    "cavity_length_nm":         ("grating.override_cavity_length_nm",    None),
    "cavity_neg_detuning_nm":   ("grating.cavity_neg_detuning_nm",       None),
    "cavity_width_nm":          ("grating.cavity_width_m",               lambda v: v * 1e-9),
    "cavity_width_option":      ("grating.cavity_width_option",          None),
    "innermost_tooth_shift_nm": ("grating.innermost_tooth_shift_m",      lambda v: v * 1e-9),
    "lengthen_cavity":          ("grating.lengthen_cavity",              None),
    "center_wavelength_nm":     ("spectral.center_wavelength_m",         lambda v: v * 1e-9),
    "scan_width_nm":            ("spectral.scan_width_nm",               None),
    "farfield":                 ("farfield.enabled",                     None),
    "polarization":             ("source.polarization",                  None),  # "TE" | "TM"
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
