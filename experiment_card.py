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


# Card field name → (dot-path on SimulationConfig, value transform or None)
_CARD_FIELD_MAP = {
    "n_periods_each_side":     ("grating.n_periods_each_side",        None),
    "center_mod_depth_nm":     ("apodization.center_mod_depth_nm",    None),
    "apod_method":             ("apodization.method",                 None),
    "tanh_steepness":          ("apodization.tanh_steepness",         None),
    "corrugation_depth_nm":    ("geometry.corrugation_depth_m",       lambda v: v * 1e-9),
    "pitch_nm":                ("grating.pitch_m",                    lambda v: v * 1e-9),
    "n_apod_periods_each_side": ("apodization.n_apod_periods_each_side", None),
    "cavity_length_nm":         ("grating.override_cavity_length_nm",   None),
    "center_wavelength_nm":     ("spectral.center_wavelength_m",        lambda v: v * 1e-9),
    "scan_width_nm":            ("spectral.scan_width_nm",              None),
}


@dataclass
class ExperimentCard:
    """Simplified experiment specification — only the knobs that vary between devices."""

    n_periods_each_side: int = 40
    center_mod_depth_nm: float = 10.0
    apod_method: str = 'none'
    tanh_steepness: float = 2.0
    corrugation_depth_nm: float = 200.0
    pitch_nm: float = 500.0
    cavity_length_nm: Optional[float] = None    # None = default (pitch / 2)
    n_apod_periods_each_side: Optional[int] = None
    center_wavelength_nm: Optional[float] = None  # None = use SimulationConfig default (1562.5 nm)
    scan_width_nm: Optional[float] = None         # None = use SimulationConfig default (200 nm)
    farfield: bool = False                        # Enable far-field monitors
    label: str = ''

    def to_config(self, base: Optional[SimulationConfig] = None) -> SimulationConfig:
        """
        Produce a full SimulationConfig by applying card values on top of defaults.

        If *base* is provided it is deep-copied first; otherwise a fresh
        SimulationConfig() with all defaults is used.
        """
        cfg = copy.deepcopy(base) if base else SimulationConfig()

        cfg.grating.n_periods_each_side = self.n_periods_each_side
        cfg.grating.pitch_m = self.pitch_nm * 1e-9
        cfg.geometry.corrugation_depth_m = self.corrugation_depth_nm * 1e-9
        cfg.apodization.center_mod_depth_nm = self.center_mod_depth_nm
        cfg.apodization.tanh_steepness = self.tanh_steepness

        if self.cavity_length_nm is not None:
            cfg.grating.override_cavity_length_nm = self.cavity_length_nm

        if self.n_apod_periods_each_side is not None:
            cfg.apodization.n_apod_periods_each_side = self.n_apod_periods_each_side

        if self.center_wavelength_nm is not None:
            cfg.spectral.center_wavelength_m = self.center_wavelength_nm * 1e-9

        if self.scan_width_nm is not None:
            cfg.spectral.scan_width_nm = self.scan_width_nm

        cfg.farfield.enabled = self.farfield

        if self.apod_method == 'none':
            cfg.apodization.enabled = False
        else:
            cfg.apodization.method = self.apod_method
            # Auto-enable apodization when center depth differs from full corrugation
            if self.center_mod_depth_nm < self.corrugation_depth_nm:
                cfg.apodization.enabled = True

        return cfg

    def to_sweep_config(
        self,
        sweep_field: str,
        values: list,
        base: Optional[SimulationConfig] = None,
    ) -> SimulationConfig:
        """
        Produce a SimulationConfig ready for run_sweep(), sweeping a card-level
        parameter.

        Example:
            cfg = card.to_sweep_config("n_periods_each_side", [30, 40, 50, 60])
        """
        cfg = self.to_config(base)

        if sweep_field not in _CARD_FIELD_MAP:
            raise ValueError(
                f"Unknown card field '{sweep_field}'. "
                f"Valid fields: {list(_CARD_FIELD_MAP)}"
            )

        dot_path, transform = _CARD_FIELD_MAP[sweep_field]
        if transform is not None:
            values = [transform(v) for v in values]

        cfg.sweep.parameter = dot_path
        cfg.sweep.values = values
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
    from run_simulation import run_single_sim

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
    from run_simulation import run_single_sim

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
