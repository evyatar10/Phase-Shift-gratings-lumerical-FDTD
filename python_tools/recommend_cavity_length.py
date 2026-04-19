"""
Step 1: Run FDE in Lumerical MODE for width_narrow and width_avg.
Step 2: Interpolate n_eff at the cavity wavelength.
Step 3: Print recommended cavity length for each cavity_width_option.

Usage:
    python python_tools/recommend_cavity_length.py
"""

import os
import sys
import numpy as np

# --- allow imports from project root ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulation_config import SimulationConfig
from sim_helpers import suggest_cavity_length
from python_tools.calc_neff_vs_wl import NeffSweeper

import config as cfg_paths


def get_neff_at_wavelength(sweeper: NeffSweeper, target_wl: float) -> float:
    """Run FDE frequency sweep and return real n_eff at target_wl [m]."""
    sweeper.build_sim()
    wl, neff = sweeper.run_sweep()
    n_real = np.real(neff)
    return float(np.interp(target_wl, wl, n_real))


def main():
    cfg = SimulationConfig()

    pitch        = cfg.grating.pitch_m
    w_narrow     = cfg.geometry.width_narrow_m
    w_avg        = cfg.geometry.avg_corrugation_width_m
    height       = cfg.geometry.core_height_m
    target_wl    = cfg.spectral.center_wavelength_m
    material_db  = cfg_paths.MATERIAL_DB_PATH

    print("=" * 60)
    print("Phase-matched cavity length calculator")
    print(f"  Pitch        : {pitch * 1e9:.1f} nm")
    print(f"  w_narrow     : {w_narrow * 1e9:.1f} nm")
    print(f"  w_avg        : {w_avg * 1e9:.1f} nm")
    print(f"  Height       : {height * 1e9:.1f} nm")
    print(f"  Target λ     : {target_wl * 1e9:.2f} nm")
    print("=" * 60)

    common_kwargs = dict(
        height=height,
        wl_start=target_wl - 50e-9,
        wl_stop=target_wl + 50e-9,
        n_points=20,
        material_db_path=material_db,
        core_material="Si3N4 (Silicon Nitride) - Luke",
        clad_material="SiO2 (Glass) - Palik",
    )

    # ── Step 1: n_eff for narrow width ─────────────────────────────────────
    print(f"\n[1/2] FDE for w_narrow = {w_narrow * 1e9:.0f} nm ...")
    sweeper_narrow = NeffSweeper(width=w_narrow, **common_kwargs)
    n_narrow = get_neff_at_wavelength(sweeper_narrow, target_wl)
    sweeper_narrow.close()
    print(f"      n_eff(narrow) = {n_narrow:.5f}")

    # ── Step 2: n_eff for avg width ─────────────────────────────────────────
    print(f"\n[2/2] FDE for w_avg = {w_avg * 1e9:.0f} nm ...")
    sweeper_avg = NeffSweeper(width=w_avg, **common_kwargs)
    n_avg = get_neff_at_wavelength(sweeper_avg, target_wl)
    sweeper_avg.close()
    print(f"      n_eff(avg)    = {n_avg:.5f}")

    # ── Step 3: Recommended L_cav ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Recommended cavity lengths (override_cavity_length_nm):")
    print("-" * 60)
    for option in ("narrow", "avg", "avg_ext"):
        L = suggest_cavity_length(pitch, option, n_narrow, n_avg)
        print(f"  cavity_width_option = {option!r:10s}  →  {L * 1e9:.2f} nm")
    print("=" * 60)

    print("\nTo apply, set in your run script:")
    print("  cfg.grating.cavity_width_option = 'avg'   # or 'avg_ext'")
    L_avg = suggest_cavity_length(pitch, "avg", n_narrow, n_avg)
    L_avg_ext = suggest_cavity_length(pitch, "avg_ext", n_narrow, n_avg)
    print(f"  cfg.grating.override_cavity_length_nm = {L_avg * 1e9:.2f}  # avg")
    print(f"  # or {L_avg_ext * 1e9:.2f}  # avg_ext")


if __name__ == "__main__":
    main()
