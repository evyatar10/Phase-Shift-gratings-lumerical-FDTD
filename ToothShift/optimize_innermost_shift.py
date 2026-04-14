"""
Innermost tooth shift optimizer for Pi-Shift Bragg Grating FDTD.

Finds the shift value in [100, 140] nm that maximizes resonance transmission
using scipy's minimize_scalar (Brent's method): golden-section bracketing
combined with parabolic interpolation for fast convergence on smooth functions.

Budget: MAX_EVALS simulations.

Usage:
    python optimize_innermost_shift.py
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import copy
import gc

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize_scalar

import config
from simulation_config import SimulationConfig
from run_sweep_innermost_shift import run_single_sim_with_shift

# ── Search parameters ──────────────────────────────────────────────────────────

SHIFT_MIN_NM = 100.0
SHIFT_MAX_NM = 140.0
MAX_EVALS    = 8      # simulation budget


# ═══════════════════════════════════════════════════════════════════════════════
# Core optimization logic
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate(cfg: SimulationConfig, shift_nm: float, lengthen_cavity: bool,
              tested_shifts: list, tested_T: list) -> float:
    """Run one simulation, record the result, and return resonance transmission."""
    shift_m = shift_nm * 1e-9
    iter_cfg = copy.deepcopy(cfg)
    try:
        results = run_single_sim_with_shift(iter_cfg, shift_m, lengthen_cavity)
        T = float(results['resonance_transmission'])
    finally:
        gc.collect()
        print("Memory cleared.\n")

    tested_shifts.append(shift_nm)
    tested_T.append(T)
    return T


def run_optimization(
    cfg: SimulationConfig,
    lengthen_cavity: bool = True,
) -> tuple[list, list, float, float]:
    """
    Brent's method optimization over innermost tooth shift.

    Returns
    -------
    tested_shifts_nm : list of floats — all evaluated shift values (in eval order)
    tested_T         : list of floats — corresponding resonance transmission values
    best_shift_nm    : float          — shift with highest transmission found
    best_T           : float          — corresponding transmission value
    """
    tested_shifts_nm: list[float] = []
    tested_T: list[float] = []

    print("=" * 60)
    print(f"Brent optimization  [{SHIFT_MIN_NM:.0f}, {SHIFT_MAX_NM:.0f}] nm  "
          f"|  budget: {MAX_EVALS} evals")
    print("=" * 60 + "\n")

    def objective(shift_nm: float) -> float:
        n = len(tested_shifts_nm) + 1
        print(f">>> EVAL {n}/{MAX_EVALS}: shift = {shift_nm:.3f} nm <<<\n")
        T = _evaluate(cfg, shift_nm, lengthen_cavity, tested_shifts_nm, tested_T)
        print(f"    T = {T:.4f}\n")
        return -T  # minimize negative → maximize transmission

    minimize_scalar(
        objective,
        bounds=(SHIFT_MIN_NM, SHIFT_MAX_NM),
        method='bounded',
        options={'maxiter': MAX_EVALS, 'xatol': 0.3},
    )

    best_idx = int(np.argmax(tested_T))
    best_shift_nm = tested_shifts_nm[best_idx]
    best_T = tested_T[best_idx]

    print("=" * 60)
    print("OPTIMIZATION COMPLETE")
    print(f"  Best shift       : {best_shift_nm:.3f} nm")
    print(f"  Best transmission: {best_T:.4f}")
    print("=" * 60)

    return tested_shifts_nm, tested_T, best_shift_nm, best_T


# ═══════════════════════════════════════════════════════════════════════════════
# Results persistence
# ═══════════════════════════════════════════════════════════════════════════════

def save_optimization_results(
    tested_shifts_nm: list,
    tested_T: list,
    best_shift_nm: float,
    best_T: float,
    output_path: str,
) -> None:
    """Save optimization history and best result to a .mat file."""
    sio.savemat(output_path, {
        'tested_shift_nm': np.array(tested_shifts_nm),
        'tested_T':        np.array(tested_T),
        'best_shift_nm':   float(best_shift_nm),
        'best_T':          float(best_T),
        'shift_range_nm':  np.array([SHIFT_MIN_NM, SHIFT_MAX_NM]),
        'max_evals':       MAX_EVALS,
    })
    print(f"Optimization results saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_optimization(
    tested_shifts_nm: list,
    tested_T: list,
    best_shift_nm: float,
    best_T: float,
) -> None:
    """
    Two-panel figure:
      Top    — Transmission vs shift. Points are colored by evaluation order
               (early = dark, late = light) to show where Brent's method probed.
      Bottom — Search progression: T at each evaluation with running best.
    """
    n = len(tested_shifts_nm)
    order = np.argsort(tested_shifts_nm)
    sorted_s = np.array(tested_shifts_nm)[order]
    sorted_t = np.array(tested_T)[order]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9),
                                   gridspec_kw={'height_ratios': [3, 1.5]})
    fig.suptitle("Innermost Tooth Shift Optimization (Brent's Method)",
                 fontsize=13, fontweight='bold')

    # ── Panel 1: T vs shift, colored by eval index ───────────────────────────
    ax1.plot(sorted_s, sorted_t, 'k--', linewidth=0.8, alpha=0.4, zorder=1)
    sc = ax1.scatter(tested_shifts_nm, tested_T,
                     c=np.arange(n), cmap='viridis', s=90, zorder=3)
    plt.colorbar(sc, ax=ax1, label='Evaluation index')
    ax1.scatter([best_shift_nm], [best_T], s=220, color='red', zorder=5,
                label=f'Best: {best_shift_nm:.2f} nm  (T = {best_T:.4f})')
    ax1.axvline(best_shift_nm, color='red', linestyle=':', linewidth=1.0, alpha=0.6)

    ax1.set_xlabel("Innermost Tooth Shift [nm]")
    ax1.set_ylabel("Resonance Transmission")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.4)

    # ── Panel 2: convergence trace ────────────────────────────────────────────
    evals = np.arange(1, n + 1)
    running_best = np.maximum.accumulate(tested_T)

    ax2.plot(evals, tested_T, 'o-', color='steelblue', linewidth=1.2,
             markersize=6, label='T at eval')
    ax2.plot(evals, running_best, 'r--', linewidth=1.5, label='Running best')
    ax2.set_xlabel("Evaluation index")
    ax2.set_ylabel("Resonance Transmission")
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.4)
    ax2.set_xticks(evals)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Redirect all outputs (layouts, per-run .mat files, opt summary) into a
    # dedicated subfolder so results don't mix with other runs.
    config.BASE_SAVE_DIR = os.path.join(config.BASE_SAVE_DIR, "DS_shift_opt")

    cfg = SimulationConfig()
    cfg.grating.n_periods_each_side = 80
    cfg.mesh.simulation_mode = "optimization"
    cfg.spectral.scan_width_nm = 16.0

    tested_shifts, tested_T, best_shift, best_T = run_optimization(cfg, lengthen_cavity=True)

    opt_path = os.path.join(config.RESULTS_DIR, "optimization_shift_results.mat")
    save_optimization_results(tested_shifts, tested_T, best_shift, best_T, opt_path)

    plot_optimization(tested_shifts, tested_T, best_shift, best_T)
