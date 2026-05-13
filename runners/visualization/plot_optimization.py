"""Shared plotting utilities for inverse-design / PSO optimization runs.

Two functions, both pure / dependency-light (matplotlib + scipy.io only):

  - plot_convergence: FOM-vs-iteration curve with optional baseline line.
  - plot_spectrum_overlay: T(λ) at initial vs final geometry, with peak
                            transmission and peak wavelength annotations.

Designed to be called by per-runner `plot_run.py` entry points so the
same plotting code serves both runners/inverse_design (lumopt) and
runners/lumerical_native_optimization (Lumerical's addsweep PSO).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")   # headless: Athena nodes have no display

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


HistoryEntry = Union[float, dict]


def _coerce_fom_series(history: Sequence[HistoryEntry]) -> List[float]:
    """Accept a list of floats OR a list of dicts containing FOM-ish keys."""
    out: List[float] = []
    for entry in history:
        if isinstance(entry, dict):
            # Try the canonical key first, then common alternatives.
            for key in ("fom", "best_fom", "peak_T", "value"):
                if key in entry:
                    out.append(float(entry[key]))
                    break
            else:
                raise ValueError(
                    f"history entry {entry!r} has no recognized FOM key "
                    f"(fom / best_fom / peak_T / value)."
                )
        else:
            out.append(float(entry))
    return out


def plot_convergence(
    history: Sequence[HistoryEntry],
    out_path: str,
    *,
    baseline_fom: Optional[float] = None,
    peak_T_history: Optional[Sequence[float]] = None,
    baseline_peak_T: Optional[float] = None,
    title: str = "Optimization convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Figure of merit",
    peak_T_label: str = "Peak T",
) -> str:
    """Write a FOM-vs-iteration PNG. Returns the output path.

    `history` is either a list of floats or a list of {"fom": ...} dicts
    (PSO histories from this codebase use the latter; lumopt's fom_history
    is the former).

    If `peak_T_history` is given, overlay it on a twin y-axis (right side)
    so the researcher sees both the cost-function FOM (left) and the
    headline peak-transmission metric (right) on the same iteration axis.
    `baseline_peak_T` adds a dashed horizontal reference on the right axis.
    """
    fom_series = _coerce_fom_series(history)
    if len(fom_series) == 0:
        raise ValueError("plot_convergence: history is empty.")

    iters = np.arange(len(fom_series))

    fig, ax = plt.subplots(figsize=(9, 5))
    fom_color = "#1f77b4"
    ax.plot(iters, fom_series, "o-", color=fom_color, linewidth=1.5, markersize=6,
            label="FOM (cost fn)")
    if baseline_fom is not None:
        ax.axhline(baseline_fom, linestyle="--", color=fom_color, alpha=0.5,
                   label=f"Initial FOM = {baseline_fom:.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color=fom_color)
    ax.tick_params(axis="y", labelcolor=fom_color)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()

    if peak_T_history is not None and len(peak_T_history) > 0:
        peak_color = "#d62728"
        peak_arr = np.asarray(peak_T_history, dtype=float)
        # Trim or pad to fom_series length so the x-axis aligns. If lumopt
        # produced more forward_*.fsp than fom_hist entries (line-search probes
        # not counted as iterations), keep only the first len(fom_series).
        n = min(len(peak_arr), len(fom_series))
        ax2 = ax.twinx()
        ax2.plot(iters[:n], peak_arr[:n], "s--", color=peak_color, linewidth=1.5,
                 markersize=5, label=peak_T_label)
        if baseline_peak_T is not None:
            ax2.axhline(baseline_peak_T, linestyle=":", color=peak_color, alpha=0.5,
                        label=f"Initial peak T = {baseline_peak_T:.4f}")
        ax2.set_ylabel(peak_T_label, color=peak_color)
        ax2.tick_params(axis="y", labelcolor=peak_color)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    ax.legend(handles, labels, loc="best", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _load_spectrum(mat_path: str):
    """Read wavelength (nm) and transmission arrays from a result_*.mat file.

    The .mat schema is set by post_processing.assemble_results — we depend on
    `wl_nm` and `T`. Falls back to `wl_m * 1e9` if `wl_nm` is missing.
    """
    data = sio.loadmat(mat_path, squeeze_me=True)
    if "wl_nm" in data:
        wl_nm = np.asarray(data["wl_nm"]).flatten()
    elif "wl_m" in data:
        wl_nm = np.asarray(data["wl_m"]).flatten() * 1e9
    else:
        raise KeyError(f"No wl_nm / wl_m field in {mat_path}.")
    if "T" not in data:
        raise KeyError(f"No T field in {mat_path}.")
    T = np.asarray(data["T"]).flatten()
    # Peak (mat may also have resonance_*; prefer those for marker label)
    resonance_lambda_nm = None
    resonance_T = None
    if "resonance_wavelength_nm" in data:
        resonance_lambda_nm = float(np.asarray(data["resonance_wavelength_nm"]).flatten()[0])
    if "resonance_transmission" in data:
        resonance_T = float(np.asarray(data["resonance_transmission"]).flatten()[0])
    return wl_nm, T, resonance_lambda_nm, resonance_T


def plot_spectrum_overlay(
    initial_mat: str,
    final_mat: str,
    out_path: str,
    *,
    title: str = "Transmission spectrum — initial vs optimized",
    initial_label: str = "Initial",
    final_label: str = "Optimized",
) -> str:
    """Overlay T(λ) from two result_*.mat files (initial vs final geometry).

    Returns the output path.
    """
    wl0, T0, lam0_peak, T0_peak = _load_spectrum(initial_mat)
    wl1, T1, lam1_peak, T1_peak = _load_spectrum(final_mat)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wl0, T0, "-", linewidth=1.5, color="#1f77b4", label=initial_label)
    ax.plot(wl1, T1, "-", linewidth=1.5, color="#d62728", label=final_label)

    # Peak markers
    if lam0_peak is not None and T0_peak is not None:
        ax.plot(lam0_peak, T0_peak, "o", color="#1f77b4", markersize=8,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(f"{T0_peak:.4f}\n@ {lam0_peak:.2f} nm",
                    xy=(lam0_peak, T0_peak), xytext=(8, 8),
                    textcoords="offset points", color="#1f77b4", fontsize=9)
    if lam1_peak is not None and T1_peak is not None:
        ax.plot(lam1_peak, T1_peak, "o", color="#d62728", markersize=8,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(f"{T1_peak:.4f}\n@ {lam1_peak:.2f} nm",
                    xy=(lam1_peak, T1_peak), xytext=(8, -22),
                    textcoords="offset points", color="#d62728", fontsize=9)

    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Transmission T")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
