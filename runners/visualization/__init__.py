"""Shared visualization helpers for optimization runners.

Exposes `plot_convergence` and `plot_spectrum_overlay` — used by the
per-runner `plot_run.py` entry points to turn a completed inverse-design
or PSO run's output directory into PNG figures.
"""

from runners.visualization.plot_optimization import (
    plot_convergence,
    plot_spectrum_overlay,
)

__all__ = ["plot_convergence", "plot_spectrum_overlay"]
