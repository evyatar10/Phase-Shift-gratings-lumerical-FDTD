"""Post-run plotter for runners/lumerical_native_optimization
(Lumerical's addsweep('Optimization') PSO path).

Reads final_params.json from a completed run and writes two PNGs in the
same directory:
  - convergence.png      best FOM vs generation
  - spectrum_overlay.png T(λ) initial vs optimized geometry

Invocation:
  python -m runners.lumerical_native_optimization.plot_run <result_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.visualization import plot_convergence, plot_spectrum_overlay


def _load_summary(result_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(result_dir, "final_params.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"No final_params.json in {result_dir!r}; pass a completed "
            f"lumerical_native_optimization run directory (e.g. start0)."
        )
    with open(summary_path, "r") as fp:
        return json.load(fp)


def plot_from_dir(result_dir: str) -> None:
    summary = _load_summary(result_dir)

    # --- Convergence: best FOM vs generation -------------------------------------
    pso_history = summary.get("pso_history") or []
    if pso_history:
        baseline = summary.get("initial_peak_T")
        out = os.path.join(result_dir, "convergence.png")
        plot_convergence(
            pso_history,
            out,
            baseline_fom=baseline,
            title=(f"Lumerical PSO convergence — {summary.get('label', '')}"
                   f"  (pop={summary.get('population_size', '?')}, "
                   f"gens={summary.get('max_generations', '?')})"),
            xlabel="Generation",
            ylabel="Best peak T",
        )
        print(f"[plot_run] wrote {out}")
    else:
        print("[plot_run] WARN: no pso_history in summary — skipping convergence plot. "
              "(Sweep may have no-op'd; check the run log.)")

    # --- Spectrum overlay: initial vs final --------------------------------------
    init_mat = summary.get("initial_results_path")
    final_mat = summary.get("final_results_path")
    if init_mat and final_mat and os.path.exists(init_mat) and os.path.exists(final_mat):
        out = os.path.join(result_dir, "spectrum_overlay.png")
        plot_spectrum_overlay(
            init_mat, final_mat, out,
            title=f"T(λ): initial vs optimized — {summary.get('label', '')}",
            initial_label=f"Initial (cavity={summary['p_initial'][-1]:.0f} nm)",
            final_label=f"Optimized (cavity={summary['p_final'][-1]:.0f} nm)",
        )
        print(f"[plot_run] wrote {out}")
    else:
        missing = [name for name, p in
                   (("initial_results_path", init_mat),
                    ("final_results_path", final_mat))
                   if not (p and os.path.exists(p))]
        print(f"[plot_run] WARN: cannot draw spectrum overlay — "
              f"missing/unreadable: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot convergence + spectrum overlay for a "
                    "lumerical_native_optimization run")
    parser.add_argument("result_dir", type=str,
                        help="Directory containing final_params.json")
    args = parser.parse_args()
    plot_from_dir(args.result_dir)


if __name__ == "__main__":
    main()
