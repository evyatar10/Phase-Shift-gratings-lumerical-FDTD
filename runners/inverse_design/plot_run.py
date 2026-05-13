"""Post-run plotter for runners/inverse_design (lumopt L-BFGS-B path).

Reads final_params.json from a completed run and writes two PNGs in the
same directory:
  - convergence.png      FOM vs L-BFGS-B iteration
  - spectrum_overlay.png T(λ) initial vs optimized geometry

Invocation (from any machine with the result directory available locally):
  python -m runners.inverse_design.plot_run <result_dir>

For Athena results synced via deploy_athena.sh --results-no-fsp, point at
e.g. ``results_from_athena/inverse_design/smoke/start0/``.
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
            f"inverse_design run directory (e.g. start0)."
        )
    with open(summary_path, "r") as fp:
        return json.load(fp)


def plot_from_dir(result_dir: str) -> None:
    summary = _load_summary(result_dir)

    # --- Convergence: FOM vs iter (+ peak T on twin axis if available) ----
    fom_history = summary.get("fom_history") or []
    peak_T_history = summary.get("peak_T_history") or None
    if fom_history:
        # lumopt's initial FOM is the first entry of fom_history; use that
        # as the baseline-FOM reference. initial_peak_T (from the baseline
        # FDTD) is the baseline-peak-T reference.
        baseline_fom = float(fom_history[0]) if fom_history else None
        baseline_peak_T = summary.get("initial_peak_T")
        out = os.path.join(result_dir, "convergence.png")
        plot_convergence(
            fom_history,
            out,
            baseline_fom=baseline_fom,
            peak_T_history=peak_T_history,
            baseline_peak_T=baseline_peak_T,
            title=(f"lumopt convergence — {summary.get('label', '')}"
                   f"  (max_iter={summary.get('max_iter', '?')})"),
            ylabel="|FOM|  (Gaussian-weighted ⟨T⟩)",
        )
        print(f"[plot_run] wrote {out}")
    else:
        print("[plot_run] WARN: no fom_history in summary — skipping convergence plot. "
              "(Either this is an old run or the optimizer never reported an iter.)")

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
        description="Plot convergence + spectrum overlay for an inverse_design run")
    parser.add_argument("result_dir", type=str,
                        help="Directory containing final_params.json")
    args = parser.parse_args()
    plot_from_dir(args.result_dir)


if __name__ == "__main__":
    main()
