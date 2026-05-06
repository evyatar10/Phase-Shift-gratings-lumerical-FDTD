"""
Aggregate per-driver results from a multi-start inverse-design run.

After all SLURM array tasks complete, this walks
    {BASE_SAVE_DIR}/inverse_design/{label}/start{K}/final_params.json
for K=0..n_starts-1, picks the best (highest true peak T), and writes a
top-level `summary.json` that the user (or downstream MATLAB plotting)
can consume directly.

Invocation:
    python -m runners.inverse_design.aggregate --label my_study
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List, Optional

import config as _cfg


def aggregate_study(label: str, output_root: Optional[str] = None) -> dict:
    if output_root is None:
        output_root = os.path.join(_cfg.BASE_SAVE_DIR, "inverse_design", label)
    if not os.path.isdir(output_root):
        raise FileNotFoundError(f"Study directory not found: {output_root}")

    pattern = os.path.join(output_root, "start*", "final_params.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No final_params.json files under {output_root}")

    drivers: List[dict] = []
    for p in paths:
        with open(p) as fp:
            drivers.append(json.load(fp))

    best = max(drivers, key=lambda d: d.get("true_peak_T", float("-inf")))
    summary = {
        "label": label,
        "n_drivers_completed": len(drivers),
        "best_driver_start_idx": best["start_idx"],
        "best_p": best["p_final"],
        "best_true_peak_T": best["true_peak_T"],
        "best_true_peak_lambda_m": best["true_peak_lambda_m"],
        "drivers": drivers,
    }
    out_path = os.path.join(output_root, "summary.json")
    with open(out_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[aggregate] wrote {out_path}")
    print(f"[aggregate] best peak T = {summary['best_true_peak_T']:.4f} "
          f"at λ = {summary['best_true_peak_lambda_m']*1e9:.3f} nm "
          f"(start{summary['best_driver_start_idx']})")
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Aggregate inverse-design multi-start results")
    ap.add_argument("--label", required=True, help="Study label (subfolder name)")
    ap.add_argument("--output-root", default=None,
                    help="Override study root directory")
    args = ap.parse_args(argv)
    aggregate_study(args.label, args.output_root)


if __name__ == "__main__":
    main()
