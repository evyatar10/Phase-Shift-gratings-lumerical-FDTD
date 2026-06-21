"""
TM example: TE-vs-TM comparison in one process — exactly ONE single wide scan
per polarization (2 sims total), then a comparison summary. See
runners/tm/_tm_vs_te_common.py for the full description.

All outputs land in the study's standard layouts/ + results/ folders, and nothing
else: layouts/ holds layout_<tag>.fsp (one per polarization); results/ holds the
per-pol result_*.mat (the TE and TM spectra) plus the comparison summary
result_tm_vs_te_summary_N<n>.mat. No plots are written — overlay the TE/TM
spectra in MATLAB from the two result_*.mat. Download with `--results-full` to
bring the .fsp layouts back.

Usage:
  local (both, sequential): python -m runners.tm.run_tm_vs_te
  Athena, PARALLEL (TE + TM as a 2-task GPU array — recommended):
           bash athena/deploy_athena.sh --option2 --run=run_tm_vs_te --pol-array
           then after download:  python -m runners.tm.run_tm_vs_te --stitch <results_dir>
  Athena, sequential (one job, TE then TM, stitched on the node):
           bash athena/deploy_athena.sh --option2 --run=run_tm_vs_te

Parallelism: with SLURM_ARRAY_TASK_ID set (the --pol-array path) task 0 runs the
TE scan and task 1 the TM scan on separate GPUs; the summary is stitched locally
afterwards. Without it, both run in one process and the summary is written there.

Scan window — edit in _tm_vs_te_common.py: COMPARE_CENTER_M / COMPARE_WIDTH_NM /
COMPARE_N_POINTS (currently 1550 nm center, 150 nm wide, 6001 points).

Environment flags (see _tm_vs_te_common):
  TM_FARFIELD=1   enable far-field monitors (default 0)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.tm import _tm_vs_te_common as tvt
from simulation_config import SimulationConfig

build_cfg = tvt.build_base_cfg
STUDY_DIR_NAME = tvt.STUDY_DIR_NAME   # shared results folder: results/tm_te/


def run_tm_vs_te(cfg: SimulationConfig = None) -> dict:
    base = cfg if cfg is not None else build_cfg(SimulationConfig())
    # Parallel GPU-array path (deploy --pol-array): each array task runs ONE
    # polarization — task 0 = TE, task 1 = TM — so the two scans run concurrently
    # on separate GPUs. No summary is written here; stitch it afterwards with
    # `python -m runners.tm.run_tm_vs_te --stitch <results_dir>` on the download.
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if task != "":
        pol = ("TE", "TM")[int(task)]
        return tvt.run_one_scan(base, pol)
    # Local / single-job path: run both in one process and stitch on the spot.
    te = tvt.run_one_scan(base, "TE")
    tm = tvt.run_one_scan(base, "TM")
    return tvt.run_stitch(te, tm)


# Auto-discovery contract for athena_run.py: top-level `run` + `build_cfg`.
run = run_tm_vs_te


if __name__ == "__main__":
    # `python -m runners.tm.run_tm_vs_te`                -> run both + stitch (local)
    # `python -m runners.tm.run_tm_vs_te --stitch <dir>` -> stitch the two
    #     per-polarization result_*.mat in <dir> (the parallel-array path)
    if len(sys.argv) >= 3 and sys.argv[1] == "--stitch":
        tvt.stitch_dir(sys.argv[2])
    else:
        run_tm_vs_te(build_cfg(SimulationConfig()))
