"""
Single wide TE scan of the default pi-shift Bragg grating: one predefined wide
spectral window (T / R / loss + 1D energy), no scout/refine two-step. Output is
one layout_*.fsp + one result_*.mat in the study's standard layouts/ + results/.

The TE counterpart of run_tm — lets you run the two polarizations separately. For
both + the comparison summary use run_tm_vs_te (with --pol-array to parallelize).

Usage:
  local:   python -m runners.tm.run_te
  Athena:  bash athena/deploy_athena.sh --option2 --run=run_te

Scan window — edit COMPARE_CENTER_M / COMPARE_WIDTH_NM / COMPARE_N_POINTS in
_tm_vs_te_common.py (currently 1550 nm center, 150 nm wide, 6001 points).

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


def run_te(cfg: SimulationConfig = None) -> dict:
    base = cfg if cfg is not None else build_cfg(SimulationConfig())
    return tvt.run_one_scan(base, "TE")


# Auto-discovery contract for athena_run.py: top-level `run` + `build_cfg`.
run = run_te


if __name__ == "__main__":
    run_te(build_cfg(SimulationConfig()))
