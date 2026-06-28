"""
Single wide TM scan of the default pi-shift Bragg grating: one predefined wide
spectral window (T / R / loss + 1D energy), no scout/refine two-step. Output is
one layout_*.fsp + one result_*.mat in the study's standard layouts/ + results/.

Use run_te for the TE polarization, or run_tm_vs_te for both + the comparison
summary (and --pol-array to run the two in parallel).

Usage:
  local:   python -m runners.tm.run_tm
  Athena:  bash athena/deploy_athena.sh --option2 --run=run_tm

Scan window — edit COMPARE_CENTER_M / COMPARE_WIDTH_NM / COMPARE_N_POINTS in
_tm_vs_te_common.py (currently 1550 nm center, 150 nm wide, 6001 points).

The standalone TM device defaults to its CALIBRATED geometry: pitch 516.14 nm
(TM resonance on the TE line) and corrugation 400 nm (TM spatial mode width = TE's,
i.e. equal kappa). Override either per run via TM_PITCH_NM / TM_CORR_NM.

Environment flags (see _tm_vs_te_common):
  TM_PITCH_NM=NNN    pitch in nm (default 516.14, the calibrated TM pitch at n 1.97)
  TM_CORR_NM=NNN     corrugation depth in nm (default 400, the TE-mode-width match)
  TM_RECORD_2D=1     record XY/YZ/XZ 2D field profiles (cavity cross-section)
                     for MATLAB plot_field_poynting.m (default 0, spectra only)
  TM_MESH=accurate   finer mesh, dx~35 nm (default optimization, dx=50 nm)
  TM_FARFIELD=1      enable far-field monitors (default 0)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.tm import _tm_vs_te_common as tvt
from simulation_config import SimulationConfig

build_cfg = tvt.build_base_cfg
STUDY_DIR_NAME = tvt.STUDY_DIR_NAME   # shared results folder: results/tm_te/


def run_tm(cfg: SimulationConfig = None) -> dict:
    base = cfg if cfg is not None else build_cfg(SimulationConfig())
    # Anchor the standalone TM device to its calibrated geometry (user 2026-06-28),
    # both DEFAULTS and both overridable per run:
    #   corrugation 400 nm (TM_CORR_NM) -> matches TE's spatial mode width / kappa,
    #     found by runners/tm/tm_match_corrugation_bisect.py (job 113571).
    #   pitch 516.14 nm  (TM_PITCH_NM)  -> matches the TM resonance to the TE line.
    base.geometry.corrugation_depth_m = float(os.environ.get("TM_CORR_NM", "400")) * 1e-9
    if "TM_PITCH_NM" not in os.environ:
        base.grating.pitch_m = 516.14e-9
    return tvt.run_one_scan(base, "TM")


# Auto-discovery contract for athena_run.py: top-level `run` + `build_cfg`.
run = run_tm


if __name__ == "__main__":
    run_tm(build_cfg(SimulationConfig()))
