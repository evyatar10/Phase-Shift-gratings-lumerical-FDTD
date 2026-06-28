"""
Single-device companion to compare_8_devices.py: the 5-parameter TM optimum
(WITH tooth shift) from the exploratory PSO (job 97635, label
transmission_gf_tm_shift). Runs through the EXACT same pipeline as the 8-device
comparison (accurate mesh, n=1.9963/1.444, pitch 518.3, scan 1565±25 nm / 2501
pts, 2D field monitors for spatial FWHM) so its transmission and mode-width FWHM
are directly comparable to the other bars.

Writes result_<geomtag>_cmp_TM_optimized_shift.mat. plot_compare_8_devices.m
reads it for the TM "Optimized" bar.

Invocation:
  bash athena/deploy_athena.sh --option2 --run=compare_tm_optimized_shift
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.single.run_simulation import run_single_sim
from runners.single.compare_8_devices import _optimized

# 5-param TM optimum (job 97635, accurate-verified true_peak_T=0.9618). Already
# at n_core=1.9963 → no index scaling. DW1/DW2 + SHIFT1/SHIFT2 + cavity.
TM_OPT_SHIFT = dict(
    dw=[95.48, 183.39],
    shift=[118.12, 137.88],
    cavity_nm=829.40,
    pitch_nm=518.3,
)


def run(_unused_cfg=None) -> dict:
    cfg = _optimized("TM", TM_OPT_SHIFT, 1.0)
    print("\n===== device: TM_optimized_shift =====")
    res = run_single_sim(cfg, show_plots=False, save_figs=False,
                         tag_suffix="_cmp_TM_optimized_shift")
    print(f"  TM_optimized_shift: T={res.get('resonance_transmission')}  "
          f"lam={res.get('resonance_wavelength_nm')}  "
          f"fwhm_um={res.get('fwhm_m', 0)*1e6:.3f}")
    return res


if __name__ == "__main__":
    run()
