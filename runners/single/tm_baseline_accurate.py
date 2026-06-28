"""
Accurate-mesh baseline for the TM transmission PSO (job 97316).

The optimizer ran at the COARSE optimization mesh and verified its converged
design at the ACCURATE mesh (true_peak_T = 0.9561). But the reported baseline
(0.9582) was measured at the COARSE mesh, so the headline Δ compared coarse vs
accurate — not a fair gain. This run re-simulates the regular-grating baseline
[300, 300, 800] at the SAME accurate mesh + scan settings the optimizer used for
its final verification, so we get an apples-to-apples accurate-vs-accurate number.

Imports the optimization BASE so n_core=1.97 / n_clad=1.444 / pitch=516.14 /
TM polarization / scan window all match exactly; only the mesh and geometry
(regular grating) differ.

Invocation:
  bash athena/deploy_athena.sh --option2 --run=tm_baseline_accurate --gpu=h200
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.single.run_simulation import run_single_sim
from runners.gradient_free_design.optimize_transmission_tm import BASE


def run(_unused_cfg=None) -> dict:
    cfg = copy.deepcopy(BASE)

    # Accurate mesh — matches the optimizer's post-opt verification (which read 0.9561).
    cfg.mesh.simulation_mode = "accurate"

    # Regular-grating baseline, identical to the PSO seed [300, 300, 0, 0, 800].
    cfg.grating.n_free_inner_teeth = 2
    cfg.grating.inner_dw_nm        = [300.0, 300.0]
    cfg.grating.inner_shift_nm     = [0.0, 0.0]
    cfg.grating.cavity_width_m     = 800e-9

    # Optimization-style: no field/far-field monitors needed for the peak-T number.
    cfg.monitors.record_2d_fields = False
    cfg.monitors.record_3d_fields = False
    cfg.farfield.enabled          = False

    return run_single_sim(cfg)


if __name__ == "__main__":
    run()
