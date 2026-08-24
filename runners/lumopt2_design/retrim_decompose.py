"""Is the re-trimmed best "just a deeper grating"? — ingredient decomposition.

Study dir: runners/lumopt2_design/  |  Created 2026-08-23  |  Job(s): TBD
Purpose (user 2026-08-23: "is it just a scaled thing?"): the re-trim design
= origin + FOUR ingredients at once — (a) uniform depth payback +52.5 nm on
the 25 free teeth, (b) cavity width 800->960.9, (c) tooth shifts 2Ss=130.6,
(d) the apodization SHAPE (corr spread 39.7 nm). This ladder adds them one
at a time at identical numerics (50 nm PVA, v2 window), so each rung's
(T, FWHM) is directly comparable to the origin (T 0.89052, 17.7136) and to
the full design (T 0.95968, 17.7550 = retrim d+52.5, job 136051).

  task 0  depth only          corr 368.47 uniform, avg 800, shifts 0, wcav 800
  task 1  + cavity width      wcav 960.9
  task 2  + shifts            2Ss 130.6 (per-tooth values from BEST_T9635)
  (full design = already measured; shape is the remainder)

Widths differ per rung by construction; normalise with the MEASURED payback
rate 0.00224 T/um (retrim curve 136051: dT 0.0063 over dFWHM 2.818 um) to
read every rung "at the origin width".

Dispatch: SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \\
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.retrim_decompose
"""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9635

MEAN_CORR_NM = 368.47        # = mean(BEST_T9635 corr) + 52.5, the payback level
PAYBACK_T_PER_UM = 0.00224   # MEASURED, retrim curve (job 136051)

_B = np.asarray(BEST_T9635, dtype=float)

SPEC = eng.CampaignSpec(label="rtdec_depth", scan_width_nm=10.0,
                        n_wl_points=501, scan_center_nm=1566.0,
                        rho_band=False)          # mean corr is 13% up on purpose
N_TASKS = 3


def _params(idx):
    """Rung idx: 0 = depth only, 1 = +cavity, 2 = +shifts."""
    p = eng.seed_params(SPEC)                    # uniform + winner comb
    p[eng.SL_CORR] = MEAN_CORR_NM                # (a) depth, flat = no shape
    p[eng.SL_AVG] = eng.AVG_W_NM
    p[eng.SL_SHIFT] = 0.0
    p[eng.I_CAV] = eng.AVG_W_NM
    if idx >= 1:
        p[eng.I_CAV] = _B[eng.I_CAV]             # (b) cavity width 960.9
    if idx >= 2:
        p[eng.SL_SHIFT] = _B[eng.SL_SHIFT]       # (c) shifts, 2Ss = 130.6
    return p


def main(task_idx=0):
    idx = int(task_idx)
    label = ["rtdec_depth", "rtdec_depth_cav", "rtdec_depth_cav_shift"][idx]
    spec = dataclasses.replace(SPEC, label=label)
    spec.seed_override = tuple(_params(idx))
    out_dir = os.path.join(config.RESULTS_DIR, "retrim_decompose")
    row = eng.run_canary(spec, out_dir)
    t, fw = row.get("t_pk"), row.get("fwhm_env_um")
    # sign: along the depth axis T and width move TOGETHER (deeper = narrower =
    # lower T), so a rung measured narrow is credited the T it would gain on
    # being let back out to the origin width.
    norm = (t + PAYBACK_T_PER_UM * (17.7136 - fw)) if (t and fw) else None
    print(f"[rtdec {idx} {label}] T {t} FWHM {fw} um | T normalised to the "
          f"origin width 17.7136: {norm} (origin 0.89052, full design 0.95968)")


if __name__ == "__main__":
    for i in range(N_TASKS):
        print("task", i, ["depth", "depth+cav", "depth+cav+shift"][i])
