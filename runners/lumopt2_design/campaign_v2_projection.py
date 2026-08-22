"""V2 PROJECTION-FIRST campaign — T-only adjoint, measured-FWHM guard, seeded
from the RE-TRIMMED BEST (the +0.0665-at-equal-width verdict design).

Study dir: runners/lumopt2_design/  |  Created 2026-08-22  |  Job(s): TBD
Architecture settled by measurement (V2_FWHM_PLAN §8): the CPU width-adjoint
costs 8.7 h/solve — priced out. This campaign optimizes softmax-T with the
VALIDATED port adjoint only (C = 1.0561+0.1239i, calibrated at these exact
numerics: 50 nm region + PVA — deliberately kept so every measured anchor and
the seed's own row stay bit-comparable; the pitch-locked/conformal migration
happens at the NEXT anchor reset with MX-GRAD's verdict). Width control:
  - measured fwhm_env guard: fwhm0 17.713551 (W2), band +2%/−5%, WidthTrip on
    accepted-best + restart/final filter (engine, shipped);
  - rho band RETIRED (rho_band=False — item-27 hole; would also crush this
    legitimately-deep-corr seed); elongation cheat-wall kept;
  - stage-boundary re-trim = the proven uniform corr-add bisection (manual,
    ~5 forwards) before any cross-width comparison.
Seed: BEST_T9635 + 52.5 nm uniform corr — MEASURED in-band at these numerics
(17.755 µm = ratio 1.0023, T 0.95968; job 136051 row d+52.5).

Dispatch:  SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \\
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_v2_projection
Resume: re-dispatch the same module (cold-start resume; REQUEUE-safe).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9635

RETRIM_DELTA_NM = 52.5           # the verdict-row payback (job 136051)

SPEC = eng.CampaignSpec(
    label="lumopt2_v2proj",
    scan_width_nm=10.0, n_wl_points=501,      # v2 window (anchored by W2)
    scan_center_nm=1566.38,                   # the seed's own measured peak
    free_comb=False,                          # user: comb fixed at winner
    rho_band=False,                           # retired; fwhm guard owns width
    fwhm0_um=17.713551,                       # MEASURED origin, W2 135971_10
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,
    trust_nm={"corr": 12.0, "avg": 10.0, "shift": 12.0, "wcav": 12.0},
    fwhm_wall=True,
    fw_anchor={"fwhm": 17.7550,          # MEASURED, retrim d+52.5 row (136051)
               "mcorr": 368.5,           # set exactly in main() from the seed
               "elong": 130.6},
    max_iter=40, max_feval=70,
)
N_TASKS = 1


def main(task_idx=0):
    p = np.asarray(BEST_T9635, dtype=float).copy()
    p[eng.SL_CORR] = np.minimum(p[eng.SL_CORR] + RETRIM_DELTA_NM,
                                SPEC.corr_max_nm)
    # Order matters: trust_nm bounds center on the module seed until
    # seed_override is set — set it FIRST so replay's bounds check (and the
    # optimizer's trust box) center on THIS seed, not the uniform one.
    SPEC.seed_override = tuple(p)
    SPEC.seed_override = tuple(eng.replay_params(SPEC, p))
    SPEC.fw_anchor["mcorr"] = float(np.mean(p[eng.SL_CORR]))
    SPEC.fw_anchor["elong"] = float(2.0 * p[eng.SL_SHIFT].sum())
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2proj] finished: best_fom {best['fom']:.5f} — delivered design "
          f"comes from the width-filtered log; re-trim before ANY comparison")


if __name__ == "__main__":
    print("projection-first campaign; seed = BEST_T9635 + 52.5 nm corr")
