"""lumopt2 campaign seedA STAGE-3: single sigma-hat wall, all parameters free.

Study dir: runners/lumopt2_design/   |   Created 2026-08-17   |   Job(s): TBD
Purpose (user-approved "continue all", 2026-08-17 afternoon): the tangent
probe (job 133512) MEASURED that the sigma-neutral cross-block trade the twin
walls forbade is real and profitable — shift +40 nm alone: T +0.0096 at
sigma +0.147 um; corr payback: -0.047 um sigma for -0.0015 T; net tangent
~ +1.2e-4 T per nm of 2*Sig_s at FIXED width. Stage-3 replaces the elongation
hinge + rho deadband with ONE hinge on the calibrated width surrogate
(engine sigma_wall; slopes 0.00368/2Ss, -3.85/rho, 0.109/w_cav; anchor
re-zeroed at each restart on the best row's MEASURED sigma). Everything is
free: shifts, corr, avg, comb, cavity. Unchanged guarantee layers: measured-
sigma tripwire (sigma0 17.493, band 0.95-1.02), width-compliant restart +
final-selection filters, recenter guard, 2kL floor, production confirm.

Seed = seedA stage-1 best (same vector as stage-2): FOM 0.68831 / T 0.9313 /
sigma 17.7519 A100 = 17.749 H200 (job 133530 baseline, the anchor value).
Runs ALONGSIDE stage-2 (frozen-shift refinement) — different question:
stage-2 asks "do corr/comb respond at fixed shifts"; stage-3 walks the
width-neutral tangent. seedB inherits the winner when IGUM returns.

Dispatch (Athena, a100 lane free after stage-2 moved to H200):
    SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=72:00:00 \\
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_c325_seedA3
Output -> results/campaign_c325_seedA3/results/lumopt2_c325_seedA3/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_c325_seedA2 import SEED  # stage-1 best vector

SIGMA0_UM = 17.493    # ORIGINAL B2b anchor — the tripwire band stays cumulative

_p0 = np.asarray(SEED, dtype=float)
SPEC = eng.CampaignSpec(
    label="lumopt2_c325_seedA3",
    box_y_um=6.8, box_z_mult=4.14,          # identical numerics throughout
    scan_center_nm=1566.0,
    seed_override=SEED,                     # warm start; NOTHING frozen
    sigma_wall=True,
    sig_anchor={"sigma": 17.749,            # MEASURED H200 baseline (133530)
                "elong": float(2.0 * _p0[eng.SL_SHIFT].sum()),
                "rho": float(_p0[eng.SL_CORR].mean() / 325.0),
                "wcav": float(_p0[eng.I_CAV])},
    max_iter=30, max_feval=55,
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,  # C-fix
)
N_TASKS = 1


def main(task_idx=0):
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    eng.run_campaign(SPEC, out_dir, sigma0_um=SIGMA0_UM)


if __name__ == "__main__":
    pen, _ = eng.make_sigma_wall(SPEC)
    print(f"stage-3 seedA: sigma-wall ON (anchor {SPEC.sig_anchor['sigma']} um, "
          f"elong {SPEC.sig_anchor['elong']:.1f} nm), penalty at seed = "
          f"{float(pen(_p0)):.6f} (must be 0), all 191 params free")
