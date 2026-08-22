"""lumopt2 campaign seedA STAGE-4: sigma-neutral tangent from the stage-2 winner.

Study dir: runners/lumopt2_design/   |   Created 2026-08-18   |   Job(s): TBD
Purpose (user 2026-08-18: "if there is a way to continue optimizing, do so —
don't just assume we're converged"): stage-2 converged in ITS subspace (last
step bought +0.0001 FOM for +0.024 um width, a 100x efficiency collapse with
93% of the width band spent), but the tooth shifts have been FROZEN since
stage-1 and are the most width-efficient lever we have measured (0.065 T/um
vs corr 0.032, tangent probe 133512). Stage-4 unfreezes EVERYTHING from the
stage-2 winner and walks the sigma-neutral tangent under the sigma-hat wall.

Differences vs the failed stage-3 (133541, cancelled):
  * seed = BEST_T9635 (the CURRENT frontier, T 0.9635 / Q_i 110,874), not the
    long-overtaken stage-1 best;
  * trust_nm ON — the fix for the failure class that killed both free-shift
    runs (stage-3 overshoot, bare lnsrch death 55343): L-BFGS-B's first probe
    is unit-norm in bounds-scaled space, so bounds ARE the step size. Radii
    give ~±(shift 20 / corr 10 / avg 5 / wcav 30) nm steps, centered on the
    seed (=> exact scaler round-trip, no duplicate-x0 tax). Comb stays at
    physical bounds: measured immobile (scaled gradient ~500x smaller), and
    the basin scan proved it sits at its optimum — not worth a trust radius.
  * engine completion path fixed (_final_fom tuple handling, job 55343 crash).

sigma-hat wall anchored on the seed's MEASURED sigma 17.7952 (H200-family
value from the 133530 log); cumulative tripwire band vs sigma0 17.493
unchanged (+2% program-wide) — seed sits at 93%... no: ratio 1.01727 = 86%,
leaving ~0.048 um for wall-neutral noise. Recenter guard + width-compliant
final filter unchanged.

Dispatch (Athena, a100/4d_1g lane freed by stopping the plateaued 133530):
    SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=72:00:00 \\
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_c325_seedA4
Output -> results/campaign_c325_seedA4/results/lumopt2_c325_seedA4/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9635

SIGMA0_UM = 17.493    # original B2b anchor — cumulative tripwire band unchanged

_p0 = np.asarray(BEST_T9635, dtype=float)
SPEC = eng.CampaignSpec(
    label="lumopt2_c325_seedA4",
    box_y_um=6.8, box_z_mult=4.14,          # identical numerics throughout
    scan_center_nm=1566.14,                 # seed resonance 1566.144
    seed_override=tuple(BEST_T9635),
    sigma_wall=True,
    sig_anchor={"sigma": 17.7952,           # MEASURED on the seed row (133530 ev4)
                "elong": float(2.0 * _p0[eng.SL_SHIFT].sum()),
                "rho": float(_p0[eng.SL_CORR].mean() / 325.0),
                "wcav": float(_p0[eng.I_CAV])},
    trust_nm={"shift": 20.0, "corr": 10.0, "avg": 5.0, "wcav": 30.0},
    max_iter=25, max_feval=45,
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,  # C-fix
)
N_TASKS = 1


def main(task_idx=0):
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    eng.run_campaign(SPEC, out_dir, sigma0_um=SIGMA0_UM)


if __name__ == "__main__":
    pen, _ = eng.make_sigma_wall(SPEC)
    b = np.array(eng.param_bounds(SPEC))
    sl = eng.SL_SHIFT
    print(f"stage-4: penalty at seed = {float(pen(_p0)):.6f} (must be 0); "
          f"shift bound widths {(b[sl,1]-b[sl,0]).min():.2f}-{(b[sl,1]-b[sl,0]).max():.2f} nm; "
          f"seed centered = {bool(np.allclose(_p0[sl]-b[sl,0], b[sl,1]-_p0[sl]))}")
