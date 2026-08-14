"""lumopt2 campaign, SEED B: dip+overshoot cusp-smoothing profile. IGUM.

Study dir: runners/lumopt2_design/   |   Created 2026-08-13   |   Job(s): TBD
Purpose: the physics-informed second start (local-minimum insurance instead of
a PSO stage): radiation comes from the envelope cusp at the pi-shift, whose
Fourier content lives at ~2 periods — so seed with a SHORT corrugation dip at
the innermost teeth and pay the kappa-integral back just outside (rho ≈ 0.99,
inside the deadband). If A and B converge to the same basin the landscape is
benign; if not, the difference is the finding.

Same engine, cost function, and gates as seed A. Shorter budget (30 iters,
~1.5 GPU-days) — IGUM part-preempt is acceptable: the engine restarts from
its eval log with at most one iteration lost.

Dispatch (after checking license seats with seed A running — pool is SHARED):
    SBATCH_MEM=160G ARRAY_TIME=72:00:00 bash igum/deploy_igum.sh \
        --lumopt2-design=runners.lumopt2_design.campaign_c325_seedB
Output -> results/lumopt2_c325_seedB/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from runners.lumopt2_design import lumopt2_design as eng

SIGMA0_UM = None      # <- REQUIRED: same value as seed A (B2b canary)

# Dip at the cusp (teeth 1-4), payback just outside (teeth 5-12), frozen-level
# beyond. mean(corr)/325 = 0.9926 — inside the deadband by construction.
DIP_PROFILE_NM = (240.0, 270.0, 295.0, 315.0) + (340.0,) * 8 + (325.0,) * 13

SPEC = eng.CampaignSpec(
    label="lumopt2_c325_seedB",
    box_y_um=6.8, box_z_mult=4.14,       # z=6.8, identical to seed A (gate A0)
    corr_seed_nm=DIP_PROFILE_NM,
    max_iter=30, max_feval=55,
)


def main(task_idx=0):
    assert SIGMA0_UM is not None, "run gate B2b first and fill SIGMA0_UM"
    assert len(DIP_PROFILE_NM) == eng.N_FREE
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    eng.run_campaign(SPEC, out_dir, sigma0_um=SIGMA0_UM)


if __name__ == "__main__":
    import numpy as np
    rho = float(np.mean(DIP_PROFILE_NM)) / eng.CORR_NM
    print(f"seed B dip profile rho = {rho:.4f} (deadband 0.95-1.02), "
          f"{eng.N_PARAMS} params, {SPEC.max_iter} iters")
