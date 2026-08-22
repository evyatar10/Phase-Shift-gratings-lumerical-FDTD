"""lumopt2 campaign, BARE: grating-only optimization (NO comb). IGUM, chained.

Study dir: runners/lumopt2_design/   |   Created 2026-08-16   |   Job(s): TBD
Purpose (user, 2026-08-16): the interpretability partner of the full campaign
— optimize the SAME 25-teeth-per-side (+ cavity width) basis with the comb
absent entirely. Comparing the two converged tooth profiles answers, directly:
does the comb's presence change what the optimizer does with the teeth, and
is the comb still worth its cost on an OPTIMIZED grating? (The full campaign
remains the main deliverable; this is the decomposition experiment.)

bare=True removes the comb from the scene; the comb parameter slots get inert
sliver bounds (engine handles it) — 76 active params (75 teeth + cavity).
Same engine, cost function, anchors, and gates as seed A. The natural width
reference for the tripwire is the BARE anchor (B2a), not the comb one.

Dispatch on IGUM, chained behind seed B so the lane never idles (license pool
is SHARED — the chain also avoids adding concurrent seat pressure):
    SBATCH_MEM=160G bash igum/deploy_igum.sh \
        --lumopt2-design=runners.lumopt2_design.campaign_c325_bare \
        --after=<seedB job id>
Output -> results/lumopt2_c325_bare/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from runners.lumopt2_design import lumopt2_design as eng

SIGMA0_UM = 17.505    # MEASURED gate B2a at PVA (job 132654) — the BARE anchor

SPEC = eng.CampaignSpec(
    label="lumopt2_c325_bare",
    box_y_um=6.8, box_z_mult=4.14,       # identical numerics to seeds A/B
    bare=True, free_comb=False,          # no comb in the scene at all
    max_iter=30, max_feval=55,
    # trust_nm added 2026-08-18 after job 55343 died ABNORMAL_TERMINATION_IN_
    # LNSRCH: full-range shift bounds made L-BFGS-B's first probes blow the
    # width out (sigma 21.3 / 19.98 um) until maxls was exhausted — 2 of 6
    # evals wasted, then termination after ONE accepted iteration. Bounds are
    # the step size (skill item 22); run_campaign re-centers them on the
    # resumed best row each attempt, so the redispatch continues from the
    # eval log instead of starting over.
    trust_nm={"shift": 20.0, "corr": 10.0, "avg": 5.0, "wcav": 30.0},
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,  # C-fix (see seed A)
)


def main(task_idx=0):
    assert SIGMA0_UM is not None, "run gate B2a first and fill SIGMA0_UM"
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    eng.run_campaign(SPEC, out_dir, sigma0_um=SIGMA0_UM)


if __name__ == "__main__":
    print(f"bare campaign: grating-only ({eng.N_FREE}x3 teeth + cavity), "
          f"no comb, {SPEC.max_iter} iters; width ref = bare anchor "
          f"{SIGMA0_UM} um")
