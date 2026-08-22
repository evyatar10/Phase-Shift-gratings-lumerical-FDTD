"""lumopt2 campaign seedB STAGE-2: the refinement seed B never received.

Study dir: runners/lumopt2_design/   |   Created 2026-08-18   |   Job(s): TBD
Purpose (user 2026-08-18: "I don't understand why we stopped seed B, it had
promising results — give it the stage-2 treatment too"): seed B was parked at
FOM 0.70045 / T 0.9460 after a stage-1-style run ONLY. Seed A was BEHIND it at
the same point (0.9313) and overtook it only once stage-2 opened corrugation
and cavity width. So "A beat B" was never a fair comparison — B never got a
refinement phase.

MEASURED reason to expect a real gain (seedB eval 17 vs seedA winner):
    corr dip     234.1 nm   vs 282.6   <- B is far more aggressive
    overshoot    339.9 x13  vs none    <- B carries the payback structure
    rho          0.9901     vs 0.9722
    cavity width 810.0 nm   vs 960.9   <- ★B NEVER USED THIS LEVER (seed 800)
The cavity-width lever produced most of seed A's stage-2 jump and is sigma-flat
(free in width terms). Seed B has it entirely unspent, on top of a distinct,
more aggressive apodization — this is an under-developed basin, not a worse one.

Recipe = the CURRENT best practice (stage-4's), not literally stage-2's: all
191 free with trust_nm, since trust regions make the old freeze-stage trick
unnecessary (freezing was only ever an accidental trust region — skill item 22).

★IDENTICAL constraint to stage-4 on purpose: the same absolute sigma-hat wall
(ceiling 17.795 um) and the same +2% cumulative tripwire vs sigma0 17.493.
Seed B starts at sigma 17.7516, i.e. 0.043 um below the ceiling where seed A's
winner sits AT it. Do NOT tighten the wall to B's own anchor: the two runs are
only comparable under an identical constraint, and comparability is the point.

Dispatch:
    SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=72:00:00 \\
        bash <cluster>/deploy_<cluster>.sh \\
        --lumopt2-design=runners.lumopt2_design.campaign_c325_seedB2
Output -> results/campaign_c325_seedB2/results/lumopt2_c325_seedB2/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import SEEDB_BEST, MEASURED

SIGMA0_UM = 17.493        # program-wide anchor — cumulative band unchanged

# ★Vector IMPORTED from best_designs, never read from results_from_igum/: the
# deploy syncs runners/ only, so a runner that opens a local result folder dies
# at import time on the cluster (job 56027, FileNotFoundError). Runners must be
# self-contained — same rule that put P_BEST inside comb_dip_ab.py.
_B = MEASURED["SEEDB_BEST"]
SEED = tuple(float(v) for v in SEEDB_BEST)
_p0 = np.asarray(SEED, dtype=float)

SPEC = eng.CampaignSpec(
    label="lumopt2_c325_seedB2",
    box_y_um=6.8, box_z_mult=4.14,          # identical numerics to every campaign
    scan_center_nm=round(float(_B["lam_pk_nm"]), 2),
    seed_override=SEED,
    sigma_wall=True,
    sig_anchor={"sigma": float(_B["sigma_um"]),      # MEASURED on the seed row
                "elong": float(2.0 * _p0[eng.SL_SHIFT].sum()),
                "rho": float(_p0[eng.SL_CORR].mean() / 325.0),
                "wcav": float(_p0[eng.I_CAV])},
    # wcav radius 120 (not stage-4's 30): seed B sits at 810 nm and seed A's
    # winning move ran 810 -> 960.9, so a 30 nm cap would fence B out of the
    # region that won. Safe to widen ONLY here — cavity width is the one block
    # measured sigma-NEUTRAL (slope ~0.01 um/nm), so a big step cannot blow the
    # width; the other blocks keep the tight radii that stop the blow-outs.
    # ★EFFECTIVE radius is 60 nm, not 120: centering clamps it to the distance
    # to the physical floor (810 - 750), giving bounds (750, 870). Deliberately
    # NOT "fixed" by lowering the program-wide floor — B should find ITS OWN
    # optimum, not chase A's number, and its corr profile differs (overshoot).
    # SATURATION AT 870 IS THE SIGNAL, not a failure: if the winner lands on
    # the upper bound, re-seed a follow-on stage centered there (the trust
    # region then re-centers automatically and walks further).
    trust_nm={"shift": 20.0, "corr": 10.0, "avg": 5.0, "wcav": 120.0},
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
    c = _p0[eng.SL_CORR]
    print(f"seedB2 seed = eval {_B['eval']}: T {_B['t_pk']:.4f} / sigma "
          f"{_B['sigma_um']:.4f} (ratio {_B['sigma_um']/SIGMA0_UM:.4f}) / FOM {_B['fom']:.5f}")
    print(f"  corr {c.min():.1f}..{c.max():.1f} (overshoot teeth {(c>325).sum()}), "
          f"rho {c.mean()/325:.4f}, wcav {_p0[eng.I_CAV]:.1f} nm, center {SPEC.scan_center_nm} nm")
    print(f"  penalty at seed = {float(pen(_p0)):.6f} (must be 0); "
          f"wcav bounds {tuple(np.round(b[eng.I_CAV],1))}; "
          f"headroom to wall ceiling {eng.SIG_CEIL_UM - _B['sigma_um']:.3f} um")
