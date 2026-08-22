"""lumopt2 campaign, SEED A: uniform corr-325 + measured winner comb. ATHENA.

Study dir: runners/lumopt2_design/   |   Created 2026-08-13   |   Job(s): TBD
Purpose: the main inverse-design run — 25 free (corr, avg, shift) teeth/side
(index-symmetric across arms) + 57 free comb radii/positions + shared d, from
the measured seed, L-BFGS-B. See lumopt2_design.py for the cost function.

PREREQUISITES (hard): gates A0 and B0-B4 PASSED; SIGMA0_UM filled from the
B2b canary printout; box fields updated if A0 chose 8.0/8.8.

Budget: ~60 iters x (1 fwd + 1 adj) x ~25 min  ≈  2.8 GPU-days. No resume in
lumopt2 — the engine logs params every eval and restarts from best on a trip.

Dispatch (single task; long stateful run — Athena non-preemptible per CLAUDE.md):
    SBATCH_MEM=160G ARRAY_TIME=168:00:00 bash athena/deploy_athena.sh \
        --lumopt2-design=runners.lumopt2_design.campaign_c325_seedA
Output -> results/lumopt2_c325_seedA/ (evals jsonl, best json, per-iter .fsp).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from runners.lumopt2_design import lumopt2_design as eng

SIGMA0_UM = 17.493    # MEASURED gate B2b at PVA (job 132654): seed-comb sigma at the
                      #    B2b canary (same estimator, same N — ratio rule)

SPEC = eng.CampaignSpec(
    label="lumopt2_c325_seedA",
    box_y_um=6.8, box_z_mult=4.14,       # z=6.8 CONFIRMED by gate A0 (job 132623)
    max_iter=60, max_feval=100,
    # ★Gradient fix (2026-08-16, root cause found + FD-validated at TWO
    # operating points, 14 params total): the adjoint's projection phase is
    # off by a UNIVERSAL 6.7° (measured 6.71°/6.67° at the two points);
    # multiplying the scaled adjoint fields by C corrects it. With the best
    # within-point C: all 14 signs correct (incl. the ×29-wrong cavity),
    # per-param magnitudes ×0.84-1.67. C's amplitude varies ~×1.5 between
    # points; the value here = geometric mean of the two measured amplitudes
    # (0.874, 1.292) at the universal phase — worst-case global bias ×1.22.
    # bc_patch/colocate measured ineffective (≤0.04% / never engaged), off.
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,
)


def main(task_idx=0):
    assert SIGMA0_UM is not None, "run gate B2b first and fill SIGMA0_UM"
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    eng.run_campaign(SPEC, out_dir, sigma0_um=SIGMA0_UM)


if __name__ == "__main__":
    print(f"seed A: uniform corr {eng.CORR_NM} + comb Λ{eng.COMB_LAM_NM:.0f}/"
          f"δx{eng.COMB_DX_NM:.0f}/r{eng.COMB_R_NM:.0f}/d{eng.COMB_D_NM:.0f}, "
          f"{eng.N_PARAMS} params, box y{SPEC.box_y_um}/zmult {SPEC.box_z_mult}")
