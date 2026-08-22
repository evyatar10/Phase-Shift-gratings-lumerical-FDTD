"""V2 campaign — EXACT in-loop FWHM gradient, origin + see-saw seed, comb FIXED.

SECOND BASIN (the running v2proj campaign explores the re-trimmed-best basin
with the projection architecture; this one explores the origin+see-saw basin
with the exact width gradient — two independent basins, the program's own
convergence-evidence standard).

Study dir: runners/lumopt2_design/  |  Created 2026-08-22  |  Job(s): TBD
The first campaign against the CORRECT spec observable (V2_FWHM_PLAN.md):
FOM = windowed softmax T − AL band penalty on the anchor-mapped FWHM
prediction (softW carrier, weighted FieldRegion adjoint on the CPU lane);
authority = measured fwhm_env (WidthTrip +2%/−5% + restart filter).

User decisions wired in (plan §9 + 2026-08-22):
- comb FIXED at the winner (free_comb=False; no count/binary params);
- single seed = origin + INNER SEE-SAW δ=+20 nm (HANDOFF §6e: the measured
  −31% loss lever the optimizer provably cannot discover from a smooth
  seed: corr [345,305,325...], avg [810,790,800...]);
- anchors = W2-measured (job 135971 t10): fwhm0 17.713551, softW 18.160689;
- W4 folded into the start: first ~3 evals are the live gate.

Dispatch AFTER the C_field fit lands (fit_c_field.py; set ADJ_FIX below):
  SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \\
    bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_v2_seesaw
Resume: re-dispatch the same module (cold-start resume from the eval log).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng

# ── W3 outputs — FILL FROM THE C_FIELD FIT BEFORE DISPATCH ─────────────────
ADJ_FIX_FIELD = None      # e.g. (0.87, 0.10); None = NOT FITTED YET → refuse
ADJ_FIX_PORT = (1.0561, 0.1239)   # the validated port C (skill item 6)

SEESAW_DELTA_NM = 20.0
FWHM0_UM = 17.713551      # MEASURED, W2 (135971 t10), v2 window
SOFTW0_UM = 18.160689     # MEASURED, same eval (single-λ twin's own sample)

_corr = [eng.CORR_NM] * eng.N_FREE
_avg = [eng.AVG_W_NM] * eng.N_FREE
_corr[0] += SEESAW_DELTA_NM; _corr[1] -= SEESAW_DELTA_NM
_avg[0] += SEESAW_DELTA_NM / 2.0; _avg[1] -= SEESAW_DELTA_NM / 2.0

SPEC = eng.CampaignSpec(
    label="lumopt2_v2_seesaw",
    scan_width_nm=10.0, n_wl_points=501,          # v2 window (plan §5a)
    corr_seed_nm=tuple(_corr),
    free_comb=False,                              # user: comb fixed at winner
    width_grad=True,
    fwhm0_um=FWHM0_UM,
    wg_anchor={"softw": SOFTW0_UM, "fwhm": FWHM0_UM},
    adj_phase_fix=True, adj_fix_re=ADJ_FIX_PORT[0], adj_fix_im=ADJ_FIX_PORT[1],
    # ★2026-08-22: the width adjoint runs ON GPU via a standard import source
    # (job 136108: 3,133 s vs 8.7-12.1 h on CPU — the FieldRegion object was
    # the blocker, not the GPU). This is what makes the in-loop EXACT FWHM
    # gradient affordable: ~2.6 h/iteration instead of ~10 h.
    wg_source="import", wg_adj_resource="GPU",
    trust_nm={"corr": 15.0, "avg": 10.0, "shift": 15.0, "wcav": 15.0},
    max_iter=40, max_feval=70,
)
N_TASKS = 1


def main(task_idx=0):
    assert ADJ_FIX_FIELD is not None, \
        "C_field not fitted — run fit_c_field on the W3 vectors first"
    SPEC.adj_fix_field_re, SPEC.adj_fix_field_im = ADJ_FIX_FIELD
    # avg seed deviates from the module default → thread it through the seed
    p0 = eng.seed_params(SPEC)
    p0[eng.SL_AVG] = np.asarray(_avg, dtype=float)
    SPEC.seed_override = tuple(p0)
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[campaign_v2_seesaw] done: best_fom {best['fom']:.5f} "
          f"(delivered design = width-filtered log, never this number)")


if __name__ == "__main__":
    print("v2 campaign; dispatch only after ADJ_FIX_FIELD is set from the fit")
