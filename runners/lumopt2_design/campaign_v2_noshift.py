"""V2 third basin — UNIFORM seed with the TOOTH SHIFTS FROZEN AT ZERO.

Study dir: runners/lumopt2_design/  |  Created 2026-08-23  |  Job(s): TBD
User question 2026-08-23: "are the tooth shifts actually necessary, or can
corrugation alone do what they aim to do while keeping the mode width?"
This campaign answers it BY CONSTRUCTION: identical to campaign_v2_uniform
(same uniform seed, same anchors, same pitch-locked mesh, comb fixed) except
`freeze_shifts=True` pins all 25 shifts at the seed value 0. Whatever T it
reaches at a given width IS the no-shift ceiling, with corrugation (per
tooth, free of any average lock), cavity width and the rest free to
compensate. Compare against basin 2 (shifts free) at MATCHED width — never
across widths (CLAUDE.md §2).

Why this and not a see-saw seed (recorded so it is not relitigated): the
see-saw direction has an ordinary nonzero gradient (corrugation up on one
tooth, down on its neighbour), so basin 2 can reach it if it pays; an
earlier claim in this repo that a smooth seed "provably cannot discover" it
was WRONG and is retracted. The shift question, by contrast, cannot be
answered by watching basin 2 alone — shifts sit ON their lower bound at 0,
so "they did not move" is ambiguous without this frozen control.

Reading the result (both campaigns must be re-trimmed to equal width first):
  noshift T >= shifts-free T  ⇒ shifts are dead weight, drop 25 params.
  noshift T <  shifts-free T  ⇒ shifts earn their place; the gap is their
                               true value at fixed width.

Dispatch:  SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \\
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_v2_noshift
Resume: re-dispatch the same module (cold-start resume; REQUEUE-safe).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_uniform import FWHM0_UM

SPEC = eng.CampaignSpec(
    label="lumopt2_v2_noshift_s2",
    scan_width_nm=10.0, n_wl_points=501,
    region_dx_nm=eng.DX_PITCHLOCK_NM,             # pitch-locked (plan §24)
    scan_center_nm=1564.614,                      # MEASURED mx_origin resonance
    free_comb=False,                              # comb fixed at the winner
    freeze_shifts=True,                           # ★the whole point
    rho_band=False,
    fwhm0_um=FWHM0_UM,                            # 18.3460, MEASURED mx_origin
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,
    fwhm_wall=True,
    fw_anchor={"fwhm": FWHM0_UM, "mcorr": eng.CORR_NM, "elong": 0.0},
    max_iter=60, max_feval=100,                   # same budgets as basin 2
)
N_TASKS = 1


def main(task_idx=0):
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2noshift] done: best_fom {best['fom']:.5f} — compare to basin 2 "
          f"ONLY after both are re-trimmed to equal width")


if __name__ == "__main__":
    print("v2 basin-3: uniform seed, shifts frozen at 0; dispatch via deploy")
