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

★s2 RAN AND WAS CANCELLED (Athena 136468, 2026-08-24, 17 h, 12 evals). It did
NOT converge and its 0.9165 is NOT the no-shift ceiling — it ran under the
RANK-1 wall, which priced all 25 corrugations by their MEAN and so handed
L-BFGS-B one identical gradient across the block. Every profile-shaping move
(the see-saw: inner teeth down, outer up) was therefore invisible to it, and
the only width lever it could see was "spend width to buy T": 3 of its last 6
probes came back out of band, and the in-band ones jammed against the 18.713
ceiling at W 18.60. Meanwhile a two-number hand ladder reached T 0.93836 at
W 18.331 — narrower AND higher — in ONE solve. That is a wrong fixed point,
not slow convergence, which is why s2 was stopped rather than extended.

★s3 (2026-08-24) carries the corrected wall: per-tooth corrugation price
(fw_tooth_w), the measured elongation threshold curve (fw_curve, inert here
since shifts are frozen at 0 but kept so the spec matches its siblings), and
the saturated hinge (fw_pen_cap). New label because FOM values are NOT
comparable across a penalty change — s2's log stays as the record of the old
regime and must never be resumed into s3.

SEED CHOICE, if you dispatch this: the module default is the strictly uniform
seed (T 0.9012), which makes s3 a clean re-run of the same experiment under
correct prices. The alternative is seeding from the see-saw d090 design
(T 0.93836 / W 18.331 / shifts already 0, from seesaw_ladder.rung_params(90,68))
so the campaign starts at the best KNOWN no-shift point and spends its whole
budget trying to beat it rather than rediscovering it. That is the stronger
test of "is 0.938 really the ceiling"; it is left as a deliberate choice
rather than baked in, because the uniform seed is what makes s3 comparable to
basin 2.

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
    label="lumopt2_v2_noshift_s3",
    scan_width_nm=10.0, n_wl_points=501,
    region_dx_nm=eng.DX_PITCHLOCK_NM,             # pitch-locked (plan §24)
    scan_center_nm=1564.614,                      # MEASURED mx_origin resonance
    free_comb=False,                              # comb fixed at the winner
    freeze_shifts=True,                           # ★the whole point
    rho_band=False,
    fwhm0_um=FWHM0_UM,                            # 18.3460, MEASURED mx_origin
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,
    fwhm_wall=True,
    fw_tooth_w=eng.FW_TOOTH_W,   # ★the s2 fix: per-tooth corrugation price, so
                                 # the see-saw direction is no longer unpriced
    fw_curve=True,               # inert with shifts frozen; kept for parity
    fw_pen_cap=2.0,              # saturated hinge (the 136640 lnsrch fix)
    fw_anchor={"fwhm": FWHM0_UM, "mcorr": eng.CORR_NM, "elong": 0.0,
               "corr_vec": (eng.CORR_NM,) * eng.N_FREE},
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
