"""Best-seeded RIDE-FROM-START campaign (b2) — the Athena GPU fast lane.

Study: lumopt2_v2_proj_b2 | created 2026-08-30 (job ID in HANDOFF/memory) |
Purpose: push T above BEST_T9636's 0.96361 AT THE SPEC WIDTH, at GPU pace
(~4 h/iterate vs b1's measured ~21 h on the IGUM CPU lane). Direct response
to the user's pace verdict 2026-08-30 ("iterations way too slow").

Differences vs b1 (each from a MEASURED lesson):
- wgp_target_um = 18.3545 = BEST's own fwhm_env  ⇒ the seed sits mid-band
  and the RIDE (gW·d=0) engages from iterate 0. b1's inherited ceiling
  target made it climb width-blind: +0.272 um bought +0.00097 T
  (0.0036 T/um, 30x below the uniform rate) — never again from a
  near-converged seed.
- wgp_step = 0.5 (2x C1's 0.25): b1 accepted every early step with zero
  filter rejections — the steps were too timid, and a reject costs only one
  bounded eval. The projection's shadow price at BEST is 0.062 ⇒ riding
  only forfeits ~6% of the T gradient.
- Athena GPU lane (this file exists so the IGUM-deployed b1 module stays
  untouched while 64279 runs).
Label b2 is fresh = fresh resume log (the label IS the resume key).
"""
import dataclasses
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9636
from runners.lumopt2_design.campaign_v2_proj import ADJ_FIX_FIELD, SPEC as C1

SPEC = dataclasses.replace(
    C1,
    label="lumopt2_v2_proj_b2",
    seed_override=BEST_T9636,
    wg_dwdlam_fit=True,
    wgp_target_um=18.3545,
    wgp_step=0.5,
)

N_TASKS = 1


def main(task_idx=0):
    assert ADJ_FIX_FIELD is not None, "C_field not fitted — see campaign_v2_proj"
    SPEC.adj_fix_field_re, SPEC.adj_fix_field_im = ADJ_FIX_FIELD
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2proj-b2] done: best_fom {best['fom']:.5f} — GPU best-seed ride "
          f"lane; read {SPEC.label}_proj.jsonl (expect phase=ride from it 0)")
