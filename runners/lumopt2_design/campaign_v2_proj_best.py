"""Best-seeded corrected campaign (b1) — the IGUM parallel lane.

Study: lumopt2_v2_proj_b1 | dispatched 2026-08-28 (job IDs in HANDOFF) |
Purpose: can the λ-chain-corrected projection improve BEST_T9636 further at
held width? The parallel lane to c1 (137960, Athena), which answers the
honest-from-uniform question. User-approved 2026-08-28 ("seed from our best
design"), cluster ruling: IGUM.

Everything inherits from campaign_v2_proj.SPEC (c1) except:
- seed_override = BEST_T9636 (191-vector from best_designs.py). param_bounds
  derives every freeze/trust box from the seed, so the frozen comb's sliver
  bounds centre on BEST's own evolved comb values — the 133395/predispatch
  trap self-resolves. trust_nm recentres the 30 nm shift box on BEST's
  shifts (e≈130.6).
- wg_dwdlam_fit=True: online re-fit of dW/dλ from this run's own accepted
  (λ_pk, W) points (engages at n≥5, span≥0.5 nm) — the ride test measured
  the ±20% coefficient error as the dominant width-leak residual.
- label generation-tagged b1 (the label IS the resume key).
Initial wg_anchor is the uniform-origin pair; ~0.0 µm off BEST's fwhm_env
18.35309 and re-anchored on the first restart — accepted.

★MEASURED LESSON (2026-08-29, b1 evals 0-2): a near-converged seed must NOT
inherit c1's ride level. At BEST the shadow price lam=0.062 (∇T aligned with
∇W — T grows only by widening), so the width-blind CLIMB to c1's
W_tgt=18.613 bought +0.00097 T for +0.272 µm (0.0036 T/µm, 30× below the
uniform rate). Any FUTURE best-seeded lane (b2+) sets
wgp_target_um = <seed fwhm_env> (18.3545 for BEST_T9636) so the ride
(gW·d=0) engages from iterate 0 — the held-width question is the only one a
converged seed can answer. Do not edit the running b1's spec.
"""
import dataclasses
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9636
from runners.lumopt2_design.campaign_v2_proj import ADJ_FIX_FIELD, SPEC as C1

SPEC = dataclasses.replace(
    C1,
    label="lumopt2_v2_proj_b1",
    seed_override=BEST_T9636,
    wg_dwdlam_fit=True,
)

N_TASKS = 1


def main(task_idx=0):
    assert ADJ_FIX_FIELD is not None, "C_field not fitted — see campaign_v2_proj"
    SPEC.adj_fix_field_re, SPEC.adj_fix_field_im = ADJ_FIX_FIELD
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    best = eng.run_campaign(SPEC, out_dir)
    print(f"[v2proj-b1] done: best_fom {best['fom']:.5f} — best-seeded lane; "
          f"read {SPEC.label}_proj.jsonl for phase/dlam_pred/dwdlam_fit")
