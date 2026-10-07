"""TE inverse design at N = 80 periods/side — both seeds, v3 engine, one array.

Study dir: runners/lumopt2_design/ | Created 2026-10-07 | Job(s): see memory
project_te_inverse_design_lane.md
Purpose: continue the TE lane on the user's practical length N=80/side (user
2026-10-07: "we can do 80 periods instead of 98"; 60 free periods/side kept, so
20 frozen outer periods remain). Goal unchanged: raise peak T at fixed spatial
envelope FWHM (each seed's own ±2 % band).

  task 0 = S1 (uniform corr-250 family), starting FROM the N98 campaign's best
           design (170253 eval 3: t_pk 0.95276 @ 19.385 µm, N98) — same 60 free
           teeth, outer periods 98 → 80. Same numerics as campaign_te_s1.
  task 1 = S2 (Itai's Nt60 overshoot), starting from its seed.

Identity vs the N98 lanes: ONLY n_periods_side differs (+ windows sized for
the lower Q). Every N98 eval row is a different device ⇒ new labels, fresh
logs; iterate 0 is the necessary forward whose fields feed the first gradient.

Calibration (C factors) — decision 2026-10-07, see CLAUDE.md §9 main-goal rule:
  S1: N98 port/field C reused. The C absorbs LOCAL tooth-wall gradient errors
      at the 60 free teeth, which are unchanged; the 18 removed outer periods
      carry no design variable.
  S2: field C = its own N98 fit; port C = (1, 0) — UNCALIBRATED. The N98 S2 FD
      fit failed, but its ±2 nm legs moved a 0.20 nm-wide resonance ~0.18 nm
      (a non-local FD) and its small-step legs sat at the 1.3e-5 noise floor,
      so that failure does not show the adjoint is wrong. The v3 QP is
      scale-free in gT (τ = cap/max|D·gT|), only the direction matters.
  Both: acceptance is on MEASURED T and W, so a gradient error costs rejected
  steps, not wrong results. Live gate = per-iterate gain ratio (measured ΔFOM /
  predicted ΔFOM in <label>_proj.jsonl).

Dispatch (Athena, both tasks in parallel):
  SBATCH_MEM=256G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \
      bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_te_n80
"""
import dataclasses
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s1 as s1
from runners.lumopt2_design import campaign_te_s2 as s2
from runners.lumopt2_design.best_designs import TE_S1_N98_EVAL3 as S1_BEST_N98

N_SIDE = 80
MAX_FEVAL = 8        # persistent forward budget per lane; raise + redeploy to extend

LANES = [
    # (base v3 spec, label, start vector, scan centre nm, window nm, recenter nm,
    #  λ trust nm/step, 2κL floor, port C, field C)
    dict(base=s1.SPEC_V3, label="lumopt2_te_s1_n80", start=S1_BEST_N98,
         # N98 best sat at 1561.044; N98→N80 moved S2 by +0.012 nm (IGUM 101434)
         scan=1561.04, width=10.0, recenter=2.0, dlam=0.25,
         # 2κL at N80 on this vector = 2.71 (DERIVED, two_kappa_L) — below the
         # 3.2 surrogate floor; N is the USER's spec choice here, not a surrogate.
         two_kl_floor=2.5,
         port=s1.ADJ_FIX_PORT, field=s1.ADJ_FIX_FIELD,
         fwhm0=s1.FWHM0_UM, softw0=s1.SOFTW0_UM),
    dict(base=s2.SPEC_V3, label="lumopt2_te_s2_n80", start=s2.SEED,
         # S2 N80 linewidth 0.575 nm (MEASURED IGUM 101434 task 1, vs 0.207 at
         # N98) ⇒ window 4 nm @ 8 pm = 72 pts/FWHM; recenter 0.8; λ trust ¼·0.575
         scan=1560.42, width=4.0, recenter=0.8, dlam=0.14,
         two_kl_floor=None,          # 2κL ≈ 4.1 > 3.5 default rule
         port=(1.0, 0.0), field=s2.ADJ_FIX_FIELD,
         fwhm0=s2.FWHM0_UM, softw0=s2.SOFTW0_UM),
]
N_TASKS = len(LANES)


def lane_spec(task_idx):
    ln = LANES[task_idx]
    spec = dataclasses.replace(
        ln["base"], label=ln["label"], n_periods_side=N_SIDE,
        seed_override=tuple(ln["start"]),
        scan_center_nm=ln["scan"], scan_width_nm=ln["width"],
        recenter_nm=ln["recenter"], wgp_v3_dlam_nm=ln["dlam"],
        two_kl_floor=ln["two_kl_floor"], max_feval=MAX_FEVAL,
        # width band unchanged: each seed's own N98 anchor ±2 %
        fwhm0_um=ln["fwhm0"], wgp_target_um=ln["fwhm0"],
        wg_anchor={"softw": ln["softw0"], "fwhm": ln["fwhm0"]},
        adj_fix_re=ln["port"][0], adj_fix_im=ln["port"][1])
    spec.adj_fix_field_re, spec.adj_fix_field_im = ln["field"]
    return spec


SPEC = lane_spec(0)      # deploy's build_sweep_list requires a top-level SPEC


def main(task_idx=0):
    spec = lane_spec(task_idx)
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    best = eng.run_campaign(spec, out_dir)
    print(f"[te-n80 {spec.label}] done: best_fom {best['fom']:.5f}")
