"""TE inverse design at N = 98 periods/side — both seeds, v3 engine, one array.

Study dir: runners/lumopt2_design/ | Created 2026-10-08 | Job(s): see memory
project_te_inverse_design_lane.md
Purpose: raise peak T at fixed spatial envelope FWHM (each seed's own ±2 % band)
at the program's constant length N = 98/side, 60 free/side (user 2026-10-08:
"everything constant at 98, not 80"). The N80 detour (archived runner
runners/archive/te_n80_20261007/) MEASURED that N80 cuts S1's width 4.4 %.

  task 0 = S1 (uniform corr-250), from the N98 campaign best (170253 eval 3:
           t_pk 0.95276 @ 19.385 µm) — same numerics as campaign_te_s1.
  task 1 = S2 (Itai's Nt60 overshoot), from its seed — numerics of campaign_te_s2.

Calibration: S1 port/field C measured at exactly this length/numerics. S2 field
C is its own fit; S2 port C = (1, 0), uncalibrated by decision — its N98 FD fit
used ±2 nm legs that moved a 0.20 nm line ~0.18 nm (non-local), and the v3 QP is
scale-free in gT. Acceptance is on MEASURED T and W; the live calibration gate is
the per-iterate gain ratio (measured ΔFOM / dT_pred in <label>_proj.jsonl).

Dispatch (Athena):
  SBATCH_MEM=256G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00 \
      bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_te_n98
"""
import dataclasses
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s1 as s1
from runners.lumopt2_design import campaign_te_s2 as s2
from runners.lumopt2_design.best_designs import TE_S1_N98_EVAL3

MAX_FEVAL = 8        # persistent forward budget per lane; raise + redeploy to extend

LANES = [
    # windows/recenter/λ-trust/2κL floor are the seed specs' own N98 values
    dict(base=s1.SPEC_V3, label="lumopt2_te_s1_n98", start=TE_S1_N98_EVAL3,
         max_feval=20,                 # extended 2026-10-08: 6/6 accepted, 0.9533->0.9701
         scan=1561.04,                 # the start design's MEASURED λ_pk
         two_kl_floor=3.0,             # 2κL 3.325 at start (spec 3.3 left no room)
         port=s1.ADJ_FIX_PORT, field=s1.ADJ_FIX_FIELD,
         fwhm0=s1.FWHM0_UM, softw0=s1.SOFTW0_UM),
    dict(base=s2.SPEC_V3, label="lumopt2_te_s2_n98", start=s2.SEED,
         scan=s2.SCAN_CENTER_NM,
         two_kl_floor=None,            # 2κL 5.11 > 3.5
         port=(1.0, 0.0), field=s2.ADJ_FIX_FIELD,
         fwhm0=s2.FWHM0_UM, softw0=s2.SOFTW0_UM),
    # task 2 (user-approved 2026-10-08): S1 with the shifts weighted like the
    # corrugations in the QP metric (legacy D starved shifts ~545x; 57/60 sat at
    # 0). Offline on the stored 170253 it3 gradient: predicted dFOM/step
    # +0.0091 -> +0.0169 (EXPECTED, 1.85x), same width/lambda rows. Replaces task 0.
    dict(base=s1.SPEC_V3, label="lumopt2_te_s1_n98_shiftw", start=TE_S1_N98_EVAL3,
         scan=1561.04, two_kl_floor=3.0,
         port=s1.ADJ_FIX_PORT, field=s1.ADJ_FIX_FIELD,
         fwhm0=s1.FWHM0_UM, softw0=s1.SOFTW0_UM,
         metric={"shift": 175.0}),
    # task 3 (2026-10-08 ~03:00): S2 with shifts weighted like ITS corrugations
    # (corr 0..750 -> half-range 375; legacy D starved S2 shifts ~2400x). Runs
    # alongside task 1 (legacy) as the S2 A/B, after S1's A/B step 1 MEASURED
    # 2.35x the legacy gain at gain ratio 0.87 (172159_2 vs 172063_0).
    dict(base=s2.SPEC_V3, label="lumopt2_te_s2_n98_shiftw", start=s2.SEED,
         scan=s2.SCAN_CENTER_NM, two_kl_floor=None,
         port=(1.0, 0.0), field=s2.ADJ_FIX_FIELD,
         fwhm0=s2.FWHM0_UM, softw0=s2.SOFTW0_UM,
         metric={"shift": 375.0}),
    # task 4 (2026-10-08 ~09:40): S2 with gλ projected out of gT/gW. MEASURED on
    # 172063_1 it0/it1: cos(gT, gλ) = +1.0000, |gT| 1.5 (S1: 0.0012) -> predicted
    # gains 0.058/0.116 vs measured 3e-4: a narrow-line phase leak. After the
    # projection |gT| = 0.00097/0.00093 (stable, S1-sized). Legacy metric.
    dict(base=s2.SPEC_V3, label="lumopt2_te_s2_n98_perp", start=s2.SEED,
         scan=s2.SCAN_CENTER_NM, two_kl_floor=None,
         port=(1.0, 0.0), field=s2.ADJ_FIX_FIELD,
         fwhm0=s2.FWHM0_UM, softw0=s2.SOFTW0_UM, perp=True),
]
N_TASKS = len(LANES)


def lane_spec(task_idx):
    ln = LANES[task_idx]
    spec = dataclasses.replace(
        ln["base"], label=ln["label"], seed_override=tuple(ln["start"]),
        scan_center_nm=ln["scan"], two_kl_floor=ln["two_kl_floor"],
        max_feval=ln.get("max_feval", MAX_FEVAL),   # per-lane extension
        fwhm0_um=ln["fwhm0"], wgp_target_um=ln["fwhm0"],
        wg_anchor={"softw": ln["softw0"], "fwhm": ln["fwhm0"]},
        adj_fix_re=ln["port"][0], adj_fix_im=ln["port"][1],
        wgp_d_halfrange_nm=ln.get("metric"),
        wgp_v3_perp_glam=ln.get("perp", False))
    spec.adj_fix_field_re, spec.adj_fix_field_im = ln["field"]
    assert spec.n_periods_side == 98
    return spec


SPEC = lane_spec(0)      # deploy's build_sweep_list requires a top-level SPEC


def main(task_idx=0):
    spec = lane_spec(task_idx)
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    best = eng.run_campaign(spec, out_dir)
    print(f"[te-n98 {spec.label}] done: best_fom {best['fom']:.5f}")
