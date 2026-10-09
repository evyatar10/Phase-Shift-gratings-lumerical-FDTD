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
from runners.lumopt2_design.best_designs import TE_S1_N98_EVAL3, TE_S1_N98_T9711, TE_S2_N98_T9739

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
         max_feval=24,                 # extended 2026-10-08 22:15: fix validated (ratios 0.64, 0.37)
         scan=s2.SCAN_CENTER_NM, two_kl_floor=None,
         port=(1.0, 0.0), field=s2.ADJ_FIX_FIELD,
         fwhm0=s2.FWHM0_UM, softw0=s2.SOFTW0_UM, perp=True),
    # task 5 (2026-10-08 ~14:00): S1 from its converging best (0.97112) with the
    # innermost-corrugation floor lifted 100 -> 0 nm (tooth 1 sat ON the 100 nm
    # bound with gT pushing down; offline QP on the 172603_0 it0 gradient:
    # predicted dFOM/step 0.00048 -> 0.0039, i.e. the optimizer wants the
    # cavity-side taper S2 already has) + moderate shift weighting with a 3 nm
    # per-step shift trust (172159 showed big shift steps break the width model).
    dict(base=s1.SPEC_V3, label="lumopt2_te_s1_n98_c0", start=TE_S1_N98_T9711,
         scan=1561.04, two_kl_floor=3.0, corr_min=0.0, trust={"shift": 3.0},
         port=s1.ADJ_FIX_PORT, field=s1.ADJ_FIX_FIELD,
         fwhm0=s1.FWHM0_UM, softw0=s1.SOFTW0_UM, metric={"shift": 175.0},
         max_feval=20),
    # task 6 (2026-10-09 ~02:00): S2 A/B — perp fix + shifts weighted like S2's
    # corrugations (half-range 375) with a 3 nm per-step shift trust (the S1 c0
    # setting: ratios 0.67/0.94; 15 nm broke the width model on 172159). Starts
    # from the perp lane's best (ev4, 0.97386) so no steps are repeated.
    dict(base=s2.SPEC_V3, label="lumopt2_te_s2_n98_perp_sw", start=TE_S2_N98_T9739,
         scan=1560.42, two_kl_floor=None, trust={"shift": 3.0, "avg": 25.0},
         port=(1.0, 0.0), field=s2.ADJ_FIX_FIELD,
         fwhm0=s2.FWHM0_UM, softw0=s2.SOFTW0_UM, perp=True,
         metric={"shift": 375.0}, max_feval=24),
    # task 7 (2026-10-09 ~02:10, user: "do the tooth shifts play the main
    # role?"): S2 SHIFTS ONLY — corr/avg/cavity frozen (trust 1e-3 nm = sliver),
    # shifts free within a 3 nm per-step trust, perp fix on. Isolates the shift
    # contribution to T at fixed width, from the same start as task 6.
    dict(base=s2.SPEC_V3, label="lumopt2_te_s2_n98_shiftonly", start=TE_S2_N98_T9739,
         scan=1560.42, two_kl_floor=None,
         trust={"shift": 3.0, "corr": 1e-3, "avg": 1e-3, "wcav": 1e-3},
         port=(1.0, 0.0), field=s2.ADJ_FIX_FIELD,
         fwhm0=s2.FWHM0_UM, softw0=s2.SOFTW0_UM, perp=True,
         metric={"shift": 375.0}, max_feval=16),
    # task 8 (2026-10-09 ~09:00): S2 with ALL 98 periods free (n_free 98, zero
    # frozen). MEASURED: the 60-tooth S2 landscape is flat at 0.974 (perp lane
    # ratios 0.02/-0.01; the best moved only inner corr, nothing at a bound) —
    # Itai's 60 teeth are a local optimum; the outer 38 bulk periods are the
    # untouched lever. Start = perp best padded with bulk (corr 494.2/avg 1000).
    # Region x_half 30.6 -> 49.2 um (x1.61 DFT memory, EXPECTED ~39 GB) -> H200.
    dict(base=s2.SPEC_V3, label="lumopt2_te_s2_n98_free98", start="free98",
         n_free=98, scan=1560.42, two_kl_floor=None,
         port=(1.0, 0.0), field=s2.ADJ_FIX_FIELD,
         fwhm0=s2.FWHM0_UM, softw0=s2.SOFTW0_UM, perp=True, max_feval=12),
]


def free98_start():
    """TE_S2_N98_T9739 (60 inner teeth) embedded in the 98-free layout; outer 38
    teeth at S2's bulk (corr 494.2 / avg 1000 / shift 0), cavity from the best."""
    base = dataclasses.replace(s2.SPEC_V3, n_free=98, seed_override=None,
                               corr_seed_nm=None, avg_seed_nm=None)
    p = eng.seed_params(base)
    L, L60 = eng.layout(98), eng.layout(60)
    for sl, sl60 in ((L.SL_CORR, L60.SL_CORR), (L.SL_AVG, L60.SL_AVG), (L.SL_SHIFT, L60.SL_SHIFT)):
        p[sl.start:sl.start + 60] = TE_S2_N98_T9739[sl60]
    p[L.I_CAV] = TE_S2_N98_T9739[L60.I_CAV]
    return p
N_TASKS = len(LANES)


def lane_spec(task_idx):
    ln = LANES[task_idx]
    start = free98_start() if isinstance(ln["start"], str) else ln["start"]
    spec = dataclasses.replace(
        ln["base"], label=ln["label"], seed_override=tuple(float(v) for v in start),
        n_free=ln.get("n_free", ln["base"].n_free),
        scan_center_nm=ln["scan"], two_kl_floor=ln["two_kl_floor"],
        max_feval=ln.get("max_feval", MAX_FEVAL),   # per-lane extension
        fwhm0_um=ln["fwhm0"], wgp_target_um=ln["fwhm0"],
        wg_anchor={"softw": ln["softw0"], "fwhm": ln["fwhm0"]},
        adj_fix_re=ln["port"][0], adj_fix_im=ln["port"][1],
        wgp_d_halfrange_nm=ln.get("metric"),
        wgp_v3_perp_glam=ln.get("perp", False))
    if "corr_min" in ln:
        spec.corr_min_nm = ln["corr_min"]
    if "trust" in ln:
        spec.trust_nm = dict(ln["trust"])
    spec.adj_fix_field_re, spec.adj_fix_field_im = ln["field"]
    assert spec.n_periods_side == 98
    return spec


SPEC = lane_spec(0)      # deploy's build_sweep_list requires a top-level SPEC


def main(task_idx=0):
    spec = lane_spec(task_idx)
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    best = eng.run_campaign(spec, out_dir)
    print(f"[te-n98 {spec.label}] done: best_fom {best['fom']:.5f}")
