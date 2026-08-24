"""See-saw ladder: inner teeth DOWN, outer teeth UP, at constant mode width.

Study dir: runners/lumopt2_design/   |   Created 2026-08-24   |   Job(s): TBD
Follows directly from the corrugation ladder (IGUM 61901 + 61979), which
measured, on this same uniform corr-325 seed:

    lower inner-8   +0.0235 T per um of widening   (the cheapest way to BUY T)
    raise outer-8   -0.0052 T per um of narrowing  (the cheapest way to PAY BACK)
    raise uniform    0.0120 | lower uniform 0.0107 | elongation 0.01056

So the productive move at CONSTANT width is a SEE-SAW: buy with the inner
teeth, pay back with the outer ones. Predicted net **+0.018 T per um cycled**,
against +0.0053 for the tooth-shift route — the 3.5x that made shifts look
unnecessary. This ladder tests that prediction directly instead of trusting the
rates to add.

CONSTRUCTION. Inner 8 teeth are lowered by a FIXED 30 nm (325 -> 295); the
outer 17 are raised by b, swept. From the measured rates the width should
balance near b ~ 130, so the three rungs BRACKET the balance point:

  task 0  b =  90   outer 415   predicted W ~ 18.64 (still wide)
  task 1  b = 130   outer 455   predicted W ~ 18.46
  task 2  b = 174   outer 499   predicted W ~ 18.25 (slightly narrow)

Predictions are EXPECTED values from linearly extrapolated single-point rates —
the outer-raise rate was measured at 8 teeth x 60 nm and is here applied at 17
teeth x up to 174 nm, i.e. ~6x more. Whether it holds is precisely what this
measures; b=174 also probes the SATURATION ceiling, since 499 nm is at the
corrugation bound and no larger payback exists.

READING IT. Compare each rung against the stored seed (18.3452 um / T 0.9012,
136466 ev1 — NOT re-run, CLAUDE.md §6):
  * a rung landing in the band [17.98, 18.71] with T clearly above 0.9012 is a
    MEASURED constant-width improvement, and confirms the see-saw rule;
  * if all three land wide, the outer-raise rate does NOT extrapolate and the
    payback capacity is smaller than 0.8 um — the see-saw saturates early and
    only a fraction of the +0.018 T/um is collectable;
  * T rising with b at fixed inner-lowering would mean the outer teeth are not
    merely inert but actively helpful, which nothing so far suggests.

lambda: raising mean corrugation BLUE-shifts the resonance -0.008 nm per nm of
mean corr (MEASURED: u345 1564.438 and u305 1564.758 against the seed's
1564.614). The largest rung moves the mean by ~+109 nm => ~-0.87 nm, well
inside the +-5 nm window, so the seed's scan centre is kept.

Dispatch:
    SBATCH_MEM=160G LUMOPT2_TIME=02:00:00 \\
        bash igum/deploy_igum.sh --lumopt2-design=runners.lumopt2_design.seesaw_ladder \\
        --max-concurrent=2
"""

import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_uniform import SPEC as UNIFORM_SPEC

N_INNER = 8           # teeth lowered (indices 0..7; index 0 = innermost)
INNER_DROP_NM = 30.0  # legacy: fixed buy-side amplitude for the first two passes
# ★REVISED 2026-08-24 after the first pass (IGUM 61993) — the first sweep
# [90, 130, 174] OVERSHOT catastrophically: predicted W 18.64/18.46, MEASURED
# 16.6207/15.7648, i.e. 2.0-2.7 um too narrow. Cause: I applied the out385
# single-point rate (8 teeth x 60 nm -> 0.1285 um) at 17 teeth x 90-130 nm,
# ~6x more amplitude — the same extrapolate-a-secant-past-its-range error this
# programme spent the night diagnosing in FW_A_ELONG and FW_A_MCORR.
# MEASURED outer-raise rate from those two rungs: -0.0214 um per nm of outer
# raise (b090 16.6207 -> b130 15.7648 over +40 nm), ~6x the assumed value, so
# the balance point sits near outer +9 nm, not +130. This sweep brackets it.
# Kept for the record: b090 (outer 415) reached T 0.91584 at W 16.6207 —
# HIGHER T and NARROWER than the seed simultaneously, so the see-saw itself is
# real; only my payback amplitude was wrong.
# ★PASS 3 (2026-08-24) — AMPLITUDE SWEEP AT CONSTANT WIDTH.
# Passes 1-2 fixed the inner drop at 30 nm and swept the outer raise; that
# located the balance point and, more importantly, measured both legs cleanly:
#   inner-lower (buy side): 0.02335 um per nm, LINEAR over -30 and -60
#       (b000 19.0458, in265 19.7490, seed 18.3452 — two points, same slope)
#   outer-raise (pay side): ~0.031 um per nm below +30 and FREE in T
#       (+0 -> +30 narrows 0.926 um while T moves +0.00002), then costs
#       0.0026 T/um to +90 and 0.0084 T/um to +130
# ⇒ net ~+0.021 T per um cycled while the outer leg stays below +30.
# Balance from those slopes: b ~ 0.75 * d. This sweep walks the AMPLITUDE d
# with b held at that ratio, so every rung should sit near 18.35 and the only
# question is how far the see-saw keeps paying before it saturates.
# Bounds allow d <= 175 (inner 150) and b <= 175 (outer 500).
# The predicted-width column in __main__ is an EXPECTATION UNDER TEST, not a
# design input — this file has already been wrong by 2.7 um once by trusting
# an out-of-range rate.
SEESAW_DB = [(60.0, 45.0), (90.0, 68.0), (120.0, 90.0), (150.0, 113.0)]


def rung_params(d, b):
    p = np.asarray(eng.seed_params(UNIFORM_SPEC), dtype=float).copy()
    corr = np.full(eng.N_FREE, eng.CORR_NM)
    corr[:N_INNER] -= d
    corr[N_INNER:] += b
    p[eng.SL_CORR] = corr
    return p


SPECS = {i: dataclasses.replace(UNIFORM_SPEC,
                                label=f"seesaw_d{int(d):03d}",
                                fwhm_wall=False, fw_anchor=None, fw_curve=False,
                                seed_override=tuple(rung_params(d, b)))
         for i, (d, b) in enumerate(SEESAW_DB)}
SPEC = SPECS[0]          # deploy contract: build_sweep_list needs a top-level SPEC
N_TASKS = len(SPECS)


def main(task_idx=0):
    idx = int(task_idx)
    spec = SPECS[idx]
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    p = eng.replay_params(spec, np.asarray(spec.seed_override, dtype=float))
    lmpt = eng.import_lumopt2()
    project, _ = eng.make_project(spec, out_dir, lmpt)
    cb = eng.make_log_callback(spec, out_dir, lmpt=lmpt)
    fom = project.compute_fom(p)
    try:
        cb.on_function_eval(project, 0, p, fom)
    except (eng.RecenterNeeded, eng.WidthTrip):
        pass                                          # ladder reports, never gates
    with open(os.path.join(out_dir, f"{spec.label}_evals.jsonl")) as f:
        row = json.loads(f.readlines()[-1])
    c = p[eng.SL_CORR]
    print(f"[seesaw {idx}] inner {c[:N_INNER].mean():.1f} outer {c[N_INNER:].mean():.1f} "
          f"mcorr {c.mean():.2f} | fwhm_env {row.get('fwhm_env_um')} um  "
          f"T {row.get('t_pk')}  lam {row.get('lam_pk_nm')} "
          f"| seed 18.3452 um / T 0.9012, band [17.98, 18.71]")


if __name__ == "__main__":
    for i, (d, b) in enumerate(SEESAW_DB):
        p = rung_params(d, b)
        c = p[eng.SL_CORR]
        bnds = np.array(eng.param_bounds(SPECS[i]))
        ok = bool(((p >= bnds[:, 0] - 1e-9) & (p <= bnds[:, 1] + 1e-9)).all())
        # EXPECTED width from the 61901 single-point rates (this is the claim under test)
        w = 18.3452 + d * 0.02335 - b * 0.031      # EXPECTATION under test
        print(f"task {i}: d={d:5.1f} b={b:5.1f}  inner={c[:N_INNER].mean():5.1f} outer={c[N_INNER:].mean():5.1f} "
              f"mcorr={c.mean():6.2f}  2kL={eng.two_kappa_L(p, SPECS[i].n_periods_side):.3f}  "
              f"W_pred={w:6.3f}  bounds_ok={ok}")
