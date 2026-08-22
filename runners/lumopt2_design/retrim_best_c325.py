"""§6b decisive experiment: does ANY campaign gain survive at CONSTANT width?

Study dir: runners/lumopt2_design/  |  Created 2026-08-22  |  Job(s): TBD
Purpose (HANDOFF §6b, the designated first experiment; user asked "do we have
anything that beats the regular comb device?"): take BEST_T9635 (T 0.9640,
FWHM 20.336 = +14.9%) and raise its corrugation UNIFORMLY — bisecting on the
MEASURED fwhm_env — until it reaches the ORIGIN's width (17.7005 µm, PVA
family, same numerics as the seed audit). Verdicts:
  T(at origin width) > 0.89265  → the campaign found a REAL constant-width
                                   gain; the trade curve says how much.
  T(at origin width) <= 0.89265 → nothing beats the regular comb device yet;
                                   v2's honest starting point stands.
En route it passes noshift's width 18.5664 (compare T vs 0.9345) — every
bisection row is a free (fwhm, T) point of the constant-basis re-trim curve.

ONE task; sequential forwards (~5-6 × ~50 min). RESUME-PROTECTED (>2h rule):
every row appends to retrim_evals.jsonl; cold start reuses stored rows.

Dispatch: SBATCH_MEM=160G LUMOPT2_QOS=12h_4g LUMOPT2_TIME=10:00:00 \\
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.retrim_best_c325
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9635

TARGET_UM = 17.7005          # origin FWHM, PVA family (fspw_origin, MEASURED)
ORIGIN_T = 0.89265           # the number to beat AT that width
NOSHIFT_UM, NOSHIFT_T = 18.5664, 0.9345
TOL_UM = 0.02                # ≈ 0.1% of target; noise floor is 0.03% of width
D_LO, D_HI = 0.0, 60.0       # uniform corr add (nm); +60 must overshoot narrow

SPEC = eng.CampaignSpec(label="retrim_best")
N_TASKS = 1


def _measure(delta_nm, out_dir):
    p = np.asarray(BEST_T9635, dtype=float).copy()
    p[eng.SL_CORR] = np.minimum(p[eng.SL_CORR] + delta_nm, SPEC.corr_max_nm)
    spec = dataclasses.replace(SPEC, label=f"retrim_d{delta_nm:05.1f}".replace(".", "p"))
    spec.seed_override = tuple(eng.replay_params(spec, p))
    row = eng.run_canary(spec, out_dir)
    return {"delta_nm": float(delta_nm), "fwhm": row.get("fwhm_env_um"),
            "t_pk": row.get("t_pk"), "lam": row.get("lam_pk_nm")}


def main(task_idx=0):
    out_dir = os.path.join(config.RESULTS_DIR, "retrim_best")
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "retrim_evals.jsonl")
    rows = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
    done = {round(r["delta_nm"], 3): r for r in rows}

    def measure(d):
        k = round(d, 3)
        if k not in done:                          # resume: reuse stored rows
            done[k] = _measure(d, out_dir)
            with open(log, "a") as f:
                f.write(json.dumps(done[k]) + "\n")
            r = done[k]
            print(f"[retrim] d+{d:.1f} nm -> fwhm {r['fwhm']:.4f} um, T {r['t_pk']:.5f}")
        return done[k]

    lo, hi = D_LO, D_HI
    r_lo, r_hi = measure(lo), measure(hi)
    assert r_lo["fwhm"] > TARGET_UM > r_hi["fwhm"], \
        f"bracket broken: fwhm({lo})={r_lo['fwhm']}, fwhm({hi})={r_hi['fwhm']}"
    while True:
        mid = 0.5 * (lo + hi)
        r = measure(mid)
        if abs(r["fwhm"] - TARGET_UM) <= TOL_UM or (hi - lo) < 1.0:
            break
        if r["fwhm"] > TARGET_UM:
            lo = mid
        else:
            hi = mid
    print(f"[retrim VERDICT] at width {r['fwhm']:.4f} um (target {TARGET_UM}): "
          f"T {r['t_pk']:.5f} vs origin {ORIGIN_T} -> "
          f"{'REAL constant-width gain' if r['t_pk'] > ORIGIN_T else 'NO gain survives'} "
          f"| noshift checkpoint: {NOSHIFT_UM} um vs T {NOSHIFT_T} (read curve rows)")


if __name__ == "__main__":
    print("1 task: bisect uniform corr-add on measured fwhm_env to", TARGET_UM)
