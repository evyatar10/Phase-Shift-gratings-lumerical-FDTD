"""Corrugation ladder on the UNIFORM seed: is profile shaping a better buy than shifts?

Study dir: runners/lumopt2_design/   |   Created 2026-08-24   |   Job(s): TBD
User question (2026-08-24, verbatim intent): "raising the average corrugation in
the center and then doing some apodization — is that actually more helpful than
the tooth shifts?"

WHY THIS EXISTS. The elongation ladder (61742/61782) measured how much T a tooth
shift buys per um of mode width, ON THE UNIFORM SEED: 0.01056 T/um at low e.
The competing number in use, 0.00223 T/um for corrugation, came from the retrim
curve — a DIFFERENT device (the apodized best design, e=130.6, mcorr 316-376).
Comparing them settles nothing. This ladder measures the corrugation rates on
the SAME device as the elongation ladder so the two are finally comparable.

Sign convention that matters: raising corr NARROWS and LOWERS T, so LOWERING
corr is the direction that widens and raises T — the same direction as
elongation, not the opposite. Both are ways of spending width to buy T; the
question is only which is cheaper.

Rungs (all e=0, comb frozen, everything else at the seed):
  task 0  uniform  corr 305 (-20)          widen leg, uniform
  task 1  uniform  corr 345 (+20)          narrow leg, uniform
  task 2  inner-8  corr 385 (+60), rest 325   mean 344.2  <- the user's "raise
                                                the centre" direction
  task 3  inner-8  corr 265 (-60), rest 325   mean 305.8
  task 4  outer-8  corr 385 (+60), rest 325   mean 344.2  <- SAME MEAN as task 2,
                                                opposite placement

Task 2 vs task 1 isolates PLACEMENT at nearly matched mean (344.2 vs 345).
Task 2 vs task 4 isolates inner vs outer at EXACTLY matched mean. Tooth index 0
is the INNERMOST tooth (verified in make_func: the right walk starts at the
cavity and steps outward), so inner-8 = indices 0..7, outer-8 = 17..24.

NOT re-run, cited from storage (CLAUDE.md §6): the seed itself, corr 325 / e=0
-> fwhm_env 18.3452 um, T 0.9012 (136466 ev1) is the in-study control.

Numerics inherited from campaign_v2_uniform via dataclasses.replace, so every
rung is directly comparable to the elongation ladder and to the campaigns.
Scan centre is left at the seed's: corrugation moves the resonance only
~+0.0036 nm per nm of mean corr (DERIVED from best-vs-seed after removing the
elongation term), i.e. <0.1 nm here against 5 nm of window headroom.

Dispatch (short lane, 5 single forwards):
    SBATCH_MEM=160G LUMOPT2_TIME=02:00:00 \\
        bash igum/deploy_igum.sh --lumopt2-design=runners.lumopt2_design.corr_profile_ladder
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

N_GROUP = 8      # teeth per shaped group; outer-8 = the measured-inert block
RUNGS = [("u305", "uniform", 305.0), ("u345", "uniform", 345.0),
         ("in385", "inner", 385.0), ("in265", "inner", 265.0),
         ("out385", "outer", 385.0)]


def rung_params(where, value):
    p = np.asarray(eng.seed_params(UNIFORM_SPEC), dtype=float).copy()
    corr = np.full(eng.N_FREE, eng.CORR_NM)
    if where == "uniform":
        corr[:] = value
    elif where == "inner":
        corr[:N_GROUP] = value
    else:
        corr[-N_GROUP:] = value
    p[eng.SL_CORR] = corr
    return p


SPECS = {i: dataclasses.replace(UNIFORM_SPEC,
                                label=f"corr_{tag}",
                                fwhm_wall=False, fw_anchor=None, fw_curve=False,
                                seed_override=tuple(rung_params(where, v)))
         for i, (tag, where, v) in enumerate(RUNGS)}
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
    print(f"[corr_ladder {idx} {spec.label}] mcorr {p[eng.SL_CORR].mean():.2f}  "
          f"fwhm_env {row.get('fwhm_env_um')} um  T {row.get('t_pk')}  "
          f"lam {row.get('lam_pk_nm')} | stored seed: 18.3452 um / T 0.9012")


if __name__ == "__main__":
    for i, (tag, where, v) in enumerate(RUNGS):
        p = rung_params(where, v)
        b = np.array(eng.param_bounds(SPECS[i]))
        ok = bool(((p >= b[:, 0] - 1e-9) & (p <= b[:, 1] + 1e-9)).all())
        print(f"task {i}: {tag:7s} {where:7s} mcorr={p[eng.SL_CORR].mean():6.2f}  "
              f"2kL={eng.two_kappa_L(p, SPECS[i].n_periods_side):.3f} (guard 3.5)  "
              f"e={2*p[eng.SL_SHIFT].sum():.1f}  bounds_ok={ok}")
