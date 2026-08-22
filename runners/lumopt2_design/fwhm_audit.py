"""FWHM audit: what has the campaign actually done to the SPEC's observable?

Study dir: runners/lumopt2_design/   |   Created 2026-08-18   |   Job(s): TBD
Purpose (user 2026-08-18: "22 is slightly too large actually... do we already
have a >0.96 result whose FWHM is close to the initial, around 20?"): the
acoustic spec is written in SPATIAL FWHM, but the campaign has controlled
sigma (a 2nd moment) for its entire duration, and FWHM was only added to the
log today. The first three FWHM numbers ever measured (sigma-neutral probe,
job 134107) came back 22.21 / 22.22 / 23.21 um — and NOBODY KNOWS what the
origin was, so we cannot yet say whether 22 um is something the optimization
CAUSED or something this device family always had.

This audit measures FWHM on three stored designs, one forward each, no new
physics — only the reference points needed to answer that:

  task 0  UNIFORM_SEED   the origin of everything (T 0.8924, sigma 17.4891)
  task 1  BEST_T9635     the banked winner        (T 0.9635, sigma 17.7952)
  task 2  BEST_T9635 with shifts x0               (       ,  sigma 17.4956)

Task 2 is the decisive control: the shift ladder showed deleting the shifts
returns sigma to ~sigma0, so if its FWHM also returns to the origin's value,
FWHM growth is driven by the SHIFTS (and is therefore controllable by the same
lever); if its FWHM stays high, the growth came from the corrugation/cavity
shaping instead, and the sigma band never had authority over it at all.

Reading, once the three land:
  FWHM(origin) ~ 20 and FWHM(best) ~ 22  -> the campaign grew the spec
      observable ~10% while sigma grew only 1.7%: the sigma band did NOT
      enforce the spec, and the constraint must move to FWHM (or to both).
  FWHM(origin) ~ 22 already                -> 22 um is the family's natural
      mode; nothing was broken, and the ~20 um target belongs to the
      PRODUCTION length (N~165-169), not to this N=100 surrogate.

Dispatch (Athena short lane; leaves stage-4's 4d_1g quota alone):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \\
        bash athena/deploy_athena.sh \\
        --lumopt2-design=runners.lumopt2_design.fwhm_audit --max-concurrent=1
Output -> results/fwhm_{origin,best,noshift}/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9635, UNIFORM_SEED


def _noshift():
    p = np.asarray(BEST_T9635, dtype=float).copy()
    p[eng.SL_SHIFT] = 0.0
    return p


CASES = [("fwhm_origin",  np.asarray(UNIFORM_SEED, dtype=float), 1566.0),
         ("fwhm_best",    np.asarray(BEST_T9635, dtype=float),   1566.14),
         ("fwhm_noshift", _noshift(),                            1564.52)]

SPECS = {i: eng.CampaignSpec(label=lbl, scan_center_nm=ctr, seed_override=tuple(v))
         for i, (lbl, v, ctr) in enumerate(CASES)}
SPEC = SPECS[0]        # deploy contract: build_sweep_list needs a top-level SPEC
N_TASKS = len(SPECS)


def main(task_idx=0):
    idx = int(task_idx)
    spec = SPECS[idx]
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    p = eng.replay_params(spec, np.asarray(spec.seed_override, dtype=float))
    lmpt = eng.import_lumopt2()
    project, _ = eng.make_project(spec, out_dir, lmpt)
    cb = eng.make_log_callback(spec, out_dir, lmpt=lmpt)     # no sigma0 -> no trips
    fom = project.compute_fom(p)
    try:
        cb.on_function_eval(project, 0, p, fom)
    except (eng.RecenterNeeded, eng.WidthTrip):
        pass                                                 # audit reports, never gates
    with open(os.path.join(out_dir, f"{spec.label}_evals.jsonl")) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[fwhm_audit {idx} {spec.label}] T {row.get('t_pk')} "
          f"sigma {row.get('sigma_um')} FWHM {row.get('fwhm_env_um')} "
          f"fwhm/sigma {row.get('fwhm_over_sigma')} lam {row.get('lam_pk_nm')}")


if __name__ == "__main__":
    for i, (lbl, v, ctr) in enumerate(CASES):
        b = np.array(eng.param_bounds(SPECS[i]))
        ok = bool(((v >= b[:, 0] - 1e-9) & (v <= b[:, 1] + 1e-9)).all())
        print(f"task {i}: {lbl:13s} 2Ss={2*v[eng.SL_SHIFT].sum():6.1f} nm "
              f"corr {v[eng.SL_CORR].min():.1f}..{v[eng.SL_CORR].max():.1f} "
              f"wcav {v[eng.I_CAV]:.1f} center {ctr} nm bounds_ok={ok}")
