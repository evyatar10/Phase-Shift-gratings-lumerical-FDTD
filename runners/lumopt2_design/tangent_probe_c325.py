"""Width-neutral tangent probe: can MORE shift be paid for by MORE corrugation?

Study dir: runners/lumopt2_design/   |   Created 2026-08-17   |   Job(s): TBD
Purpose (user, 2026-08-17): the two independent walls (elongation, rho)
structurally forbid the sigma-NEUTRAL cross-block trade "shift up (loss down,
width up) + corr up (width back down)". Before building the v1.5 combined
sigma-hat wall, measure the trade directly at seedA's best point (base vector
= campaign_c325_seedA2.SEED, the banked stage-1 best: FOM 0.68831 / T 0.9313 /
sigma 17.7519 = ratio 1.0148). Three replay evaluations, identical numerics:

  task 0  shift   shifts x1.3063 (2*Sig_s 130.6 -> 170.6 nm), corr unchanged
                  -> measures d(sigma), d(T) along the shift lever PAST the wall
  task 1  corr    corr +5.0 nm on all 25 free teeth (rho_free +0.0154)
                  -> measures the payback lever where the field lives
  task 2  combo   shifts x1.6126 (+80 nm) AND corr +7.54 nm (prior-predicted
                  sigma-neutral) -> the actual trade. SUCCESS = T > 0.9313
                  with sigma ratio <= ~1.02.

Fitted model behind the design (12 seedA rows, R2 0.997): sigma-hat = 17.49
+ 0.0051*(2Sig_s) + 0.109*(w_cav-800); rho term from the measured kappa~corr
law (prior -17.5 um per unit rho — task 1 measures the true free-region value).
These are PROBES (no sigma0 passed -> no tripwire; nothing is accepted-best).
CONCLUSIONS ONLY — stage-3 design follows only with the user.

Dispatch (Athena 2h_2g; tasks serialize on the lane mem cap, ~1 h each):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \\
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.tangent_probe_c325
Output -> results/tangent_probe_c325/results/tangent_{shift,corr,combo}/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_c325_seedA2 import SEED  # stage-1 best vector

N_TASKS = 3
SHIFT_F = {0: 1.3063, 2: 1.6126}      # 2*Sig_s 130.6 -> 170.6 / 210.6 nm
CORR_D  = {1: 5.0, 2: 7.54}           # uniform nm on the 25 free teeth
LABELS  = {0: "tangent_shift", 1: "tangent_corr", 2: "tangent_combo"}


def probe_params(task_idx):
    p = np.asarray(SEED, dtype=float).copy()
    p[eng.SL_SHIFT] = p[eng.SL_SHIFT] * SHIFT_F.get(task_idx, 1.0)
    p[eng.SL_CORR] = p[eng.SL_CORR] + CORR_D.get(task_idx, 0.0)
    return p


def main(task_idx=0):
    task_idx = int(task_idx)
    spec = eng.CampaignSpec(label=LABELS[task_idx], scan_center_nm=1566.0)
    p = eng.replay_params(spec, probe_params(task_idx))
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    lmpt = eng.import_lumopt2()
    project, _ = eng.make_project(spec, out_dir, lmpt)
    cb = eng.make_log_callback(spec, out_dir, lmpt=lmpt)   # no sigma0 -> no trips
    fom = project.compute_fom(p)
    try:
        cb.on_function_eval(project, 0, p, fom)
    except (eng.RecenterNeeded, eng.WidthTrip):
        pass
    with open(os.path.join(out_dir, f"{spec.label}_evals.jsonl")) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[tangent task {task_idx} {spec.label}] FOM {fom:.5f}  "
          f"T {row.get('t_pk')}  sigma {row.get('sigma_um')}  "
          f"lam {row.get('lam_pk_nm')}  Q_i {row.get('q_i')}")


SPEC = eng.CampaignSpec(label="tangent_probe_c325")   # deploy contract only

if __name__ == "__main__":
    for t in range(N_TASKS):
        p = probe_params(t)
        print(f"task {t} {LABELS[t]}: 2Ss {2*p[eng.SL_SHIFT].sum():.1f} nm, "
              f"rho_free {p[eng.SL_CORR].mean()/325.0:.4f}")
