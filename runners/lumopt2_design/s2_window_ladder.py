"""S2 seed, identical device, FOUR scan-window centres: how much does S2's T move
with a few-pm window shift? (2026-10-10)

Study dir: runners/lumopt2_design/ | Created 2026-10-10 | Job(s): see memory
project_te_inverse_design_lane.md
Purpose: the S2 seed measured (MEASURED, A100, PVA, N98, 2 nm/501 pts):
  window 1560.464  t_pk 0.97309 (anchor 168530_11; sweep ctrl 173624_0)
  window 1560.46   t_pk 0.96909 (campaign path: 172063_1, 174284_9 — run_campaign
                   rounds the centre to 2 decimals)
  window 1560.42   t_pk 0.97208 (lanes 173310_6/7)
λ and W agree to 1e-5 across all three; only T differs, by up to 0.004 — the size
of every S2 optimizer gain. Peak sampling on a 4 pm grid can explain ≤ 0.0004 of
it. This ladder tells whether T(window) is smooth, a step (a one-cell knife edge
somewhere in the auto-meshed region), or erratic (precision noise amplified by
the resonance). Same canary path as the sweep control, so 1560.464 = ctrl row.

Dispatch (Athena A100, one at a time):
    SBATCH_MEM=256G LUMOPT2_QOS=24h_1g LUMOPT2_TIME=06:00:00 \
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.s2_window_ladder \
        --array-tasks=0-3%1 --gpu=a100
"""
import dataclasses
import json
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s2 as s2

CENTERS_NM = [1560.460, 1560.462, 1560.466, 1560.470]
N_TASKS = len(CENTERS_NM)


def spec_of(idx):
    c = CENTERS_NM[idx]
    return dataclasses.replace(
        s2.SPEC_V3, label=f"s2win_{c:.3f}".replace(".", "p"),
        seed_override=tuple(float(v) for v in s2.SEED), scan_center_nm=c,
        fwhm0_um=None, wg_anchor={"softw": 19.0, "fwhm": 19.0}, two_kl_floor=1.0)


SPEC = spec_of(0)      # deploy contract: build_sweep_list needs a top-level SPEC


def main(task_idx=0):
    spec = spec_of(int(task_idx))
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    log = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    if os.path.exists(log):           # preemption resume: one forward, never twice
        row = json.loads(open(log).readlines()[-1])
    else:
        row = eng.run_canary(spec, out_dir)
    print(f"[s2win {task_idx} {spec.label}] centre {spec.scan_center_nm} nm: T {row.get('t_pk')}  "
          f"fom {row.get('fom')}  λ {row.get('lam_pk_nm')}  Q {row.get('q_loaded')}  "
          f"FWHM {row.get('fwhm_env_um')} µm")
