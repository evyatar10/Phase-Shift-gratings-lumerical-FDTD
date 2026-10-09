"""Overshoot (S2) seed + innermost-tooth shifts: a 7-row forward sweep (user, 2026-10-09).

Study dir: runners/lumopt2_design/ | Created 2026-10-09 | Job(s): see memory
project_te_inverse_design_lane.md
Purpose: does simply shifting the innermost teeth of Itai's overshoot device
raise T at fixed width? Plain forwards through the optimizer's own device path
(run_canary, PVA, the S2 campaign window 2 nm/501 @ 1560.464) so the control
row is the campaign's iterate 0 (172063_1 ev0: t_pk 0.96909, W 19.636) and
every row is comparable to the optimizer lanes.

Facts that shaped the rows (MEASURED, S2 perp gradient at the seed):
  - tooth 1 has corr 0 at the seed (width = cavity width 951.4) → it is
    invisible; dT/ds_1 = dλ/ds_1 = dW/ds_1 = 0. The innermost VISIBLE tooth is 2.
  - a shift s_d moves teeth d..N outward (cavity +2s per shifted tooth).
  - dλ/ds grows outward, 0.0005 (tooth 2) … 0.0057 nm/nm (tooth 6): every row
    moves λ by < 0.2 nm → one common window, no recentring.
Rows: 0 control | 1-3 tooth 2 shifted 5/10/20 nm | 4 teeth 2-4 by 5 nm each |
5 teeth 2-6 by 5 nm | 6 teeth 2-6 by 10 nm.

Dispatch (Athena A100, one at a time, behind the confirm array):
    SBATCH_MEM=256G LUMOPT2_QOS=24h_1g LUMOPT2_TIME=06:00:00 \
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.s2_shift_sweep \
        --array-tasks=0-6%1 --gpu=a100 --after=173313
"""
import dataclasses
import json
import os

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s2 as s2

# (label, {tooth index d (1 = innermost): shift nm})
ROWS = [
    ("ctrl",      {}),
    ("t2_5",      {2: 5.0}),
    ("t2_10",     {2: 10.0}),
    ("t2_20",     {2: 20.0}),
    ("t2-4_5",    {2: 5.0, 3: 5.0, 4: 5.0}),
    ("t2-6_5",    {d: 5.0 for d in range(2, 7)}),
    ("t2-6_10",   {d: 10.0 for d in range(2, 7)}),
    # rows 7-10 (2026-10-09 23:30, after row 5 MEASURED +0.0024 fom at W -0.06 %):
    # extend the graded stretch outward and in amplitude
    ("t2-10_5",   {d: 5.0 for d in range(2, 11)}),
    ("t2-15_5",   {d: 5.0 for d in range(2, 16)}),
    ("t2-6_15",   {d: 15.0 for d in range(2, 7)}),
    ("t2-10_10",  {d: 10.0 for d in range(2, 11)}),
]
N_TASKS = len(ROWS)


def spec_of(idx):
    name, shifts = ROWS[idx]
    L = eng.layout(60)
    p = np.asarray(s2.SEED, dtype=float).copy()
    for d, s_nm in shifts.items():
        p[L.SL_SHIFT.start + d - 1] = s_nm
    return dataclasses.replace(
        s2.SPEC_V3, label=f"s2shift_{name}", seed_override=tuple(float(v) for v in p),
        scan_center_nm=s2.SCAN_CENTER_NM,
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
    print(f"[s2shift {task_idx} {spec.label}] T {row.get('t_pk')}  fom {row.get('fom')}  "
          f"λ {row.get('lam_pk_nm')}  Q {row.get('q_loaded')}  FWHM {row.get('fwhm_env_um')} µm")
