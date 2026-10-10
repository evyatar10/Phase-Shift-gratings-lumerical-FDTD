"""Tooth-shift forward sweep, BOTH TE seeds, at the converged fixed sim time (2026-10-10).

Study dir: runners/lumopt2_design/ | Created 2026-10-10 | Job(s): see memory
project_te_inverse_design_lane.md
Purpose (user: "did we check shifting a single tooth? does it apply to both
seeds? very important"): the only shift sweep so far (s2_shift_sweep.py) was
S2-only and ran at the production auto-shutoff, whose cut-time scatter is
±0.004 (s2_window_ladder) — the size of its own gains. Repeat the decisive
rows at 150 ps with the shutoff off (s2_simtime_ladder: 150 vs 250 ps agree to
7e-5), and run the same rows on S1 for the first time.
Controls are REUSED, not re-run (CLAUDE.md §6): S2 seed @1560.464/150 ps =
174406_0 (0.98069 / W 19.636); S1 best @1561.04/150 ps = 174645_3 (0.97439 /
W 19.394). Tooth index d = 1 is the innermost tooth (S2's tooth 1 has corr 0,
so its innermost VISIBLE tooth is 2; S1's tooth 1 is a full 250 nm tooth).
Rows (label, seed, window, {tooth: shift nm}):
  0-2  S2: t2_10 (single tooth), t2-6_5, t2-6_10 (graded stretch, the 63 ps
       sweep's best: +0.0026/+0.0028 — to be confirmed or retracted)
  3-7  S1: t1_5, t1_10, t1_20 (single tooth), t1-5_5, t1-5_10 (stretch)
Pass/read: dT vs the reused control at 1e-4 repeatability; W must stay in band.

Dispatch (Athena A100, behind the live lanes, QOS cap permitting):
    SBATCH_MEM=256G LUMOPT2_QOS=24h_1g LUMOPT2_TIME=08:00:00 \
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.te_shift_sweep_fixt \
        --array-tasks=0-7%2 --gpu=a100
"""
import dataclasses
import json
import os

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s1 as s1
from runners.lumopt2_design import campaign_te_s2 as s2
from runners.lumopt2_design.best_designs import TE_S1_N98_T9711

SIM_TIME_PS = 150.0

# (label, base spec, design vector, scan centre nm, {tooth d: shift nm})
ROWS = [
    ("s2_t2_10",   s2.SPEC_V3, s2.SEED,         1560.464, {2: 10.0}),
    ("s2_t2-6_5",  s2.SPEC_V3, s2.SEED,         1560.464, {d: 5.0 for d in range(2, 7)}),
    ("s2_t2-6_10", s2.SPEC_V3, s2.SEED,         1560.464, {d: 10.0 for d in range(2, 7)}),
    ("s1_t1_5",    s1.SPEC_V3, TE_S1_N98_T9711, 1561.04,  {1: 5.0}),
    ("s1_t1_10",   s1.SPEC_V3, TE_S1_N98_T9711, 1561.04,  {1: 10.0}),
    ("s1_t1_20",   s1.SPEC_V3, TE_S1_N98_T9711, 1561.04,  {1: 20.0}),
    ("s1_t1-5_5",  s1.SPEC_V3, TE_S1_N98_T9711, 1561.04,  {d: 5.0 for d in range(1, 6)}),
    ("s1_t1-5_10", s1.SPEC_V3, TE_S1_N98_T9711, 1561.04,  {d: 10.0 for d in range(1, 6)}),
]
N_TASKS = len(ROWS)


def spec_of(idx):
    name, base, seed, c, shifts = ROWS[idx]
    L = eng.layout(60)
    p = np.asarray(seed, dtype=float).copy()
    for d, s_nm in shifts.items():
        p[L.SL_SHIFT.start + d - 1] += s_nm
    return dataclasses.replace(
        base, label=f"shiftfixt_{name}", seed_override=tuple(float(v) for v in p),
        scan_center_nm=c, sim_time_ps=SIM_TIME_PS, auto_shutoff_min=1e-12,
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
    print(f"[shiftfixt {task_idx} {spec.label}] {SIM_TIME_PS} ps: T {row.get('t_pk')}  fom {row.get('fom')}  "
          f"λ {row.get('lam_pk_nm')}  Q {row.get('q_loaded')}  FWHM {row.get('fwhm_env_um')} µm")
