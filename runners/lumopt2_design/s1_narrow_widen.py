"""Narrow-then-widen test, S1 (user 2026-10-10 evening: "have we tried minimizing
the mode width and then enlarging it?").

Study dir: runners/lumopt2_design/ | Created 2026-10-10 | Job(s): see memory
project_te_inverse_design_lane.md
Question: does optimising the plain grating at a NARROWER mode (~17 um, deeper
uniform corrugation) and then retrimming the corrugation back to 19.4 um beat
optimising at 19.4 um directly (TE_S1_N98_T9711: 0.97439 at 50 ps, MEASURED)?
Row 0 measures the narrow seed (predict-q3db: uniform corr 283 nm -> 17.0 um,
EXPECTED +-5 %, 68 periods below its calibration) at the converged 50 ps; its
measured FWHM / softW become the width anchors of campaign_te_n98 task 18.
Rows 1+ (added after task 18 converges): the task-18 best with its corrugation
scaled back toward 19.4 um (two-row ladder), compared with 0.97439 at equal W.
Numerics identical to the S1 lanes (PVA, N=98, 10 nm/501, 50 ps, shutoff off).

Dispatch (Athena A100, behind the shift sweep for seats):
    SBATCH_MEM=256G LUMOPT2_QOS=24h_1g LUMOPT2_TIME=04:00:00 \
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.s1_narrow_widen \
        --array-tasks=0 --gpu=a100 --after=174828
"""
import dataclasses
import json
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s1 as s1

SIM_TIME_PS = 50.0        # s2_simtime_ladder 174823_6/7: S1 converged at 50 ps (6e-5 vs 150 ps)

# (label, uniform corrugation nm, scan centre nm)
ROWS = [
    ("c283", 283.0, 1561.04),
]
N_TASKS = len(ROWS)


def spec_of(idx):
    name, corr, c = ROWS[idx]
    return dataclasses.replace(
        s1.SPEC_V3, label=f"s1narrow_{name}", seed_override=None, corr_seed_nm=corr,
        scan_center_nm=c, sim_time_ps=SIM_TIME_PS, auto_shutoff_min=1e-12,
        fwhm0_um=None, wg_anchor={"softw": 17.0, "fwhm": 17.0}, two_kl_floor=1.0)


SPEC = spec_of(0)      # deploy contract: build_sweep_list needs a top-level SPEC


def main(task_idx=0):
    spec = spec_of(int(task_idx))
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    log = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    if os.path.exists(log):           # preemption resume: one forward, never twice
        row = json.loads(open(log).readlines()[-1])
    else:
        row = eng.run_canary(spec, out_dir)
    print(f"[s1narrow {task_idx} {spec.label}] corr {ROWS[int(task_idx)][1]} nm, {SIM_TIME_PS} ps: "
          f"T {row.get('t_pk')}  λ {row.get('lam_pk_nm')}  Q {row.get('q_loaded')}  "
          f"FWHM {row.get('fwhm_env_um')} µm  softW {row.get('softw_adj_um')}")
