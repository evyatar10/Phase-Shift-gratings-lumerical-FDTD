"""S2 seed with a FIXED simulation time (auto-shutoff disabled): is the 63 ps
auto-shutoff cut the source of S2's ~0.004 T scatter? (2026-10-10)

Study dir: runners/lumopt2_design/ | Created 2026-10-10 | Job(s): see memory
project_te_inverse_design_lane.md
Purpose (MEASURED trigger, s2win_1560p460/fwd_default_p0.log): the production S2
forward terminates by auto-shutoff 1e-7 after 63 ps (cap 2000 ps) while the
resonance (Q_L 7650, amplitude ring-down 2Q/ω ≈ 12.6 ps) is still at ~0.7 % of
peak. The window-centre ladder then showed T 0.96909 / 0.97029 / 0.97309 at
1560.460 / .462 / .464 with identical sampled wavelengths and identical λ, W —
the signature of a truncation error whose size depends on the exact cut moment.
Rows (identical device = S2 seed; shutoff disabled at 1e-12):
  0  window 1560.464, 150 ps     (12 amplitude time constants)
  1  window 1560.464, 250 ps     (20)   -> convergence check vs row 0
  2  window 1560.460, 250 ps            -> window independence vs row 1
Pass: |T(250) − T(150)| < 1e-3 and |T(1560.460) − T(1560.464)| < 1e-3 at 250 ps.
Cost: ~2.5 / 4 / 4 h on an A100 (63 ps took ~70 min). The converged T replaces
0.97309 as S2's reference, and S2 lanes then need the same setting (slower, but
their deltas become trustworthy to ~1e-4).

Dispatch (Athena A100, one at a time, behind the window ladder):
    SBATCH_MEM=256G LUMOPT2_QOS=24h_1g LUMOPT2_TIME=08:00:00 \
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.s2_simtime_ladder \
        --array-tasks=0-2%1 --gpu=a100 --after=174370
"""
import dataclasses
import json
import os

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import campaign_te_s2 as s2

# (label, scan centre nm, simulation time ps)
ROWS = [
    ("w464_150ps", 1560.464, 150.0),
    ("w464_250ps", 1560.464, 250.0),
    ("w460_250ps", 1560.460, 250.0),
]
N_TASKS = len(ROWS)


def spec_of(idx):
    name, c, t_ps = ROWS[idx]
    return dataclasses.replace(
        s2.SPEC_V3, label=f"s2time_{name}",
        seed_override=tuple(float(v) for v in s2.SEED), scan_center_nm=c,
        sim_time_ps=t_ps, auto_shutoff_min=1e-12,
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
    print(f"[s2time {task_idx} {spec.label}] {spec.sim_time_ps} ps @ {spec.scan_center_nm}: "
          f"T {row.get('t_pk')}  fom {row.get('fom')}  λ {row.get('lam_pk_nm')}  "
          f"Q {row.get('q_loaded')}  FWHM {row.get('fwhm_env_um')} µm")
