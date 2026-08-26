"""Lengthen the inverse-designed device to the -3 dB point: N, and Q there.

Study dir: runners/lumopt2_design/   |   Created 2026-08-25   |   Job(s): TBD
Purpose (user 2026-08-25): "do a conformal mesh, check the resonance and stuff,
and then lengthen it and do the q3db". Wave 1 (prod_confirm) answered the SHORT
device at N=100 under the project's regular mesher; this is the lengthening --
walk n_periods_side up until peak T crosses -3 dB, and report the crossing N and
the loaded Q there, the same figure the regular devices were locked at.

★WHY THIS RUNS THROUGH lumopt2 AND NOT A PLAIN SweepSpec RUNNER. It is the only
path that builds the EXACT measured device. The obvious plain-builder version was
written (runners/sweeps/invdesign_q3db_20um.py) and is BLOCKED: a zero-GPU scene
gate found bragg_device.py:1180 shortens R_narrow_d by s_(d-1) while lumopt2's
make_func uses s_d, so the regular builder places the right arm up to 6.43 nm
differently and cannot be re-indexed into agreement. Job 63202_0 prices what that
convention costs; until it reports, EXACT beats artifact-free.
★THE PRICE OF THIS PATH, already MEASURED (prod_confirm row 0, job 62750): the
lumopt2 optimization-region mesh override reads T **+0.0079 HIGH** and Q_loaded
**7% LOW** against the stored regular-pipeline anchor, at N=100. lambda (+0.025 nm)
and mode width (+0.019%) are unaffected. -3 dB is an absolute-T question, so the
crossing N reported here is biased LONG by that offset -- quote it with the
bias stated, and read the in-batch shape (dB vs N) as the reliable part.

Rows: n_periods_side per task. N=100 is NOT repeated -- prod_confirm row 3 is
that point at identical numerics (CLAUDE.md section 6 no-rerun).
Compare against, same family numerics, never re-run:
  ctrl corr-325 N=165   T 0.4906 = -3.09 dB / Q 13930      (job 130458)
  winner comb  N=169    -3.04 dB / Q 16203 / mode 19.91 um (comb_q3db wave 2)

NUMERICS: identical to prod_confirm -- conformal variant 0, box y 8.0 / z-mult
5.42, window 20 nm @ 1559.5 / 4001 pts, z-sym, region dx pitch-locked, comb
frozen at the winner lattice. Only n_periods_side changes.

Resume-protected (CLAUDE.md >2h rule): each task's row is appended to its own
evals.jsonl and a cold restart reuses it instead of re-solving.

Dispatch (after the ladder is pointed at the measured T(N=100)):
    SBATCH_MEM=300G LUMOPT2_TIME=10:00:00 \
        bash igum/deploy_igum.sh \
        --lumopt2-design=runners.lumopt2_design.prod_q3db_ladder --max-concurrent=2
"""

import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9636
from runners.lumopt2_design.prod_confirm import BASE

# ★POINTED AT THE MEASURED SHORT-DEVICE T (prod_confirm row 3). The bracket is
# wide on purpose: CLAUDE.md section 5 says bisect a threshold in ONE cheap array,
# never one expensive point per dispatch. EXPECTED crossing ~195-215 from a
# TWO-POINT dB~N^4.1 fit on the stored bare rows -- a sizing estimate, not a result.
N_LADDER = [150, 180, 200, 220]

# -3 dB of the device's own off-resonance reference = T 0.5 absolute, the same
# convention the stored family rows use (ctrl N165 T 0.4906 = -3.09 dB).
T_3DB = 0.5


def spec_of(idx):
    n = N_LADDER[idx]
    spec = dataclasses.replace(BASE, label=f"prodq3db_N{n}", n_periods_side=n)
    spec.seed_override = tuple(eng.replay_params(spec, np.asarray(BEST_T9636, float)))
    return spec


N_TASKS = len(N_LADDER)
SPEC = spec_of(0)        # deploy contract: build_sweep_list needs a top-level SPEC


def main(task_idx=0):
    spec = spec_of(int(task_idx))
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    log = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    if os.path.exists(log):                  # preemption resume: never solve twice
        row = json.loads(open(log).readlines()[-1])
    else:
        row = eng.run_canary(spec, out_dir)
    t, q = row.get("t_pk"), row.get("q_loaded")
    db = 10.0 * np.log10(t) if t else float("nan")
    print(f"[prodq3db N={spec.n_periods_side}] T {t}  ({db:+.3f} dB)  "
          f"lam {row.get('lam_pk_nm')}  Q {q}  Q_i {row.get('q_i')}  "
          f"FWHM {row.get('fwhm_env_um')} um  | -3 dB at T {T_3DB} "
          f"| stored: ctrl N165 -3.09 dB Q13930, comb N169 -3.04 dB Q16203")


if __name__ == "__main__":
    for i, n in enumerate(N_LADDER):
        print(f"  task {i}: N={n:4d}  half-device {n * 0.51683:6.1f} um  "
              f"label prodq3db_N{n}")
    s = spec_of(0)
    print(f"mesh={s.mesh_refinement} | region dx={s.region_dx_nm:.3f} nm | "
          f"box y{s.box_y_um}/zmult{s.box_z_mult} | window {s.scan_width_nm} nm @"
          f"{s.scan_center_nm}/{s.n_wl_points} pts | free_comb={s.free_comb}")
    print("N=100 NOT re-run — prod_confirm row 3 is that point (section 6)")
