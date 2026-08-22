"""Comb post COUNT on the optimized device: does more/less comb help?

Study dir: runners/lumopt2_design/   |   Created 2026-08-17   |   Job(s): TBD
Purpose (user, 2026-08-17 night): "take the optimized device and check what
happens when I considerably increase or decrease the number of circles in the
comb. Just two simulations." This is also the standing follow-up from memory
[[feedback_optimize_structural_counts]]: 57 posts was FROZEN at design lock
while the user was explicitly unsure of it, so the count owes us a measurement.

Base = BEST_T9609 (stage-2 job 133530 eval 3, THE program best: T 0.9609,
Q_i 103,149, sigma 17.7914). The grating is replayed exactly; only the number
of comb posts changes. The comb is FROZEN in both tasks (free_comb=False), so
the 191-param vector's 57 comb slots stay inert and the scene's post count is
what varies -- the count is a scene property, not a parameter.

  task 0  n=29  posts (half 14)  -- roughly HALF the comb, spans +/- 7.4 um
  task 1  n=113 posts (half 56)  -- roughly DOUBLE,        spans +/- 29.9 um

The doubled comb still sits well inside the 51.8 um grating half-length, and
both keep the seed lattice (531 nm pitch, 270 deg offset, r 80, d 1.9 um) --
only its EXTENT changes. Control: BEST_T9609 itself at n=57, T 0.9609, already
measured at these exact numerics (CLAUDE.md section 6 -> no control row).

Reading: the comb suppresses radiation where the mode still has amplitude. If
T rises with n=113, the comb is being truncated too early and the outer mode
tail is still leaking; if n=29 matches n=57, the outer posts are dead weight
(a fab simplification); if both lose, 57 is already the right extent.

Target 1566.0 nm, +/-3 nm at 20 pm -- identical recording to the campaigns.

Dispatch (Athena, short lane; does NOT touch stage-2's 4d_1g quota):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \\
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.comb_count_scan
Output -> results/comb_count_n{29,113}/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9609

SCAN_CENTER_NM = 1566.0
COUNTS = [14, 56]          # half-counts -> 29 and 113 posts (control = 28 -> 57)

SPECS = {i: eng.CampaignSpec(label=f"comb_count_n{2*k+1}",
                             scan_center_nm=SCAN_CENTER_NM,
                             seed_override=tuple(BEST_T9609),
                             free_comb=False,      # required by comb_n_half
                             comb_n_half=k)
         for i, k in enumerate(COUNTS)}
SPEC = SPECS[0]            # deploy contract: build_sweep_list needs a top-level SPEC
N_TASKS = len(SPECS)


def main(task_idx=0):
    spec = SPECS[int(task_idx)]
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    p = eng.replay_params(spec, BEST_T9609)     # inert comb slots reset + bounds assert
    lmpt = eng.import_lumopt2()
    project, _ = eng.make_project(spec, out_dir, lmpt)
    cb = eng.make_log_callback(spec, out_dir, lmpt=lmpt)   # no sigma0 -> no trips
    fom = project.compute_fom(p)
    try:
        cb.on_function_eval(project, 0, p, fom)
    except (eng.RecenterNeeded, eng.WidthTrip):
        pass                                    # scan reports, never gates
    with open(os.path.join(out_dir, f"{spec.label}_evals.jsonl")) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[comb_count {task_idx} n={eng.comb_count(spec)}] FOM {fom:.5f}  "
          f"T {row.get('t_pk')}  lam {row.get('lam_pk_nm')}  Q_i {row.get('q_i')}  "
          f"sigma {row.get('sigma_um')} um | control n=57 T 0.9609 Q_i 103149")


if __name__ == "__main__":
    for i, k in enumerate(COUNTS):
        n = eng.comb_count(SPECS[i])
        span = (k * eng.COMB_LAM_NM + eng.COMB_DX_NM) / 1000.0
        print(f"task {i}: {SPECS[i].label:16s} n={n:4d} posts  half={k:3d}  "
              f"spans +/-{span:5.1f} um  (control n=57, +/-15.3 um)")
