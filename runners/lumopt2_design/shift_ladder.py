"""Tooth-shift ladder on the stage-2 winner: are the shifts at the right SCALE?

Study dir: runners/lumopt2_design/   |   Created 2026-08-18   |   Job(s): TBD
Purpose (user 2026-08-17: "I'm still afraid the tooth shifts are not in the
best configuration. How can we check that?"): the shifts (2*Sig_s = 130.6 nm)
were fitted in stage-1 against a corrugation profile the device NO LONGER HAS
(corr_1 was 325-ish, now 282.6; cavity 800 -> 960.9), then frozen through all
of stage-2. Stage-4 explores them LOCALLY (trust radius +/-20 nm); this ladder
tests the SCALE globally with three single forwards on the winner:

  task 0  x0.0   shifts zeroed      (2*Sig_s =   0 nm)  - dip-apod alone: do
                                      the shifts still earn their place AT ALL?
                                      (also the user's shift-vs-apodization
                                      question, measured on the current device)
  task 1  x0.5   shifts halved      (2*Sig_s =  65.3 nm)
  task 2  x1.5   shifts x1.5        (2*Sig_s = 195.9 nm)

With the stored x1.0 winner row (T 0.9635 / sigma 17.7952, 133530 ev4 - the
in-study control, NOT re-run) this gives a 4-point curve of T and sigma vs
shift scale: peak at x1.0 = shifts already right; rising toward x1.5 = the
frozen value is too small; flat/rising at x0.5 = stage-1's value overshoots.
Device length is constant by construction at every scale (cavity absorbs
2*Sig_s exactly - verified in make_func). Expect lambda to move ~-0.6 nm at
x0 (stage-3 measured +1.6 nm per +374 nm of 2*Sig_s); window +/-3 nm covers.

Dispatch (Athena, short lane, queued behind the count study):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \\
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.shift_ladder
Output -> results/shift_x{000,050,150}/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9635, BEST_T9636

# ★PASS 2 — WIDTH RECOVERY (2026-08-24). The pass-1 rows (shift_x000/050/150,
# run overnight on BEST_T9635) used the DEFAULT spec: +-3 nm/301 window and
# dx=50 nm — the non-pitch-locked mesh whose fwhm_env mis-reads by up to 3.9%
# (plan §24), so their WIDTHS are VOID (T/lambda stand). This pass re-runs the
# three scales at the v2 numerics on the CURRENT best (BEST_T9636, the
# converged 136465 design) with fresh labels; the x1.0 control is that
# campaign's own eval-12 row (T 0.96361 / W 18.35309, pitch-locked — stored,
# NOT re-run, CLAUDE.md §6). This is the experiment that turns the shift
# verdict (apodization saturates ~0.938; shifts worth +0.025) from strong to
# FINAL: if T falls toward ~0.938 as shifts scale to 0 while width stays in
# band, shifts are PROVEN to buy transmission at constant width on the best
# device itself.
SCAN_CENTER_NM = 1566.444       # MEASURED eval-12 resonance; x0 expected ~-0.6 nm
SCALES = [0.0, 0.5, 1.5]        # x1.0 control = the campaign's own eval-12 row


def scaled_vector(f):
    p = np.asarray(BEST_T9636, dtype=float).copy()
    p[eng.SL_SHIFT] *= f
    return p


SPECS = {i: eng.CampaignSpec(label=f"shiftw_x{int(f*100):03d}",
                             scan_width_nm=10.0, n_wl_points=501,
                             region_dx_nm=eng.DX_PITCHLOCK_NM,
                             scan_center_nm=SCAN_CENTER_NM,
                             seed_override=tuple(scaled_vector(f)))
         for i, f in enumerate(SCALES)}
SPEC = SPECS[0]      # deploy contract: build_sweep_list needs a top-level SPEC
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
        pass                                                 # ladder reports, never gates
    with open(os.path.join(out_dir, f"{spec.label}_evals.jsonl")) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[shift_ladder {idx} x{SCALES[idx]:.1f}] 2Ss={2*p[eng.SL_SHIFT].sum():.1f} nm  "
          f"FOM {fom:.5f}  T {row.get('t_pk')}  lam {row.get('lam_pk_nm')}  "
          f"Q_i {row.get('q_i')}  sigma {row.get('sigma_um')} um "
          f"| control x1.0 stored: T 0.9635, sigma 17.7952")


if __name__ == "__main__":
    for i, f in enumerate(SCALES):
        p = scaled_vector(f)
        b = np.array(eng.param_bounds(SPECS[i]))
        ok = bool(((p >= b[:, 0] - 1e-9) & (p <= b[:, 1] + 1e-9)).all())
        print(f"task {i}: x{f:3.1f}  2Ss={2*p[eng.SL_SHIFT].sum():6.1f} nm  "
              f"cavity x-span={eng.PITCH_NM/2 + 2*p[eng.SL_SHIFT].sum():.1f} nm  bounds_ok={ok}")
