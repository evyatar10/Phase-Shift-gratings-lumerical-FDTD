"""Can REDISTRIBUTING the corrugation buy transmission at CONSTANT width?

Study dir: runners/lumopt2_design/   |   Created 2026-08-18   |   Job(s): TBD
Purpose (user 2026-08-18: "we do not want to cheat in any way", and "the
uniform corrugation must be kept the same — changing that to match the FWHM to
around 20 does not count"): every transmission gain this campaign booked came
with a larger mode, and the width was policed by sigma, which is nearly blind
to apodization. NO grating-side gain at constant width has ever been shown.

★All previously logged widths (sigma AND FWHM) are VOID: they were sampled on
one off-axis y-row instead of the y-integral (see profile_line's bug note), so
the trade line and the "seed B beats it" ranking that earlier versions of this
docstring quoted have been deleted rather than repeated. What survives is the
QUALITATIVE point and the design idea below; the numbers must be re-measured.

Seed B's dip+overshoot cusp-smoothing profile was built rho-NEUTRAL — dip at
the cusp, payback just outside — so it changes the SHAPE of kappa(x) while
barely moving its mean. The cusp is the sharpest feature of a pi-shift mode and
therefore the natural place radiation comes from, so smoothing it at fixed mean
is the one lever the optimizer never pulled (it always just lowered the mean).
This study makes that profile exactly rho-neutral and sweeps how hard to push it:

    corr(a) = 325 + a * (DIP - 325 - mean(DIP - 325))      mean == 325 EXACTLY

    task 0  a = 0.5   gentle
    task 1  a = 1.0   seed B's shape, mean-corrected to rho = 1.000000
    task 2  a = 1.5
    task 3  a = 2.0   innermost tooth 159.8 nm (bound is 150)

a = 0 IS the uniform origin (T 0.8926 MEASURED, job 134217 task 0 — that
number is a PORT quantity and stands; only its width does not). Its FWHM is
being re-measured by job 134299, and that value is the reference this ladder is
read against. Shifts stay 0 throughout so the corrugation shape is isolated.
Comb = the winner seed.

WHY THIS RESPECTS THE USER'S CONSTRAINT: the mean corrugation is pinned to the
baseline's 325 nm on every row. Nothing is re-matched to make a target land
conveniently; only the DISTRIBUTION of a fixed corrugation budget changes.

READING (judge ONLY on the measured fwhm_env_um, vs the origin's own
re-measured fwhm_env_um from job 134299):
  T > 0.8926 at the origin's FWHM -> the first honest fixed-width gain in the
      program; this profile family seeds the corrected campaign.
  T ~ 0.8926, or FWHM grows anyway -> the mean corrugation does NOT pin the
      width (shape matters), and width must be held by measurement alone.
Either outcome is decisive.

Dispatch (Athena short lane, serial; run AFTER 134299 drains):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \\
        bash athena/deploy_athena.sh \\
        --lumopt2-design=runners.lumopt2_design.rho_neutral_shape --max-concurrent=1
Output -> results/rhoneut_a{050,100,150,200}/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_c325_seedB import DIP_PROFILE_NM

SCAN_CENTER_NM = 1564.5          # seed B's own seed peaked at 1564.304
AMPLITUDES = [0.5, 1.0, 1.5, 2.0]

_DEV = np.asarray(DIP_PROFILE_NM, dtype=float) - eng.CORR_NM
_DEV -= _DEV.mean()              # mean-zero => every row has rho == 1.000000


def shape_profile(a):
    return eng.CORR_NM + a * _DEV


SPECS = {i: eng.CampaignSpec(label=f"rhoneut_a{int(a * 100):03d}",
                             box_y_um=6.8, box_z_mult=4.14,
                             scan_center_nm=SCAN_CENTER_NM,
                             corr_seed_nm=tuple(shape_profile(a)))
         for i, a in enumerate(AMPLITUDES)}
SPEC = SPECS[0]        # deploy contract: build_sweep_list needs a top-level SPEC
N_TASKS = len(SPECS)


def main(task_idx=0):
    idx = int(task_idx)
    spec = SPECS[idx]
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    lmpt = eng.import_lumopt2()
    project, _ = eng.make_project(spec, out_dir, lmpt)
    cb = eng.make_log_callback(spec, out_dir, lmpt=lmpt)     # no anchors -> no trips
    p = eng.seed_params(spec)
    fom = project.compute_fom(p)
    try:
        cb.on_function_eval(project, 0, p, fom)
    except (eng.RecenterNeeded, eng.WidthTrip):
        pass                                                 # probe reports, never gates
    with open(os.path.join(out_dir, f"{spec.label}_evals.jsonl")) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[rhoneut {idx} a={AMPLITUDES[idx]}] T {row.get('t_pk')} "
          f"sigma {row.get('sigma_um')} FWHM {row.get('fwhm_env_um')} Q_i {row.get('q_i')} "
          f"lam {row.get('lam_pk_nm')} | origin T 0.8926")


if __name__ == "__main__":
    for i, a in enumerate(AMPLITUDES):
        prof = shape_profile(a)
        p = eng.seed_params(SPECS[i])
        b = np.array(eng.param_bounds(SPECS[i]))
        ok = bool(((p >= b[:, 0] - 1e-9) & (p <= b[:, 1] + 1e-9)).all())
        print(f"task {i}: a={a:.1f}  corr {prof.min():.1f}..{prof.max():.1f} nm  "
              f"rho={prof.mean() / eng.CORR_NM:.6f}  2kL={eng.two_kappa_L(p, 100):.3f}  "
              f"bounds_ok={ok}")
