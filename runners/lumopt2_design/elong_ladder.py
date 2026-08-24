"""Elongation ladder on the UNIFORM corr-325 seed: the width-vs-elongation curve.

Study dir: runners/lumopt2_design/   |   Created 2026-08-23   |   Job(s): TBD
Purpose: the fwhm_wall's elongation term is calibrated on the fspw family
(6 points, noshift/best/d020-d080), where the local slope rises 0.0135 ->
0.061 um/nm and a quadratic FW_C_ELONG = 1.07e-4 fits to <=0.25 um. But the
uniform corr-325 device — the one campaigns 136466/136468 actually optimise —
has only TWO measured points of its own, and the quadratic UNDER-predicts its
far one (8.9 predicted vs 13.9 um measured at elong 287.5). Stored data cannot
distinguish super-quadratic curvature from a device difference (Fable audit
2026-08-23). This ladder measures the curve directly, one forward per rung.

Elongation is the COMMON MODE of the shift block, 2*sum(shift); each rung sets
every free shift to the same s = e / (2*N_FREE), which isolates that mode
exactly (a uniform s has no differential component at all).

  task 0   e =  60.0 nm    s = 1.20
  task 1   e = 120.0 nm    s = 2.40
  task 2   e = 180.0 nm    s = 3.60
  task 3   e = 240.0 nm    s = 4.80
  task 4   e = 287.5 nm    s = 5.75

NOT re-run, cited from storage (CLAUDE.md §6 — never re-measure a stored point):
  e = 0      -> fwhm_env 18.345 um, T 0.9012   (136466 eval 1, its own seed)
  e = 287.5  -> fwhm_env 32.268 um, T 0.9558   (136466 eval 2)
The stored 287.5 point is NOT the same design as task 4: that probe's shifts
were non-uniform (mean 5.75, max 7.82), so it carries a differential component
too. Task 4 is the pure common mode at the same elongation — the difference
between them is exactly how much of the far-tail gap is pattern, not curvature.

Numerics come from campaign_v2_uniform's SPEC via dataclasses.replace, so every
rung is bit-identical to the campaign it explains (§2: compare only within
identical numerics). The wall is switched OFF here — this study MEASURES width,
it never steers on it.

Dispatch (short lane, 5 single forwards ~33 min each):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \\
        bash <cluster>/deploy_<cluster>.sh --lumopt2-design=runners.lumopt2_design.elong_ladder
Output -> results/elong_{060,120,180,240,287}/.
"""

import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.campaign_v2_uniform import SPEC as UNIFORM_SPEC

ELONGS_NM = [60.0, 120.0, 180.0, 240.0, 287.5]

# Elongation drags the resonance redward, so each rung records on its OWN
# centre or the peak walks toward the window edge. DERIVED from the two stored
# 136466 rows (MEASURED lam_pk 1564.6140 at e=0, 1568.2224 at e=287.5):
# +3.6084 nm / 287.5 nm = 0.012551. Without this the top rung sits 1.39 nm from
# the +5 nm edge — inside, but tighter than the +-2.5*FWHM the FOM window wants
# and with no slack if a UNIFORM pattern drifts differently from that
# non-uniform probe. Re-centring keeps dlambda (10 nm / 501 pts) and the region
# mesh identical, so widths stay comparable to the campaign (§2); it is exactly
# what run_campaign itself does on RecenterNeeded.
LAM_PER_ELONG = 0.012551         # nm of lam_pk per nm of 2*sum(shift)


def rung_params(e_nm):
    """Uniform seed with every free shift set to the common mode e/(2*N_FREE)."""
    p = np.asarray(eng.seed_params(UNIFORM_SPEC), dtype=float).copy()
    p[eng.SL_SHIFT] = e_nm / (2.0 * eng.N_FREE)
    return p


SPECS = {i: dataclasses.replace(UNIFORM_SPEC,
                                label=f"elong_{int(e):03d}",
                                fwhm_wall=False, fw_anchor=None,
                                scan_center_nm=round(UNIFORM_SPEC.scan_center_nm
                                                     + LAM_PER_ELONG * e, 3),
                                seed_override=tuple(rung_params(e)))
         for i, e in enumerate(ELONGS_NM)}
SPEC = SPECS[0]          # deploy contract: build_sweep_list needs a top-level SPEC
N_TASKS = len(SPECS)


def main(task_idx=0):
    idx = int(task_idx)
    spec = SPECS[idx]
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    p = eng.replay_params(spec, np.asarray(spec.seed_override, dtype=float))
    lmpt = eng.import_lumopt2()
    project, _ = eng.make_project(spec, out_dir, lmpt)
    cb = eng.make_log_callback(spec, out_dir, lmpt=lmpt)
    fom = project.compute_fom(p)
    try:
        cb.on_function_eval(project, 0, p, fom)
    except (eng.RecenterNeeded, eng.WidthTrip):
        pass                                          # ladder reports, never gates
    with open(os.path.join(out_dir, f"{spec.label}_evals.jsonl")) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[elong_ladder {idx}] e={2*p[eng.SL_SHIFT].sum():.1f} nm  "
          f"fwhm_env {row.get('fwhm_env_um')} um  T {row.get('t_pk')}  "
          f"lam {row.get('lam_pk_nm')}  Q_i {row.get('q_i')} "
          f"| stored e=0: 18.345 um / T 0.9012")


if __name__ == "__main__":
    for i, e in enumerate(ELONGS_NM):
        p = rung_params(e)
        b = np.array(eng.param_bounds(SPECS[i]))
        ok = bool(((p >= b[:, 0] - 1e-9) & (p <= b[:, 1] + 1e-9)).all())
        print(f"task {i}: e={2*p[eng.SL_SHIFT].sum():6.1f} nm  s={p[eng.SL_SHIFT][0]:5.2f}  "
              f"corr_mean={p[eng.SL_CORR].mean():.1f}  bounds_ok={ok}")
