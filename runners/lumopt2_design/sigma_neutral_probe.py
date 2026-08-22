"""sigma-NEUTRAL trade probe: more transmission at the SAME mode width.

Study dir: runners/lumopt2_design/   |   Created 2026-08-18   |   Job(s): TBD
Purpose (user 2026-08-18: "keep the sigma at the intended ratio, it's already
big enough, and see if there's anything with bigger transmission"): the shift
ladder proved T rises monotonically with tooth shift and that the WIDTH SPEC —
not physics — is what stops it. So the only remaining direction is the trade:
push the shifts further UP while paying the width back through corrugation,
landing on the SAME sigma. stage-4 is searching that direction with gradients
and creeping; this measures it directly along the calibrated line.

The payback recipe uses the MEASURED slopes (tangent probe 133512):
    dsigma/d(2*Sig_s) = +0.00368 um/nm      (shifts widen the mode)
    dsigma/drho       = -3.85 um            (more corrugation narrows it)
=> to hold sigma fixed while adding D nm of total shift:
       d_rho  = 0.00368*D/3.85       and    d_corr = d_rho * 325 nm on every
       free tooth. Shifts are SCALED (preserving the stage-1 bump shape), not
       offset, matching how the ladder moved them.

  task 0  2*Sig_s +20 nm  (x1.153)  corr +6.2 nm
  task 1  2*Sig_s +40 nm  (x1.306)  corr +12.4 nm
  task 2  2*Sig_s +60 nm  (x1.459)  corr +18.6 nm
  task 3  2*Sig_s +80 nm  (x1.613)  corr +24.9 nm

Base + control = BEST_T9635 (T 0.9635 / sigma 17.7952 / ratio 1.0173), already
measured at these numerics -> NO control row is re-run.
READING: sigma should stay ~17.795 on every row (that is the point — it also
re-tests the width surrogate at the frontier). If T RISES along the ladder the
trade is alive and a new campaign stage should be seeded from the best row; if
T is flat or falls, the trade is exhausted and stage-4's creeping is explained,
which closes the optimization at this constraint.
Note x1.5-scaled shifts ALONE measured sigma 18.062 (ratio 1.0325, OUT) in the
ladder — so any row here that lands near 17.795 is proof the payback works.

Dispatch (Athena short lane; does NOT touch stage-4's 4d_1g quota):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \\
        bash athena/deploy_athena.sh \\
        --lumopt2-design=runners.lumopt2_design.sigma_neutral_probe --max-concurrent=1
Output -> results/signeut_d{020,040,060,080}/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9635

SCAN_CENTER_NM = 1566.14          # BEST_T9635 resonance 1566.144
DELTAS_NM = [20.0, 40.0, 60.0, 80.0]      # added total shift 2*Sig_s
A_SHIFT, A_RHO, CORR0 = eng.SIG_A_SHIFT, abs(eng.SIG_A_RHO), 325.0

_P = np.asarray(BEST_T9635, dtype=float)
_ELONG0 = 2.0 * _P[eng.SL_SHIFT].sum()


def traded_vector(d_nm):
    """BEST_T9635 with +d_nm of total shift and the corrugation payback."""
    p = _P.copy()
    p[eng.SL_SHIFT] *= (_ELONG0 + d_nm) / _ELONG0        # scale, keep the bump shape
    p[eng.SL_CORR] += (A_SHIFT * d_nm / A_RHO) * CORR0   # width payback, uniform
    return p


SPECS = {i: eng.CampaignSpec(label=f"signeut_d{int(d):03d}",
                             scan_center_nm=SCAN_CENTER_NM,
                             seed_override=tuple(traded_vector(d)))
         for i, d in enumerate(DELTAS_NM)}
SPEC = SPECS[0]        # deploy contract: build_sweep_list needs a top-level SPEC
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
        pass                                                 # probe reports, never gates
    with open(os.path.join(out_dir, f"{spec.label}_evals.jsonl")) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[signeut {idx} d+{DELTAS_NM[idx]:.0f}] 2Ss={2*p[eng.SL_SHIFT].sum():.1f} nm "
          f"corr1={p[0]:.1f} | T {row.get('t_pk')} sigma {row.get('sigma_um')} "
          f"fwhm {row.get('fwhm_env_um')} Q_i {row.get('q_i')} "
          f"| control T 0.9635 sigma 17.7952")


if __name__ == "__main__":
    for i, d in enumerate(DELTAS_NM):
        p = traded_vector(d)
        b = np.array(eng.param_bounds(SPECS[i]))
        ok = bool(((p >= b[:, 0] - 1e-9) & (p <= b[:, 1] + 1e-9)).all())
        el = 2 * p[eng.SL_SHIFT].sum()
        rho = p[eng.SL_CORR].mean() / CORR0
        sig_hat = 17.7952 + eng.SIG_A_SHIFT * (el - _ELONG0) + eng.SIG_A_RHO * (
            rho - _P[eng.SL_CORR].mean() / CORR0)
        print(f"task {i}: +{d:4.0f} nm -> 2Ss {el:6.1f} (x{el/_ELONG0:.3f}), "
              f"corr {p[eng.SL_CORR].min():.1f}..{p[eng.SL_CORR].max():.1f}, "
              f"predicted sigma {sig_hat:.4f} um (target 17.7952), bounds_ok={ok}")
