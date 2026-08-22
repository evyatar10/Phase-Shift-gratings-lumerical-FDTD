"""Comb BASIN scan: is the comb globally right for the apodized+shifted device?

Study dir: runners/lumopt2_design/   |   Created 2026-08-17   |   Job(s): TBD
Purpose (user, 2026-08-17 evening): the gradient leaves the comb motionless
(MEASURED: 3 accepted stage-2 steps move it <0.03 nm while cavity width moves
+25.8 nm; its scaled gradient is ~500x smaller than the shift/cavity ones), and
the A/B says the comb is WORTH +0.0048 T on this profile. Both facts are LOCAL.
The comb geometry (pitch 531, phase 270 deg, r 80, d 1.9 um) was optimized in
the q3db decoration program against a UNIFORM corr-325 grating with no tooth
shifts; this device now has a corrugation dip (234..339 nm) and 2*Sig_s = 124 nm
of shift. This scan asks the question a gradient cannot: is there a DIFFERENT
comb basin that matches the new tooth arrangement better?

Base = seedB eval-5 (imported from comb_dip_ab, not re-pasted), which already
has BOTH anchors measured at these exact numerics (CLAUDE.md section 6 -> no
control row here): comb as-is T 0.94629 / Q_i 75,361, comb REMOVED T 0.94147.
Every task below is one forward at those same numerics, so all dT are directly
comparable to that pair.

Physics being tested, in order of expected effect:
  phase  - tooth shifts move the local Bragg phase; the comb's 270 deg offset
           was set against the unshifted lattice. Scans the full 531 nm period.
  pitch  - momentum matching. 516.83 = commensurate with the grating (the
           "locked" hypothesis); 524/540 bracket the current 531.
  r, d   - scatterer strength / evanescent overlap; already optimized against
           the same mode width, so these are falsification checks, not hopes.

  task 0-2  phase  +132.75 / +265.50 / +398.25 nm   (90 / 180 / 270 deg)
  task 3-5  pitch  516.83 / 524.0 / 540.0 nm        (center site held at 401)
  task 6-7  radius 70 / 100 nm  (70 = campaign floor)
  task 8    comb distance d 1700 nm  (from 1899)

Target resonance 1565.9 nm, window +/-3 nm at 20 pm (301 pts) - identical
recording to the campaigns and to the A/B, so nothing is re-measured.
CONCLUSIONS ONLY: no campaign change follows without the user.

Dispatch (short lane, keeps the campaign GPUs; ~55 min/task):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.comb_basin_scan
Output -> results/comb_basin_<label>/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.comb_dip_ab import P_BEST      # seedB eval-5 vector

SCAN_CENTER_NM = 1565.9
LAM_COMB_NM = 531.0        # current comb pitch
X_CENTER_NM = 401.0        # current center site (site 28 of 57) = the 270 deg offset
N_SITES = 57

# (label, kind, value) - one forward each. kind is applied to the comb block only.
VARIANTS = [
    ("ph090", "phase", 132.75),
    ("ph180", "phase", 265.50),
    ("ph270", "phase", 398.25),
    ("pitch516", "pitch", 516.83),
    ("pitch524", "pitch", 524.0),
    ("pitch540", "pitch", 540.0),
    ("r070", "radius", 70.0),      # 70 = the campaign radius floor, not a free choice
    ("r100", "radius", 100.0),
    ("d1700", "dist", 1700.0),
]
N_TASKS = len(VARIANTS)


def comb_variant(kind, value):
    """P_BEST with one comb property changed. Grating block never touched."""
    p = P_BEST.copy()
    if kind == "phase":
        p[eng.SL_X] += value
    elif kind == "pitch":
        p[eng.SL_X] = X_CENTER_NM + (np.arange(N_SITES) - 28) * value
    elif kind == "radius":
        p[eng.SL_R] = value
    elif kind == "dist":
        p[eng.I_DCOMB] = value
    else:
        raise ValueError(kind)
    return p


# seed_override = the MODIFIED vector, so param_bounds centers the comb-x box on
# the new sites (bounds are seed +/-100 nm; a pitch change moves outer sites by
# up to ~400 nm and would otherwise trip the bounds assert - the 133395 trap).
SPECS = {i: eng.CampaignSpec(label=f"comb_basin_{lbl}", scan_center_nm=SCAN_CENTER_NM,
                             seed_override=tuple(comb_variant(kind, val)))
         for i, (lbl, kind, val) in enumerate(VARIANTS)}
SPEC = SPECS[0]      # deploy contract: build_sweep_list requires a top-level SPEC


def main(task_idx=0):
    idx = int(task_idx)
    lbl, kind, val = VARIANTS[idx]
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
        pass                                                 # scan reports, never gates
    with open(os.path.join(out_dir, f"{spec.label}_evals.jsonl")) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[comb_basin {idx} {lbl}] {kind}={val}  FOM {fom:.5f}  T {row.get('t_pk')}  "
          f"lam {row.get('lam_pk_nm')}  Q_i {row.get('q_i')}  sigma {row.get('sigma_um')} um "
          f"| anchors at same numerics: comb 0.94629, no-comb 0.94147")


if __name__ == "__main__":
    for i, (lbl, kind, val) in enumerate(VARIANTS):
        p = comb_variant(kind, val)
        b = np.array(eng.param_bounds(SPECS[i]))
        ok = bool(((p >= b[:, 0] - 1e-9) & (p <= b[:, 1] + 1e-9)).all())
        print(f"task {i}: {lbl:9s} {kind:6s}={val:8.2f}  x[0]={p[eng.SL_X][0]:10.2f} "
              f"x[-1]={p[eng.SL_X][-1]:9.2f}  r={p[eng.SL_R][0]:6.2f}  d={p[eng.I_DCOMB]:7.1f}"
              f"  bounds_ok={ok}")
