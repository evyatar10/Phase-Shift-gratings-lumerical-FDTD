"""Comb-under-dip A/B: is the comb worth anything ON the dip+shift design?

Study dir: runners/lumopt2_design/   |   Created 2026-08-17   |   Job(s): TBD
Purpose (user, 2026-08-17 night): the comb helps at the UNIFORM origin
(MEASURED +0.0107 T job 132631, +0.0112 at PVA job 132654) but the optimizer
holds it frozen on the dip profile — zero gradient cannot distinguish
"already optimal" from "does nothing under this profile" (precedent: trench
nulled under apod-20). So measure the VALUE, not the derivative: one forward
each of seedB's current-best geometry WITH and WITHOUT the comb, identical
numerics. ★CONCLUSIONS ONLY — no campaign change follows without the user.

  task 0  comb    seedB eval-5 params as-is          (label comb_dip_ab_comb)
  task 1  nocomb  same grating params, bare=True     (label comb_dip_ab_nocomb)

Target λ ≈ 1565.9 (seedB eval-5 peak), window 6 nm / 301 pts — same recording
as the campaigns, re-centered so the reader window is not edge-clipped.
Verdict guide: dT = T(comb) − T(nocomb) vs +0.0107 at origin and vs the
±0.001–0.002 numerics floor; also dQ_i and the σ ratio (comb was
width-neutral at origin, ratio 0.9993).

Dispatch (Athena, 2h_2g lane — separate QOS quota from the 4d_1g campaign,
so seedA keeps its GPU; ~70 min/task expected):
    SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \
        bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.comb_dip_ab
Output -> results/comb_dip_ab_{comb,nocomb}/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng

N_TASKS = 2
SCAN_CENTER_NM = 1565.9

# seedB best-so-far (eval 5: FOM 0.7001, T 0.9451, λ 1565.905, σ 17.705 µm;
# IGUM job 54968, lumopt2_c325_seedB_evals.jsonl, read 2026-08-17).
# Layout: 25 corr | 25 avg | 25 shift | 57 r | 57 x | d_comb | cavity_w (nm).
P_BEST = np.array([
    234.220433, 266.999918, 292.662907, 313.041778, 337.747386, 338.513102,
    338.868204, 339.173875, 339.569427, 339.660395, 339.767617, 339.906175,
    325.051014, 324.987326, 325.122352, 325.058290, 325.043761, 325.106477,
    325.004002, 324.961285, 324.984792, 324.880222, 324.849398, 324.873420,
    324.786902, 800.097356, 800.067599, 800.104314, 800.104830, 800.059967,
    800.036790, 800.020984, 799.997221, 799.985702, 799.975149, 799.960052,
    799.955087, 799.956335, 799.952833, 799.960518, 799.967248, 799.968733,
    799.980375, 799.987171, 799.988642, 799.998034, 800.002448, 800.002993,
    800.010294, 800.013221, 1.995267, 2.176694, 3.417239, 4.604850,
    5.364206, 5.937691, 5.871152, 5.341738, 5.070075, 4.320660,
    3.462680, 2.739073, 1.812653, 1.172269, 0.790345, 0.775681,
    0.766201, 0.765371, 0.773650, 0.783937, 0.800272, 0.820955,
    0.842778, 0.869234, 0.897290, 80.009344, 80.008978, 80.010109,
    80.014795, 80.016846, 80.019259, 80.028867, 80.032144, 80.034440,
    80.044964, 80.047895, 80.049455, 80.058182, 80.059118, 80.060860,
    80.065556, 80.056351, 80.046412, 80.038618, 80.020593, 80.001951,
    79.993247, 79.985984, 79.996844, 80.011815, 80.018848, 79.988961,
    79.998194, 79.989929, 79.997674, 80.025061, 80.015067, 79.991376,
    79.985205, 79.991989, 80.004832, 80.023097, 80.040711, 80.051581,
    80.059045, 80.067155, 80.063314, 80.062380, 80.056244, 80.051960,
    80.051354, 80.041759, 80.035572, 80.033062, 80.026346, 80.021692,
    80.019289, 80.013515, 80.010871, 80.010824, 80.006964, 80.006247,
    -14467.000140, -13936.000832, -13405.002791, -12874.003079, -12343.003994, -11812.006068,
    -11281.005396, -10750.007284, -10219.008565, -9688.008700, -9157.010709, -8626.009844,
    -8095.011855, -7564.013481, -7033.009599, -6502.012965, -5971.009112, -5440.002960,
    -4909.003952, -4377.994665, -3846.990656, -3315.991343, -2784.987970, -2253.996162,
    -1723.025593, -1192.045739, -660.997150, -129.967204, 400.961882, 931.993207,
    1463.050594, 1994.020399, 2524.992173, 3055.987719, 3586.986927, 4117.992335,
    4648.994225, 5179.996125, 5711.005128, 6242.004749, 6773.009816, 7304.011475,
    7835.008610, 8366.011337, 8897.009005, 9428.007101, 9959.008954, 10490.006186,
    11021.006185, 11552.005692, 12083.003514, 12614.004240, 13145.002986, 13676.001573,
    14207.001367, 14738.000437, 15268.999589, 1899.129210, 809.578246,
])
assert P_BEST.shape == (eng.N_PARAMS,)

SPECS = {
    0: eng.CampaignSpec(label="comb_dip_ab_comb", scan_center_nm=SCAN_CENTER_NM),
    1: eng.CampaignSpec(label="comb_dip_ab_nocomb", scan_center_nm=SCAN_CENTER_NM,
                        bare=True, free_comb=False),
}
SPEC = SPECS[0]   # deploy contract: build_sweep_list requires top-level SPEC
                  # (first dispatch 133394 FAILED 8 s on exactly this — and the
                  # deploy submitted anyway on a fallback 1-task list; deploy
                  # should abort on sweep-list build failure, parked fix)


def main(task_idx=0):
    spec = SPECS[int(task_idx)]
    out_dir = os.path.join(config.RESULTS_DIR, spec.label)
    p = eng.replay_params(spec, P_BEST)   # resets inert comb slots for bare
                                          # specs + asserts bounds (skill #17)
    lmpt = eng.import_lumopt2()
    project, _ = eng.make_project(spec, out_dir, lmpt)
    cb = eng.make_log_callback(spec, out_dir, lmpt=lmpt)  # no sigma0 → no trips
    fom = project.compute_fom(p)
    try:
        cb.on_function_eval(project, 0, P_BEST, fom)
    except (eng.RecenterNeeded, eng.WidthTrip):
        pass                                    # A/B only reports, never gates
    path = os.path.join(out_dir, f"{spec.label}_evals.jsonl")
    with open(path) as f:
        row = json.loads(f.readlines()[-1])
    print(f"[comb_dip_ab task {task_idx} {spec.label}] FOM {fom:.5f}  "
          f"T {row.get('t_pk')}  λ {row.get('lam_pk_nm')}  Q {row.get('q_loaded')}  "
          f"Q_i {row.get('q_i')}  σ {row.get('sigma_um')} µm")


if __name__ == "__main__":
    for i, s in SPECS.items():
        print(f"task {i}: {s.label}  bare={s.bare}  center {s.scan_center_nm} nm")
