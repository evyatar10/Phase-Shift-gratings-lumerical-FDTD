"""lumopt2 campaign seedA STAGE-2: shifts FROZEN, refine corr/avg/comb/cavity.

Study dir: runners/lumopt2_design/   |   Created 2026-08-17   |   Job(s): TBD
Purpose (skill item 19, user-approved 2026-08-17 morning): stage-1 seedA
(job 133276) plateaued ON the elongation wall - the shift direction is
exhausted and width-coupled, yet L-BFGS-B kept spending ~half its solves
probing it (zero gradient inside the deadband). Stage-2 warm-starts from
stage-1's best and FREEZES the shift block at its discovered values
(sliver bounds via freeze_shifts), so every solve goes to corr/avg/comb/
cavity - the directions where genuine (Q_i/sigma^2-raising) gains live.
Width safety is STRICTER than stage-1: the shift lever is gone entirely and
the sigma tripwire keeps the ORIGINAL anchor 17.493, so total drift stays
capped at +2 % cumulative (seed sits at ratio 1.0148 -> ~0.5 % headroom).

Seed = stage-1 best (MEASURED eval 8, job 133276, 2026-08-17 ~09:00):
FOM 0.68831 / T 0.9313 / lam 1565.975 / sigma 17.7519 (2*Sig_shift 132.6 nm).

Dispatch (Athena; stop stage-1 133276 first - its lane and seat):
    SBATCH_MEM=160G LUMOPT2_QOS=4d_1g LUMOPT2_TIME=72:00:00         bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.campaign_c325_seedA2
Output -> results/campaign_c325_seedA2/results/lumopt2_c325_seedA2/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from runners.lumopt2_design import lumopt2_design as eng

SIGMA0_UM = 17.493    # ORIGINAL B2b anchor - cumulative +2 % width cap (user rule)

SEED = (
    316.785926, 321.202970, 322.324412, 322.799698, 322.983956, 323.449596,
    323.750924, 324.046881, 324.430165, 324.521862, 324.637055, 324.872853,
    324.870286, 324.903441, 325.034589, 324.966743, 324.947677, 325.006419,
    324.905712, 324.865813, 324.889348, 324.785703, 324.756495, 324.780908,
    324.695255, 800.099473, 800.066435, 800.114426, 800.103422, 800.055714,
    800.019375, 799.995384, 799.972881, 799.964484, 799.957647, 799.947219,
    799.951468, 799.955645, 799.954383, 799.964850, 799.972057, 799.973248,
    799.983891, 799.989998, 799.991122, 799.999843, 800.003862, 800.004239,
    800.011153, 800.013952, 3.224022, 2.842583, 4.227598, 5.345013,
    5.813710, 6.291518, 5.948353, 5.182922, 4.734409, 3.867847,
    2.968978, 2.258583, 1.445545, 0.902473, 0.889229, 0.881279,
    0.880372, 0.886740, 0.897358, 0.912543, 0.933026, 0.953834,
    0.978873, 1.007924, 1.035239, 80.008544, 80.008090, 80.009024,
    80.013323, 80.014981, 80.016871, 80.025610, 80.028212, 80.029861,
    80.039594, 80.042778, 80.046246, 80.057706, 80.061386, 80.064345,
    80.070516, 80.067120, 80.059040, 80.052868, 80.036417, 80.011806,
    79.992273, 79.978185, 79.992732, 80.011328, 80.028165, 79.985100,
    80.006709, 79.994499, 79.997407, 80.035288, 80.013199, 79.988259,
    79.981112, 79.995815, 80.021485, 80.043336, 80.055903, 80.061825,
    80.070731, 80.067363, 80.063340, 80.063275, 80.053114, 80.045914,
    80.043993, 80.034981, 80.029890, 80.028248, 80.022494, 80.018615,
    80.016738, 80.011610, 80.009344, 80.009480, 80.005960, 80.005422,
    -14467.000319, -13936.001027, -13405.002918, -12874.003229, -12343.004153, -11812.006126,
    -11281.005655, -10750.007516, -10219.008853, -9688.009493, -9157.011784, -8626.011894,
    -8095.014818, -7564.016804, -7033.013380, -6502.017822, -5971.015505, -5440.009062,
    -4909.010124, -4377.998586, -3846.991223, -3315.988013, -2784.982236, -2253.991502,
    -1723.025865, -1192.054712, -660.988320, -129.956269, 400.947273, 931.984642,
    1463.060959, 1994.019079, 2524.989754, 3055.984021, 3586.984531, 4117.996792,
    4649.001622, 5180.002684, 5711.012067, 6242.012220, 6773.014456, 7304.014766,
    7835.012193, 8366.013783, 8897.010920, 9428.008383, 9959.009350, 10490.006644,
    11021.006392, 11552.005841, 12083.003779, 12614.004326, 13145.003170, 13676.001799,
    14207.001553, 14738.000669, 15268.999815, 1899.076864, 812.666958,
)

SPEC = eng.CampaignSpec(
    label="lumopt2_c325_seedA2",
    box_y_um=6.8, box_z_mult=4.14,          # identical numerics to stage-1
    scan_center_nm=1566.0,                  # window centered on the seed peak
    seed_override=SEED, freeze_shifts=True, # stage-2: shift block pinned
    max_iter=30, max_feval=55,
    adj_phase_fix=True, adj_fix_re=1.0561, adj_fix_im=0.1239,  # C-fix
)
N_TASKS = 1


def main(task_idx=0):
    out_dir = os.path.join(config.RESULTS_DIR, SPEC.label)
    eng.run_campaign(SPEC, out_dir, sigma0_um=SIGMA0_UM)


if __name__ == "__main__":
    p = eng.seed_params(SPEC); b = eng.param_bounds(SPEC)
    frozen = sum(1 for i in range(50, 75) if b[i][1] - b[i][0] < 0.01)
    print(f"stage-2 seedA: {frozen}/25 shifts frozen, seed FOM ref 0.68831, "
          f"center {SPEC.scan_center_nm} nm, sigma0 {SIGMA0_UM}")
