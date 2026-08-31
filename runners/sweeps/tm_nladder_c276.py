"""TM corr-276 saturation-onset rung: does Q_i bend where the predictor says?

Study dir: runners/sweeps/   |   Created 2026-09-01   |   Job(s): IGUM TBD (1 task)
Purpose: first live validation rung for the q3db predictive engine
(python_tools/predict_q3db.py, memory project_q3db_predictive_engine.md).
The tm_bare_c276 family fit (rows N=110-165, all stored) predicts at N=200:
    T = 0.5641 (-2.49 dB)  [band 0.5257..0.5998]
    Q_L = 19599            [band 18375..20879]
    width = 23.94 um, lambda = 1559.93 nm            (all PREDICTED, pre-registered)
The run DECIDES the c276 Q_i saturation onset (fitted sat ~ 8.2e4): if Q_i has
not begun saturating, measured Q_L lands ABOVE the band. Stored rows cannot
answer this — N=165 is the family's highest rung.

Reused points (CLAUDE.md section 6, no re-runs): c276 bare rows N=110/125/140/
150/165 from results_from_igum/trench_q3db_20um/results/ (IGUM 47910 family
jobs) at these EXACT numerics. NO control row — nothing on the section-2
numerics list changed vs those stored rows.

Physics line (section 4): TM h350, pitch 516.83, corr 276, W800; target
resonance ~1559.9 nm, window 20 nm centered 1559.5 (1549.5-1569.5, 4001 pts =
5 pm; predicted Q_L ~2e4 -> line ~80 pm -> ~16 samples, adequate; no high-Q
window needed). Expected solve ~1.5-2 h (tau ~ 17 ps).

Dispatch (IGUM, queue-empty check + license probe first):
    ARRAY_TIME=04:00:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.sweeps.tm_nladder_c276 --max-concurrent=1
Output -> results/tm_nladder_c276/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_NM        = 276.0
N_LADDER       = [200]

BOX_Y_UM       = 8.0         # q3db family numerics exactly (trench_q3db_20um ctrl arm)
SCAN_CENTER_NM = 1559.5
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.center_wavelength_m = SCAN_CENTER_NM * 1e-9
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.spectral.n_wl_points = N_WL_POINTS
assert BASE.symmetry.use_z_symmetry, "bare device is z-symmetric — keep the 2x z saving"

assert max(N_LADDER) > 165, "N<=165 rows exist stored (trench_q3db ctrl arm) — never re-run them"

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM] * len(N_LADDER),
    n_periods_each_side  = list(N_LADDER),
    center_wavelength_nm = [SCAN_CENTER_NM] * len(N_LADDER),
    scatterer_radius_nm  = [0.0] * len(N_LADDER),
    mode  = "zipped",
    label = "tm_nladder_c276",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"  task 0: corr={CORR_NM:.0f} N={N_LADDER[0]}  bare device")
    print("pre-registered PREDICTION: T 0.5641 [0.5257..0.5998], Q_L 19599 [18375..20879],")
    print("width 23.94 um, lambda 1559.93 nm — run decides c276 Q_i saturation onset")
