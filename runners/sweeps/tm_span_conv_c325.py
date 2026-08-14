"""TM corr-325 transverse-domain CONVERGENCE at the inverse-design surrogate (N=100).

Study dir: runners/sweeps/   |   Created 2026-08-12   |   Job(s): TBD
Purpose (user): the corr-325 campaign inherited y=8.0 um / z=8.8 um from the q3db
family, but that box was converged on corr-400, which radiates ~35% MORE power
(radiated ~ kappa^2; kappa 0.0440 vs 0.0353 /um). If corr-325 converges in a
smaller box, every inverse-design iteration gets cheaper by the transverse-area
ratio -- y 8.0->5.8 and z 8.8->6.8 would cut cells 70.4 -> 39.4 um^2 (-44%).
This study finds the smallest (y, z) whose observables still match the big box.

Device: bare corr-325, N=100 per side (the settled surrogate: 2*kappa*L = 3.65),
h350 / pitch 516.83 / W800 / n 1.97-1.444, optimization mesh, q3db spectral window.

Rows (zipped; y_span_um sets Y alone, span_mult then drives Z alone):
  0-2  Y-LADDER at the known-safe z=8.8 um : y = 4.8, 5.8, 6.8
  3-5  Z-LADDER at y=6.8 um                : z = 5.8, 6.8, 10.8  (mult 3.50/4.14/6.70)
  6    CHEAP CORNER (cross-term)            : y = 5.8, z = 6.8
  => 7 tasks, every one SMALLER or equal in cells than the current numerics
     except row 5 (z=10.8), which is the "is 8.8 actually enough?" upper check.

NO control row (CLAUDE.md section 6 no-rerun): the (y 8.0, z 8.8) point IS the
stored tm_nladder_c325 N=100 row -- IGUM job 51736, T 0.910 / Q 1760 / mode
19.24 um / lambda 1559.01 -- at these exact numerics. All deltas are vs that file.

Acceptance: converged = successive box steps move T by less than the measured
dx=50 nm jitter floor (~0.002). Read T, lambda_res, Q and the DERIVED
Q_i = Q/(1-sqrt(T)) -- Q_i is what the campaign actually ranks on, so it is the
quantity that must be box-independent, not just T. z is the axis that burned us
on corr-400 (round 2, job 116870: T moved +9 points from z 3.2 -> 5.8 and had
not converged), so treat any z shrink as guilty until row 3/4 prove otherwise.

NOTE (section 7): convergence-study results are KEEP-FOREVER data.

Dispatch (queue must be EMPTY of other --option3 arrays -- shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_span_conv_c325
Output -> results/tm_span_conv_c325/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_NM        = 325.0
N_PERIODS      = 100         # settled surrogate, 2*kappa*L = 3.65 (see memory: tm_nladder_surrogate)

SCAN_CENTER_NM = 1559.5      # q3db family numerics, identical to tm_nladder_c325
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001

# Stored reference point that this study is measured against -- never re-run.
REF_Y_UM, REF_Z_MULT = 8.0, 5.42

# z_span = core_height + span_mult * lambda_center  ->  mult = (z - 0.35) / 1.5595
_Y_UM   = [4.8,  5.8,  6.8,  6.8,  6.8,  6.8,  5.8]
_Z_MULT = [5.42, 5.42, 5.42, 3.50, 4.14, 6.70, 4.14]

assert len(_Y_UM) == len(_Z_MULT) == 7
assert len(set(zip(_Y_UM, _Z_MULT))) == 7, "duplicate (y, z) row -> file-tag collision"
assert (REF_Y_UM, REF_Z_MULT) not in set(zip(_Y_UM, _Z_MULT)), "that point is stored (job 51736)"

BASE = _common.build_ports_base()
BASE.scatterer.enabled = False          # bare device -- this is a domain study
BASE.y_span_override_m = None           # per-row via y_span_um
BASE.spectral.center_wavelength_m = SCAN_CENTER_NM * 1e-9
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.spectral.n_wl_points = N_WL_POINTS
assert BASE.symmetry.use_z_symmetry, "bare device is z-symmetric -- keep the 2x z saving"

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM] * len(_Y_UM),
    n_periods_each_side  = [N_PERIODS] * len(_Y_UM),
    center_wavelength_nm = [SCAN_CENTER_NM] * len(_Y_UM),
    scatterer_radius_nm  = [0.0] * len(_Y_UM),
    y_span_um            = list(_Y_UM),
    span_mult            = list(_Z_MULT),
    mode  = "zipped",
    label = "tm_span_conv_c325",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (y, m) in enumerate(zip(_Y_UM, _Z_MULT)):
        z = 0.35 + m * SCAN_CENTER_NM * 1e-3
        print(f"  task {i}: y={y:.1f} um  z={z:.2f} um (mult {m:.2f})  area={y * z:5.1f} um^2")
    z_ref = 0.35 + REF_Z_MULT * SCAN_CENTER_NM * 1e-3
    print(f"reference (job 51736 N=100, NOT re-run): y={REF_Y_UM} z={z_ref:.2f} um "
          f"area={REF_Y_UM * z_ref:.1f} um^2 -> T 0.910 / Q 1760 / mode 19.24 um")
    print("converged = |dT| < 0.002 between successive box steps, Q_i = Q/(1-sqrt(T)) flat")
