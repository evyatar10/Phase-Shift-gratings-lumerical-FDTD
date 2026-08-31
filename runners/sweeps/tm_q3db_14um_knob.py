"""Corr-knob live test: design a -3 dB / 14 um TM device in ONE run, no ladder.

Study dir: runners/sweeps/   |   Created 2026-09-01   |   Job(s): IGUM TBD (1 task)
Purpose: first LIVE test of predict_q3db's corrugation knob + cross-corr scaling
(user example: "let's say it's a fourteen micron mode"). The tool retuned the
tm_bare_c325 family to a 14.0 um width target via the measured per-TM knob line
(corr* = 448.4 nm) and solved the -3 dB point (N* = 97.5 -> integer 98).
Pre-registered PREDICTION (EXPECTED-grade — the corr rescaling uses kappa prop.
corr [MEASURED 0.1-1.3%] and Q_i prop. corr^-2.9 [MEASURED at N=150 only]):
    T = 0.4937 [band 0.4576..0.5280]
    Q_L = 4639 [band 4383..4902]
    width = 13.86 um, lambda ~ 1555.9 nm (corr-ladder line, EXPECTED)
The run DECIDES whether the corr knob designs a Q3dB device at a NEW width in
one shot (pass bands: T +-0.03 of prediction, width +-5%, Q_L +-10%).

Reused points (CLAUDE.md section 6): corr-450 fwhm 13.86 um row cited from
results_from_athena/tm_match_corr/results/corr_bisect_log.csv (knob-line
source); no stored device exists at corr~448 / N=98 — nothing re-run.
NO control row (no section-2 numerics change vs the q3db family).

Physics line (section 4): TM h350, pitch 516.83, corr 448.4, W800; expected
resonance ~1555.9 nm, window 20 nm centered 1559.5 (1549.5-1569.5, 4001 pts;
Q_L ~4.6k -> line ~335 pm -> ~67 samples). 2*kappa*L ~ 4.9 (well-formed).
Expected solve ~20-40 min (tau ~ 4 ps).

Dispatch (IGUM; serialize deploys, own queue may hold 67731 — different study,
per-study sweep lists, no shared-file overlap):
    bash igum/deploy_igum.sh --option3 --spec=runners.sweeps.tm_q3db_14um_knob --max-concurrent=1
Output -> results/tm_q3db_14um_knob/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_NM        = 448.4
N_EACH_SIDE    = 98

BOX_Y_UM       = 8.0         # q3db family numerics exactly
SCAN_CENTER_NM = 1559.5
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.center_wavelength_m = SCAN_CENTER_NM * 1e-9
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.spectral.n_wl_points = N_WL_POINTS
assert BASE.symmetry.use_z_symmetry, "bare device is z-symmetric — keep the 2x z saving"

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM],
    n_periods_each_side  = [N_EACH_SIDE],
    center_wavelength_nm = [SCAN_CENTER_NM],
    scatterer_radius_nm  = [0.0],
    mode  = "zipped",
    label = "tm_q3db_14um_knob",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print(f"  task 0: corr={CORR_NM} N={N_EACH_SIDE}  bare device")
    print("pre-registered PREDICTION: T 0.4937 [0.4576..0.5280], Q_L 4639 [4383..4902],")
    print("width 13.86 um, lambda ~1555.9 nm — ONE-SHOT -3dB/14um design test")
