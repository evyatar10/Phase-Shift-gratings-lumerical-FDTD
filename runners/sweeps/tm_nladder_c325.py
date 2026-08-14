"""TM surrogate-N ladder, corr-325 family: where is the shortest well-formed device?

Study dir: runners/sweeps/   |   Created 2026-08-11   |   Job(s): IGUM 51736 (5 tasks)
Purpose (user): pick the surrogate device length for the inverse-design campaign.
Measure resonance lambda / peak T / Q vs n_periods_each_side for the BARE corr-325
device at the q3db study numerics, descending below the production N. Verdict =
smallest N whose resonance is well-formed: finder hit inside the window, peak
>= ~3 FWHM from the stopband shoulder, T_peak <= ~0.9.

Rows (zipped, bare device, radius 0): N = 60, 70, 80, 100, 120 per side.
NO control row (CLAUDE.md section 6 no-rerun): anchor = job 130458 ctrl row
(corr 325, N=165, T 0.4906 / -3.09 dB, Q 13930, lambda ~1558.4) at these EXACT
numerics (same build_ports_base + y 8.0 um box + 20 nm / 4001 pt window + z-sym).
Nothing on the section-2 numerics list changed -> stored anchor serves as control.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 325, W800;
target resonance ~1558.4 nm, window 20 nm centered 1559.5 (1549.5-1569.5,
4001 pts) — q3db study numerics. NOTE: at N < ~78 the half-device is shorter
than 2x the 20 um mode FWHM — the envelope is end-truncated. That is the
surrogate operating point being characterized (end-coupling IS the coupling),
not an artifact; expect Q to fall and T_peak to rise as N drops.

Dispatch (queue must be EMPTY of other --option3 arrays — CLAUDE.md section 6;
serialize with tm_nladder_c400, this one first):
    ARRAY_TIME=04:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.sweeps.tm_nladder_c325 --max-concurrent=4
Output -> results/tm_nladder_c325/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

CORR_NM        = 325.0
N_LADDER       = [60, 70, 80, 100, 120]

BOX_Y_UM       = 8.0         # q3db family numerics exactly (trench_flush_q3db / comb_q3db)
SCAN_CENTER_NM = 1559.5
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001        # 5 pm; broadest expected line (N=60, Q ~ few hundred) still >100 pts/FWHM

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.center_wavelength_m = SCAN_CENTER_NM * 1e-9
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.spectral.n_wl_points = N_WL_POINTS
assert BASE.symmetry.use_z_symmetry, "bare device is z-symmetric — keep the 2x z saving"

assert len(set(N_LADDER)) == len(N_LADDER), "rows must be tag-unique (N is the tag)"
assert max(N_LADDER) < 165, "N=165 exists (job 130458 ctrl) — never re-run it"

SPEC = SweepSpec(
    corrugation_depth_nm = [CORR_NM] * len(N_LADDER),
    n_periods_each_side  = list(N_LADDER),
    center_wavelength_nm = [SCAN_CENTER_NM] * len(N_LADDER),
    scatterer_radius_nm  = [0.0] * len(N_LADDER),
    mode  = "zipped",
    label = "tm_nladder_c325",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, n in enumerate(N_LADDER):
        print(f"  task {i}: corr={CORR_NM:.0f} N={n}  bare device")
    print("anchor (130458 ctrl, same numerics): N165 T 0.4906 Q 13930 lambda ~1558.4")
    print("verdict wanted: smallest N with in-window, shoulder-separated, T_peak<=0.9 resonance")
