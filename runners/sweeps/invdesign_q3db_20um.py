"""The inverse-designed pi-shift grating at the q3db operating point: N for -3 dB, and Q.

★STATUS 2026-08-25: task 0 (N=100) HAS RUN -- IGUM 63202_0, 1348.8 s, exit 0.
MEASURED: T 0.97228 | lambda 1560.947 nm | Q_loaded 1818.6 | mode FWHM 19.1709 um.
Against the STORED regular-pipeline bare anchor at these exact numerics (IGUM
51736, never re-run: T 0.9104 / 1559.006 / Q 1760 / 19.2448) that is +0.0619 T,
cavity loss 0.0896 -> 0.0277 (-69%), and mode width -0.38% (KEPT). Tasks 1-4 are
the lengthening ladder to the -3 dB crossing.

★KNOWN CAVEAT, do not lose it: this path builds the design as the REGULAR BUILDER
builds it, which differs from the optimizer's own layout on the RIGHT ARM only --
bragg_device.py:1180 shortens R_narrow_d by s(d-1), lumopt2 make_func by s(d), up
to 6.43 nm per tooth (zero-GPU scene gate, 75 props, left arm exact). It cannot be
re-indexed into agreement and bragg_device must NOT be changed (it would silently
alter every stored distributed-shift result). prod_confirm row 3 (job 63195_3) is
the exact-optimizer device at N=100 and prices the difference. Everything here is
internally consistent and is the device the regular builder would fabricate.

Study dir: runners/sweeps/   |   Created 2026-08-25   |   Job(s): TBD
Purpose (user 2026-08-25): take the inverse-design winner (BEST_T9636, the one
that KEPT its mode width) to the same -3 dB / 20 um-mode operating point the
regular devices were locked at, and report how many periods it takes and what Q
it reaches. Same question, same numerics, as te_q3db_20um / comb_q3db.

★WHY THIS IS A PLAIN SweepSpec RUNNER AND NOT A lumopt2 CANARY (the point of
the file): the lumopt2 path cannot drop its optimization-region mesh override,
and job 62750 task 0 PRICED that residue -- at N=100 conformal it reproduced the
stored regular anchor's lambda to +0.025 nm and mode width to +0.019%, but read
T +0.0079 HIGH and Q_loaded 7% LOW. -3 dB is an ABSOLUTE-T question, so that
offset would bias the crossing. Built here through the regular builder the device
is the regular device, and the stored family crossings below are usable directly.

Rows (zipped): N per side = 100, 150, 180, 200, 220 -- the dB(N) curve, wide
enough to BRACKET the crossing rather than guess it (CLAUDE.md section 5: bisect
in ONE cheap array, never one expensive point per dispatch). A final confirming
rung at the interpolated crossing follows as a 1-task dispatch.

NO control row (CLAUDE.md section 6 no-rerun). Compare against, at these EXACT
numerics, never re-run:
  bare corr-325 N=100   T 0.9104 / lambda 1559.006 / Q 1760 / mode 19.2448  (IGUM 51736)
  ctrl corr-325 N=165   T 0.4906 = -3.09 dB / Q 13930                       (job 130458)
  winner comb  N=169    -3.04 dB / Q 16203 / mode 19.91                     (comb_q3db wave 2)
EXPECTED crossing N ~ 195-215: the design loses roughly half as much per period
as the bare device, and the family's own dB(N) slope is steep (dB ~ N^4.1 fitted
across the stored N=100 and N=165 bare rows). That exponent is a TWO-POINT fit
and is here only to size the ladder -- it is not a result.

THE DEVICE. BEST_T9636 (runners/lumopt2_design/best_designs.py), MEASURED by the
optimizer at PVA/N=100: T 0.96361 / fwhm_env 18.35309 um (ratio 1.0004 vs its own
origin -- that is the width-keeping claim). Its 191-vector unpacks 25 corr | 25
avg | 25 shift | 57 comb r | 57 comb x | d_comb | cavity_w, all nm, innermost
first. The comb block is INERT: the v2 campaigns ran free_comb=False, so the
measured device carries the SEED comb lattice (Lambda 531 / dx 401 / r 80 /
d 1900), built explicitly below exactly as comb_q3db builds it.
The slice layout is asserted, and the built scene is gated against the lumopt2
parametrization by scratchpad/gate_invdesign_scene.py (zero GPU) before dispatch.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, W800, outer teeth at
corr 325; box y 8.0 um, window 20 nm centered 1559.5 (4001 pts), z-sym.

Dispatch:
    SBATCH_MEM=160G ARRAY_TIME=08:00:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.sweeps.invdesign_q3db_20um --max-concurrent=2
Output -> results/invdesign_q3db_20um/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common
from runners.lumopt2_design.best_designs import BEST_T9636

N_FREE      = 25             # free inner teeth per side, innermost first
CORR_NM     = 325.0          # frozen outer-tooth corrugation
N_LADDER    = [100, 150, 180, 200, 220, 240, 280, 320]   # 240 APPENDED 2026-08-26: the
# ★RE-POINTED 2026-08-26 on MEASURED data: N=100 (-0.122 dB) and N=150 (-0.389 dB)
# give loss_dB ~ N^2.859, which puts the -3 dB crossing at **N ~ 306**, not the
# 220-235 first projected off the BARE device's alpha=4.04 (that transfer was
# wrong -- the design's own exponent is much shallower). 280 and 320 STRADDLE it
# (predicted -2.32 and -3.40 dB). Original note kept for the record:
# crossing projects to N~220-235, i.e. at the top of the original bracket, so one
# rung above it is needed to STRADDLE rather than extrapolate. Append-only —
# indices 0-4 keep their meaning, so in-flight arrays reading them stay correct.

# Comb: the winner lattice the v2 campaigns froze (identical to comb_q3db).
COMB_LAM_NM, COMB_DX_NM, COMB_R_NM, COMB_D_NM, COMB_N_HALF = 531.0, 401.0, 80.0, 1900.0, 28

BOX_Y_UM       = 8.0         # q3db family numerics exactly
SCAN_CENTER_NM = 1559.5
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 4001        # 5 pm; ~19 pts across the narrowest expected line (Q ~ 16k)

# ── unpack the design (layout asserted, not assumed) ───────────────────────
p = np.asarray(BEST_T9636, dtype=float)
assert p.shape == (191,), "BEST_T9636 layout changed — re-read best_designs.py"
corr, avg, shift = p[0:25], p[25:50], p[50:75]
CAVITY_W_NM = float(p[190])
W_NARROW = list(avg - corr / 2.0)      # bragg_device takes these verbatim, d=1..25
W_WIDE   = list(avg + corr / 2.0)
SHIFTS   = list(shift)
assert 200.0 < min(W_NARROW) and max(W_WIDE) < 1400.0, "unpacked widths off-family"
assert 0.0 <= min(SHIFTS) and max(SHIFTS) < 258.415, "shift must stay inside a half-pitch"

X_COMB = [round(k * COMB_LAM_NM + COMB_DX_NM, 1) for k in range(-COMB_N_HALF, COMB_N_HALF + 1)]

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.center_wavelength_m = SCAN_CENTER_NM * 1e-9
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.spectral.n_wl_points = N_WL_POINTS
assert BASE.symmetry.use_z_symmetry, "design is z-symmetric — keep the 2x z saving"

assert len(set(N_LADDER)) == len(N_LADDER), "rows must be tag-unique (N is the tag)"
assert min(N_LADDER) * 0.51683 >= 2.0 * 20.0, "half-device must hold 2x the 20 um mode"
assert COMB_N_HALF * COMB_LAM_NM + COMB_DX_NM + COMB_R_NM + 2000.0 <= min(N_LADDER) * 516.83
assert COMB_D_NM + COMB_R_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0      # y-PML clearance

# ★PER-ROW SPECTRAL WINDOW (added 2026-08-26 — a MEASUREMENT-ADEQUACY fix, not a
# physics change). The 20 nm / 4001 pt family window is 5 pm per sample. That is
# ~30 samples across the line at N=150 (MEASURED Q_L 10494) but only ~2 at N=280
# and ~1 at N=320, where Q_L projects to 1.4e5-2.4e5 and the line is 6-11 pm wide.
# Under-sampling does not just spoil Q — it puts the true peak BETWEEN grid points,
# biasing peak T low, which would move the -3 dB crossing itself. So the long rungs
# keep 4001 points but NARROW the window onto the resonance. Rows 0-5 keep the exact
# family window so every already-measured rung stays bit-comparable (section 6/2).
# Resonance drift is small and measured: 1560.947 (N=100) -> 1560.857 (N=150), so a
# 2-3 nm window centred 1560.6-1560.7 keeps >=1 nm of margin against an off-window miss.
_WIN = {200: (1560.7, 3.0), 220: (1560.7, 3.0), 240: (1560.7, 3.0),
        280: (1560.7, 3.0), 320: (1560.6, 2.0)}        # N -> (centre nm, width nm)
# ★EXTENDED to 200/220/240 (2026-08-26): the crossing model is NOT settled -- the
# same two measured points give N~230 (coupled-cavity), ~306 (power law) or ~640
# (ln T linear). Under the N~230 branch, N=220 gets 3.1 samples across its line and
# N=240 gets 1.8 on the 5 pm family grid. Under-sampling biases peak T LOW, which
# would drag the apparent crossing DOWN and look self-consistent. So every rung that
# could plausibly sit near -3 dB gets a resolved line, whichever model turns out right.
# N=180 keeps the family window: >=10 samples under every branch.
SCAN_CENTER_LIST = [_WIN.get(n, (SCAN_CENTER_NM, SCAN_WIDTH_NM))[0] for n in N_LADDER]
SCAN_WIDTH_LIST  = [_WIN.get(n, (SCAN_CENTER_NM, SCAN_WIDTH_NM))[1] for n in N_LADDER]
for _n_, _c_, _w_ in zip(N_LADDER, SCAN_CENTER_LIST, SCAN_WIDTH_LIST):
    assert _w_ / (N_WL_POINTS - 1) * 1000 <= 5.0, "window/point grid coarser than the family's 5 pm"

_n = len(N_LADDER)
SPEC = SweepSpec(
    n_periods_each_side       = list(N_LADDER),
    corrugation_depth_nm      = [CORR_NM] * _n,          # outer teeth (beyond the 25)
    width_narrow_per_tooth_nm = [W_NARROW] * _n,
    width_wide_per_tooth_nm   = [W_WIDE] * _n,
    inner_shift_list_nm       = [SHIFTS] * _n,
    n_free_inner_teeth        = [N_FREE] * _n,
    cavity_width_nm           = [CAVITY_W_NM] * _n,
    center_wavelength_nm      = SCAN_CENTER_LIST,
    scan_width_nm             = SCAN_WIDTH_LIST,
    scatterer_radius_nm       = [COMB_R_NM] * _n,
    scatterer_x_list_nm       = [X_COMB] * _n,
    scatterer_y_list_nm       = [[COMB_D_NM] * len(X_COMB)] * _n,
    scatterer_height_nm       = [350.0] * _n,
    mode  = "zipped",
    label = "invdesign_q3db_20um",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, n in enumerate(N_LADDER):
        print(f"  task {i}: N={n:4d}  half-device {n * 0.51683:6.1f} um")
    print(f"design: mcorr {corr.mean():.2f} nm | e=2*sum(shift) {2 * shift.sum():.1f} nm "
          f"| w_cav {CAVITY_W_NM:.2f} nm | narrow {W_NARROW[0]:.2f}..{W_NARROW[-1]:.2f} "
          f"| wide {W_WIDE[0]:.2f}..{W_WIDE[-1]:.2f}")
    print("compare (stored, NOT re-run): bare N100 0.9104/Q1760 | ctrl N165 0.4906/-3.09dB/Q13930 "
          "| comb N169 -3.04dB/Q16203")
