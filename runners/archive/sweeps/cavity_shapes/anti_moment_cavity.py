"""
ANTI-MOMENT CAVITY families — the last in-scope shot at beating rect 1050.

Motivation (radiation_kspace_diag, 2026-07-05, zero-GPU): the light-cone map
attributes only ~29% of the radiating weight to |x| <= 3 pitches; rect 1050
(-30% loss, ACC-confirmed) has plausibly harvested that local budget already
(the 1050->1400 reversal = the lowest radiating moment crossing its null).
These rows scan the residual local moment through the null with the two
mathematically-motivated one-knob families (both EVEN about the defect —
odd parity provably nulls, cf. asym_dw_study):

  Family B (fine width trim == cos(2*beta) wall by the measured equal-area
  equivalence; barrel IS cos(2*beta) over the half-pitch segment):
    rect cavity 1000 / 1025 / 1075 / 1100 nm around the 1050 champion.
  Family A (anti-moment tooth pair, ZERO net area -> fwhm-neutral):
    on the 1050 base, tooth +-1 wide width 1000+delta, tooth +-2 1000-delta,
    delta = +-10 / +-20 / +-30 nm (sign flips which tooth donates).

Rows (zipped, 13 tasks, ALL accurate mesh dx~35 — expected effects are below
the dx=50 floor ~0.002; accurate floor ~1e-4, pillar precedent), converged box
y=6.8/z-mult 5.42, window 1558.5 nm center / 40 nm / 3001 pts (accurate-mesh
lambda_res ~1555.9 in-window; target resonance 1555.9-1558.6):
   0 control W800 (anchor)
   1 rect 1050 (champion reference)
   2 rect 1052 (jitter partner: sub-mesh geometry change at the flat optimum
     -> measures the numerical floor in-study)
   3-6  rect 1000 / 1025 / 1075 / 1100        (Family B)
   7-12 ptw delta = +10/-10/+20/-20/+30/-30   (Family A, on 1050)
Registered predictions: ladder = shallow parabola, minimum near 1050; row 2
== row 1 within floor. Family A: linear-in-delta trend if a residual local
moment exists (one sign helps); all-flat at the 1050 level = local moment
already nulled -> phase closed, rect 1050 final.

Tags: rect rows differ by cavity width; ptw rows carry _ptw2W{w0}to{w1} —
all 13 distinct, no collisions.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.anti_moment_cavity
Output -> results/anti_moment_cavity/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

# Anchored device teeth: avg 800 / corr 400 -> narrow 600 / wide 1000 nm.
_NARROW = [600.0, 600.0]
def _pair(delta):
    return [1000.0 + delta, 1000.0 - delta]

_cwid = [None,  1050.0, 1052.0, 1000.0, 1025.0, 1075.0, 1100.0] + [1050.0] * 6
_ptw_n = [None] * 7 + [_NARROW] * 6
_ptw_w = [None] * 7 + [_pair(d) for d in (10.0, -10.0, 20.0, -20.0, 30.0, -30.0)]

_n = 13
assert len(_cwid) == len(_ptw_n) == len(_ptw_w) == _n
_keys = [(w, tuple(pw) if pw else None) for w, pw in zip(_cwid, _ptw_w)]
assert len(set(_keys)) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    cavity_width_nm           = _cwid,
    width_narrow_per_tooth_nm = _ptw_n,
    width_wide_per_tooth_nm   = _ptw_w,
    simulation_mode           = ["accurate"] * _n,
    mode  = "zipped",
    label = "anti_moment_cavity",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
