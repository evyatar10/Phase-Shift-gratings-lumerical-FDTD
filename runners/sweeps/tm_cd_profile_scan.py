"""
BATCH 1c — counterdiabatic per-tooth-translation profile scan (accurate mesh).

WHY. Batch-1 (job 118618) found — against the Phase-0 kernel's ~null prediction
— that the counterdiabatic (shortcut-to-adiabaticity) quadrature profile, a
per-tooth position modulation over the 14 innermost teeth, reduces the stack's
resonant loss with a clear SIGN dependence and a MONOTONIC trend in amplitude:

  scale  +2     +1      0(stack)  -1      -2
  loss   0.0627 0.0560  0.0545    0.0527  0.0504    (accurate, fwhm +0.4%, 1 res)

Negative scale is the loss-reducing direction (the counter-term's correct
sign); the trend has not turned by -2. This scan densely maps the NEGATIVE
branch to LOCATE the optimum and CONFIRM the effect is physical (a smooth,
monotone-then-turning curve at accurate mesh cannot be mesh noise — the
opt-mesh jitter floor is ~0.0018 and is non-monotonic; here the accurate-mesh
points move smoothly by ~0.002/scale). Row 0 (scale 0) is the in-batch stack
control at identical numerics; the whole curve is read as Δ vs it.

Profile CD_PROFILE is the constrained-solve output from
python_tools/phase0_counterdiabatic.py; cd_shifts(scale) = [20,20]+0*12 (the
stack) + scale*CD_PROFILE over 14 teeth (lengthen_cavity absorbs 2*Σ). All
|shift| stay < 60 nm (« half_pitch 258 nm) and gaps > 100 nm across the range.

Predictions (honesty gate): the negative branch keeps dropping until a turning
point (expected around scale -3..-5); best loss projected ~0.047-0.050 at
fwhm within ~1%, single resonance. If instead loss FLATTENS immediately or the
curve is ragged (non-monotone > 0.001), the batch-1 signal was a fluctuation —
report that. Survivor's formal half-cell mesh-jitter / finer-mesh confirm is
batch-1d.

Rows (zipped, 13 tasks, accurate mesh, stack single-device base, converged box
y=6.8 / z-mult 5.42, window 1556.5/40/3001):
   0-6  CD-shape amplitude scan: scales {0,-1,-2,-3,-4,-5,-6} (14-tooth profile).
        scale 0 = the stack (control). Maps loss vs total shift for the CD shape.
   7-12 DISTRIBUTION CONTROLS at 3 matched totals (~42.2/44.4/46.6 nm), each:
        uniform-14-tooth (spread) + lumped-2-tooth — same cavity length as the
        CD point at that total. Decides SHAPE vs total-shift (see below).

Dispatch (queue must be EMPTY — serialize after FW-BIC job 118734 drains):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_cd_profile_scan
Output -> results/tm_cd_profile_scan/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5565e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled = False

STACK_PTW = [1040.0, 980.0]
STACK_PTN = [600.0, 600.0]
# constrained-solve counterdiabatic quadrature profile (nm), 14 teeth:
CD_PROFILE = [-0.2, 8.6, -4.9, -4.0, 0.1, 0.6, -0.9, -2.8,
              2.6, 2.6, 1.4, -0.3, -2.3, -1.6]


def cd_shifts(scale):
    base = [20.0, 20.0] + [0.0] * 12
    return [round(b + scale * s, 2) for b, s in zip(base, CD_PROFILE)]


# Integer scales only: the profile sum is -1.1 nm/scale, so consecutive
# half-steps would round to the SAME total-shift file tag (dsh...S{tot}) and
# clobber each other. Integer steps (Δtotal ≥ 1.1 nm) stay tag-distinct while
# the ~0.002 loss/scale step is well above the accurate-mesh jitter floor
# (~1e-4). Extend more negative if the batch shows the optimum past -7.
SCALES = [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]

# ── DISTRIBUTION CONTROLS (the decisive experiment) ──────────────────────────
# The CD amplitude scan confounds two things: the PROFILE SHAPE and the TOTAL
# shift (which lengthens the cavity by 2*Σshift). To attribute the loss drop to
# the counterdiabatic SHAPE (not just "more total shift / longer cavity"), at
# each of three matched totals compare the CD profile against the SAME total
# redistributed two trivial ways — lumped on 2 teeth, and spread uniformly over
# 14 teeth. All three share the identical cavity length (lengthen_cavity absorbs
# 2*Σ, equal for equal totals), so ONLY the distribution differs.
#   CD wins at fixed total  → the shape is real (counterdiabatic genuine).
#   three-way tie           → it is JUST total shift; CD adds nothing special.
CTRL_SCALES = [-2.0, -4.0, -6.0]         # totals ~42.2, 44.4, 46.6 nm


def uniform_shifts(total):
    return [round(total / 14.0, 3)] * 14


def lumped_shifts(total):
    return [round(total / 2.0, 2), round(total / 2.0, 2)]


# rows: (inner_shift_list, n_free)
rows = [(cd_shifts(s), 14) for s in SCALES]                       # CD-shape scan
for s in CTRL_SCALES:
    T = sum(cd_shifts(s))
    rows.append((uniform_shifts(T), 14))                          # uniform control
    rows.append((lumped_shifts(T), 2))                            # lumped control

_n = len(rows)
assert _n == 13, _n
# tag-uniqueness: (n_free, S_total-rounded, s0-rounded) must be distinct
_sig = [(nf, round(sum(sh)), round(sh[0])) for sh, nf in rows]
assert len(set(_sig)) == _n, f"file-tag collision: {_sig}"

SPEC = SweepSpec(
    inner_shift_list_nm       = [sh for sh, _ in rows],
    n_free_inner_teeth        = [nf for _, nf in rows],
    cavity_width_nm           = [1050.0] * _n,
    width_narrow_per_tooth_nm = [STACK_PTN] * _n,
    width_wide_per_tooth_nm   = [STACK_PTW] * _n,
    simulation_mode           = ["accurate"] * _n,
    mode  = "zipped",
    label = "tm_cd_profile_scan",
)


if __name__ == "__main__":
    print(SPEC.describe())
    for sh, nf in rows:
        print(f"  n_free={nf:2d}  sum={sum(sh):5.1f}  s0={sh[0]:5.2f}  {sh}")
    run_sweep_spec(SPEC, target="local", base=BASE)
