"""
LATERAL STRIP REFLECTOR — the main recycling study of the loss program
(docs/tm_loss_program_phase0_2026-07-05.md, plan "2b"). DISPATCH ONLY AFTER
THE PHASE-1 CHECKPOINT (user agreement) and after tm_center_completion drains.

Physics + measured basis (polarimetry job 117907, rows 0-2):
  - Rect-1050 remaining loss ~0.078 splits ~55/45 in-plane/vertical;
    the in-plane share is ~92% ARM radiation (defect-local share collapsed 27->8%
    when rect-1050 cancelled the cavity's broadside lobe).
  - The radiated field keeps the TM polarization (f_TE ~ 0): with E along z it is
    s-POLARIZED w.r.t. a vertical strip sidewall -> the high-reflectance branch
    (Fresnel R_s ~ 0.48 per interface at grazing 80 deg; theory section C).
  - STACKING GOAL (user): everything runs on the 1050+see-saw stack
    (cavity 1050 + wide teeth 1020/980, loss 0.0810 accurate); target is the
    combined device at T ~ 0.95.

Geometry: SiN strip(s) (same litho layer, full core height, n=1.97) parallel to
the guide at center offset d = scatterer_y_nm, mirrored +-y. Full-arm strips
(L=84 um) attack the ~3.4% absolute in-plane arm loss; short strips (L=6 um)
are the user-requested "reflectors near the cavity" probing the defect residue.
Return phase: near-axial lobes give a d-period of ~3.1 um (> usable range), so
the WIDTH dimension (two-interface reflection phase) supplies the rest of the
phase scan — hence the (d x width) grid.

Registered predictions (filed before dispatch):
  P1 Loss vs (d, width) shows structure with BOTH signs — destructive
     placements must WORSEN loss. All-flat within 2x the dx=50 floor (~0.002)
     = null -> the recycling route is closed for good.
  P2 Best-case magnitude if half the arm in-plane loss recycles coherently:
     DT ~ +0.017 (loss 0.078 -> ~0.06 at opt mesh). Registered honest ceiling.
  P3 Near-cavity short strips: small (<= few 1e-3) — the cavity's own in-plane
     residue is only ~8% of the side flux after rect-1050.
  P4 Drain discriminator: recycling changes loss AT RESONANCE only; a strip
     that DRAINS shows a broadband T drop (checked off-resonance in analysis).
     Guided-mode tail at d>=1.2 um is <= 0.12 amplitude (theory, gamma=1.75).

Rows (zipped, 24 tasks, optimization mesh dx=50 for the map; winners get an
accurate confirm block later). Box: y_span 7.6 um for ALL rows incl. control
(strips reach |y|=2.4 um; keeps >=1.3 um > lambda/n_clad clearance to the y
PML), z-mult 5.42. Window 1558.5 / 40 nm / 3001 (1050+see-saw stack lambda_res
~1558.8 at this mesh, in-window).
   0     control: 1050+see-saw stack, NO strip (spans None), same box
   1-18  full-arm strip L=84 um: d {1.2,1.4,1.6,1.8,2.0,2.2} um x
         width {198, 300, 400} nm
  19-21  near-cavity strip L=6 um, width 198: d {1.2, 1.5, 1.8} um
  22     jitter partner: d=1.625 um (+25 nm = half mesh cell), w 198, L 84
  23     jitter partner: d=1.6 um, w 202 (sub-mesh width change), L 84

Tags: _scRECT_L{L}xW{w}_X0_Y{d}_pair — all distinct; control carries no scat
tag (spans None => _has_scatterer False, identical-numerics control).

Dispatch (serialize behind tm_center_completion; queue must be EMPTY):
    smoke: bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_strip_reflector --array-tasks=0
    full:  bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_strip_reflector
Output -> results/tm_strip_reflector/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 7.6            # strips reach |y|=2.4 um -> keep >= lambda/n_clad PML clearance
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

# Base = the frontier-best in-bound stack (job 118214, 2026-07-06):
# cavity 1050 + gap-shift pair [+20,+20] + see-saw (1040, 980) — loss 0.0545,
# T 0.9449, fwhm +0.9%. Strips recycle THIS device's residual arm radiation.
BASE.grating.cavity_width_m = 1050e-9
BASE.grating.inner_shift_nm = [20.0, 20.0]
BASE.grating.n_free_inner_teeth = 2
BASE.grating.width_narrow_per_tooth_m = [600e-9, 600e-9]
BASE.grating.width_wide_per_tooth_m = [1040e-9, 980e-9]

# Strip machinery: enabled study-wide; per-row spans (None = control row)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "rect"
BASE.scatterer.x_m = 0.0
BASE.scatterer.mirrored_y = True

_L, _W, _D = [], [], []


def row(L_um=None, w_nm=None, d_um=1.5):
    _L.append(L_um); _W.append(w_nm); _D.append(d_um * 1000.0)


row()                                            # 0 control (no strip)
for d in (1.2, 1.4, 1.6, 1.8, 2.0, 2.2):         # 1-18 full-arm (d x width)
    for w in (198.0, 300.0, 400.0):
        row(L_um=84.0, w_nm=w, d_um=d)
for d in (1.2, 1.5, 1.8):                        # 19-21 near-cavity short strips
    row(L_um=6.0, w_nm=198.0, d_um=d)
row(L_um=84.0, w_nm=198.0, d_um=1.625)           # 22 half-cell d jitter
row(L_um=84.0, w_nm=202.0, d_um=1.6)             # 23 sub-mesh width jitter

_n = 24
assert all(len(v) == _n for v in (_L, _W, _D))
assert len({(l, w, d) for l, w, d in zip(_L, _W, _D)}) == _n, "duplicate row"

SPEC = SweepSpec(
    scatterer_x_span_um = _L,
    scatterer_y_span_nm = _W,
    scatterer_y_nm      = _D,
    mode  = "zipped",
    label = "tm_strip_reflector",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
