"""
PHASE-3 — LATERAL AIR-TRENCH light-cone / TIR recycling (novel, 2026-07-07).

Distinct from the (dead) SiN strip-reflector study: those used HIGH-index (n=1.97)
strips → partial Fresnel reflection of the arm radiation, and were null (the
broadside channel was already cancelled by rect-1050). This uses LOW-index AIR
trenches (n=1.0) → a different mechanism:

  The residual radiation is NEAR-AXIAL (|ux| ≈ 0.98 measured): it hits a
  guide-parallel interface at ≈79° from the normal. The oxide→air critical angle
  is asin(1.0/1.444) = 43.8°, so 79° > 43.8° ⇒ the near-axial in-plane radiation
  is TOTALLY INTERNALLY REFLECTED back toward the guide. Two air trenches (±y)
  form a lateral low-index cladding that SHRINKS the in-plane light cone the mode
  can radiate into — the planar analogue of the suspended-membrane lever that
  makes air-clad nanobeam cavities high-Q. Radiation that cannot propagate past
  the trenches is either recycled into the mode (constructive return phase) or at
  least prevented from escaping to the PML.

HONESTY / registered predictions:
  P1 The trench MUST show a return-phase dependence on offset d (both signs) — a
     flat null within 2× the opt-mesh floor (~0.002) means the near-axial arm
     radiation does not coherently re-couple (report null). Best case if it TIRs
     and recycles: DT up to ~+0.01–0.02 (loss 0.0545 → ~0.04).
  P2 The AIR trench must beat the SiN strip control at the same (d, width): if
     they are equal (both null), the low-index/TIR mechanism adds nothing over the
     dead high-index reflector, and the whole external-reflector route is closed.
  P3 Drain check: recycling changes loss AT RESONANCE only; a trench that just
     perturbs the guided mode shows a broadband T shift (checked off-resonance).
  Caveat: trench height 2 µm covers the core + main evanescent/radiation z-extent
     but not the full vertical spread — it targets the IN-PLANE (62%) channel.

Base = the stack (cavity 1050 + [+20,+20] + see-saw 1040/980, loss 0.0545),
accurate mesh, box y=8.0 µm (trench edge |y|≤2.8 µm keeps ≥1.2 µm PML clearance),
z-mult 5.42, window 1556.5/40/3001.

Rows (zipped, 12 tasks):
   0     control: the stack, NO trench (spans None) — Δ reference
   1-5   air trench (n=1.0) L=84 µm, width 800 nm, d {1.2,1.5,1.8,2.1,2.4} µm
   6-8   air trench L=84 µm, width 400 nm, d {1.5,1.8,2.1} µm
   9     air trench L=84 µm, width 800 nm, d=1.825 µm (half-cell jitter of d=1.8)
   10    SiN strip (n=1.97) L=84 µm, width 800 nm, d=1.8 — high-index control (P2)
   11    air trench SHORT L=20 µm, width 800 nm, d=1.8 — extent check

Dispatch (queue EMPTY — serialize after phase-2 drains):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_air_trench
Output -> results/tm_air_trench/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 8.0
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5565e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001
BASE.mesh.simulation_mode = "accurate"

# Base device = the stack
BASE.grating.cavity_width_m = 1050e-9
BASE.grating.inner_shift_nm = [20.0, 20.0]
BASE.grating.n_free_inner_teeth = 2
BASE.grating.width_narrow_per_tooth_m = [600e-9, 600e-9]
BASE.grating.width_wide_per_tooth_m = [1040e-9, 980e-9]

# Trench machinery: rect scatterer, tall (covers core + evanescent), mirrored ±y.
BASE.scatterer.enabled = True
BASE.scatterer.shape = "rect"
BASE.scatterer.x_m = 0.0
BASE.scatterer.mirrored_y = True
BASE.scatterer.height_m = 2.0e-6            # taller than core to wall the in-plane leak

_L, _W, _D, _IDX = [], [], [], []


def row(L_um=None, w_nm=None, d_um=1.8, idx=1.0):
    _L.append(L_um); _W.append(w_nm); _D.append(d_um * 1000.0); _IDX.append(idx)


row(idx=1.0)                                     # 0 control (spans None => no trench)
for d in (1.2, 1.5, 1.8, 2.1, 2.4):              # 1-5 air trench w800
    row(L_um=84.0, w_nm=800.0, d_um=d, idx=1.0)
for d in (1.5, 1.8, 2.1):                        # 6-8 air trench w400
    row(L_um=84.0, w_nm=400.0, d_um=d, idx=1.0)
row(L_um=84.0, w_nm=800.0, d_um=1.825, idx=1.0)  # 9 half-cell d jitter
row(L_um=84.0, w_nm=800.0, d_um=1.8, idx=1.97)   # 10 SiN high-index control (P2)
row(L_um=20.0, w_nm=800.0, d_um=1.8, idx=1.0)    # 11 short-extent air trench

_n = 12
assert all(len(v) == _n for v in (_L, _W, _D, _IDX))
# tag distinctness: control has no scat tag; others distinct by (L,w,d,idx)
_sig = list(zip(_L, _W, _D, _IDX))
assert len(set(_sig)) == _n, "duplicate row"

SPEC = SweepSpec(
    scatterer_x_span_um = _L,
    scatterer_y_span_nm = _W,
    scatterer_y_nm      = _D,
    scatterer_index     = _IDX,
    mode  = "zipped",
    label = "tm_air_trench",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
