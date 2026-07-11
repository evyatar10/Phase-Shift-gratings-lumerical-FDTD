"""
PHASE-2 novel-mechanism array (2026-07-07). The theory (docs/
novelty_analysis_2026-07-07.md) argues that for a single-resonance defect cavity
at fixed width, loss is the mode-envelope's light-cone tail ⇒ envelope
optimization ⇒ the inverse-design (tooth-shift/width/apod/counterdiabatic) space,
and that the escapes (2nd-resonance FW-BIC, external secondary source) FAILED in
FDTD this session. This array TESTS the last untested planar ideas empirically so
the "envelope optimization is the ceiling" conclusion is measured, not asserted.
All on the STACK baseline, accurate mesh, converged box — Δ vs row 0.

Groups (each theory-gated with honest confidence):

  A. REAL-α-optimal external post (Green's function, user-requested). Batch-1
     placed posts at COMPLEX-α-optimal sites (wrong phase for a passive cylinder)
     and they added loss. `phase0_greens_cluster.py` finds the REAL-α (sign-
     correct) optimum at x≈0.50, y≈0.95 µm (models ~13% cancel). Test r∈{100,150,
     200} there. Confidence LOW (the in-plane model ignores the vertical scatter +
     guided back-reflection that sank batch-1's posts). Decisive: if the correctly
     -sited post STILL adds loss, the passive-scatterer route is closed by
     parasitics, not siting.

  B. OPPOSITE-SIGN external element (α<0). The real-α map's global best needs
     α<0 (index below cladding) at x≈0.12, y≈2.0 µm (models ~26% cancel) — never
     tried (batch-1 used SiN posts, α>0). Test an AIR void (index 1.0) there, and
     a SiN post (α>0) at the SAME site as a sign control. Confidence LOW-MED; fab
     caveat (an air void in the oxide is a second etch — a physics test, not
     necessarily "same device").

  C. COUNTERDIABATIC × vertical 2Λ combo. CD (the in-plane win) + a weak
     every-other-tooth 2Λ width modulation (the vertical-channel knob) target
     DIFFERENT loss channels; batch-1 tested them separately (CD win, 2Λ null on
     rect-1050). Their COMBINATION on the CD device is untested. Confidence
     LOW-MED. Row C0 = CD alone (in-batch accurate control for the combo).

Rows (zipped, 9 tasks, accurate mesh, stack base, box y=6.8/z-mult5.42,
window 1556.5/40/3001):
   0   stack control (no scatterer, no 2Λ) — Δ reference
   1-3 A: SiN post r={100,150,200} at (500, 950) nm       [real-α site]
   4   B: air void (index 1.0) r=150 at (120, 2000) nm    [α<0 optimum]
   5   B: SiN post r=150 at (120, 2000) nm                 [sign control]
   6   C0: CD profile scale -2 (14-tooth), no 2Λ           [combo control]
   7   C: CD(-2) + 2Λ wide-tooth ±4 nm on teeth 3-16
   8   C: CD(-2) + 2Λ wide-tooth ±8 nm on teeth 3-16

Dispatch (queue EMPTY — serialize after FW-BIC 118734 AND batch-1c drain):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_novel_phase2
Output -> results/tm_novel_phase2/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = True
BASE.scatterer.mirrored_y = True
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5565e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled = False

# stack geometry
STACK_SHIFT = [20.0, 20.0]
STACK_PTW = [1040.0, 980.0]
STACK_PTN = [600.0, 600.0]
CD_PROFILE = [-0.2, 8.6, -4.9, -4.0, 0.1, 0.6, -0.9, -2.8,
              2.6, 2.6, 1.4, -0.3, -2.3, -1.6]


def cd_shifts(scale):
    return [round(b + scale * s, 2) for b, s in zip([20.0, 20.0] + [0.0] * 12, CD_PROFILE)]


def cd_plus_2lambda(dw):
    """CD(-2) shifts (14 teeth) + width array: teeth 1-2 see-saw 1040/980,
    teeth 3-16 = 1000 ± dw alternating (2Λ)."""
    shifts = cd_shifts(-2.0)
    ptw = [1040.0, 980.0] + [1000.0 + (dw if j % 2 == 0 else -dw) for j in range(14)]
    ptn = [600.0] * 16
    return shifts, ptw, ptn


# row builder: each returns dict of the swept fields
def scat_row(shifts, nfree, ptn, ptw, r, x, y, index):
    return dict(shifts=shifts, nfree=nfree, ptn=ptn, ptw=ptw,
                r=r, xl=[x], yl=[y], idx=index)


rows = []
# 0 control
rows.append(scat_row(STACK_SHIFT, 2, STACK_PTN, STACK_PTW, 0.0, 0.0, 1000.0, None))
# 1-3 A: SiN post at real-alpha site (500,950)
for r in (100.0, 150.0, 200.0):
    rows.append(scat_row(STACK_SHIFT, 2, STACK_PTN, STACK_PTW, r, 500.0, 950.0, 1.97))
# 4 B: air void at alpha<0 optimum (120,2000)
rows.append(scat_row(STACK_SHIFT, 2, STACK_PTN, STACK_PTW, 150.0, 120.0, 2000.0, 1.0))
# 5 B: SiN post at same site (sign control)
rows.append(scat_row(STACK_SHIFT, 2, STACK_PTN, STACK_PTW, 150.0, 120.0, 2000.0, 1.97))
# 6 C0: CD(-2) alone, no scatterer
rows.append(scat_row(cd_shifts(-2.0), 14, STACK_PTN, STACK_PTW, 0.0, 0.0, 1000.0, None))
# 7-8 C: CD(-2) + 2Λ dw={4,8}
for dw in (4.0, 8.0):
    sh, ptw, ptn = cd_plus_2lambda(dw)
    rows.append(scat_row(sh, 14, ptn, ptw, 0.0, 0.0, 1000.0, None))

assert len(rows) == 9, len(rows)

SPEC = SweepSpec(
    inner_shift_list_nm       = [r["shifts"] for r in rows],
    n_free_inner_teeth        = [r["nfree"] for r in rows],
    cavity_width_nm           = [1050.0] * 9,
    width_narrow_per_tooth_nm = [r["ptn"] for r in rows],
    width_wide_per_tooth_nm   = [r["ptw"] for r in rows],
    scatterer_radius_nm       = [r["r"] for r in rows],
    scatterer_x_list_nm       = [r["xl"] for r in rows],
    scatterer_y_list_nm       = [r["yl"] for r in rows],
    scatterer_index           = [r["idx"] for r in rows],
    simulation_mode           = ["accurate"] * 9,
    mode  = "zipped",
    label = "tm_novel_phase2",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
