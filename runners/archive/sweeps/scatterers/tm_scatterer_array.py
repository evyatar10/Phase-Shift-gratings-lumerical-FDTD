"""
Multi-pillar ARRAY study — does the recycling add coherently, and along which line?

Placement physics (paper_8 Eq. 15-16): the round-trip phase of each pillar's
returned light is 2*k_clad*rho, rho = RADIAL distance from the defect — so the
constructive comb is spaced d = lambda_res/(2*n_clad) ~ 0.5397 um in RHO, not in x
(user-caught correction: at y=1 um the second site is x=1531 nm, NOT 810+540).
Three competing geometries are tested head-to-head, all r-provisional 100 nm,
each site an (x, ±y) mirrored pair:

  idx 0: control (no scatterers)
  idx 1: N=1 [ (810,1000) ]                      — confirmed best single, reference
  idx 2: RHO-COMB line N=3 @ y=1000 nm           — x = 810, 1531, 2145 (rho steps of d)
  idx 3: RHO-COMB line N=6 @ y=1000 nm           — x = 810, 1531, 2145, 2723, 3283, 3833
  idx 4: MEASURED-WINNER set N=4 @ y=1000 nm     — x = 810, 4050, 4590, 6075
                                                   (each independently verified constructive)
  idx 5: SAME-ARC N=3 (user's diagonal idea A)   — constant rho = 4.174 um, spread in angle:
                                                   (4050,1000), (3982,1250), (3895,1500)
                                                   — identical phase by construction
  idx 6: LOBE-RAY rho-comb N=3 (diagonal idea B) — along the ray through (4050,1000)
                                                   (~13.9 deg), rho steps of d:
                                                   (4050,1000), (4574,1129), (5097,1259)

Why the main line stays at y = 1000 nm: the leaky field decays ~e^(-1.8 y/um)
laterally, so climbing the diagonal costs ~2.6x field per arc step — the fixed-y
line keeps every site in the strongest reachable field, and its angle from the
defect approaches the lobe direction as x grows anyway. The arc and ray rows put
the user's angle hypothesis to a direct empirical test at equal cost.

Honest expectation: coherent addition would give roughly the SUM of the single-
pair gains (idx 2/3: ~+0.003..0.005; idx 4: ~+0.003) — N^2 only in the ideal
far-zone limit; mutual re-scattering may erode it; the arc/ray rows start from a
weaker (+0.0006) anchor so even doubling is a small absolute number.

Numerics: ACCURATE mesh; converged transverse box — SET BOX_Y_UM / BOX_Z_MULT
from tm_span_convergence(2) before dispatch. Own control at identical numerics.

Dispatch (queue EMPTY of other --option3 arrays; AFTER tm_scatterer_radius):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_scatterer_array
Output -> results/tm_scatterer_array/results/.
"""

import os
import sys

# Make the project root importable when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


# Converged box (jobs 116854/116870): y 6.8 um, z 8.8 um (mult 5.42).
BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42
R_NM = 100.0         # confirmed optimum (tm_scatterer_radius, job 116896: dT +0.0026)

Y0_NM = 1000.0       # main-line lateral offset
D_NM = 539.7         # recoupling period lambda_res/(2 n_clad) at 1558.6 nm


def _rho_comb_x(x0, y0, n):
    """x positions on the fixed-y line with RADIAL spacing D_NM (paper phase rule)."""
    import math
    rho0 = math.hypot(x0, y0)
    return [round(math.sqrt((rho0 + k * D_NM) ** 2 - y0 ** 2), 1) for k in range(n)]


BASE = build_base()
BASE.mesh.simulation_mode = "accurate"
BASE.y_span_override_m = BOX_Y_UM * 1e-6
if BOX_Z_MULT is not None:
    BASE.span_multiplier_override = BOX_Z_MULT

_line3 = _rho_comb_x(810.0, Y0_NM, 3)            # [810, 1531.2, 2144.9]
_line6 = _rho_comb_x(810.0, Y0_NM, 6)

_rows_x = [[0.0], [810.0], _line3, _line6,
           [810.0, 4050.0, 4590.0, 6075.0],
           [4050.0, 3981.6, 3894.7],             # same-arc, rho = 4173.7 nm
           [4050.0, 4573.7, 5097.4]]             # lobe-ray rho-comb (~13.9 deg)
_rows_y = [None, None, None, None, None,
           [1000.0, 1250.0, 1500.0],
           [1000.0, 1129.3, 1258.6]]
_rs     = [0.0] + [R_NM] * 6

assert len(_rows_x) == len(_rows_y) == len(_rs) == 7
_keys = {(r, len(xs), round(xs[0]), round(xs[-1]),
          round(ys[0]) if ys else 0, round(ys[-1]) if ys else 0)
         for r, xs, ys in zip(_rs, _rows_x, _rows_y)}
assert len(_keys) == 7, "array rows must differ in (r, N, first/last x, first/last y)"

SPEC = SweepSpec(
    scatterer_radius_nm = _rs,
    scatterer_x_list_nm = _rows_x,
    scatterer_y_list_nm = _rows_y,
    scatterer_y_nm      = [Y0_NM] * 7,
    mode  = "zipped",
    label = "tm_scatterer_array",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
