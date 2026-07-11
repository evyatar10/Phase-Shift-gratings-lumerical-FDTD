"""
CLADDING REFLECTOR study (2026-07-08) — reflect the lateral leak back toward the
mode with (A) a 1D SiN/oxide BRAGG MIRROR (DBR) and (B) a 2D PHOTONIC-CRYSTAL
lattice, both PLANAR (350 nm, same layer as the core) and same length as the
regular pi-shift device. TM. Distinct from the (dead) side-cavity FW: this is a
passive REFLECTOR in the cladding, not a coupled resonator.

THEORY GATE (python_tools/phase0_cladding_reflector.py,
docs/... ) — from the MEASURED leak's angular spectrum:
  * air TIR (the trench) reflects only the GRAZING share (|kx|/kc>0.69, ~46%);
    it LETS THROUGH the more-normal lateral leak. Measured trench gain -0.012.
  * a 1D SiN/oxide DBR reflects a TUNABLE stopband of angles (pure Bragg, no TIR
    -- both indices >= incidence oxide). Best-tuned ceiling ~60-65% reflected
    (it can grab the near-normal part TIR misses). Complementary to the trench.
  * a 2D PhC COMPLETE gap reflects ALL angles -> full lateral channel (ceiling
    loss 0.0545 -> ~0.023). SiN/oxide contrast is only 1.36 => expect a PARTIAL /
    directional gap -> real PhC sits BETWEEN the DBR (~65%) and the ceiling.
HONESTY: reflected-flux is an UPPER BOUND; the loss drop needs the reflected light
to RE-COUPLE into the localized mode, which is kx-weighted toward GRAZING (what
the trench already catches). Whether the DBR/PhC's extra (near-normal) reflection
CONVERTS is exactly what this FDTD settles. The air trench already PROVED the
reflect->recouple mechanism works once (-0.012), so this tests 'does a Bragg/PhC
mirror beat TIR', not 'does reflection help at all'.

Registered predictions (filed before dispatch):
  P1 A working reflector shows loss BELOW the in-study control (row 0) AND a
     structured dependence on design/offset (both better- and worse- than control
     placements). All-flat within ~2x the mesh floor = null -> reflector re-coupling
     is the bottleneck, route closed.
  P2 Best-case magnitude if the near-normal reflection re-couples like the grazing
     did: dLoss up to ~ -0.02 (DBR) / more (PhC). Honest ceiling; re-coupling may
     eat it (grazing re-couples best, and that's the trench's job already).
  P3 DBR design B (grazing 0.7) should behave most like the trench; design A
     (near-normal 0.5) tests the NEW angular content. PhC lattice-constant scan
     brackets any partial gap.
  P4 Length control: the central-only PhC (row 11, +-20 um) vs full-length (row 6)
     -- if full-length is needed, short underperforms (as the trench L=20 did).

Geometry: DBR = SiN rect strips (n=1.97, planar 350 nm) parallel to the guide,
quarter-wave stack in y (thickness t_H, period Lambda from the design angle),
N periods from offset d, mirrored +-y, length = device length. PhC = SiN rods
(cylinders, planar 350 nm) on a square lattice (constant a, radius r), N_rows in
y from offset d, spanning +-L_dev/2 in x, mirrored +-y. Background = oxide.

Box: y_span 16 um (reflector reaches |y|<=~5.9 um; keeps >=1.5 um PML clearance),
z-mult 5.42, accurate mesh (resolves the ~210 nm strips / r~110 nm rods; opt mesh
under-resolves them), monitors OFF (T/R spectra only -> memory ~ convergence run).
Window 1556.5 / 40 nm / 3001 (stack lambda_res ~1556.6 accurate, in-window).

Dispatch (queue EMPTY -- ONE array, DBR+PhC rows mixed so a single --option3 keeps
the shared sweep_list serialize-safe):
    smoke:  bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_cladding_reflector --array-tasks=0
    full:   bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_cladding_reflector
Output -> results/tm_cladding_reflector/results/.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


# ── physics constants for the DBR quarter-wave design ───────────────────────────
LAM_UM = 1.5566
N_CLAD = 1.444
N_SIN = 1.97
K0 = 2 * math.pi / LAM_UM
KC = N_CLAD * K0

BOX_Y_UM = 16.0
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.source.polarization = "TM"                 # explicit (build_base is TM); user-confirmed

# stack baseline (same device the trench/CD studies used)
BASE.grating.cavity_width_m = 1050e-9
BASE.grating.inner_shift_nm = [20.0, 20.0]
BASE.grating.n_free_inner_teeth = 2
BASE.grating.width_narrow_per_tooth_m = [600e-9, 600e-9]
BASE.grating.width_wide_per_tooth_m = [1040e-9, 980e-9]

BASE.mesh.simulation_mode = "accurate"
BASE.spectral.center_wavelength_m = 1.5565e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled = False

# scatterer machinery: planar (height None -> core 350 nm), mirrored pair
BASE.scatterer.enabled = True
BASE.scatterer.mirrored_y = True
BASE.scatterer.x_m = 0.0

# device length = the regular pi-shift device (user default): 2*N*pitch + cavity
N_PER = BASE.grating.n_periods_each_side
PITCH_UM = BASE.grating.pitch_m * 1e6
L_DEV_UM = round(2 * N_PER * PITCH_UM + BASE.grating.cavity_width_m * 1e6, 2)
XHALF_UM = L_DEV_UM / 2.0


def dbr_design(frac):
    """Quarter-wave SiN/oxide DBR at incidence |kx|/kc=frac -> (t_H nm, Lambda um)."""
    kx = frac * KC
    kzH = math.sqrt((N_SIN * K0) ** 2 - kx ** 2)
    kzL = math.sqrt(KC ** 2 - kx ** 2)
    tH = (math.pi / 2) / kzH
    tL = (math.pi / 2) / kzL
    return round(tH * 1000, 1), tH + tL


def dbr_ylist(d_um, N, Lam_um):
    return [round((d_um + m * Lam_um) * 1000, 1) for m in range(N)]


def phc_grid(a_um, n_rows, d_um, xhalf_um):
    """Square lattice of rods: x in [-xhalf, xhalf] step a, y in d..d+(n_rows-1)a."""
    nx = int(round(2 * xhalf_um / a_um))
    xs = [round((-xhalf_um + i * a_um) * 1000, 1) for i in range(nx + 1)]
    ys = [round((d_um + r * a_um) * 1000, 1) for r in range(n_rows)]
    xl, yl = [], []
    for yv in ys:
        for xv in xs:
            xl.append(xv)
            yl.append(yv)
    return xl, yl


tH_A, Lam_A = dbr_design(0.5)                    # near-normal; period 524 nm ~= device pitch 517 nm
tH_B, Lam_B = dbr_design(0.7)                    # moderate/grazing (trench-like)

# Reflector length = CENTRAL, not full device: the radiation is the mode-envelope
# tail (~15 um FWHM localized at center); the arms are mostly Bragg MIRRORS that
# radiate weakly along their length. A central reflector captures the dominant
# leak, builds robustly, and avoids the full-length rods at the grating edge. DBR
# strips are continuous (ends far out in the weak-radiation zone) so they keep the
# full device length like the trench; the PhC uses a +-25 um central span.
PHC_HALF_UM = 25.0

rows = []


def add(shape, xspan=None, yspan=None, r=0.0, ynm=1500.0, idx=1.97, xl=None, yl=None, label=""):
    rows.append(dict(shape=shape, xspan=xspan, yspan=yspan, r=r, y=ynm,
                     idx=idx, xl=xl, yl=yl, label=label))


# 0 control (no reflector, same box) — identical-numerics Δ reference.
# xspan=None -> _has_scatterer False -> nothing drawn; dummy xl/yl only satisfy the
# expand() list transform (which maps float over the list; None is not iterable).
add("rect", xspan=None, yspan=None, r=0.0, ynm=1500.0, idx=None, xl=[0.0], yl=[1500.0], label="control")

# 1-5 DBR (SiN strips, full device length). Design A period ~= device pitch, so
# the user's 'same period as the device' intuition is tested by design A.
add("rect", xspan=L_DEV_UM, yspan=tH_A, ynm=1500.0, xl=[0.0] * 5, yl=dbr_ylist(1.5, 5, Lam_A), label="DBR_A_d1p5_N5")
add("rect", xspan=L_DEV_UM, yspan=tH_A, ynm=1500.0, xl=[0.0] * 10, yl=dbr_ylist(1.5, 10, Lam_A), label="DBR_A_d1p5_N10")
add("rect", xspan=L_DEV_UM, yspan=tH_A, ynm=1800.0, xl=[0.0] * 8, yl=dbr_ylist(1.8, 8, Lam_A), label="DBR_A_d1p8_N8")
add("rect", xspan=L_DEV_UM, yspan=tH_B, ynm=1500.0, xl=[0.0] * 8, yl=dbr_ylist(1.5, 8, Lam_B), label="DBR_B_d1p5_N8")
add("rect", xspan=L_DEV_UM, yspan=tH_A, ynm=1525.0, xl=[0.0] * 10, yl=dbr_ylist(1.525, 10, Lam_A), label="DBR_A_jitter")

# 6-11 PhC (SiN rods, square lattice, central +-25 um). a=500 ~= device pitch;
# a-scan brackets the partial-gap frequency for the leak.
# PhC OFFSET (distance from the pi-shift) is scanned d = 1.2/1.5/1.8/2.1 um on the
# a500/r110 lattice — the trench showed d is CRITICAL (too close = enters the mode
# and detunes/hurts; too far = catches less). Plus gap-tune (a) and fill (r) rows.
for (a, r, nrows, d, half, lab) in [
    (0.50, 110.0, 6, 1.2,  PHC_HALF_UM, "PhC_a500_d1p2"),    # distance scan (close)
    (0.50, 110.0, 6, 1.5,  PHC_HALF_UM, "PhC_a500_r110"),    # distance scan (ref)
    (0.50, 110.0, 6, 1.8,  PHC_HALF_UM, "PhC_a500_d1p8"),    # distance scan
    (0.50, 110.0, 6, 2.1,  PHC_HALF_UM, "PhC_a500_d2p1"),    # distance scan (far)
    (0.50, 140.0, 6, 1.5,  PHC_HALF_UM, "PhC_a500_r140"),    # higher fill
    (0.46, 105.0, 7, 1.5,  PHC_HALF_UM, "PhC_a460_r105"),    # gap tune down
    (0.58, 130.0, 5, 1.5,  PHC_HALF_UM, "PhC_a580_r130"),    # gap tune up
    (0.50, 110.0, 6, 1.525, PHC_HALF_UM, "PhC_a500_jitter"),  # half-cell d jitter
]:
    xl, yl = phc_grid(a, nrows, d, half)
    add("cylinder", r=r, ynm=round(d * 1000, 1), xl=xl, yl=yl, label=lab)

# length control: SHORT central PhC (+-12 um) vs the +-25 um ref
xl, yl = phc_grid(0.50, 6, 1.5, 12.0)
add("cylinder", r=110.0, ynm=1500.0, xl=xl, yl=yl, label="PhC_a500_short12")

_n = len(rows)
assert _n == 15, _n

SPEC = SweepSpec(
    scatterer_shape     = [r["shape"] for r in rows],
    scatterer_x_span_um = [r["xspan"] for r in rows],
    scatterer_y_span_nm = [r["yspan"] for r in rows],
    scatterer_radius_nm = [r["r"] for r in rows],
    scatterer_y_nm      = [r["y"] for r in rows],
    scatterer_index     = [r["idx"] for r in rows],
    scatterer_x_list_nm = [r["xl"] for r in rows],
    scatterer_y_list_nm = [r["yl"] for r in rows],
    mode  = "zipped",
    label = "tm_cladding_reflector",
)


if __name__ == "__main__":
    print(f"device length L_dev = {L_DEV_UM} um  (N={N_PER}/side, pitch {PITCH_UM*1000:.1f} nm)")
    print(f"DBR design A (|kx|/kc=0.5): t_H={tH_A} nm, Lambda={Lam_A*1000:.1f} nm, "
          f"N=8 reaches y={1.5+7*Lam_A+tH_A/2000:.2f} um")
    print(f"DBR design B (|kx|/kc=0.7): t_H={tH_B} nm, Lambda={Lam_B*1000:.1f} nm, "
          f"N=8 reaches y={1.5+7*Lam_B+tH_B/2000:.2f} um")
    for i, r in enumerate(rows):
        nrod = 0 if not r["xl"] else (2 * len(r["xl"]))
        print(f"  row {i:2d} {r['label']:20s} shape={r['shape']:8s} "
              f"{'strips/rods='+str(nrod) if nrod else 'CONTROL'}")
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
