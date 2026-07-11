"""
BATCH 1 of the BIC/Kerker/counterdiabatic program (Phase-0 survivors, user-
approved 2026-07-06). Gate verdicts: docs/phase0_gate_verdicts_2026-07-06.md.

One combined zipped array (35 rows), ordered by likelihood: controls ->
forward-Huygens scatterers (validated +0.0026 anchor) -> vertical anti-phase
2*Lambda alternation -> TE Huygens -> counterdiabatic falsifier. The FW-BIC
side-cavity scan (needs a builder extension: device-2 n_periods_2/pitch_2)
ships separately as batch 1b.

Physics line: target resonance 1558.6 nm (opt-mesh rows) / 1556.6 nm
(accurate rows); scan window 1538.5-1578.5 nm (40 nm, 3001 pts) covers both
plus any alternation-induced detuning/partner. TM rows: h=350 nm, pitch
516.83 nm, corr 400 (wide 1000 / narrow 600); TE rows: corr 300 co-resonant
at the same pitch. Converged box y=6.8 um / z-mult 5.42.

REGISTERED PREDICTIONS (honesty gate, from the Phase-0 models):
  * Huygens rows (accurate, stack base, loss 0.0545): point-source placement
    ceiling ~+0.008 T for one pair at the map optimum; Kerker directionality
    (F:B 65:1 at r~250 vs 2.3:1 at the tested r=150) should push the achieved
    dT TOWARD that ceiling: prediction dT(r250@A) > dT(r150@L) ~ +0.0026,
    with dT(r250) in +0.003..+0.008. Effects are 2-25x the accurate-mesh
    jitter floor (~1e-4-3e-4, re-measured here by rows 8-9).
  * Alternation rows (opt mesh, rect-1050 base, loss ~0.077, vertical share
    ~0.030): a 2*Lambda width alternation is a first-order vertical
    out-coupler; amplitude ratio 0.14 of the vertical leak needs only
    nm-scale dW. Prediction: loss changes NON-monotonic in dW with a sign
    asymmetry between +/- (phase 0 vs pi) and between wide/narrow families
    (phase quadrature pair); best row cuts vertical-channel loss by tens of
    percent of 0.030 if the leak is coherent; a null everywhere = incoherent
    vertical leak (report as such).
  * TE Huygens rows (accurate): TE has a sharp 2D Huygens point at r=260
    (F:B ~1e5): prediction dT(260) > dT(200), dT(320) — a bracket peak.
  * Counterdiabatic rows (accurate, stack base): kernel predicts ~NULL
    (optimal quadrature profile == uniform pair to within the floor); the
    x2 and -1 scales are the falsifiers. Expected |dLoss| <~ 0.003.

Rows (zipped, 35 tasks):
   0     rect-1050 control, opt mesh (alternation Delta-reference)
   1     the stack control, accurate (Huygens/CD Delta-reference)
   2-4   Huygens TM r={200,250,300} at map site A (x=380, y=1020) nm
   5     Huygens TM r=250 at map site B (x=620, y=1400)
   6-7   plain r=150 / Huygens r=250 at the legacy validated site (810, 1000)
   8-9   jitter partners: r=250 @A and @L with x + 17 nm (half accurate cell)
   10    two-pair row: r=250 at [A, map-partner (1000, 870)]
   11-18 wide-tooth alternation +-{2,4,8,16} nm, 16 teeth/side (phase 0/pi)
   19-24 narrow-tooth alternation +-{4,8,16} nm, 16 teeth/side (phase +-pi/2)
   25-26 wide alternation +-8 nm, 8 teeth/side (extent check)
   27    TE control, accurate (corr 300)
   28-30 TE Huygens r={200,260,320} at (810, 1000)
   31-34 counterdiabatic per-tooth-translation profile x {+1,-1,+2,-2}
         on the stack (shift lists len 14; profile from phase0_counterdiabatic)

Dispatch (queue must be EMPTY of other --option3 arrays):
    bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_bic_kerker_batch1
Output -> results/tm_bic_kerker_batch1/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = True         # radius 0 rows draw nothing (identical numerics)
BASE.scatterer.mirrored_y = True
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001
BASE.monitors.record_2d_fields = False
BASE.monitors.record_3d_fields = False
BASE.farfield.enabled = False

# ── The stack (current best in-bound device) ────────────────────────────────
STACK_SHIFT = [20.0, 20.0]
STACK_PTW = [1040.0, 980.0]
STACK_PTN = [600.0, 600.0]

# ── Alternation lists (2*Lambda superperiod = vertical out-coupler) ─────────
N_ALT = 16


def alt(base_w, dw, n):
    return [base_w + (dw if j % 2 == 0 else -dw) for j in range(n)]


# ── Counterdiabatic quadrature profile (phase0_counterdiabatic.py solve) ────
CD_PROFILE = [-0.2, 8.6, -4.9, -4.0, 0.1, 0.6, -0.9, -2.8,
              2.6, 2.6, 1.4, -0.3, -2.3, -1.6]


def cd_shifts(scale):
    base = [20.0, 20.0] + [0.0] * 12
    return [round(b + scale * s, 2) for b, s in zip(base, CD_PROFILE)]


# ── Row table (zipped) ───────────────────────────────────────────────────────
rows = []           # (mode, pol, corr, cavW, shifts, nfree, ptn, ptw, r, x, y)
A = (380.0, 1020.0)
B = (620.0, 1400.0)
L = (810.0, 1000.0)

def add(mode, pol, corr, cavW, shifts, nfree, ptn, ptw, r, xy, xlist=None, ylist=None):
    x, y = xy if xlist is None else (None, None)
    rows.append(dict(mode=mode, pol=pol, corr=corr, cavW=cavW, shifts=shifts,
                     nfree=nfree, ptn=ptn, ptw=ptw, r=r,
                     xl=xlist if xlist is not None else [x],
                     yl=ylist if ylist is not None else [y]))

STK = dict(shifts=STACK_SHIFT, nfree=2, ptn=STACK_PTN, ptw=STACK_PTW)
# 0-1 controls
add("optimization", "TM", 400.0, 1050.0, None, 1, None, None, 0.0, (0.0, 1000.0))
add("accurate", "TM", 400.0, 1050.0, r=0.0, xy=(0.0, 1000.0), **STK)
# 2-10 Huygens TM (stack base, accurate)
for r, site in ((200.0, A), (250.0, A), (300.0, A), (250.0, B),
                (150.0, L), (250.0, L)):
    add("accurate", "TM", 400.0, 1050.0, r=r, xy=site, **STK)
add("accurate", "TM", 400.0, 1050.0, r=250.0, xy=(A[0] + 17.0, A[1]), **STK)
add("accurate", "TM", 400.0, 1050.0, r=250.0, xy=(L[0] + 17.0, L[1]), **STK)
add("accurate", "TM", 400.0, 1050.0, r=250.0, xy=None,
    xlist=[A[0], 1000.0], ylist=[A[1], 870.0], **STK)
# 11-18 wide alternation (rect-1050 base, opt mesh)
for dw in (2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0):
    add("optimization", "TM", 400.0, 1050.0, None, 1,
        [600.0] * N_ALT, alt(1000.0, dw, N_ALT), 0.0, (0.0, 1000.0))
# 19-24 narrow alternation
for dw in (4.0, -4.0, 8.0, -8.0, 16.0, -16.0):
    add("optimization", "TM", 400.0, 1050.0, None, 1,
        alt(600.0, dw, N_ALT), [1000.0] * N_ALT, 0.0, (0.0, 1000.0))
# 25-26 extent check (8 teeth)
for dw in (8.0, -8.0):
    add("optimization", "TM", 400.0, 1050.0, None, 1,
        [600.0] * 8, alt(1000.0, dw, 8), 0.0, (0.0, 1000.0))
# 27-30 TE rows (corr 300 co-resonant, accurate)
add("accurate", "TE", 300.0, None, None, 1, None, None, 0.0, (0.0, 1000.0))
for r in (200.0, 260.0, 320.0):
    add("accurate", "TE", 300.0, None, None, 1, None, None, r, L)
# 31-34 counterdiabatic falsifier (stack base, accurate)
for sc in (1.0, -1.0, 2.0, -2.0):
    add("accurate", "TM", 400.0, 1050.0, cd_shifts(sc), 14,
        STACK_PTN, STACK_PTW, 0.0, (0.0, 1000.0))

assert len(rows) == 35, f"task count changed: {len(rows)}"
_sig = [(r["mode"], r["pol"], r["corr"], r["cavW"], tuple(r["shifts"] or []),
         tuple(r["ptn"] or []), tuple(r["ptw"] or []), r["r"],
         tuple(r["xl"]), tuple(r["yl"])) for r in rows]
assert len(set(_sig)) == len(rows), "duplicate row -> file-tag collision"

SPEC = SweepSpec(
    simulation_mode           = [r["mode"] for r in rows],
    polarization              = [r["pol"] for r in rows],
    corrugation_depth_nm      = [r["corr"] for r in rows],
    cavity_width_nm           = [r["cavW"] for r in rows],
    inner_shift_list_nm       = [r["shifts"] for r in rows],
    n_free_inner_teeth        = [r["nfree"] for r in rows],
    width_narrow_per_tooth_nm = [r["ptn"] for r in rows],
    width_wide_per_tooth_nm   = [r["ptw"] for r in rows],
    scatterer_radius_nm       = [r["r"] for r in rows],
    scatterer_x_list_nm       = [r["xl"] for r in rows],
    scatterer_y_list_nm       = [r["yl"] for r in rows],
    mode  = "zipped",
    label = "tm_bic_kerker_batch1",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
