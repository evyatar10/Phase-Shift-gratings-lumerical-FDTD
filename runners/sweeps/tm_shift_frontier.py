"""
TOOTH-SHIFT FRONTIER — follow-up to tm_center_completion (job 117927), which
found the tooth-shift family is the strongest fwhm-SAFE lever ever measured on
this device:
    W1050 + gap-shift pair [+20,+20]: loss 0.0823 -> 0.0549, T 0.9444,
    fwhm_m +1.0% (AT the user's <=1% bound), Q 1384 -> 1403 (mode NOT
    delocalized — unlike cavity lengthening, which hit -16e-3 only at
    fw +4.2% and is parked as constraint-violating).
Single shifts were NOT saturated (+10/-5.0, +20/-14.6, +30/-18.4 e-3), and the
2-tooth pair beat the single at the same per-tooth dose. This study maps the
frontier: dose, distribution over 1-3 teeth, interaction with cavity width /
length / the see-saw, and where the fwhm bound bites.

Mechanism reading (for the writeup): positive gap shifts pull the innermost
teeth toward the cavity with the cavity absorbing 2*sum (lengthen_cavity) —
a minimal 1-2 tooth interface impedance-matching section (the literature's #1
route, Lalanne-style, without a long taper). The fixed-cavity variant WIDENS
the mode (fw +2.6%) — the cavity-length compensation is load-bearing.

Registered predictions (filed before dispatch):
  P1 DOSE: loss falls with total shift dose but fwhm grows with it
     (+0.4%/+0.7%/+1.0% at 20/30/40 nm total); expect the loss-optimal
     IN-BOUND point near total dose ~40-55 nm. Rows past the bound (e.g.
     [+30,+30]) are frontier-mapping diagnostics, not candidates.
  P2 DISTRIBUTION: at fixed total dose, spreading over more teeth is SMOOTHER
     (better matching, lower loss) but may delocalize more — genuinely
     unknown which wins; that is the point of the triple rows.
  P3 SEE-SAW ADDITIVITY: pair + see-saw(40,-20) ~ additive (see-saw is
     width-space, shifts are position-space; 117927 already showed
     det x see-saw additivity) -> predict ~0.0531 at fw ~ +0.9%.
  P4 WIDTH INTERACTION: the cavity-width optimum may move under the pair
     (the pair reshapes the defect); W1000/W1100 x pair rows check. Weak
     prior: stays at 1050.
  P5 LENGTH TRADE: det +20 on the pair (cavity extra 80->60 nm) trades loss
     for fwhm margin; det -20 (extra 100 nm) the reverse. Maps the local
     Pareto corner around the best point.

Rows (zipped, 20 tasks, ALL accurate mesh, converged box y=6.8/z-mult 5.42,
window 1558.5 / 40 nm / 3001 — same numerics as 117927/117814):
   0   control W800 avg              (anchor 0.1174)
   1   W1050 + pair [+20,+20]        (reproduce best 0.0549)
   2   W1050 + pair [+20,+21]        (jitter partner, sub-mesh dose change)
   3   W1050 + single +40            (dose ladder, scalar tag _S40)
   4   W1050 + single +50            (scalar tag _S50)
   5   W1050 + pair [+25,+25]        (dose 50)
   6   W1050 + pair [+30,+30]        (dose 60 — expect fw violation, frontier)
   7   W1050 + pair [+30,+20]        (asymmetric dose split)
   8   W1050 + pair [+20,+30]        (reversed split)
   9   W1050 + triple [+15,+15,+15]  (dose 45 over 3 teeth)
  10   W1050 + triple [+20,+20,+20]  (dose 60 over 3 teeth)
  11   W1050 + triple [+20,+20,+10]  (tapered dose 50)
  12   W1050 + pair + see-saw(20,-20)  (ptw 1020/980)
  13   W1050 + pair + see-saw(40,-20)  (ptw 1040/980 — plane best)
  14   W1000 + pair [+20,+20]        (width interaction)
  15   W1100 + pair [+20,+20]
  16   W1050 + pair + det +20        (cavity extra 80->60: fwhm margin trade)
  17   W1050 + pair + det -20        (extra 100: loss trade)
  18   W1050 + single +35            (fill the single ladder knee)
  19   W1050 + pair [+15,+15]        (dose 30 — in-bound safety point)

Tags: scalar rows _S{n}_..._dsh1S{n}s{n}; list rows _dsh{k}S{tot}s{s0}; combos
add _ptw/_D prefixes. All 20 verified distinct (see assert).

Dispatch (queue must be EMPTY of other --option3 arrays):
    smoke: bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_shift_frontier --array-tasks=0
    full:  bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_shift_frontier
Output -> results/tm_shift_frontier/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps.tm_scatterer_scan import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

_NARROW = [600.0, 600.0]

_cw, _det, _ssc, _ishl, _nfree, _ptwn, _ptww = [], [], [], [], [], [], []


def row(w=None, det=0.0, ssc=0.0, ishl=None, nfree=1, ptwn=None, ptww=None):
    _cw.append(w); _det.append(det); _ssc.append(ssc); _ishl.append(ishl)
    _nfree.append(nfree); _ptwn.append(ptwn); _ptww.append(ptww)


PAIR = [20.0, 20.0]
row()                                                    # 0 control
row(w=1050.0, ishl=PAIR, nfree=2)                        # 1 best (reproduce)
row(w=1050.0, ishl=[20.0, 21.0], nfree=2)                # 2 jitter partner
row(w=1050.0, ssc=40.0)                                  # 3 single +40
row(w=1050.0, ssc=50.0)                                  # 4 single +50
row(w=1050.0, ishl=[25.0, 25.0], nfree=2)                # 5 pair 25/25
row(w=1050.0, ishl=[30.0, 30.0], nfree=2)                # 6 pair 30/30 (frontier)
row(w=1050.0, ishl=[30.0, 20.0], nfree=2)                # 7 pair 30/20
row(w=1050.0, ishl=[20.0, 30.0], nfree=2)                # 8 pair 20/30
row(w=1050.0, ishl=[15.0] * 3, nfree=3)                  # 9 triple 15x3
row(w=1050.0, ishl=[20.0] * 3, nfree=3)                  # 10 triple 20x3
row(w=1050.0, ishl=[20.0, 20.0, 10.0], nfree=3)          # 11 triple tapered
row(w=1050.0, ishl=PAIR, nfree=2,
    ptwn=_NARROW, ptww=[1020.0, 980.0])                  # 12 pair + see-saw(20,-20)
row(w=1050.0, ishl=PAIR, nfree=2,
    ptwn=_NARROW, ptww=[1040.0, 980.0])                  # 13 pair + see-saw(40,-20)
row(w=1000.0, ishl=PAIR, nfree=2)                        # 14 width interaction
row(w=1100.0, ishl=PAIR, nfree=2)                        # 15
row(w=1050.0, det=20.0, ishl=PAIR, nfree=2)              # 16 fwhm-margin trade
row(w=1050.0, det=-20.0, ishl=PAIR, nfree=2)             # 17 loss trade
row(w=1050.0, ssc=35.0)                                  # 18 single +35
row(w=1050.0, ishl=[15.0, 15.0], nfree=2)                # 19 pair 15/15

_n = 20
assert all(len(v) == _n for v in (_cw, _det, _ssc, _ishl, _nfree, _ptwn, _ptww))
_keys = [(w, d, s, tuple(l) if l else None, tuple(pw) if pw else None)
         for w, d, s, l, pw in zip(_cw, _det, _ssc, _ishl, _ptww)]
assert len(set(_keys)) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    cavity_width_nm           = _cw,
    cavity_neg_detuning_nm    = _det,
    innermost_tooth_shift_nm  = _ssc,
    inner_shift_list_nm       = _ishl,
    n_free_inner_teeth        = _nfree,
    width_narrow_per_tooth_nm = _ptwn,
    width_wide_per_tooth_nm   = _ptww,
    simulation_mode           = ["accurate"] * _n,
    mode  = "zipped",
    label = "tm_shift_frontier",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
