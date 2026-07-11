"""
CENTER-REGION COMPLETION — cavity LENGTH x WIDTH 2D, two-edge discriminators,
tooth-shift retest (user-requested), and small-knob additivity. All ACCURATE
mesh (expected effects at/below the dx=50 floor ~0.002; accurate floor ~1e-4).
Phase 1.5 of the loss program (docs/tm_loss_program_phase0_2026-07-05.md).

Combines what the plan called 2c + 2d into ONE array (shared controls, single
--option3 serialize slot). Window matches anti_moment_cavity (1558.5 / 40 nm /
3001) so rows cross-compare directly with job 117814 (champion 0.0823, ptw
best 0.0810, control 0.1174, jitter ~2e-4).

Registered predictions (filed before dispatch):
  P1 cavity LENGTH (det = pitch/2 - L_cav, +-20/40 nm) x WIDTH {1000,1050,1100}:
     shallow paraboloid, optimum at/near (det 0, 1050). The pitch was trimmed
     for co-resonance, not loss — but the local moment is now harvested, so
     expect <=1-2% relative effects. Closure scan.
  P2 TWO-EDGE DISCRIMINATOR (from lateral_radiation_theory.py section B): if
     the cavity-width ladder is two-sidewall interference at ky ~ kc, there is
     a loss MAXIMUM near W~1600 and a SECOND MINIMUM near W~2100. If it is a
     local moment-null, loss keeps worsening monotonically past 1400. Rows 15
     (W1600) and 16 (W2100) decide — the single cheapest physics test in the
     program.
  P3 TOOTH-SHIFT RETEST (user asked; TE-era lever, never tested at TM
     pitch 516.83/corr 400): registered guess = tooth-1 shift is largely the
     cavity-length knob in disguise (lengthen_cavity=True absorbs 2*shift into
     the cavity) -> compare with P1 rows; the fc (fixed-cavity) rows and the
     2-tooth rows carry the independent information.
  P4 ADDITIVITY: ptw(1020,980) [117814's -1.6%] stacked with det -+20 —
     additive if the two small moments are independent.
  P5 SEE-SAW GENERALIZATION (user 2026-07-05: "we can change innermost teeth
     as well as cavity"): 117814 probed only the CONSTRAINED see-saw axis
     (delta2 = -delta1 on the wide teeth). Rows 37-43 open the (delta1,
     delta2) plane — registered guess: the saturation at (20,-20)..(30,-30)
     is a ridge along the zero-area axis; the true 2D optimum is nearby with
     |gain| <= ~2x the see-saw's -1.6%. Rows 44-45 extend to tooth 3
     (+-10 on top of the (20,-20) best). Rows 46-48 open the NARROW-tooth
     see-saw (600+-delta) — an untouched even family; narrow teeth sit at the
     field antinodes' gaps so the moment weight differs; no sign prediction
     registered (genuinely unknown).

Rows (zipped, 49 tasks, accurate mesh, converged box y=6.8/z-mult 5.42):
   0     control W800 avg          (cross-study anchor: expect loss 0.1174)
   1     W1050                     (champion: expect 0.0823)
   2     W1052                     (jitter partner: expect |delta| ~ 2e-4)
   3-6   W1000 x det {-40,-20,+20,+40}
   7-10  W1050 x det {-40,-20,+20,+40}
  11-14  W1100 x det {-40,-20,+20,+40}
  15     W1600                     (P2: predicted max if two-edge)
  16     W2100                     (P2: predicted 2nd minimum if two-edge)
  17-18  W1050 + ptw(1020,980) x det {-20,+20}   (P4)
  19-21  W800  tooth-1 shift +10/+20/+30 (legacy scalar -> tags _S10/_S20/_S30)
  22-24  W800  tooth-1 shift -10/-20/-30 (list path -> _dsh1Sm10sm10 ...)
  25-27  W1050 tooth-1 shift +10/+20/+30
  28-30  W1050 tooth-1 shift -10/-20/-30
  31-32  W800  teeth-1&2 shift [+20,+20] / [-20,-20]
  33-34  W1050 teeth-1&2 shift [+20,+20] / [-20,-20]
  35-36  W800 / W1050 shift +20 with lengthen_cavity=False (tag _S20_fc)
  37-43  W1050 off-axis see-saw (d1,d2): (20,-30) (20,-40) (30,-20) (30,-40)
         (40,-20) (40,-30) (40,-40)                                    (P5)
  44-45  W1050 tooth-3 extension: (1020,980,1010) / (1020,980,990)     (P5)
  46-48  W1050 narrow see-saw: wide(1020,980)+narrow(620,580) /
         wide(1020,980)+narrow(580,620) / wide(1000,1000)+narrow(620,580)
         (narrow lists get the new _ptn tag — sim_helpers extended)    (P5)

Tag notes (why the shift rows are split between the scalar and list knobs):
positive scalar shifts tag _S{nm} (+ _fc when lengthen_cavity False) but
NEGATIVE scalar shifts are untagged -> negatives go through inner_shift_list
(_dsh{n}S{tot}s{s0}, m-prefix for minus). All 37 tags verified distinct.

Dispatch (queue must be EMPTY of other --option3 arrays — shared sweep_list.txt;
tm_radiation_polarimetry goes first):
    smoke: bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_center_completion --array-tasks=0
    full:  bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_center_completion
Output -> results/tm_center_completion/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


BOX_Y_UM = 6.8
BOX_Z_MULT = 5.42

BASE = build_base()
BASE.scatterer.enabled = False        # build_base leaves the scatterer ON (r=150 default)!
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.span_multiplier_override = BOX_Z_MULT
BASE.spectral.center_wavelength_m = 1.5585e-6   # match anti_moment window exactly
BASE.spectral.scan_width_nm = 40.0
BASE.spectral.n_wl_points = 3001

_NARROW = [600.0, 600.0]
_PTW_BEST = [1020.0, 980.0]           # 117814 Family-A best (-1.6% vs champion)
_DETS = [-40.0, -20.0, 20.0, 40.0]

_cw, _det, _ssc, _ishl, _nfree, _lenc, _ptwn, _ptww = [], [], [], [], [], [], [], []


def row(w=None, det=0.0, ssc=0.0, ishl=None, nfree=1, lenc=True,
        ptwn=None, ptww=None):
    # nfree/lenc must be REAL values on every row: zipped sweeps assign None
    # entries onto the config (None n_free crashes the builder's tooth loop;
    # None lengthen_cavity is falsy -> silently behaves as False).
    _cw.append(w); _det.append(det); _ssc.append(ssc); _ishl.append(ishl)
    _nfree.append(nfree); _lenc.append(lenc); _ptwn.append(ptwn); _ptww.append(ptww)


row()                                             # 0  control W800
row(w=1050.0)                                     # 1  champion
row(w=1052.0)                                     # 2  jitter partner
for w in (1000.0, 1050.0, 1100.0):                # 3-14  L x W grid
    for d in _DETS:
        row(w=w, det=d)
row(w=1600.0)                                     # 15 two-edge max
row(w=2100.0)                                     # 16 two-edge 2nd minimum
for d in (-20.0, 20.0):                           # 17-18 ptw additivity
    row(w=1050.0, det=d, ptwn=_NARROW, ptww=_PTW_BEST)
for w in (None, 1050.0):                          # 19-30 tooth-1 shifts
    for s in (10.0, 20.0, 30.0):
        row(w=w, ssc=s)
    for s in (-10.0, -20.0, -30.0):
        row(w=w, ishl=[s], nfree=1)
for w in (None, 1050.0):                          # 31-34 teeth-1&2 pairs
    row(w=w, ishl=[20.0, 20.0], nfree=2)
    row(w=w, ishl=[-20.0, -20.0], nfree=2)
row(ssc=20.0, lenc=False)                         # 35 W800 fixed-cavity
row(w=1050.0, ssc=20.0, lenc=False)               # 36 W1050 fixed-cavity
for d1, d2 in ((20.0, -30.0), (20.0, -40.0), (30.0, -20.0), (30.0, -40.0),
               (40.0, -20.0), (40.0, -30.0), (40.0, -40.0)):
    row(w=1050.0, ptwn=_NARROW, ptww=[1000.0 + d1, 1000.0 + d2])   # 37-43
for d3 in (10.0, -10.0):                          # 44-45 tooth-3 extension
    row(w=1050.0, ptwn=[600.0] * 3, ptww=[1020.0, 980.0, 1000.0 + d3])
row(w=1050.0, ptwn=[620.0, 580.0], ptww=_PTW_BEST)          # 46 narrow see-saw +
row(w=1050.0, ptwn=[580.0, 620.0], ptww=_PTW_BEST)          # 47 narrow see-saw -
row(w=1050.0, ptwn=[620.0, 580.0], ptww=[1000.0, 1000.0])   # 48 narrow-only

_n = 49
assert all(len(v) == _n for v in (_cw, _det, _ssc, _ishl, _nfree, _lenc, _ptwn, _ptww))
_keys = [(w, d, s, tuple(l) if l else None, lc,
          tuple(pw) if pw else None, tuple(pn) if pn else None)
         for w, d, s, l, lc, pw, pn in zip(_cw, _det, _ssc, _ishl, _lenc, _ptww, _ptwn)]
assert len(set(_keys)) == _n, "duplicate row -> tag collision"

SPEC = SweepSpec(
    cavity_width_nm           = _cw,
    cavity_neg_detuning_nm    = _det,
    innermost_tooth_shift_nm  = _ssc,
    inner_shift_list_nm       = _ishl,
    n_free_inner_teeth        = _nfree,
    lengthen_cavity           = _lenc,
    width_narrow_per_tooth_nm = _ptwn,
    width_wide_per_tooth_nm   = _ptww,
    simulation_mode           = ["accurate"] * _n,
    mode  = "zipped",
    label = "tm_center_completion",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
