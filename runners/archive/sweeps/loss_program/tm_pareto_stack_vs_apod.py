"""
STACK vs APODIZATION — the loss-vs-mode-width PARETO comparison + modularity
test (user framing 2026-07-05: the deliverable is transferable MODULES and
whether the cavity-only stack competes with apodization WITHOUT paying the
mode-width price; later phases combine with apodization / inverse design).
DISPATCH ONLY AFTER THE PHASE-1 CHECKPOINT (order: after tm_center_completion
and tm_strip_reflector drain — shared sweep_list.txt, serialize).

The question, precisely: apodization buys loss reduction at a KNOWN mode-width
cost (fwhm_m grows). The center-region stack — UPDATED 2026-07-06 after jobs
117927/118214 to **cavity 1050 + gap-shift pair [+20,+20] + see-saw (1040,980)**
— buys loss 0.1174 -> 0.0545 (T 0.9449) at fwhm +0.9%, and the shift-frontier
measured the family's tradeoff slope: ~ -1e-3 loss per +0.1% fwhm. On the
(fwhm_m, loss) plane, is that a Pareto segment apodization cannot reach at
small widths? And do the modules TRANSFER under an apodized envelope (both the
cavity-width null and the NEW shift-pair module)?

Registered predictions (filed before dispatch):
  P1 Apod ladder reproduces the known trend: loss falls, fwhm_m grows with
     n_apod (sanity anchor against the 2026-04 tm_te_apod study).
  P2 PARETO (the money question): at fwhm change <= +1%, the stack's loss
     (~0.081 accurate) beats the apodization curve interpolated to the same
     width. No prediction beyond +2% width — apodization likely wins at large
     widths (it attacks the envelope tail, a bigger budget).
  P3 MODULARITY: Delta-loss from adding cavity-1050 to an apodized device ~
     the standalone Delta (the null is defect-LOCAL). Registered risk: the
     apodized envelope changes the local mode width -> the 1050 null detunes;
     a shifted-but-present null (better at 1000 or 1100) still confirms
     modularity of the MECHANISM, only the operating point moves.

Rows (zipped, 12 tasks, ALL accurate mesh, converged box y=6.8/z-mult 5.42,
window 1558.5 / 40 nm / 3001 — same numerics as tm_center_completion/118214
for direct cross-comparison):
   0  control W800 (anchor; expect loss 0.1174)
   1  stack: 1050 + pair[+20,+20] + see-saw(1040,980)  (expect 0.0545)
   2  apod n=5,  linear (default depth 100 nm)
   3  apod n=10, linear
   4  apod n=20, linear
   5  apod n=10 + cavity 1050                    (P3 width-module transfer)
   6  apod n=10 + cavity 1000                    (P3: does the null shift?)
   7  apod n=10 + cavity 1100                    (P3: null-shift bracket)
   8  apod n=10 + full stack                     (P3 full-local transfer)
   9  apod n=10 + 1050 + pair[+20,+20]           (P3 SHIFT-module transfer)
  10  apod n=20 + cavity 1050
  11  jitter partner: apod n=10 + cavity 1052    (in-study accurate floor)
NOTE apodization tapers the innermost n_apod teeth's depth while the pair
shifts the innermost 2 gaps — the builder applies per-tooth arrays VERBATIM
for d <= len and the apod envelope beyond (verified, no guard conflict).

Track BOTH loss and fwhm_m per row -> deliverable is the Pareto scatter
(matlab_plotting/plot_pareto_stack_vs_apod.m, written at fetch time).

Tags: apod rows carry _A{n}; cavity/see-saw rows carry _W{..}/_ptw{..} — all
11 distinct (control has neither).

Dispatch (queue must be EMPTY of other --option3 arrays):
    smoke: bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_pareto_stack_vs_apod --array-tasks=0
    full:  bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_pareto_stack_vs_apod
Output -> results/tm_pareto_stack_vs_apod/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.sweeps._tm_base import build_base


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
_SEESAW = [1040.0, 980.0]             # frontier-best see-saw (job 118214)
_PAIR = [20.0, 20.0]                  # gap-shift pair (the main new module)

_cw, _apm, _apn, _ptwn, _ptww, _ishl, _nfree = [], [], [], [], [], [], []


def row(w=None, apod=0, ptwn=None, ptww=None, ishl=None, nfree=1):
    _cw.append(w)
    _apm.append("linear" if apod else "none")
    _apn.append(apod)
    _ptwn.append(ptwn)
    _ptww.append(ptww)
    _ishl.append(ishl)
    _nfree.append(nfree)


row()                                             # 0  control
row(w=1050.0, ptwn=_NARROW, ptww=_SEESAW,
    ishl=_PAIR, nfree=2)                          # 1  the stack (0.0545)
for n in (5, 10, 20):                             # 2-4 apod ladder
    row(apod=n)
row(w=1050.0, apod=10)                            # 5  apod + cavity null
row(w=1000.0, apod=10)                            # 6  null-shift probe
row(w=1100.0, apod=10)                            # 7  null-shift bracket
row(w=1050.0, apod=10, ptwn=_NARROW, ptww=_SEESAW,
    ishl=_PAIR, nfree=2)                          # 8  apod + full stack
row(w=1050.0, apod=10, ishl=_PAIR, nfree=2)       # 9  apod + shift pair only
row(w=1050.0, apod=20)                            # 10 stronger apod + null
row(w=1052.0, apod=10)                            # 11 jitter partner

_n = 12
assert all(len(v) == _n for v in (_cw, _apm, _apn, _ptwn, _ptww, _ishl, _nfree))

SPEC = SweepSpec(
    cavity_width_nm            = _cw,
    apod_method                = _apm,
    n_apod_periods_each_side   = _apn,
    width_narrow_per_tooth_nm  = _ptwn,
    width_wide_per_tooth_nm    = _ptww,
    inner_shift_list_nm        = _ishl,
    n_free_inner_teeth         = _nfree,
    simulation_mode            = ["accurate"] * _n,
    mode  = "zipped",
    label = "tm_pareto_stack_vs_apod",
)


if __name__ == "__main__":
    print(SPEC.describe())
    run_sweep_spec(SPEC, target="local", base=BASE)
