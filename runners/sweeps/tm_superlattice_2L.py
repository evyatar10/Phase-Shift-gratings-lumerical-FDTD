"""2-Lambda anti-phase superlattice — vertical-channel cancellation (phase-0 4.1c).

Study dir: runners/sweeps/   |   Created 2026-07-19   |   Job(s): TBD
Purpose: band-edge resonance (beta ~ pi/Lambda) => an every-other-PERIOD wide-
tooth width alternation W_wide +/- delta (superperiod 2*Lambda, mean preserved)
is a first-order VERTICAL out-coupler driven by the resonant envelope. Tune
(delta, which-tooth phase) to ANTI-PHASE the device's intrinsic vertical leak
and null that channel at the source. Theory gate PASS 2026-07-06
(docs/phase0_gate_verdicts_2026-07-06.md 4.1c): matching the vertical leak
needs amplitude ratio ~0.14 -> nm-scale delta; 2 real knobs vs 1 complex
target. W800 baseline first (user 2026-07-19); goal = beat the width-lever
family at zero mode-width cost.

REGISTERED PREDICTIONS (pre-dispatch): if the vertical leak is spatially
coherent, ONE phase reduces loss with an optimum at some delta* (ceiling ~
vertical share ~27% x loss 0.110 ~ +0.03 T) while the OPPOSITE phase worsens
it; if incoherent, BOTH phases worsen ~delta^2 and the route dies. Mean width
is preserved -> lambda_res stays ~1558.61 (program sidecar remains valid);
mode width should be UNTOUCHED (this is the low-profile requirement).

Rows (zipped, 10 tasks): ctrl | delta {2,5,12,30} x phase {A: +first, B: -first}
| jitter twin (delta 2.05, phase A) -> in-family floor.
Far-field ON: the TOP-monitor E2 is the direct vertical-channel readout.

Dispatch (queue MUST be empty — another chat runs nights; serialize, no
exceptions):
    ARRAY_TIME=02:00:00 bash athena/deploy_athena.sh --option3 \
        --spec=runners.sweeps.tm_superlattice_2L --max-concurrent=3
Output -> results/tm_superlattice_2L/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

N_TEETH   = 80
W_WIDE    = 1000.0     # nm (corr 400 around avg 800: narrow 600 / wide 1000)
W_NARROW  = 600.0
DELTAS_NM = [2.0, 5.0, 12.0, 30.0]

LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR   # W800 mean geometry unchanged

BASE = _common.build_ff_base()
BASE.scatterer.enabled = False                # no pillars in this study


def alternation(delta, first_sign):
    """80-long wide-tooth list [W+d, W-d, ...] (innermost first)."""
    return [round(W_WIDE + first_sign * delta * (1 if i % 2 == 0 else -1), 2)
            for i in range(N_TEETH)]


_ww, _wn = [None], [None]                     # row 0: control (uniform)
for d in DELTAS_NM:
    for sign in (+1, -1):                     # phase A (+first) / phase B (-first)
        _ww.append(alternation(d, sign))
        _wn.append([W_NARROW] * N_TEETH)
_ww.append(alternation(2.05, +1))             # jitter twin of (2.0, A)
_wn.append([W_NARROW] * N_TEETH)

# Tag uniqueness: ptw{n}W{first}to{last} -> first element differs per row.
_firsts = {None if w is None else w[0] for w in _ww}
assert len(_firsts) == len(_ww), "superlattice rows must differ in first tooth width"

SPEC = SweepSpec(
    width_wide_per_tooth_nm   = _ww,
    width_narrow_per_tooth_nm = _wn,
    mode  = "zipped",
    label = "tm_superlattice_2L",
)

if __name__ == "__main__":
    print(SPEC.describe()[:600])
    print(f"-> {len(_ww)} rows; deltas {DELTAS_NM} x phases A/B + ctrl + jitter")
