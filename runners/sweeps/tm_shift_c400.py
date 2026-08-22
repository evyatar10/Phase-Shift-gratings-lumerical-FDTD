"""Innermost-tooth shift on the ANCHORED corr-400 TM device — WHICH SEGMENT SHRINKS?

Study dir: runners/sweeps/   |   Created 2026-08-12, wide-target half added 2026-08-19
Jobs: narrow-target half = (2026-08-12 dispatch); wide-target half = (fill at dispatch)

The shift removes length from ONE segment of the innermost tooth and gives it to
the cavity (total grating length preserved). Until 2026-08-19 the builder could
only shorten the NARROW segment; `shift_target="wide"` shortens the wide one
instead. Both leave the local period at pitch - s, so the pair differs ONLY in
which side of duty-cycle 0.5 the tooth lands on:

    narrow-target :  narrow = HP - s,  wide = HP      -> wide fraction > 0.5
    wide-target   :  narrow = HP,      wide = HP - s  -> wide fraction < 0.5   (HP = pitch/2)

This is NOT the same as a negative shift, which lengthens the narrow segment and
so moves the PERIOD the other way (2HP + s) — that confounds duty cycle with
period. Negative shifts are already measured and are uniformly bad for TM
(results_from_athena/distributed_shift_study, 5 rows: T 0.8864 -> 0.838-0.861,
mode 15.53 -> 16.1-19.9 um, lambda falling monotonically).

QUESTION THIS DECIDES: at identical s and identical period, does it matter which
segment is shortened?
  * T(wide) ~ T(narrow) -> the lever is duty-cycle-MAGNITUDE driven; nothing new.
  * T(wide) <  T(narrow) -> confirms the <n_eff> / light-cone mechanism: only the
    direction that RAISES the wide fraction (hence <n_eff>, hence the
    n_eff - n_clad light-cone margin that limits TM radiation) helps.
  * T(wide) >  T(narrow), or less mode widening per unit T -> the whole shift
    axis was explored on the wrong side and the campaign basis must be reopened.

PREDICTION ON RECORD (EXPECTED, CLAUDE.md §9): wide-target is the worse half for
TM. In the stored narrow-target ladder lambda rises monotonically with s
(1558.617 -> 1559.27), i.e. <n_eff> goes UP; shortening the wide segment must
move it DOWN and shrink the light-cone margin instead.

MEASURED narrow-target half (reused, NOT re-run — CLAUDE.md §6):
  results_from_athena/tm_shift_c400/results/  +  s=0 anchor from
  results_from_athena/asym_dw_study/results/result_N80_TM_avg_Ybox6p8_Zbox8p8.mat
    s (nm)     lambda        T_res      Q_L     mode (um)
      0.00    1558.617      0.8864    1311.4      15.532
     51.68    1558.946      0.9038    1333.5      15.707
    103.37    1559.186      0.9179    1342.3      16.260
    155.05    1559.266      0.9279    1347.6      16.983
    206.73    1559.196      0.9335    1350.5      16.947

ROWS: s = 51.68 / 103.37 nm (20 % / 40 % of the half-pitch) at shift_target
"wide" — the exact mirrors of the first two stored rungs. Two rows is enough to
settle a sign question whose stored counterpart is monotone; if the pair splits
the way the prediction says, the axis closes without the 60/80 % rungs.

NO CONTROL ROW (CLAUDE.md §6): the s=0 anchor above is at these EXACT numerics
(build_ports_base: box y 6.8 um / z-mult 5.42, window 30 nm / 3001 pts, mesh
optimization, ports-only, scatterer OFF) and is shift-target-independent.

FILE TAGS: wide-target rows carry an extra "w" (_S52w / _S103w) — without it they
would collide with the stored _S52 / _S103 narrow rows and clobber them.

Physics line (CLAUDE.md §4): TM h350, pitch 516.83, corr 400, W800, n 1.97/1.444;
target resonance ~1558.6 nm, window 30 nm centered 1558.5 (3001 pts, ~10 pm) —
the stored ladder spans only 0.65 nm, so the window is ample either way.

Dispatch (2 tasks, ~30 min each; default ARRAY_QOS=24h_1g is ample. Do NOT pass
ARRAY_TIME= — athena.conf plain-assigns it after sourcing, so it is silently
ignored; change walltime via the conf knobs and verify with sacct):
    bash athena/deploy_athena.sh \
        --option3 --spec=runners.sweeps.tm_shift_c400 --max-concurrent=4
Output -> results/tm_shift_c400/results/  (tags _S52w / _S103w).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

HALF_PITCH_NM = 516.83 / 2.0
SHIFT_PCT     = [20, 40]                  # of the half-pitch; mirrors the stored rungs
SHIFTS_NM     = [round(p / 100.0 * HALF_PITCH_NM, 2) for p in SHIFT_PCT]

BASE = _common.build_ports_base()   # scatterer-program numerics verbatim (y 6.8 / z 8.8 um)

assert max(SHIFTS_NM) < HALF_PITCH_NM, "shift must stay below half-pitch (wide segment would vanish)"
assert len({round(s) for s in SHIFTS_NM}) == len(SHIFTS_NM), "rows must be tag-unique"

SPEC = SweepSpec(
    innermost_tooth_shift_nm = SHIFTS_NM,
    shift_target             = ["wide"] * len(SHIFTS_NM),
    scatterer_radius_nm      = [0.0] * len(SHIFTS_NM),   # build_ports_base leaves the scatterer ON
    mode  = "zipped",
    label = "tm_shift_c400",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for pct, s in zip(SHIFT_PCT, SHIFTS_NM):
        print(f"  shift {s:6.2f} nm ({pct} % half-pitch) | narrow {HALF_PITCH_NM:6.2f} nm "
              f"| wide {HALF_PITCH_NM - s:6.2f} nm | period {2 * HALF_PITCH_NM - s:7.2f} nm "
              f"| wide frac {(HALF_PITCH_NM - s) / (2 * HALF_PITCH_NM - s):.4f}")
    print("stored narrow-target mirrors: +51.68 -> T 0.9038 / mode 15.707 um ; "
          "+103.37 -> T 0.9179 / mode 16.260 um ; s=0 -> T 0.8864 / mode 15.532 um")
