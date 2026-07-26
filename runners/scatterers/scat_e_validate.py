"""Stage E — FDTD validation of the chosen scatterer COMBINATIONS.

Study dir: runners/scatterers/   |   Created 2026-07-12   |   Job(s): TBD
Purpose: build each candidate combination (list of x positions, all RADIUS_NM/Y0_NM
mirrored pairs) in ONE simulation and measure the real figure of merit — far-field
power, T, R, loss at resonance — against an identical-numerics r=0 control.
Compare with stage D's predictions via `solve_response_matrix.py validate`.

>>> USER STEP: paste the position lists printed by
>>>     python runners/scatterers/solve_response_matrix.py solve
>>> into CANDIDATE_ARRAYS below (and MIN_SPACING_NM from the stage-B gates output).

Dispatch (AFTER stage D; queue empty of other --option3 arrays):
    bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_e_validate
Output -> results/scat_e_validate/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR

# ── PASTE solve output here ────────────────────────────────────────────────────
# Each entry = one combination = list of (x_nm, y_nm) sites (each drawn as a
# mirrored +/-y pair, radius RADIUS_NM).
# ROUND 1 (job 121239, all COMPLETED): [135] 30.0%/dT+0.0203 | [0,270] 32.7%/+0.0227
#   | [0,270,5535] 32.9%/+0.0227 | comb [-3240,135,3510] 30.4%/+0.0201.
# ROUND 2 (job 121372, COMPLETED): off-grid interpolation test [0,229] — measured FF
#   reduction 33.4% (interpolation +0.7pp confirmed vs [0,270]'s 32.7%) but dT +0.0224
#   = tie with [0,270]'s +0.0227 at the jitter floor. Landscape top is FLAT in T;
#   final device stays [0,270]. Off-grid positions are legal: the builder takes
#   continuous coordinates; the 135-nm grid was only the MEASUREMENT set.
# ROUND 3 (job 121754, all COMPLETED): 2D-grid combinations from the full 293-column
#   joint solve (rows y=700..1510 all free, Euclidean >=270 pair rule). Predicted FF:
#   A +30.83  B +30.53  C +31.41  D +31.41  E +7.59  F +32.39 (%).
#   MEASURED dT vs control T=0.8862: A +0.0227 (round-1 reproduced exactly) |
#   B +0.0207 | C +0.0208 | D +0.0231 | E +0.0045 | F +0.0211.
#   VERDICT: D beats A by +0.0004 << 0.0018 jitter floor -> TIE; no 2D combo beats
#   the plain [0,270]@700 pair; F (best predicted FF) LOSES in T (proxy decoupling).
#   E (pure-2D, no row 700) measured +0.0045 ~= predicted ~+0.005: cross-row
#   interference is real and the linear model is quantitative — just small.
#   FINAL DEVICE stays [0,270] r=80 y=700. No accurate-mesh confirm triggered.
# ROUND 4 (job 121797, COMPLETED): inner-standoff test — [0,270] moved 700->650
#   (70 nm oxide gap; tooth tips at 500 = 0.5*width_wide; an earlier "tips at 600"
#   claim was WRONG). MEASURED: T 0.9127, dT +0.0265 vs ctrl 0.8862 — BEATS the
#   y=700 winner (+0.0227) by +0.0038 = 2.1x the 0.0018 floor. T-saturation does
#   NOT hold for standoff reduction (it did for site count / rows / off-grid x).
#   CANDIDATE result: gap spans ~1.5 dy cells (dy pinned ~46-50 nm in all modes);
#   dy=25 nm refinement confirm PARKED by user ("later").
# ROUND 5 (job 121802, COMPLETED): y=600 (20 nm gap) T 0.9158 dT +0.0296 |
#   y=580 (tangent to wide-tooth tips) T 0.9170 dT +0.0308, Q up 1336->1342,
#   lam drift +0.11 nm. Monotonic ladder 700/650/600/580 — strong-coupling
#   regime keeps paying; trend not yet peaked at contact.
# ROUND 6 (job 121816, COMPLETED): y=500 (20 nm above cavity) dT +0.0358 |
#   y=480 (edge touching the cavity body at 400) T 0.9239 dT +0.0377, Q 1347 —
#   monotonic all the way down; spatial mode width preserved to +2%.
# ROUND 7 (2026-07-15): MECHANISM DISCRIMINATION — recycling vs effective cavity
#   widening. Facts used: cavity spans |x|<~129 (length pitch/2), first tooth on
#   each side is WIDE (reaches y=500), first NARROW segment centered at x~517
#   (half-width 300). Rows (zipped, cavity_width_nm now swept too; W800 control
#   = avg-equivalence canary, must reproduce T=0.8862):
#     0 ctrl W800                       3 plain W1050 (TE-program optimum), no pillars
#     1 W800 + [(0,480),(516.8,380)]    4 W1050 + pair touching it (edge 525, y=605)
#       (narrow-touch: right pillar     5 W1050 + pair 100 nm above it (y=705)
#        moved to the narrow segment)
#     2 plain W960 (uniform widening equal to the bump height), no pillars
# ROUND 7 RESULTS (job 121830, all COMPLETED): ctrl W800 0.8862 (canary exact) |
#   plain W960 +0.0293 | plain W1050 +0.0354 (~94% of the pillar-descent best
#   +0.0377!) | W1050+pair touching (605) +0.0116 and +100nm (705) +0.0238 —
#   pillars HURT a widened cavity | X0to517 row: right pillar BURIED in R_wide_1
#   (useless as narrow-touch; useful accident = single (0,480) pillar +0.0304).
#   GEOMETRY TRUTH (asymmetric arms): wide tooth LEFT of cavity, NARROW section
#   RIGHT of cavity (x 129-388, sidewall y=300); first right wide tooth 388-646.
#   VERDICT: descent-ladder gain was mostly effective-cavity-widening physics.
# ROUND 8 (job 121843, COMPLETED): user-corrected narrow-touch — (0,480) touching
#   the cavity + (270,380) touching the NARROW section sidewall (y=300).
#   RESULT: T 0.9310, dT +0.0448 — BEST DEVICE TO DATE (beats pair@480 +0.0377
#   and plain W1050 +0.0354). Loss 0.0672 (−39%), Q 1352, mode FWHM 16.02µm
#   (+3.2%). Saved: memory project_narrow_touch_design.md + LOCAL fsp.
# ROUND 9 (job 121922, COMPLETED): narrow-touch transplanted onto W1050 =
#   [(0,605),(270,380)]. RESULT: T 0.9068 = −0.0148 vs plain W1050 (0.9216
#   fresh) — narrow-touch does NOT transfer; W800-only design.
# ROUND 10 (2026-07-15, user request): GIANT scatterers — r=400 (5x, ~25x
#   scattering amplitude; OUTSIDE the linear-response regime, auxiliary-
#   resonator territory) at x=0, mirrored pairs, three cladding gaps from the
#   cavity edge, on BOTH bases. Expectation (stated): C4 weights say amplitude
#   is unwanted on W1050; W800 r-ladder had optimum r~100. Ground-truth test.
#   Farthest row capped at 900 nm gap to keep > lambda/n_clad PML clearance.
_R5 = 400.0
ROWS = [
    # (cavity_width_nm, radius_nm, [(x, y), ...])
    (800.0,  0.0,  [(0.0, 700.0)]),
    (1050.0, 0.0,  [(0.0, 700.0)]),
    (800.0,  _R5,  [(0.0, 1000.0)]),    # gap 200 from W800 cavity edge 400
    (800.0,  _R5,  [(0.0, 1300.0)]),    # gap 500
    (800.0,  _R5,  [(0.0, 1700.0)]),    # gap 900
    (1050.0, _R5,  [(0.0, 1125.0)]),    # gap 200 from W1050 cavity edge 525
    (1050.0, _R5,  [(0.0, 1425.0)]),    # gap 500
    (1050.0, _R5,  [(0.0, 1925.0)]),    # gap 900 (edge 2325, PML clearance 1075)
]
MIN_SPACING_NM = 270.0        # Euclidean center-to-center (measured dimer gate)
# ───────────────────────────────────────────────────────────────────────────────

_half_span_nm = _common.SPAN_UM * 1000.0 / 2.0
for _, r, arr in ROWS:
    arr.sort()
    assert all(abs(x) <= _half_span_nm + 1.0 for x, _ in arr), \
        f"x outside the measured grid span (+/-{_half_span_nm} nm): {arr}"
    assert all(y + max(r, _common.RADIUS_NM) < 3400.0 - 800.0 for _, y in arr), \
        f"site too close to the transverse PML: {arr}"
    if r > 0:
        for i, (xa, ya) in enumerate(arr):
            for xb, yb in arr[i + 1:]:
                d = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
                assert d >= MIN_SPACING_NM - 1.0, \
                    f"pair closer than {MIN_SPACING_NM} nm Euclidean: {arr}"

_cavw    = [w for w, _, _ in ROWS]
_rs      = [r for _, r, _ in ROWS]
_rows_x  = [[x for x, _ in arr] for _, _, arr in ROWS]
_rows_yl = [[y for _, y in arr] for _, _, arr in ROWS]

# File-tag collision guard (tag encodes W, r, N sites, first/last x/y): every
# row's key must be unique or two array tasks would share output names (§6).
_keys = {(w, r, len(xs), round(xs[0], 1), round(xs[-1], 1), round(ys[0], 1), round(ys[-1], 1))
         for w, r, xs, ys in zip(_cavw, _rs, _rows_x, _rows_yl)}
assert len(_keys) == len(_rs), \
    "stage-E rows must differ in (W, r, N, x/y endpoints) — rename/trim a candidate"

BASE = _common.build_ff_base()

SPEC = SweepSpec(
    cavity_width_nm     = _cavw,
    scatterer_radius_nm = _rs,
    scatterer_x_list_nm = _rows_x,
    scatterer_y_list_nm = _rows_yl,
    mode  = "zipped",
    label = "scat_e_validate",
)

if __name__ == "__main__":
    print(SPEC.describe())
