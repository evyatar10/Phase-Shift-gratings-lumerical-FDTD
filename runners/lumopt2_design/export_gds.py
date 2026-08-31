"""Export a named design from best_designs.py to GDS (top view, SiN core layer).

Study: lumopt2 inverse design | 2026-08-28 | one-off deliverable generator.
Usage: python runners/lumopt2_design/export_gds.py  (writes <name>.gds beside it)

Geometry source of truth = the optimizer's own make_func (exact free-region
layout, 25 periods/side + cavity), extended with the uniform outer periods
(seed corr/avg, no shift — exactly how the surrogate device is built) and
2 um feed stubs. Comb posts drawn from the vector (frozen values). Layer 1 =
SiN core (height 350 nm, not represented in 2D GDS). Unit um, precision nm.
"""
import sys

sys.path.insert(0, r"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes")
import gdstk
import numpy as np

from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design.best_designs import BEST_T9636
from runners.lumopt2_design.campaign_v2_proj_best import SPEC

NAME = "BEST_T9636"
P = np.asarray(BEST_T9636, dtype=float)
UM = 1e6  # meters -> um

lib = gdstk.Library(unit=1e-6, precision=1e-9)
cell = lib.new_cell(NAME)

# ── free region + cavity: exact optimizer layout via make_func ──────────────
props = eng.make_func(SPEC)(P)
rects = {}
for key, val in props.items():
    if "::" not in key or "field_profile" in key:
        continue
    name, prop = key.split("::")
    rects.setdefault(name, {})[prop] = float(val)
for name, d in rects.items():
    if "x span" not in d or "y span" not in d:
        continue
    cx, sx, sy = d["x"] * UM, d["x span"] * UM, d["y span"] * UM
    cell.add(gdstk.rectangle((cx - sx / 2, -sy / 2), (cx + sx / 2, sy / 2),
                             layer=1))

# ── outer uniform periods (seed geometry, no shift), d = 26 .. n_side ───────
hp = eng.PITCH_NM / 2.0 / 1000.0                      # half pitch, um
w_n = (eng.AVG_W_NM - eng.CORR_NM / 2.0) / 1000.0     # 637.5 nm
w_w = (eng.AVG_W_NM + eng.CORR_NM / 2.0) / 1000.0     # 962.5 nm
x_out = (eng.PITCH_NM / 4.0 + eng.N_FREE * eng.PITCH_NM) / 1000.0
n_outer = SPEC.n_periods_side - eng.N_FREE
for k in range(n_outer):
    x0 = x_out + k * 2 * hp
    for sgn in (+1, -1):
        a, b = sorted((sgn * x0, sgn * (x0 + hp)))
        cell.add(gdstk.rectangle((a, -w_n / 2), (b, w_n / 2), layer=1))
        a, b = sorted((sgn * (x0 + hp), sgn * (x0 + 2 * hp)))
        cell.add(gdstk.rectangle((a, -w_w / 2), (b, w_w / 2), layer=1))
x_end = x_out + n_outer * 2 * hp
for sgn in (+1, -1):
    a, b = sorted((sgn * x_end, sgn * (x_end + 2.0)))
    cell.add(gdstk.rectangle((a, -0.4), (b, 0.4), layer=1))   # 2 um feed stub

# ── comb posts: two symmetric rows from the (frozen) vector values ──────────
r_nm, x_nm, d_nm = P[eng.SL_R], P[eng.SL_X], P[eng.I_DCOMB]
for r, x in zip(r_nm, x_nm):
    for sgn in (+1, -1):
        cell.add(gdstk.ellipse((x / 1000.0, sgn * d_nm / 1000.0), r / 1000.0,
                               layer=1, tolerance=1e-4))

out = rf"c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\{NAME}.gds"
lib.write_gds(out)
n_polys = len(cell.polygons)
print(f"WROTE {out}  ({n_polys} polygons, device span "
      f"{2 * (x_end + 2.0):.1f} x {2 * (d_nm / 1000.0 + 0.3):.1f} um)")
