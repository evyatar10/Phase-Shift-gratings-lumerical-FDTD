"""BORN-DESIGNED POST ARRAY on the TE OVERSHOOT (Nt80x): hundreds of mirrored SiN posts (full 350 nm height) beside both arms, sites from a model.

Study dir: runners/sweeps/ | Created 2026-10-23 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Why: the single comb (+12 um, 51 deg, r63) gave Q_i +7.5 % (production) / +8.3 % (dy29); one comb = one u_x channel and each comb radiates
TWO mirror lobes (standing wave), so x2 needs many independent channels. A Born dipole model (drive = control tube side-plane field back-
propagated to the post plane, cylinder polarisability, free-oxide dipole far field; the model itself has no fitted parameter: |B|^2 within ~10 %
on 8 FDTD comb rows, pattern overlap ~0.6; the DESIGN uses an empirical response scale 0.70 measured at dy29) gives a convex design problem (amplitude = (r/63)^2 >= 0). Discrete design with
r 50-100 nm, edge gap >= 100 nm in a row, rows 300 nm apart, |x| <= 24 um: PREDICTED far-field leak P/Pc 0.693 (3 rows y 1.2/1.5/1.8 um,
238 posts/side) and 0.813 (2 rows y 1.5/1.8 um, 136 posts/side). The widest Nt80x teeth are 1800 nm (wall |y| 0.9 um at x ~10-30 um), so
rows must sit at y > 0.9 um (model drive needs pure oxide between the row and the y 2.1 um sampling plane; a y 0.9 row was discarded).
Designs in postarray_overshoot_designs.json (scratch comb/design.py).
Control = stored Nt80x T340 (Q_i 5.35e5, same numerics; no control row). Reading: solve_v4.py --short --energy; Q_i, T, W, P/Pc vs prediction.
Pre-registered: mechanism real if Q_i >= +20 % with W held (|dW| <= 0.1 um); model scale confirmed if measured P/Pc within +-0.1 of predicted.
Dispatch (Athena): SBATCH_MEM=250G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_postarray_overshoot \
                     --max-concurrent=2 --qos=4d_1g --gpu=a100
"""
import copy
import json

import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, N_SIDE, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

POL, PITCH_NM, CORR_NM, AVG_NM, CAV_W_NM, TEETH, LAM_NM, WIN_NM, BOX_Y, BOX_Z = FF_ROWS[0]
DESIGNS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "postarray_overshoot_designs.json")))
NAMES = ["pa3_r100", "pa2_r100"]
# variants (2026-10-23, user go "launch things on servers right now"), selected by env FF15_PA_SET via wrappers:
#   "big":  pa4_r120_x30 (rows 1.2-2.1, |x| <= 30 um, r 50-120, 319 posts/side, model 0.654) + pa5_r120_x30 (rows 1.2-2.4, 379/side, 0.646).
#           NOTE the far-field tube side face sits at y 2.104 um: posts at y 2.1 / 2.4 straddle or lie outside it, so P/Pc (far-field
#           projection) is INVALID for these two; Q_i (trace decay + port T) and the mode width are unaffected.
#   "dy29": pa3_r100 at y_mesh_cells 26 (dy 28.8 nm) vs the stored dy29 control 267605_0 (Q_i 5.67e5) = mesh check of the +63 %.
PA_SET = os.environ.get("FF15_PA_SET", "")
if PA_SET == "big":
    NAMES = ["pa4_r120_x30", "pa5_r120_x30"]
elif PA_SET == "dy29":
    NAMES = ["pa3_r100"]
#   "fine" / "finectrl" (GPT review 01a16842: the posts sit on the 50 nm background mesh): pa3_r100 and a NO-POST control at IDENTICAL
#           refined numerics: dx = pitch/20 = 25.7 nm (mode fine_x10, period-aligned) + a y-only 25 nm override over |y| 1.05-1.95 um.
elif PA_SET in ("fine", "finectrl"):
    NAMES = ["pa3_r100"]
#   "r7": 7 rows (y 1.2-2.7 um, 0.25 um pitch, |x| <= 30 um, r 50-120 / outer row <= 100, all-pairs gap >= 50 nm) in the LARGER box
#         8.8/8.8 um at dy 28.8 nm (y_mesh_cells 26) = identical numerics to the stored control 267605_1 (dy29_box8p8, Q_i 5.70e5): no control
#         re-run; posts 1.6 um from the PML (box 6.8 would leave 0.6 um). Tube side face moves out with the box (check in __main__).
elif PA_SET == "r7":
    NAMES = ["pa7_r120_x30"]
BIGBOX = PA_SET == "r7"
FINE = PA_SET in ("fine", "finectrl")
FINE_DY_M, FINE_BAND_M = 25e-9, (1.05e-6, 1.95e-6)
ROWS = [(nm, DESIGNS[nm]) for nm in NAMES]

BASE = copy.deepcopy(BASE_FF)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "cylinder"
BASE.scatterer.mirrored_y = True
if PA_SET in ("dy29", "r7"):
    BASE.mesh.y_mesh_cells = 26
if FINE:
    BASE.mesh.simulation_mode = "fine_x10"
    BASE.mesh.scatterer_mesh_dy_m = FINE_DY_M
    BASE.mesh.scatterer_mesh_y_band_m = FINE_BAND_M
if PA_SET == "finectrl":
    BASE.scatterer.enabled = False
n = len(ROWS)
SPEC = SweepSpec(
    polarization              = [POL] * n,
    pitch_nm                  = [PITCH_NM] * n,
    corrugation_depth_nm      = [CORR_NM] * n,
    avg_width_nm              = [AVG_NM] * n,
    cavity_width_nm           = [CAV_W_NM] * n,
    width_narrow_per_tooth_nm = [TEETH[0]] * n,
    width_wide_per_tooth_nm   = [TEETH[1]] * n,
    n_periods_each_side       = [N_SIDE] * n,
    center_wavelength_nm      = [LAM_NM] * n,
    scan_width_nm             = [WIN_NM] * n,
    y_span_um                 = [8.8 if BIGBOX else BOX_Y] * n,
    span_mult                 = [zmult(8.8 if BIGBOX else BOX_Z, LAM_NM)] * n,
    scatterer_shape           = ["cylinder"] * n,
    scatterer_radius_nm       = [75.0] * n,                      # shared default; per-site radii below win
    scatterer_height_nm       = [350.0] * n,
    scatterer_index           = [1.97] * n,
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [d["x_nm"] for _, d in ROWS],
    scatterer_y_list_nm       = [d["y_nm"] for _, d in ROWS],
    scatterer_r_list_nm       = [d["r_nm"] for _, d in ROWS],
    mode  = "zipped",
    label = "farfield_sph_15um_postarray_overshoot" + (f"_{PA_SET}" if PA_SET else ""),
)
if PA_SET == "finectrl":
    for f_ in ("scatterer_shape", "scatterer_radius_nm", "scatterer_height_nm", "scatterer_index", "scatterer_mirrored_y",
               "scatterer_x_list_nm", "scatterer_y_list_nm", "scatterer_r_list_nm"):
        setattr(SPEC, f_, None)

if __name__ == "__main__":
    # file tags encode (count, first/last x, first/last y) only -> rows must differ there (TRAP 2026-10-23, 268992 clobber)
    _tags = [(len(xl), xl[0], xl[-1], yl[0], yl[-1]) for xl, yl in zip(SPEC.scatterer_x_list_nm or [], SPEC.scatterer_y_list_nm or [])]
    assert len(set(_tags)) == len(_tags), ('file-tag collision between rows', _tags)
    if PA_SET == "finectrl":
        cfg = SPEC.expand(base=BASE)[0]; kw = cfg.to_device_kwargs()
        assert not cfg.scatterer.enabled and kw["cells_per_half_period"] == 10 and kw["scatterer_mesh_dy_m"] == FINE_DY_M and kw["scatterer_mesh_y_band_m"] == FINE_BAND_M, "finectrl numerics"
        print(f"finectrl: NO posts, dx {PITCH_NM/20:.2f} nm, band dy {FINE_DY_M*1e9:.0f} nm over |y| {FINE_BAND_M}, box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f}, short {cfg.farfield.short_run}")
        sys.exit(0)
    wall = max(max(TEETH[0]), max(TEETH[1]), CAV_W_NM) / 2
    for (name, d), cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        xb = [v * 1e9 for v in kw["scatterer_x_list_m"]]; yb = [v * 1e9 for v in kw["scatterer_y_list_m"]]; rb = [v * 1e9 for v in kw["scatterer_r_list_m"]]
        # property asserts: per-site lists carried through intact, radii 50-100, wall clearance >= 150 nm, in-row edge gap >= 99 nm, box/ff span
        assert len(xb) == len(d["x_nm"]) > 50 and all(abs(a - b) < 0.01 for a, b in zip(rb, d["r_nm"])), ("lists", name)
        assert 49.9 <= min(rb) and max(rb) <= (120.1 if PA_SET in ("big", "r7") else 100.1), ("radius", name)
        assert min(y - r for y, r in zip(yb, rb)) - wall >= 150, ("wall clearance", name)
        assert max(y + r for y, r in zip(yb, rb)) <= cfg.y_span * 1e9 / 2 - 600, ("box edge", name)
        assert max(abs(x) for x in xb) + 100 <= cfg.farfield.farfield_x_span_m * 1e9 / 2, ("ff span", name)
        for yr in sorted(set(yb)):
            s = sorted((x, r) for x, y, r in zip(xb, yb, rb) if y == yr)
            assert all(b[0] - a[0] - a[1] - b[1] >= (49 if BIGBOX else 99) for a, b in zip(s[:-1], s[1:])), ("in-row gap", name, yr)
        if BIGBOX:   # all-pairs gap (any two posts, any rows) >= 50 nm
            X, Y, Rr = np.array(xb), np.array(yb), np.array(rb)
            gap = np.hypot(X[:, None] - X[None], Y[:, None] - Y[None]) - Rr[:, None] - Rr[None]; np.fill_diagonal(gap, 1e9)
            assert gap.min() >= 49.5, ("all-pairs gap", name, gap.min())
        if FINE:
            assert kw["cells_per_half_period"] == 10 and kw["scatterer_mesh_dy_m"] == FINE_DY_M and kw["scatterer_mesh_y_band_m"] == FINE_BAND_M, ("fine numerics", name)
            assert FINE_BAND_M[0] <= min(y - r for y, r in zip(yb, rb)) * 1e-9 and max(y + r for y, r in zip(yb, rb)) * 1e-9 <= FINE_BAND_M[1], ("posts outside the fine band", name)
        rows_txt = ", ".join(f"y{yr:.0f}:{sum(1 for y in yb if y == yr)}" for yr in sorted(set(yb)))
        print(f"{name}: {len(xb)} posts/side ({rows_txt}) r {min(rb):.0f}-{max(rb):.0f} x {min(xb)/1e3:.2f}..{max(xb)/1e3:.2f} um | wall {wall:.0f} "
              f"| Born P/Pc {d['born_P_over_Pc']} | short {cfg.farfield.short_run} ps {os.environ.get('TM_SIM_TIME_PS')} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} "
              f"dy {kw['width_narrow']*1e9/kw['y_mesh_cells']:.1f}")
