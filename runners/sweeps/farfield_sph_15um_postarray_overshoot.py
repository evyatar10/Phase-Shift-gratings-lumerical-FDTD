"""BORN-DESIGNED POST ARRAY on the TE OVERSHOOT (Nt80x): hundreds of mirrored SiN posts (full 350 nm height) beside both arms, sites from a model.

Study dir: runners/sweeps/ | Created 2026-10-23 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Why: the single comb (+12 um, 51 deg, r63) gave Q_i +7.5 % (production) / +8.3 % (dy29); one comb = one u_x channel and each comb radiates
TWO mirror lobes (standing wave), so x2 needs many independent channels. A Born dipole model (drive = control tube side-plane field back-
propagated to the post plane, cylinder polarisability, free-oxide dipole far field; no fitted parameter: |B|^2 within ~10 % on 8 FDTD comb
rows, pattern overlap ~0.6, measured scale 0.70 at dy29) gives a convex design problem (amplitude = (r/63)^2 >= 0). Discrete design with
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
ROWS = [(nm, DESIGNS[nm]) for nm in NAMES]

BASE = copy.deepcopy(BASE_FF)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "cylinder"
BASE.scatterer.mirrored_y = True
if PA_SET == "dy29":
    BASE.mesh.y_mesh_cells = 26
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
    y_span_um                 = [BOX_Y] * n,
    span_mult                 = [zmult(BOX_Z, LAM_NM)] * n,
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

if __name__ == "__main__":
    # file tags encode (count, first/last x, first/last y) only -> rows must differ there (TRAP 2026-10-23, 268992 clobber)
    _tags = [(len(xl), xl[0], xl[-1], yl[0], yl[-1]) for xl, yl in zip(SPEC.scatterer_x_list_nm, SPEC.scatterer_y_list_nm)]
    assert len(set(_tags)) == len(_tags), ('file-tag collision between rows', _tags)
    wall = max(max(TEETH[0]), max(TEETH[1]), CAV_W_NM) / 2
    for (name, d), cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        xb = [v * 1e9 for v in kw["scatterer_x_list_m"]]; yb = [v * 1e9 for v in kw["scatterer_y_list_m"]]; rb = [v * 1e9 for v in kw["scatterer_r_list_m"]]
        # property asserts: per-site lists carried through intact, radii 50-100, wall clearance >= 150 nm, in-row edge gap >= 99 nm, box/ff span
        assert len(xb) == len(d["x_nm"]) > 50 and all(abs(a - b) < 0.01 for a, b in zip(rb, d["r_nm"])), ("lists", name)
        assert 49.9 <= min(rb) and max(rb) <= (120.1 if PA_SET == "big" else 100.1), ("radius", name)
        assert min(y - r for y, r in zip(yb, rb)) - wall >= 150, ("wall clearance", name)
        assert max(y + r for y, r in zip(yb, rb)) <= cfg.y_span * 1e9 / 2 - 600, ("box edge", name)
        assert max(abs(x) for x in xb) + 100 <= cfg.farfield.farfield_x_span_m * 1e9 / 2, ("ff span", name)
        for yr in sorted(set(yb)):
            s = sorted((x, r) for x, y, r in zip(xb, yb, rb) if y == yr)
            assert all(b[0] - a[0] - a[1] - b[1] >= 99 for a, b in zip(s[:-1], s[1:])), ("in-row gap", name, yr)
        rows_txt = ", ".join(f"y{yr:.0f}:{sum(1 for y in yb if y == yr)}" for yr in sorted(set(yb)))
        print(f"{name}: {len(xb)} posts/side ({rows_txt}) r {min(rb):.0f}-{max(rb):.0f} x {min(xb)/1e3:.2f}..{max(xb)/1e3:.2f} um | wall {wall:.0f} "
              f"| Born P/Pc {d['born_P_over_Pc']} | short {cfg.farfield.short_run} ps {os.environ.get('TM_SIM_TIME_PS')} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} "
              f"dy {kw['width_narrow']*1e9/kw['y_mesh_cells']:.1f}")
