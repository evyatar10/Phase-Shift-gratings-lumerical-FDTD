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
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, N_SIDE, zmult, pad, NT80X_NARROW_NM, NT80X_WIDE_NM  # noqa: E402
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
elif PA_SET in ("fine3", "fine3ctrl"):        # CONVERGED-MESH check of the best 2-D device v21_seedfy (user 10-25) + matched control; fine2 numerics but
    NAMES = ["v21_seedfy"]                    # windows RE-CENTRED to 1615.3 nm (the 25 nm mesh moved lambda -4.2 nm: 270754/755 lost T + far field) and the
    LAM_NM = 1615.3                           # fine y band widened to 0.65-2.05 um for the v21 posts (y-r >= 0.70, y+r <= 2.0); ctrl = same scene, posts off
elif PA_SET in ("fine2", "fine2ctrl"):        # REFINED-MESH re-baseline of the best device (2026-10-25, mesh probe 270533): control + as-drawn + radius-mapped
    NAMES = ["nw_r2pack", "nw_r2pack_map"] if PA_SET == "fine2" else ["nw_r2pack"]   # ctrl: one placeholder row, posts switched off
elif PA_SET in ("fine", "finectrl"):
    NAMES = ["pa3_r100"]
#   "c29": an array DESIGNED on the converged dy29 control (rows 1.2/1.5/1.8, |x| <= 30, r <= 100, gaps >= 100; calibrated prediction x1.50)
#          run at dy29 (y_mesh_cells 26, box 6.8) = identical to the stored control 267605_0 (Q_i 5.67e5); pa3 there measured x1.33.
elif PA_SET == "c29":
    NAMES = ["c29_pa3_r100_x30"]
#   "c29r12": the best ONE-row (y 1.2, 36 posts, calibrated x1.30) and TWO-row (1.2/1.5, 90 posts, x1.46) designs on the same converged
#             control, run at dy29 (user 2026-10-23: is there meaning to 3 rows vs a better 1 row?). Control 267605_0.
elif PA_SET == "c29r12":
    NAMES = ["c29_r1_x30", "c29_r2_x30"]
#   "r7": 7 rows (y 1.2-2.7 um, 0.25 um pitch, |x| <= 30 um, r 50-120 / outer row <= 100, all-pairs gap >= 50 nm) in the LARGER box
#         8.8/8.8 um at dy 28.8 nm (y_mesh_cells 26) = identical numerics to the stored control 267605_1 (dy29_box8p8, Q_i 5.70e5): no control
#         re-run; posts 1.6 um from the PML (box 6.8 would leave 0.6 um). Tube side face moves out with the box (check in __main__).
elif PA_SET == "r7":
    NAMES = ["pa7_r120_x30"]
#   "w1" (2026-10-23 night, wave 1 of the 2-D / measured-residual program; dy29 box 6.8 = stored control 267605_0): the free 2-D
#        Born design (504 posts, y 1.1-2.0 um, |x| <= 30, Fable 2-D agent) and a DWBA-1 design (python_tools/radcancel/postarray_dwba*.py:
#        target = c29's MEASURED far field with the model's estimate of c29's own posts removed). Both record the z=0 in-situ drive map.
#   "nfctrl": the bare Nt80x at the same numerics + the same drive map (fields are not stored for 267605_0 -> this is NOT a re-measure;
#        its Q_i/T must reproduce 267605_0 exactly = free check that the extra monitor changes nothing).
elif PA_SET == "w1":
    NAMES = ["twod_x30_y2000"]
elif PA_SET == "w1b":                       # the DWBA-1 design, dispatched after its review (same numerics + drive map as w1)
    NAMES = ["dwba1_c29"]
elif PA_SET == "nfctrl":
    NAMES = ["c29_pa3_r100_x30"]            # placeholder row; scatterers are switched off below
elif PA_SET == "w2a":                       # single factor vs c29_r2_x30 (x1.43): same rows 1.2/1.5, packed by postarray_place2d (211 vs 90 posts)
    NAMES = ["r2pack_x30"]
elif PA_SET == "w2b":                       # free 2-D design under Fable's predictive model (2026-10-24)
    NAMES = ["fm2d_y19"]
#   "w3" (2026-10-24): NEAR-WALL rows driven by the DIRECT in-situ map of 269582 (postarray_nearwall.py; edge >= 250 nm from the LOCAL
#        tooth wall, r <= 80, |x| <= 10 um): nw_cal = standalone calibration row; nw_r2pack = r2pack_x30 (measured x1.65) + 9 near-wall posts
#        designed on r2pack's MEASURED far field and in-situ map (kept posts unchanged = bounded Gauss-Newton step). Far-field window widened to +-0.75 nm
#        (269581 moved lambda +0.41 nm out of +-0.25 nm); Q_i/T/width are unaffected by the monitor window.
elif PA_SET == "w3":
    NAMES = ["nw_cal", "nw_r2pack"]
elif PA_SET == "w3b":                       # single factor vs r2pack (x1.65): a third packed row at y 1.8 (333 posts)
    NAMES = ["r3pack_x30"]
elif PA_SET == "w3c":                       # Fable x2 candidate: r2pack + 9 near-cavity posts + DWBA row y 1.8 (pre-reg x1.96, 1.91-2.02)
    NAMES = ["nw9_row18"]
#   "m3db" / "m3dbctrl" (winner validation, 2026-10-24): the winner FF15_WIN (design name) and the bare device at N = FF15_N periods/side
#        (teeth padded with the bulk 750/1250 nm exactly as runners/sweeps/farfield_sph_15um_m3db_overshoot.py) and run FF15_PS ps
#        (~2 amplitude decay times at the expected -3 dB Q_L), dy29 box 6.8, drive map off, far-field window +-0.75 nm.
elif PA_SET == "w4":                        # Fable wave 4 on nw_r2pack (x1.89): bounded resize + 44 posts (pre-reg x2.00) and resize only (x1.97)
    NAMES = ["w4_rs50", "w4_rsonly50"]
elif PA_SET == "w5":                        # Fable wave 5: w4 resize + 88 posts with r >= 35 nm (pre-reg x2.04, 1.96-2.06; r < 50 = mesh risk)
    NAMES = ["w5_r35freers"]
elif PA_SET == "w6e":                       # element-shape test on nw_r2pack: near-cavity circles -> equal-area ellipses, 2 orientations
    NAMES = ["ell_nw_r0", "ell_nw_r90"]
elif PA_SET == "w6":                        # Fable 'beyond' designs on nw_r2pack: resize + 83 posts r 35-59 (pre-reg Q_i x2.23) / r >= 50 twin (x2.15)
    NAMES = ["w6_r35", "w6_r50"]
elif PA_SET == "kn":                        # KNOB wave on nw_r2pack (x1.89): FDTD-measured slopes of 4 global knobs (resize model failed 269918)
    NAMES = ["k_rowR_m10", "k_rowR_p10s", "k_rows_p50", "k_nearR60"]
elif PA_SET in ("x20", "x15", "kn40"):       # exploration probes on nw_r2pack: truncation to |x| <= 20 / 15 um (compactness), near-cavity r 40
    NAMES = {"x20": ["nwr2_x20"], "x15": ["nwr2_x15"], "kn40": ["k_nearR40"]}[PA_SET]
elif PA_SET == "ext":                       # nw_r2pack + rows extended to abs(x) <= 42 um (89 add-on posts; model x1.06 over nw_r2pack)
    NAMES = ["nw_r2pack_ext42"]
elif PA_SET == "x42":                       # add-ons on the MEASURED nw_r2pack_ext42 (x2.08, 270593): near-wall row 0.95 + rows 1.2/1.5 to 49.5 um (pre-reg x2.16) / rows 0.90+1.05 (x2.14)
    NAMES = ["x42_j3", "x42_r0910"]
elif PA_SET == "uni":                       # UNIFORM-array control for ext42 (same rows/area/extent, pitch 500 nm, uniform r; model x0.95) + half-cell x shift of ext42 (noise floor)
    NAMES = ["uni_sub500", "ext42_hx"]
elif PA_SET == "c2d":                       # FRESH full-footprint free 2-D cloud (2-D gate test; pre-reg x1.6-1.9, kill < 1.65)
    NAMES = ["c2d_full"]
elif PA_SET == "xs":                        # x-shift knobs on ext42 (Fable route B): global -25.7, right rows +12, left rows -12, near-cavity +25.7
    NAMES = ["ext42_xm26", "ext42_R12", "ext42_L12m", "ext42_nc26"]
elif PA_SET == "v21":                       # model v2.1 designs (2-D): ext42 x with free y (pre-reg x2.44) + free cloud from scratch (x2.14, 2-D gate test)
    NAMES = ["v21_seedfy", "v21_scratch"]
elif PA_SET == "v21b":                      # model v2.1 designs round b: rows redesign, clouds y<=1.5 / y<=1.7, conservative-fab cloud (150/150)
    NAMES = ["v21b_rows", "v21b_y1500", "v21b_y1700", "v21b_acons"]
elif PA_SET == "rep":                       # NOISE FLOOR: identical repeat of nw_r2pack (reversed list order only), other GPU type
    NAMES = ["nw_r2pack_rep"]
elif PA_SET == "w8":                        # Fable fresh COMPACT free 2-D design (42 posts, |x|<=19 um, 9 layers; pre-reg Q_i x1.87, 1.73-2.01)
    NAMES = ["w8_c2d"]
elif PA_SET in ("m3db", "m3dbctrl"):
    NAMES = [os.environ["FF15_WIN"]]
    N_SIDE = int(os.environ["FF15_N"]); TEETH = pad(NT80X_NARROW_NM, NT80X_WIDE_NM, N_SIDE)
    os.environ["TM_SIM_TIME_PS"] = os.environ["FF15_PS"]
NFMAP = PA_SET in ("w1", "w1b", "w2a", "w2b", "w3", "w3b", "w3c", "w4", "w5", "w6e", "w6", "kn", "x20", "x15", "kn40", "w8", "rep", "ext", "x42", "uni", "c2d", "xs", "v21", "v21b", "fine2", "fine2ctrl", "fine3", "fine3ctrl", "nfctrl")          # z=0 XY drive map: |x| <= 36 um, y 0..2.3 um, 21 points in the far-field window
BIGBOX = PA_SET == "r7"
FINE = PA_SET in ("fine", "finectrl", "fine2", "fine2ctrl", "fine3", "fine3ctrl")
FINE_DY_M, FINE_BAND_M = 25e-9, ((0.70e-6, 1.70e-6) if PA_SET in ("fine2", "fine2ctrl") else (0.65e-6, 2.05e-6) if PA_SET in ("fine3", "fine3ctrl") else (1.05e-6, 1.95e-6))
ROWS = [(nm, DESIGNS[nm]) for nm in NAMES]

BASE = copy.deepcopy(BASE_FF)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "cylinder"
BASE.scatterer.mirrored_y = True
if PA_SET in ("dy29", "r7", "fine", "finectrl", "c29", "c29r12", "w1", "w1b", "w2a", "w2b", "w3", "w3b", "w3c", "w4", "w5", "w6e", "w6", "kn", "x20", "x15", "kn40", "w8", "rep", "ext", "x42", "uni", "c2d", "xs", "v21", "v21b", "fine2", "fine2ctrl", "fine3", "fine3ctrl", "nfctrl", "m3db", "m3dbctrl"):   # fine/finectrl also refine the core (GPT 01a1691c: otherwise only x + post band)
    BASE.mesh.y_mesh_cells = 26
if FINE:
    BASE.mesh.simulation_mode = "fine_x10"
    BASE.mesh.scatterer_mesh_dy_m = FINE_DY_M
    BASE.mesh.scatterer_mesh_y_band_m = FINE_BAND_M
if PA_SET in ("finectrl", "nfctrl", "m3dbctrl", "fine2ctrl", "fine3ctrl"):
    BASE.scatterer.enabled = False
if PA_SET in ("w3", "w3b", "w3c", "w4", "w5", "w6e", "w6", "kn", "x20", "x15", "kn40", "w8", "rep", "ext", "x42", "uni", "c2d", "xs", "v21", "v21b", "fine2", "fine2ctrl", "fine3", "fine3ctrl", "m3db", "m3dbctrl"):
    BASE.monitors.monitor_2d_span_nm = 1.5
if PA_SET in ("fine3", "fine3ctrl"):
    BASE.monitors.monitor_2d_center_nm = LAM_NM          # far field / field profile / XY map windows follow the fine-mesh resonance
if NFMAP:
    BASE.monitors.record_2d_fields = True
    BASE.monitors.field_2d_x_span_m = 72e-6
    BASE.monitors.field_2d_xy_band_m = (0.0, 2.3e-6)
    BASE.spectral.n_2d_monitor_points = 21
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
    scatterer_r2_list_nm      = [d.get("r2_nm") for _, d in ROWS] if any("r2_nm" in d for _, d in ROWS) else None,
    scatterer_rot_list_deg    = [d.get("rot_deg") for _, d in ROWS] if any("rot_deg" in d for _, d in ROWS) else None,
    mode  = "zipped",
    label = "farfield_sph_15um_postarray_overshoot" + (f"_{PA_SET}" if PA_SET else ""),
)
if PA_SET in ("finectrl", "nfctrl", "m3dbctrl", "fine2ctrl", "fine3ctrl"):
    for f_ in ("scatterer_shape", "scatterer_radius_nm", "scatterer_height_nm", "scatterer_index", "scatterer_mirrored_y",
               "scatterer_x_list_nm", "scatterer_y_list_nm", "scatterer_r_list_nm"):
        setattr(SPEC, f_, None)

if __name__ == "__main__":
    # file tags encode (count, first/last x, first/last y) only -> rows must differ there (TRAP 2026-10-23, 268992 clobber)
    _rots = [next((a for a, v in zip(rl or [], r2 or []) if v is not None), None) for rl, r2 in
             zip(SPEC.scatterer_rot_list_deg or [None] * n, SPEC.scatterer_r2_list_nm or [None] * n)]   # sim_helpers adds _E<n>r<rot> for ellipses
    _tags = [(len(xl), xl[0], xl[-1], yl[0], yl[-1], ro) for xl, yl, ro in zip(SPEC.scatterer_x_list_nm or [], SPEC.scatterer_y_list_nm or [], _rots)]
    assert len(set(_tags)) == len(_tags), ('file-tag collision between rows', _tags)
    if PA_SET == "nfctrl":   # every kwarg identical to the c29 run (= stored control 267605_0 numerics) except posts and the drive map
        cfg = SPEC.expand(base=BASE)[0]; kw = cfg.to_device_kwargs()
        os.environ["FF15_PA_SET"] = "c29"; import importlib, runners.sweeps.farfield_sph_15um_postarray_overshoot as m29; importlib.reload(m29)
        kw29 = m29.SPEC.expand(base=m29.BASE)[0].to_device_kwargs()
        diff = {k for k in set(kw) | set(kw29) if not k.startswith("scatterer") and k not in ("record_2d_fields_top_and_cross", "field_2d_x_span_m", "field_2d_xy_band_m")
                and str(kw.get(k)) != str(kw29.get(k))}
        assert not cfg.scatterer.enabled and not diff, ("nfctrl numerics differ from c29", diff)
        print(f"nfctrl: NO posts, numerics == c29 run, drive map x {kw['field_2d_x_span_m']} band {kw['field_2d_xy_band_m']} pts {cfg.spectral.n_2d_monitor_points}")
        sys.exit(0)
    if PA_SET in ("m3db", "m3dbctrl"):   # length/time knobs reached the build: N periods, padded teeth, run time
        cfg = SPEC.expand(base=BASE)[0]; kw = cfg.to_device_kwargs(); nar = [v * 1e9 for v in kw["width_narrow_per_tooth_m"]]
        assert kw["n_periods_each_side"] == N_SIDE == len(nar) and all(abs(a - b) < 1e-6 for a, b in zip(nar[:81], NT80X_NARROW_NM)) and round(nar[-1], 3) == 750.0
        assert cfg.scatterer.enabled == (PA_SET == "m3db") and kw["y_mesh_cells"] == 26
        print(f"{PA_SET}: N {N_SIDE}, teeth {len(nar)}, run {os.environ['TM_SIM_TIME_PS']} ps, posts {len(kw.get('scatterer_x_list_m') or [])}")
        if PA_SET == "m3dbctrl": sys.exit(0)
    if PA_SET in ("finectrl", "fine2ctrl", "fine3ctrl"):
        cfg = SPEC.expand(base=BASE)[0]; kw = cfg.to_device_kwargs()
        assert not cfg.scatterer.enabled and kw["cells_per_half_period"] == 10 and kw["scatterer_mesh_dy_m"] == FINE_DY_M and kw["scatterer_mesh_y_band_m"] == FINE_BAND_M, "finectrl numerics"
        print(f"finectrl: NO posts, dx {PITCH_NM/20:.2f} nm, band dy {FINE_DY_M*1e9:.0f} nm over |y| {FINE_BAND_M}, box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f}, short {cfg.farfield.short_run}")
        sys.exit(0)
    wall = max(max(TEETH[0]), max(TEETH[1]), CAV_W_NM) / 2
    for (name, d), cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        xb = [v * 1e9 for v in kw["scatterer_x_list_m"]]; yb = [v * 1e9 for v in kw["scatterer_y_list_m"]]; rb = [v * 1e9 for v in kw["scatterer_r_list_m"]]
        if kw.get("scatterer_r2_list_m"):   # ellipses: check gaps/clearance with the LARGER semi-axis (conservative), echo the ellipse sites
            r2 = kw["scatterer_r2_list_m"]; rot = kw.get("scatterer_rot_list_deg") or [0] * len(rb)
            ell = [(round(x), round(y), round(r), round(v * 1e9, 2), a) for x, y, r, v, a in zip(xb, yb, rb, r2, rot) if v is not None]
            assert len(ell) == 9 and all(abs(np.pi * r * v - np.pi * 50 ** 2) < 30 for _, _, r, v, _ in ell), ("ellipse area", name)
            print(f"{name}: ellipses {ell}")
            rb = [max(r, (v or 0) * 1e9) for r, v in zip(rb, r2)]
        # property asserts: per-site lists carried through intact, radii 50-100, wall clearance >= 150 nm, in-row edge gap >= 99 nm, box/ff span
        assert len(xb) == len(d["x_nm"]) > 20 and all(abs(a - b) < 0.01 for a, b in zip(rb, d["r_nm"])), ("lists", name)
        assert (34.9 if PA_SET in ("w5", "w6", "kn40") else 44.9 if PA_SET == "fine2" else 49.9) <= min(rb) and max(rb) <= (120.1 if PA_SET in ("big", "r7") else 100.1), ("radius", name)
        if PA_SET in ("w1", "w1b", "w2a", "w2b", "w3", "w3b", "w3c", "w4", "w5", "w6e", "w6", "kn", "x20", "x15", "kn40", "w8", "rep", "ext", "x42", "uni", "c2d", "xs", "v21", "v21b", "fine2", "fine3", "m3db"):   # free 2-D designs: >= 100 nm from the LOCAL tooth wall (max half-width of every period within r + 100 nm in x)
            edges = CAV_W_NM / 2 + PITCH_NM * np.arange(len(TEETH[0]) + 1)
            for x_, y_, r_ in zip(xb, yb, rb):
                near = [max(TEETH[0][i], TEETH[1][i]) / 2 for i in range(len(TEETH[0])) if edges[i] - r_ - 100 <= abs(x_) <= edges[i + 1] + r_ + 100]
                assert y_ - r_ - max(near + [CAV_W_NM / 2]) >= 100 - 1e-6, ("local wall clearance", name, x_, y_, r_)
        else:
            assert min(y - r for y, r in zip(yb, rb)) - wall >= 150, ("wall clearance", name)
        assert max(y + r for y, r in zip(yb, rb)) <= cfg.y_span * 1e9 / 2 - 600, ("box edge", name)
        assert max(abs(x) for x in xb) + 100 <= cfg.farfield.farfield_x_span_m * 1e9 / 2, ("ff span", name)
        for yr in sorted(set(yb)):
            s = sorted((x, r) for x, y, r in zip(xb, yb, rb) if y == yr)
            assert all(b[0] - a[0] - a[1] - b[1] >= (49 if BIGBOX else 99) for a, b in zip(s[:-1], s[1:])), ("in-row gap", name, yr)
        if BIGBOX or PA_SET in ("w1", "w1b", "w2a", "w2b", "w3", "w3b", "w3c", "w4", "w5", "w6e", "w6", "kn", "x20", "x15", "kn40", "w8", "rep", "ext", "x42", "uni", "c2d", "xs", "v21", "v21b", "fine2", "fine3", "m3db"):   # all-pairs gap (any two posts, any rows) >= 50 nm (r7) / 100 nm (w1)
            X, Y, Rr = np.array(xb), np.array(yb), np.array(rb)
            gap = np.hypot(X[:, None] - X[None], Y[:, None] - Y[None]) - Rr[:, None] - Rr[None]; np.fill_diagonal(gap, 1e9)
            assert gap.min() >= (49.5 if BIGBOX else 99.5), ("all-pairs gap", name, gap.min())
        if FINE:
            assert kw["cells_per_half_period"] == 10 and kw["scatterer_mesh_dy_m"] == FINE_DY_M and kw["scatterer_mesh_y_band_m"] == FINE_BAND_M, ("fine numerics", name)
            assert FINE_BAND_M[0] <= min(y - r for y, r in zip(yb, rb)) * 1e-9 and max(y + r for y, r in zip(yb, rb)) * 1e-9 <= FINE_BAND_M[1], ("posts outside the fine band", name)
        rows_txt = ", ".join(f"y{yr:.0f}:{sum(1 for y in yb if y == yr)}" for yr in sorted(set(yb)))
        print(f"{name}: {len(xb)} posts/side ({rows_txt}) r {min(rb):.0f}-{max(rb):.0f} x {min(xb)/1e3:.2f}..{max(xb)/1e3:.2f} um | wall {wall:.0f} "
              f"| Born P/Pc {d['born_P_over_Pc']} | short {cfg.farfield.short_run} ps {os.environ.get('TM_SIM_TIME_PS')} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} "
              f"dy {kw['width_narrow']*1e9/kw['y_mesh_cells']:.1f}")
