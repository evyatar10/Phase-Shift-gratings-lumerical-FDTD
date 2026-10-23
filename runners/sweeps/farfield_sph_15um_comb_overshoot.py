"""COMB on the TE OVERSHOOT (Nt80x): a displaced row of SiN posts whose first order sends the guided wave into the grazing leak, phase circle.

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Why (user 2026-10-22: on the TM comb a strong NEGATIVE effect flipped to POSITIVE when the phase changed): the earlier verdict "comb is
the wrong tool for TE" was made on the PLAIN TE leak (broad lobes); the overshoot leak is grazing (52 % |u_x| > 0.8, 25 % > 0.95,
MEASURED from the stored Nt80x far field) like the TM needle the comb cancelled. Design by the TM rule (runners/scatterers/
scat_offcentre2.py): comb period LAMBDA_C so that beta - K_c lands at u_x ~ -0.96 (beta = pi/514 nm, k_ox at 1619.5 nm), posts far
from the wall (no near-field damage: the in-plane strips at <= 0.6 um RAISED loss, 267135/267138), comb displaced from the cavity
(TM: displacement removed the phase-independent pedestal), phase = 360 dx / LAMBDA_C from the comb centre. Three phases fix
P(phi) = A + B cos(phi - phi0) exactly; the minimum and the pedestal decide the next run (optimum + mirror comb = pair).
Control = stored Nt80x T340 far field (same box/numerics; no control row). Reading: solve_v4.py --short --energy + Q_L from the
trace decay; pass = Q_i up / radiation down beyond the per-row floor with lambda/W held.
Dispatch (Athena): SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_comb_overshoot \
                     --max-concurrent=3 --qos=4d_1g --gpu=a100
"""
import copy
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, N_SIDE, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

POL, PITCH_NM, CORR_NM, AVG_NM, CAV_W_NM, TEETH, LAM_NM, WIN_NM, BOX_Y, BOX_Z = FF_ROWS[0]
U_AIM = 0.96                                                    # grazing target |u_x|
BETA = math.pi / (PITCH_NM * 1e-3); K_OX = 2 * math.pi * 1.444 / (LAM_NM * 1e-3)
LAMBDA_C_NM = round(2 * math.pi / (BETA + K_OX * U_AIM) * 1e3, 1)   # beta - K_c = -k_ox u_aim
N_POSTS, R_NM, D_NM, XC_UM, HEIGHT_NM = 31, 80.0, 1600.0, 12.0, 350.0   # x_c +12 um (GPT 01a1607e: start past the strongest leak)
PHASES_DEG = [0.0, 120.0, 240.0]


def comb_x(n, xc_um, dx_nm):
    return [round(k * LAMBDA_C_NM + dx_nm + xc_um * 1e3, 1) for k in range(-(n - 1) // 2, (n - 1) // 2 + 1)]


ROWS = [(f"comb_xc{XC_UM:+.0f}_ph{int(p)}", comb_x(N_POSTS, XC_UM, p / 360.0 * LAMBDA_C_NM), R_NM) for p in PHASES_DEG]
# rows 3-4 APPENDED 2026-10-22 after 267523 (MEASURED Q_i 5.25e5 / 4.97e5 / 3.90e5 at 0/120/240 deg vs control 5.35e5; radiation 1.036 / 1.099
# / 1.396): exact 3-point fit R = 1.177 + 0.222 cos(phi - 230.6 deg) -> optimum phi 51 deg; pedestal ~0.18 scales as amplitude^2 and the
# swing ~0.22 as amplitude -> optimum amplitude 0.62x (r = 80 sqrt(0.62) = 63 nm, amplitude ~ r^2), predicted Q_i ~+7 %. Row 3 = that single
# comb; row 4 = the pair (+12 and -12 um, same phase convention; TM pair was ~additive), predicted ~+14 %. Indices 0-2 must not move.
PH_OPT, R_OPT = 51.0, 63.0
ROWS += [(f"comb_xc+12_ph51_r63", comb_x(N_POSTS, XC_UM, PH_OPT / 360.0 * LAMBDA_C_NM), R_OPT),
         (f"comb_pair12_ph51_r63", comb_x(N_POSTS, XC_UM, PH_OPT / 360.0 * LAMBDA_C_NM) + comb_x(N_POSTS, -XC_UM, PH_OPT / 360.0 * LAMBDA_C_NM), R_OPT)]
# rows 5-7 APPENDED 2026-10-22 19:30 after 267628 (MEASURED: single +12/51 deg/r63 Q_i 5.75e5 (+7.5 %), T 0.682; PAIR with the same phase at -12 um
# Q_i 5.29e5 (-1 %): the left comb at that phase cancels the gain). The device has no x mirror (both arms built in the same order), so the
# LEFT comb needs its own phase circle (GPT 01a16454): -12 um alone, r 63, 0/120/240 deg. Indices 0-4 must not move.
ROWS += [(f"combL_xc-12_ph{int(p)}_r63", comb_x(N_POSTS, -XC_UM, p / 360.0 * LAMBDA_C_NM), R_OPT) for p in PHASES_DEG]
# row 8 APPENDED 2026-10-23 after 268443 (MEASURED left circle Q_i 4.52e5 / 5.73e5 / 5.00e5 at 0/120/240 deg -> fit optimum 147 deg, ~+9 %):
# the INDEPENDENTLY phased pair, right +12 um at 51 deg + left -12 um at 147 deg, both r 63 (GPT 01a165bf: tests whether phased channels add;
# pre-registered prediction from the stored complex far fields c + B_R + B_L). Indices 0-7 must not move.
PH_L_OPT = 147.0
ROWS += [("comb_pairRL_ph51_147_r63", comb_x(N_POSTS, XC_UM, PH_OPT / 360.0 * LAMBDA_C_NM) + comb_x(N_POSTS, -XC_UM, PH_L_OPT / 360.0 * LAMBDA_C_NM), R_OPT)]
# VALIDATION variant (env FF15_COMB_DY29=1, wrapper farfield_sph_15um_comb_overshoot_dy29): the single winner (row 3) at the converged mesh
# dy 28.8 nm (y_mesh_cells 26), compared with the stored dy-29 control (267605 task 0, same box 6.8/6.81, same short-run numerics).
DY29 = os.environ.get("FF15_COMB_DY29") == "1"
if DY29:
    ROWS = [ROWS[3]]
# TWO-ROW variant (env FF15_COMB_2ROW=1, wrapper farfield_sph_15um_comb_overshoot_2row; user go 2026-10-22, GPT 01a16454 position): row 1 = the
# single winner unchanged (31 posts r 63, y +-1.6 um, Lambda_c 546.8, 51 deg, +12 um); row 2 = 29 posts r 90 at y +-1.9 um (147 nm gap to row 1),
# period aimed at |u_x| = 0.85 (the 0.8-0.95 band = 27 % of the leak), centred +12 um, phase circle 0/120/240 deg. Drive ~0.92 x row 1
# (29/31 (90/63)^2 x field ratio 0.48). Same dy 28.8 nm as the DY29 validation, so the reference is that run, not the dy-50 control.
TWOROW = os.environ.get("FF15_COMB_2ROW") == "1"
U2_AIM, N2_POSTS, R2_NM, D2_NM = 0.85, 29, 90.0, 1900.0
LAMBDA_C2_NM = round(2 * math.pi / (BETA + K_OX * U2_AIM) * 1e3, 1)
if TWOROW:
    row1 = comb_x(N_POSTS, XC_UM, PH_OPT / 360.0 * LAMBDA_C_NM)
    ROWS = []
    for p in PHASES_DEG:
        row2 = [round(k * LAMBDA_C2_NM + p / 360.0 * LAMBDA_C2_NM + XC_UM * 1e3, 1) for k in range(-(N2_POSTS - 1) // 2, (N2_POSTS - 1) // 2 + 1)]
        ROWS.append((f"comb2row_r1ph51_r2ph{int(p)}", row1 + row2, R_OPT,
                     [D_NM] * N_POSTS + [D2_NM] * N2_POSTS, [R_OPT] * N_POSTS + [R2_NM] * N2_POSTS))
# V3 variant (env FF15_COMB_V3=1, wrapper farfield_sph_15um_comb_overshoot_v3; user 2026-10-23 "one-line version" + "two rows on each side").
# After 268986 (MEASURED pair R51/L147: Q_i 6.25e5, +16.8 %). Designed with the Born dipole model on the stored Nt80x control field
# (scratch comb/combscan2.py; ranks 4/4 right, under-predicts gains ~3x): pair = P/Pc 0.958 in that model.
#   oneline: ONE lattice, period 539 nm (aims u_x ~0.99), offset 105 deg from x = 0, posts at 2.0 <= |x| <= 20.3 um (none over the cavity
#            centre; posts through the cavity score 0.965 = worse), r 63, y +-1.6 -> model 0.943.
#   pair2row: the measured pair + a 2nd row per arm: right y +-1.3 um, period 536.6 (u 0.999), r 50, 10 deg, centred +12 um; left y +-1.9 um,
#            period 541.5 (u 0.98), r 57, 280 deg, centred -12 um (31 posts each) -> model 0.940.
# Production mesh (same as the pair, control = stored Nt80x T340).
V3 = os.environ.get("FF15_COMB_V3") == "1"
L_ONE, OFF_ONE, GAP_ONE_NM = 539.0, 105.0, 2000.0
R2ND = dict(R=(1300.0, 536.6, 50.0, 10.0, 12.0), L=(1900.0, 541.5, 57.0, 280.0, -12.0))   # y, period, r, phase, centre um
def lattice(L, ph, xc_um, n):
    return [round(k * L + ph / 360.0 * L + xc_um * 1e3, 1) for k in range(-(n - 1) // 2, (n - 1) // 2 + 1)]
if V3:
    one = [x for x in lattice(L_ONE, OFF_ONE, 0.0, 121) if GAP_ONE_NM <= abs(x) <= 20300.0]
    pr = comb_x(N_POSTS, XC_UM, PH_OPT / 360.0 * LAMBDA_C_NM) + comb_x(N_POSTS, -XC_UM, PH_L_OPT / 360.0 * LAMBDA_C_NM)
    xs2, ys2, rs2 = list(pr), [D_NM] * len(pr), [R_OPT] * len(pr)
    for side in ("R", "L"):
        y2, L2, r2, ph2, xc2 = R2ND[side]; c2 = lattice(L2, ph2, xc2, N_POSTS)
        xs2 += c2; ys2 += [y2] * len(c2); rs2 += [r2] * len(c2)
    ROWS = [("comb_oneline_L539_off105_gap2", one, R_OPT, [D_NM] * len(one), [R_OPT] * len(one)),
            ("comb_pair2row_R13_L19", xs2, R_OPT, ys2, rs2)]

BASE = copy.deepcopy(BASE_FF)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "cylinder"
BASE.scatterer.mirrored_y = True
if DY29 or TWOROW:
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
    scatterer_radius_nm       = [r[2] for r in ROWS],
    scatterer_height_nm       = [HEIGHT_NM] * n,
    scatterer_index           = [1.97] * n,
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [r[1] for r in ROWS],
    scatterer_y_list_nm       = [(r[3] if (TWOROW or V3) else [D_NM] * len(r[1])) for r in ROWS],
    mode  = "zipped",
    label = "farfield_sph_15um_comb_overshoot" + ("_dy29" if DY29 else "") + ("_2row" if TWOROW else "") + ("_v3" if V3 else ""),
)
if TWOROW or V3:
    SPEC.scatterer_r_list_nm = [r[4] for r in ROWS]

if __name__ == "__main__":
    nar, wid = TEETH
    def half_w(x):
        s = abs(x) - PITCH_NM / 4
        if s < 0:
            return (CAV_W_NM or AVG_NM) / 2
        j = int(s // PITCH_NM); f = (s - j * PITCH_NM) / PITCH_NM
        return ((nar[j] if f < 0.5 else wid[j]) if x > 0 else (wid[j] if f < 0.5 else nar[j])) / 2
    # file tags encode (count, first/last x, first/last y) only -> rows must differ there (TRAP 2026-10-23, 268992 clobber)
    _tags = [(len(xl), xl[0], xl[-1], yl[0], yl[-1]) for xl, yl in zip(SPEC.scatterer_x_list_nm, SPEC.scatterer_y_list_nm)]
    assert len(set(_tags)) == len(_tags), ('file-tag collision between rows', _tags)
    print(f"beta {BETA:.4f}/um k_ox {K_OX:.4f}/um -> Lambda_c {LAMBDA_C_NM} nm (beta-K_c = {BETA - 2*math.pi/(LAMBDA_C_NM*1e-3):.3f} = -{U_AIM} k_ox)")
    for row, cfg in zip(ROWS, SPEC.expand(base=BASE)):
        name, xs, rr = row[:3]
        kw = cfg.to_device_kwargs()
        xb = [v * 1e9 for v in kw["scatterer_x_list_m"]]; yb = [v * 1e9 for v in kw["scatterer_y_list_m"]]
        wall = max(half_w(x) for x in range(int(min(xb)) - 200, int(max(xb)) + 201, 5))
        # property asserts: posts clear the widest wall by >= 500 nm, inside the box / far-field span, period, phase of EACH comb, radius
        assert V3 or min(yb) - rr - wall >= 500.0, ("posts too close to the wall", name, wall)   # V3 has its own (300 nm) check
        assert max(yb) + rr <= cfg.y_span * 1e9 / 2 - 600, ("posts too close to the box edge", name)
        assert max(abs(x) for x in xb) + rr <= cfg.farfield.farfield_x_span_m * 1e9 / 2, ("comb outside the far-field span", name)
        if V3:
            rb = [v * 1e9 for v in kw["scatterer_r_list_m"]]
            assert len(xb) == len(row[1]) and all(abs(a - b) < 0.01 for a, b in zip(rb, row[4])) and all(abs(a - b) < 0.01 for a, b in zip(yb, row[3])), ("lists", name)
            for yr in sorted(set(yb)):   # every row: in-row edge gap, wall clearance (widest tooth 900 nm), single period per (row, side)
                for sgn in (1, -1):
                    s_ = sorted((x, r) for x, y, r in zip(xb, yb, rb) if y == yr and x * sgn > 0)
                    assert all(b[0] - a[0] - a[1] - b[1] >= 99 for a, b in zip(s_[:-1], s_[1:])), ("gap", name, yr)
                    assert len({round(b[0] - a[0], 1) for a, b in zip(s_[:-1], s_[1:])}) <= 2, ("one period per row/side", name, yr)
            assert min(y - r for y, r in zip(yb, rb)) - 900 >= 300, ("wall", name)
            ys_ = sorted(set(yb)); assert all(b - a >= 250 for a, b in zip(ys_[:-1], ys_[1:])), ("row spacing", name)
            if "oneline" in name:
                d_ = sorted(xb); assert min(abs(x) for x in xb) >= 2000 and len({round(b - a, 1) for a, b in zip(d_[:-1], d_[1:]) if abs(a) > 1 and a * b > 0}) <= 2, ("lattice", name)
                ph0 = (min(x for x in xb if x > 0) % L_ONE) / L_ONE * 360; assert abs(ph0 - OFF_ONE) < 0.5, ("offset", name, ph0)
            print(f"{name}: {len(xb)} posts/side, rows {[(int(y), sum(1 for v in yb if v == y)) for y in ys_]}, r {min(rb):.0f}-{max(rb):.0f}, "
                  f"x {min(xb)/1e3:.2f}..{max(xb)/1e3:.2f} um, dy {kw['width_narrow']*1e9/kw['y_mesh_cells']:.1f}, short {cfg.farfield.short_run}")
            continue
        if TWOROW:
            rb = [v * 1e9 for v in kw["scatterer_r_list_m"]]
            c1, c2 = xb[:N_POSTS], xb[N_POSTS:]
            assert len(c2) == N2_POSTS and all(abs(a - b) < 0.01 for a, b in zip(c1, comb_x(N_POSTS, XC_UM, PH_OPT / 360.0 * LAMBDA_C_NM))),("row 1 must equal the single winner", name)
            assert all(abs(v - D_NM) < 0.01 and abs(q - R_OPT) < 0.01 for v, q in zip(yb[:N_POSTS], rb[:N_POSTS])), ("row 1 y/r", name)
            assert all(abs(v - D2_NM) < 0.01 and abs(q - R2_NM) < 0.01 for v, q in zip(yb[N_POSTS:], rb[N_POSTS:])), ("row 2 y/r", name)
            assert all(abs((b - a) - LAMBDA_C2_NM) < 0.2 for a, b in zip(c2[:-1], c2[1:])), ("row 2 period", name)
            ph2 = (c2[N2_POSTS // 2] - XC_UM * 1e3) / LAMBDA_C2_NM * 360
            assert abs(ph2 - float(name.split("_r2ph")[1])) < 0.5, ("row 2 phase", name, ph2)
            assert D2_NM - R2_NM - (D_NM + R_OPT) >= 140, ("row gap", name)
            assert kw["y_mesh_cells"] == 26, ("mesh dy29", name)
            print(f"{name}: row1 {len(c1)} r{R_OPT} y{D_NM:.0f} x {c1[0]:.1f}..{c1[-1]:.1f} | row2 {len(c2)} r{R2_NM} y{D2_NM:.0f} "
                  f"L {LAMBDA_C2_NM} ph {ph2:.1f} x {c2[0]:.1f}..{c2[-1]:.1f} | wall {wall:.0f} gap {D2_NM-R2_NM-D_NM-R_OPT:.0f} "
                  f"dy {kw['width_narrow']*1e9/kw['y_mesh_cells']:.2f} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} short {cfg.farfield.short_run}")
            continue
        combs = [xb[i:i + N_POSTS] for i in range(0, len(xb), N_POSTS)]
        assert len(xb) % N_POSTS == 0 and len(combs) == (2 if "pair" in name else 1), ("comb count", name)
        for c in combs:
            assert all(abs((b - a) - LAMBDA_C_NM) < 0.2 for a, b in zip(c[:-1], c[1:])), ("period", name)
            xc = 1e3 * (XC_UM if c[N_POSTS // 2] > 0 else -XC_UM)
            ph = (c[N_POSTS // 2] - xc) / LAMBDA_C_NM * 360
            want = (PH_OPT if xc > 0 else PH_L_OPT) if "pairRL" in name else float(name.split("_ph")[1].split("_")[0])
            assert abs(ph - want) < 0.5, ("phase", name, ph, want)
        assert kw["scatterer_shape"] == "cylinder" and abs(kw["scatterer_radius_m"] - rr * 1e-9) < 1e-15, ("post shape/radius", name)
        print(f"{name}: {len(combs)} comb(s) x {N_POSTS} posts r {rr} at y +-{yb[0]:.0f} nm (wall {wall:.0f}), x {min(xb):.0f}..{max(xb):.0f} nm | "
              f"short {cfg.farfield.short_run} sim ps {os.environ.get('TM_SIM_TIME_PS')} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f}")
