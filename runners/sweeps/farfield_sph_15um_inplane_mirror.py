"""IN-PLANE MIRROR beside the cavity: planar SiN strips parallel to the waveguide (same layer, full height), gap scan.

Study dir: runners/sweeps/ | Created 2026-10-22 (overnight, autonomous) | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Why (user 2026-10-22: no envelope / tooth / cavity-size levers; planar only; clear FDTD by morning): stored far fields show the
overshoot (Nt80x) leak is 52 % grazing (|u_x| > 0.8), 57 % inside |u_z| < 0.3 (within +-17 deg of the chip plane) and 87 %
E_phi (horizontally polarised); the plain leak 28 % / 41 %. FDE: every planar element tried (centre slot, core-index drop,
composites) radiates 10-20x more per Bragg strength than the sidewall step, and in-plane dark supermodes give R >= 0.8 (C(D) <= 0.17)
-> the one planar class left that is neither envelope nor cavity is the ELECTROMAGNETIC ENVIRONMENT: a mirror in the plane whose
image source interferes with the in-plane part of the leak (Hardy-Welch-Streifer 1989: reflector PHASE sets the radiated power).
Mirror: 3 SiN strips per side (200 nm wide, 470 nm period = quarter-wave 198/270 nm at normal incidence), 24 um long centred
on the cavity, y-mirrored; inner edge = (max wall half-width over the strip length) + gap. Gap scan 200/400/600 nm (Nt80x walls reach 884 nm half-width; 600 keeps >= 0.6 um to the stored 6.8-um box edge) ~ a quarter
to half lateral wavelength = the image-phase scan. Box (6.8/6.81 um) and numerics identical to the stored controls -> NO control
rows (CLAUDE.md 6): plain control = farfield_sph_15um/results/result_N96_avg_C316_*_ff.mat; Nt80x control =
farfield_sph_15um_c_nt80/results/*T340ps_ff.mat. Device selected by env FF15_MIRROR_DEV = PLAIN | NT80X (original overshoot).
Reading: solve_v4.py (--energy; --short for NT80X) + Q_L from the trace decay; pass = radiation per stored energy <= 0.7 of the
control at any gap with lambda/W held (|dW| <= 2 %).
Dispatch (Athena):
  plain: FF15_MIRROR_DEV=PLAIN bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_inplane_mirror --max-concurrent=3 --gpu=a100
  nt80x: FF15_MIRROR_DEV=NT80X SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_inplane_mirror \
           --max-concurrent=3 --qos=4d_1g --gpu=a100
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEV = os.environ.get("FF15_MIRROR_DEV", "NT80X")
os.environ["FF15_ROUND"] = "C_NT80" if DEV == "NT80X" else "A"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, N_SIDE, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

STRIP_W_NM, STRIP_PERIOD_NM, N_STRIPS, STRIP_LEN_UM = 200.0, 470.0, 3, 24.0
GAPS_NM = [200.0, 400.0, 600.0]
POL, PITCH_NM, CORR_NM, AVG_NM, CAV_W_NM, TEETH, LAM_NM, WIN_NM, BOX_Y, BOX_Z = FF_ROWS[0]
NAR = TEETH[0] if TEETH else [AVG_NM - CORR_NM / 2] * N_SIDE
WID = TEETH[1] if TEETH else [AVG_NM + CORR_NM / 2] * N_SIDE
CAV_W = CAV_W_NM if CAV_W_NM else AVG_NM


def half_width_nm(x_nm):
    """Local core half-width (same rule as bragg_device: right arm narrow then wide per period, left arm wide then narrow)."""
    s = abs(x_nm) - PITCH_NM / 4
    if s < 0:
        return CAV_W / 2
    j = int(s // PITCH_NM); frac = (s - j * PITCH_NM) / PITCH_NM
    return ((NAR[j] if frac < 0.5 else WID[j]) if x_nm > 0 else (WID[j] if frac < 0.5 else NAR[j])) / 2


MAX_HALF_NM = max(half_width_nm(x) for x in range(-int(STRIP_LEN_UM * 500), int(STRIP_LEN_UM * 500) + 1, 5))
ROWS = [(f"mirror_g{int(g)}", [MAX_HALF_NM + g + STRIP_W_NM / 2 + k * STRIP_PERIOD_NM for k in range(N_STRIPS)], STRIP_W_NM, STRIP_LEN_UM)
        for g in GAPS_NM]          # (name, strip-centre y list nm, strip width nm, strip length um)
# rows 3-5 APPENDED 2026-10-22 (wave 2, GPT 01a1607e): at the gap G_STAR chosen from the plain read -- A: the same 3 strips but
# 48 um long at the SAME y (near-axial leak passes the 24-um ends; Nt80x walls reach 900 nm beyond 12 um -> 184-nm clearance there); B: ONE 1-um-wide strip (stronger transverse reflection, no inter-strip gaps);
# C: ONE 200-nm strip (do extra strips add parasitic channels?). Indices 0-2 (267135/267138) must not move.
G_STAR_NM = float(os.environ.get("FF15_MIRROR_GSTAR", "600"))   # 600: Nt80x far-field window +-0.25 nm (g400 shifted 0.31 nm, 267135)
ROWS += [(f"mirrorL48_g{int(G_STAR_NM)}", ROWS[GAPS_NM.index(G_STAR_NM)][1], STRIP_W_NM, 48.0),
         (f"slab1um_g{int(G_STAR_NM)}", [MAX_HALF_NM + G_STAR_NM + 500.0], 1000.0, STRIP_LEN_UM),
         (f"strip1_g{int(G_STAR_NM)}", [MAX_HALF_NM + G_STAR_NM + STRIP_W_NM / 2], STRIP_W_NM, STRIP_LEN_UM)]

BASE = copy.deepcopy(BASE_FF)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "rect"
BASE.scatterer.mirrored_y = True
n = len(ROWS)
SPEC = SweepSpec(
    polarization              = [POL] * n,
    pitch_nm                  = [PITCH_NM] * n,
    corrugation_depth_nm      = [CORR_NM] * n,
    avg_width_nm              = [AVG_NM] * n,
    cavity_width_nm           = [CAV_W_NM] * n,
    width_narrow_per_tooth_nm = [TEETH[0] if TEETH else None] * n,
    width_wide_per_tooth_nm   = [TEETH[1] if TEETH else None] * n,
    n_periods_each_side       = [N_SIDE] * n,
    center_wavelength_nm      = [LAM_NM] * n,
    scan_width_nm             = [WIN_NM] * n,
    y_span_um                 = [BOX_Y] * n,
    span_mult                 = [zmult(BOX_Z, LAM_NM)] * n,
    scatterer_shape           = ["rect"] * n,
    scatterer_x_span_um       = [r[3] for r in ROWS],
    scatterer_y_span_nm       = [r[2] for r in ROWS],
    scatterer_index           = [1.97] * n,
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [[0.0] * len(r[1]) for r in ROWS],
    scatterer_y_list_nm       = [r[1] for r in ROWS],
    mode  = "zipped",
    label = f"farfield_sph_15um_inplane_mirror_{DEV.lower()}",
)

if __name__ == "__main__":
    for (name, ys, sw, sl), cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        yb = [v * 1e9 for v in kw["scatterer_y_list_m"]]; xb = [v * 1e9 for v in kw["scatterer_x_list_m"]]
        g = float(name.split("_g")[1])
        # property asserts (CLAUDE.md 5): innermost strip clears the widest wall by exactly the gap, >= 0.6 um to the box edge,
        # strip width/length as named; rows 0-2 unchanged (dispatched)
        assert abs(min(yb) - sw / 2 - MAX_HALF_NM - g) < 1e-6, ("wrong gap", name)
        assert max(yb) + sw / 2 <= cfg.y_span * 1e9 / 2 - 600, ("strip too close to the box edge", name, cfg.y_span)
        assert all(abs(x) < 1e-9 for x in xb) and abs(kw["scatterer_x_span_m"] - sl * 1e-6) < 1e-12, ("strip x", name)
        assert abs(kw["scatterer_y_span_m"] - sw * 1e-9) < 1e-15, ("strip width", name)
        wall = max(half_width_nm(x) for x in range(-int(sl * 500), int(sl * 500) + 1, 5))   # widest wall over the strip length
        assert min(yb) - sw / 2 - wall >= 150.0, ("strip too close to the wall over its length", name, wall)
        if name.startswith("slab") or name.startswith("strip1"):
            assert len(yb) == 1, ("single strip expected", name)
        print(f"{DEV} {name}: strip centres y {[round(v, 1) for v in yb]} nm (mirrored), width {sw} nm, len {sl} um, max half-width "
              f"{MAX_HALF_NM:.1f} nm | short {cfg.farfield.short_run} sim ps {os.environ.get('TM_SIM_TIME_PS')} "
              f"box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} lam {cfg.spectral.center_wavelength_m*1e9:.3f}")
