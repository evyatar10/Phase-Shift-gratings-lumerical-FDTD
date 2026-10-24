"""OFF-LATTICE sidewall tabs = the second-teeth antenna with its PHASE set by position in the period.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why: the sub-teeth rows A-F (farfield_sph_15um_subteeth) were snapped onto the 514-nm tooth lattice, so they were
fine-grained tooth reshaping (R 0.971, inside the inverse-design space), not an antenna; the lever that made the TM comb
work -- the feature's position inside the period -- was never varied. Here the same tab set S is drawn at two
REGISTRATIONS, r0 and r0 + half a period, on (1) the Nt80x +20 nm contrast parent (the best lab device, far field
MEASURED today) and (2) the plain TE pi-shift device of this study (pure pi-shift cusp, cheap, 150 ps steady state).
Tab set S (per arm, same |x| on both arms, 7 sites = 6 gaps): gap i = lambda / (n_eff + n_clad (0.85 - 0.05 i)), i = 0..5,
the first-order scattering period that sends the +-beta waves to |u_x| = 0.85 .. 0.60 (578 .. 663 nm Nt80x, 560 .. 641 nm
plain), from x0 = 0.6 um; 100-nm SiN rects, protrusion d, both sidewalls, full height. Registration = SIGNED global
translation x -> x + delta (the phase formula's convention), which commutes with the clamp on the uniform half grid. Off-lattice, but the builder
draws ONE rect per site at the local half-width, so a footprint must stay on one flat wall: the offset from the nearest
half-period centre is clamped to +-(pitch/4 - 55 nm).
Reading (scratch solve_v4.py): b_r = row - parent per stored energy; r = P/P_parent; bounded nonnegative 2-row solve on
{b_r0, b_rh}; harmonic tables; half-depth row = linearity gate of r0 (||b5 - 2 b2.5||/||b5|| <= 0.25). Pursue if r <= 0.90 at
either registration or the solve predicts <= 0.85 (above the noise floor); otherwise THIS basis fails at these depths
(not the family). Plain rows: 10 nm (its leak per stored energy is far larger) + a 5-nm r0 row for linearity.
Reviewed by GPT (session 01a15a66, 2026-10-20): chirp arithmetic, signed translation, plain = mechanism check only.
Device selection: env FF15_PHASE_DEV = NT80X (default) | PLAIN (wrapper farfield_sph_15um_phase_plain).
Dispatch (Athena):
  plain : bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_phase_plain --max-concurrent=2
  Nt80x rows 0-2 (265802): SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_phase \
            --array-tasks=0-2 --max-concurrent=3 --qos=4d_1g --gpu=a100   (rows 3-4 = 265969: --array-tasks=3-4; never re-run 0-2)
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEV = os.environ.get("FF15_PHASE_DEV", "NT80X")
os.environ["FF15_ROUND"] = "C_NT80" if DEV == "NT80X" else "A"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS, BASE as BASE_FF, N_SIDE, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

N_CLAD, U_AIM, DU, X0_NM, N_SITES = 1.444, 0.85, 0.05, 600.0, 7
X0_MIRROR_NM = 195.0                           # leak's own mirror plane (Fable review a373bcb3, verified)
SITE_LEN_NM, SITE_Y_SPAN_NM = 100.0, 60.0
seed = ROWS[0]
POL, PITCH_NM, CORR_NM, AVG_NM, CAV_W_NM, TEETH, LAM_NM, WIN_NM, BOX_Y, BOX_Z = seed
if DEV == "NT80X":                                    # parent = the +20 nm contrast device (farfield_sph_15um_contrast)
    from runners.sweeps.farfield_sph_15um_contrast import contrast_teeth  # noqa: E402
    TEETH = contrast_teeth((0, 1), 20.0)
    # rows 3-4 added 2026-10-21 (x-parity, user): the same sets on the RIGHT arm only. With rows 0-1 (both arms) they give each
    # arm's response separately (b_left = b_both - b_right, linear at 5 nm), so the solve can weight the arms independently and
    # reach the leak's x-even (43 %) and x-odd (57 %) parts separately. APPENDED: indices 0-2 (265802) must not move.
    VARIANTS = [("r0_d5", 0.0, 5.0), ("rh_d5", PITCH_NM / 2, 5.0), ("r0_d2p5", 0.0, 2.5), ("r0_d5_R", 0.0, 5.0), ("rh_d5_R", PITCH_NM / 2, 5.0)]
    # rows 5-6 added 2026-10-21 03:40 (PARITY-MATCHED, after 265802): the leak is 87 % x-EVEN about its own plane x0 = +0.195 um, while
    # the sets mirrored about x = 0 radiate 76-91 % x-ODD (measured). Here the left tabs are the mirror images of the right ones about
    # X0_MIRROR_NM (clamped to one wall, <= 22.4 nm moves), so the pair radiates into the even sector. 
    # GPT (01a15a66) + Fable review: run r0_M at 2.5 and 1.25 nm (same-depth comparison with the stored r0 2.5 nm row); no depth
    # extrapolation (the r0 overlap changed sign between 2.5 and 5 nm). 4 of 7 image pairs land on the opposite wall type, so the
    # pair is a position mirror only (odd residual expected).
    VARIANTS += [("r0_M_d2p5", 0.0, 2.5), ("r0_M_d1p25", 0.0, 1.25)]
else:
    VARIANTS = [("r0_d10", 0.0, 10.0), ("rh_d10", PITCH_NM / 2, 10.0), ("r0_d5", 0.0, 5.0)]
NAR = TEETH[0] if TEETH else [AVG_NM - CORR_NM / 2] * N_SIDE
WID = TEETH[1] if TEETH else [AVG_NM + CORR_NM / 2] * N_SIDE
CAV_W = CAV_W_NM if CAV_W_NM else AVG_NM
N_EFF = LAM_NM / (2 * PITCH_NM)                       # Bragg condition
GAPS_NM = [LAM_NM / (N_EFF + N_CLAD * (U_AIM - DU * i)) for i in range(N_SITES - 1)]
CLAMP_NM = PITCH_NM / 4 - SITE_LEN_NM / 2 - 5.0


def half_width_nm(x_nm):
    """Local core half-width: right arm draws narrow then wide within each period, left arm wide then narrow (bragg_device)."""
    s = abs(x_nm) - PITCH_NM / 4
    if s < 0:
        return CAV_W / 2
    j = int(s // PITCH_NM); frac = (s - j * PITCH_NM) / PITCH_NM
    narrow_first = x_nm > 0
    return ((NAR[j] if frac < 0.5 else WID[j]) if narrow_first else (WID[j] if frac < 0.5 else NAR[j])) / 2


def on_one_wall(x_nm):
    """Nearest half-period centre and the clamped site centre (footprint stays on one flat wall)."""
    sgn = 1 if x_nm >= 0 else -1
    s = abs(x_nm) - PITCH_NM / 4
    c = sgn * (PITCH_NM / 4 + (round((s - PITCH_NM / 4) / (PITCH_NM / 2)) + 0.5) * PITCH_NM / 2)
    return c, c + max(-CLAMP_NM, min(CLAMP_NM, x_nm - c))


def sites(reg_shift_nm, d_nm, arms=(+1, -1)):
    """(x list nm, y list nm): chirped set S, shifted by reg_shift along +x, clamped to one wall; y = wall + d - span/2."""
    xs_arm = [X0_NM + sum(GAPS_NM[:k]) for k in range(N_SITES)]
    raw = [] if arms == "mirror_x0" else [sgn * x + reg_shift_nm for sgn in arms for x in xs_arm]
    if arms == "mirror_x0":                        # right arm + its mirror images about X0_MIRROR_NM on the left arm
        right = [on_one_wall(x + reg_shift_nm)[1] for x in xs_arm]
        raw = right + [2 * X0_MIRROR_NM - x for x in right]
    xs = [on_one_wall(x)[1] for x in raw]
    assert len(set(round(v, 1) for v in xs)) == len(xs), "duplicate sites"
    return xs, [half_width_nm(x) + d_nm - SITE_Y_SPAN_NM / 2 for x in xs]


BASE = copy.deepcopy(BASE_FF)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "rect"
BASE.scatterer.mirrored_y = True
XL = [sites(sh, d, (+1,) if name.endswith("_R") else "mirror_x0" if "_M_" in name else (+1, -1)) for name, sh, d in VARIANTS]
n = len(VARIANTS)
SPEC = SweepSpec(
    polarization              = [POL] * n,
    pitch_nm                  = [PITCH_NM] * n,
    corrugation_depth_nm      = [CORR_NM] * n,
    avg_width_nm              = [AVG_NM] * n,
    cavity_width_nm           = [CAV_W_NM] * n,
    width_narrow_per_tooth_nm = [None if TEETH is None else TEETH[0]] * n,
    width_wide_per_tooth_nm   = [None if TEETH is None else TEETH[1]] * n,
    n_periods_each_side       = [N_SIDE] * n,
    center_wavelength_nm      = [LAM_NM] * n,
    scan_width_nm             = [WIN_NM] * n,
    y_span_um                 = [BOX_Y] * n,
    span_mult                 = [zmult(BOX_Z, LAM_NM)] * n,
    scatterer_shape           = ["rect"] * n,
    scatterer_x_span_um       = [SITE_LEN_NM * 1e-3] * n,
    scatterer_y_span_nm       = [SITE_Y_SPAN_NM] * n,
    scatterer_index           = [1.97] * n,
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [xl[0] for xl in XL],
    scatterer_y_list_nm       = [xl[1] for xl in XL],
    mode  = "zipped",
    label = "farfield_sph_15um_phase" + ("" if DEV == "NT80X" else "_plain"),
)

if __name__ == "__main__":
    print(f"{DEV}: pitch {PITCH_NM} lam {LAM_NM} n_eff {N_EFF:.4f} gaps {[round(g) for g in GAPS_NM]} nm clamp +-{CLAMP_NM:.1f} nm | teeth[0:2] nar {NAR[:2]} wid {WID[:2]}")
    if DEV == "NT80X":
        assert abs(NAR[0] - 940.3) < 1e-6 and abs(WID[0] - 960.3) < 1e-6, "parent is not the +20 nm contrast device"
    for (name, sh, d), (xs_v, _) in zip(VARIANTS, XL):   # property asserts (CLAUDE.md 5, incident 2026-10-21)
        right = [x for x in xs_v if x > 0]; left = [x for x in xs_v if x < 0]
        if name.endswith("_R"):
            assert not left and len(right) == N_SITES, name
        elif "_M_" in name:
            assert all(abs(xl - (2 * X0_MIRROR_NM - xr)) <= CLAMP_NM for xl, xr in zip(left, right)), ("not mirrored about x0", name)
        else:
            assert all(abs(xl + xr - 2 * sh) <= 2 * CLAMP_NM for xl, xr in zip(left, right)), ("not mirrored about the shift", name)
    for (name, sh, d), cfg in zip(VARIANTS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        xs = [v * 1e9 for v in kw["scatterer_x_list_m"]]; ys = [v * 1e9 for v in kw["scatterer_y_list_m"]]
        prot = [y + SITE_Y_SPAN_NM / 2 - half_width_nm(x) for x, y in zip(xs, ys)]
        for x, y in zip(xs, ys):                      # the whole 100-nm footprint sits on one flat wall
            for e in (-SITE_LEN_NM / 2, SITE_LEN_NM / 2):
                assert abs(half_width_nm(x + e) - half_width_nm(x)) < 1e-6, ("footprint straddles a step", name, x)
        desc = " ".join(f"{x:+.0f}{'N' if (x > 0) == (((abs(x) - PITCH_NM / 4) % PITCH_NM) < PITCH_NM / 2) else 'W'}{x - on_one_wall(x)[0]:+.0f}" for x in xs[:N_SITES])
        if "_M_" in name:
            desc += " | left: " + " ".join(f"{x:+.0f}" for x in xs[N_SITES:])
        print(f"{name:8s}: {len(xs)} sites, d {min(prot):.1f}-{max(prot):.1f} nm | right arm x(half,offset): {desc}")
        print(f"          sim ps {os.environ.get('TM_SIM_TIME_PS')} short {cfg.farfield.short_run} window {cfg.spectral.center_wavelength_m*1e9:.3f}+-{cfg.spectral.scan_width_nm/2} "
              f"ff2d {getattr(cfg.monitors, 'monitor_2d_center_nm', None)} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} rect {kw['scatterer_x_span_m']*1e9:.0f}x{kw['scatterer_y_span_m']*1e9:.0f}")
