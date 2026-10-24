"""Nt80x OVERSHOOT + second tooth in every period: the plain_te_second_tooth position sweep, on Itai's Nt80 overshoot.

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see dispatch note in the session)
Why (user 2026-10-22): the plain sweep (plain_te_second_tooth, 267575) moved T/Q/width through the coupling strength;
test the same second tooth on the overshoot device (farfield_sph_15um round C_NT80: pitch 514, N96, per-period widths,
cavity 950.3 nm; stored control 265046 = farfield_sph_15um_c_nt80, 340 ps short run, Q_L 9.9e4).
Tooth (same definition as the plain sweep, scaled per period): 3 abutting rect units of pitch/10 = 51.4 nm (154.2 nm),
each protruding (wide_j - narrow_j)/4 above the LOCAL wall of the period j it sits in -- half of that period's own tooth
(the plain sweep's 79 nm = 316/4) -- so the apodization profile is scaled, not swamped. Rect y-span is one value per row,
so it is 2 * the largest protrusion and each unit is centred to put its outer edge at wall + prot_j (inner part buried
in the core). Position DELTA = k * pitch/10 from each arm's narrow-half centre (+x, both arms on their own lattice,
like the plain sweep); units in the cavity or past the grating end are dropped. Unit edges sit on the dx = pitch/10
mesh lines.
Steps (skill highq-short-run; Q can move far from 1e5):
  NT2_STEP=LOC  (this module): 60 ps, 30-nm window centred 1630 nm (EXPECTED shift +8..+14 nm from the added mean
                width), far field + short_run on (traces are exported only then) -> lambda and Q_L per row.
  NT2_STEP=PROD (wrapper nt80x_second_tooth_prod, after LOC): per-row lambda, 4-nm source, far field in its own
                +-0.25 nm window, run = max(60, 2 * 2Q_L/omega) ps per row. Control: stored 265046 (no control row).
Dispatch (Athena): SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.nt80x_second_tooth --max-concurrent=3 --gpu=a100
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STEP = os.environ.get("NT2_STEP", "LOC")
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, N_SIDE, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

POL, PITCH_NM, CORR_NM, AVG_NM, CAV_W_NM, TEETH, LAM0_NM, _, BOX_Y, BOX_Z = FF_ROWS[0]
NAR, WID = TEETH
UNIT_NM = PITCH_NM / 10
N_UNITS = 3
DELTAS_UNITS = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
DELTAS_NM = [k * UNIT_NM for k in DELTAS_UNITS]
PROT_NM = [(w - n) / 4 for n, w in zip(NAR, WID)]           # per period: half of the period's own tooth (per side)
SPAN_NM = 2 * max(PROT_NM)
END_NM = PITCH_NM / 4 + N_SIDE * PITCH_NM
LOC_LAM_NM, LOC_WIN_NM, LOC_PS = 1630.0, 30.0, 60.0
# PROD: {DELTA units: (lambda_res nm, Q_L)} from the LOC run -- filled after LOC, asserted non-trivial before dispatch
PROD_LAM_Q = {   # MEASURED, LOC 267814 (2026-10-22): lambda from the port finder; Q_L from the cavity-trace decay
    -4: (1623.021, 217902), -3: (1625.837, 112224), -2: (1629.082, 26698), -1: (1633.424, 4072),   # (fit from 30 ps,
    0: (1634.812, 1259), 1: (1633.274, 3969), 2: (1628.933, 26192), 3: (1625.763, 111002),          #  halves agree < 0.1 %;
    4: (1623.036, 219306), 5: (1623.230, 217561)}                                                    #  DELTA 0 = lam/FWHM, rung down)


def period_of(x_nm):
    """(period index j, narrow?) of a point on either arm; None inside the cavity."""
    s = abs(x_nm) - PITCH_NM / 4
    if s < 0:
        return None, None
    j = int(s // PITCH_NM)
    first_half = (s - j * PITCH_NM) < PITCH_NM / 2
    return j, first_half == (x_nm > 0)                       # right arm narrow first, left arm wide first


def half_width_nm(x_nm):
    j, narrow = period_of(x_nm)
    if j is None:
        return CAV_W_NM / 2
    return (NAR[j] if narrow else WID[j]) / 2


def sites(delta_nm):
    """Unit centres x and y (rect centre = wall + prot_j - SPAN/2, so the outer edge is wall + prot_j)."""
    xs = []
    for j in range(N_SIDE):
        for c in (PITCH_NM / 4 + (j + 0.25) * PITCH_NM + delta_nm,
                  -(PITCH_NM / 4 + (j + 0.75) * PITCH_NM) + delta_nm):
            for k in range(N_UNITS):
                x = c + (k - (N_UNITS - 1) / 2) * UNIT_NM
                if PITCH_NM / 4 < abs(x) < END_NM and (x > 0) == (c > 0) and PROT_NM[period_of(x)[0]] >= 0.5:
                    xs.append(x)                             # period 0 is flat (narrow = wide = cavity width): no tooth there
    return xs, [half_width_nm(x) + PROT_NM[period_of(x)[0]] - SPAN_NM / 2 for x in xs]


TASK = int(os.environ.get("SWEEP_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID") or 0)
BASE = copy.deepcopy(BASE_FF)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "rect"
BASE.scatterer.mirrored_y = True
BASE.farfield.short_run = True               # caps + time traces; traces are exported only with far field + short_run on
if STEP == "LOC":
    os.environ["TM_SIM_TIME_PS"] = str(LOC_PS)
    BASE.farfield.use_2d_window = False      # far field over the source window: LOC only needs the traces
    LAMS, WINS = [LOC_LAM_NM] * len(DELTAS_NM), [LOC_WIN_NM] * len(DELTAS_NM)
else:
    assert STEP == "PROD" and sorted(PROD_LAM_Q) == sorted(DELTAS_UNITS), "PROD needs the LOC lambda/Q_L of every row"
    assert all(1500 < l < 1700 and q > 100 for l, q in PROD_LAM_Q.values()), "LOC values look wrong"
    assert 0 <= TASK < len(DELTAS_UNITS), ("task index out of range", TASK)
    LAMS, WINS = [PROD_LAM_Q[d][0] for d in DELTAS_UNITS], [4.0] * len(DELTAS_UNITS)
    lam_t, q_t = PROD_LAM_Q[DELTAS_UNITS[TASK]]
    tau_ps = 2 * q_t / (2 * 3.14159265 * 299792458 / (lam_t * 1e-9)) * 1e12
    os.environ["TM_SIM_TIME_PS"] = str(round(max(60.0, 2 * tau_ps)))
    BASE.farfield.use_2d_window = True
    BASE.monitors.monitor_2d_center_nm = lam_t
    BASE.monitors.monitor_2d_span_nm = 0.5
XY = [sites(d) for d in DELTAS_NM]
n = len(DELTAS_NM)
SPEC = SweepSpec(
    polarization              = [POL] * n,
    pitch_nm                  = [PITCH_NM] * n,
    corrugation_depth_nm      = [CORR_NM] * n,
    avg_width_nm              = [AVG_NM] * n,
    cavity_width_nm           = [CAV_W_NM] * n,
    width_narrow_per_tooth_nm = [NAR] * n,
    width_wide_per_tooth_nm   = [WID] * n,
    n_periods_each_side       = [N_SIDE] * n,
    center_wavelength_nm      = LAMS,
    scan_width_nm             = WINS,
    y_span_um                 = [BOX_Y] * n,
    span_mult                 = [zmult(BOX_Z, lam) for lam in LAMS],
    scatterer_shape           = ["rect"] * n,
    scatterer_x_span_um       = [UNIT_NM * 1e-3] * n,
    scatterer_y_span_nm       = [SPAN_NM] * n,
    scatterer_index           = [1.97] * n,
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [xy[0] for xy in XY],
    scatterer_y_list_nm       = [xy[1] for xy in XY],
    mode  = "zipped",
    label = "nt80x_second_tooth" + ("" if STEP == "LOC" else "_prod"),
)

if __name__ == "__main__":
    assert abs(PITCH_NM - 514.0) < 1e-9 and abs(NAR[1] - 919.5) < 1e-9 and abs(WID[1] - 982.6) < 1e-9, "not the Nt80x device"
    print(f"Nt80x: pitch {PITCH_NM}, unit {UNIT_NM:.1f} nm, prot range {min(PROT_NM):.1f}..{max(PROT_NM):.1f} nm, span {SPAN_NM:.1f} nm")
    for d, cfg in zip(DELTAS_NM, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        assert cfg.farfield.enabled and cfg.farfield.short_run and kw.get("short_run"), ("no time traces -> no Q_L", d)
        xs = [v * 1e9 for v in kw["scatterer_x_list_m"]]; ys = [v * 1e9 for v in kw["scatterer_y_list_m"]]
        span, ux = kw["scatterer_y_span_m"] * 1e9, kw["scatterer_x_span_m"] * 1e9
        assert len(xs) >= 2 * N_UNITS * (N_SIDE - 2), ("teeth missing", d, len(xs))   # period 0 is flat
        for arm in (1, -1):                                  # units sit at DELTA from this arm's narrow-half centres
            ref = PITCH_NM / 2 if arm > 0 else -PITCH_NM            # narrow-half centre of period 0 (right 257, left -514)
            offs = [((x - d - ref) + PITCH_NM / 2) % PITCH_NM - PITCH_NM / 2 for x in xs if x * arm > 0]
            assert all(min(abs(o - u) for u in (-UNIT_NM, 0.0, UNIT_NM)) < 1e-6 for o in offs), ("unit not in a DELTA tooth", d, arm)
            assert period_of(ref + arm * PITCH_NM)[1], ("reference is not a narrow centre", arm)
        for x, y in zip(xs, ys):                             # one flat wall, edges on mesh lines, outer edge = wall + prot_j, core-buried
            hw, j = half_width_nm(x), period_of(x)[0]
            assert abs(half_width_nm(x - ux / 2 + 1e-3) - hw) < 1e-6 and abs(half_width_nm(x + ux / 2 - 1e-3) - hw) < 1e-6, ("straddles", d, x)
            edge = (x - ux / 2 - UNIT_NM / 2) / UNIT_NM
            assert abs(edge - round(edge)) < 1e-6, ("unit edge off the mesh lines", d, x)
            assert abs(y + span / 2 - (hw + PROT_NM[j])) < 1e-6 and y - span / 2 >= -hw - 1e-6, ("outer edge / inner edge outside the core", d, x)
        right = sorted(x for x in xs if x > 0)[3:6]; left = sorted((x for x in xs if x < 0), reverse=True)[3:6]
        tag = lambda u: "".join("N" if period_of(v)[1] else "W" for v in u)
        print(f"DELTA {d:+7.1f}: {len(xs)} units x2 | right {[round(v) for v in right]} {tag(right)} prot {[round(PROT_NM[period_of(v)[0]]) for v in right]}"
              f" | left {[round(v) for v in left]} {tag(left)} | {STEP} window {cfg.spectral.center_wavelength_m*1e9:.2f}+-{cfg.spectral.scan_width_nm/2}"
              f" ps {os.environ.get('TM_SIM_TIME_PS')} ff {cfg.farfield.enabled} short {cfg.farfield.short_run}")
