"""PLAIN TE second tooth: one extra tooth in EVERY period (same period as the grating); sweep its POSITION in the period.

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see dispatch note in the session)
Why (user 2026-10-22): what a second tooth does to the far field, the harmonic projection and peak T of the plain TE
pi-shift grating (corr 316 / pitch 500 / W800 / N96 = round A of farfield_sph_15um). Step 1 = position only.
Tooth: ~half the first tooth -- 150 nm long (first tooth = the 250-nm wide half), protrusion corr/4 = 79 nm above the LOCAL
wall on both sidewalls (+-y). Drawn as 3 abutting 50-nm rect units, each at its own local wall + 79 nm, so a tooth that
crosses a step follows it. dx = pitch/10 = 50 nm with mesh lines at 25 + 50 k nm: every unit edge and every grating step
lies on a mesh line, so all positions see the same discretisation (no sub-cell jitter between rows).
Position: DELTA = tooth-centre offset (+x) from the NARROW-half centre of its own period, on BOTH arms (each arm's own
lattice: the builder starts the right arm narrow and the left arm wide next to the cavity, so the left narrow centres are
at -(500 + k P) and the right at 250 + k P). The composite unit cell is therefore one translational lattice with the same
half-period cavity insert as the plain device. DELTA 0 = narrow centre, 250 = wide centre, 125 = the N->W step (+x).
Units that would land inside the cavity (|x| < P/4) or beyond the grating end are dropped (only the innermost/outermost tooth).
EXPECTED (first-order sidewall Fourier + local FDE n_eff, scratch fde_w.py): narrow -> kappa x~0.6, lambda ~+10.7 nm;
wide -> kappa x~1.4, lambda ~+6.2 nm; step positions add a quadrature term (resonance can detune inside the gap).
Control (MEASURED, stored round-A result_N96_avg_C316_Ybox6p8_Zbox6p8_AS1em12_ff.mat): lambda 1558.51, Q 3224, W 15.29 um.
Timing (skill highq-short-run): Q_L here <= ~1.5e4 EXPECTED (kappa x1.4) -> tau = 2Q/omega <= 25 ps; ONE 100-ps run = >= 4 tau
(>= 2 tau even at Q 3e4): Q_L from the cavity trace, T x gamma^2 (ringdown_budget.py), far field 201 points over the 20-nm
window = 0.1 nm grid (<= half a linewidth at Q 1.5e4). Rows whose resonance is not interior are re-centred and re-run alone.
Numerics vs control: window 20 nm / 2001 pts (control 10 nm), 100 ps + short_run caps/traces (control 150 ps); mesh, box,
tube identical. dy unchanged (width_narrow unchanged). No control row.
Dispatch (Athena): bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.plain_te_second_tooth --max-concurrent=2 --gpu=a100
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "A"
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, N_SIDE, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

POL, PITCH_NM, CORR_NM, AVG_NM, _, _, _, _, BOX_Y, BOX_Z = FF_ROWS[0]
TOOTH_LEN_NM, UNIT_NM, PROT_NM = 150.0, 50.0, CORR_NM / 4
DELTAS_NM = [-200.0, -150.0, -100.0, -50.0, 0.0, 50.0, 100.0, 150.0, 200.0, 250.0]
LAM_NM, WIN_NM, SIM_PS, FF_POINTS = 1567.0, 20.0, 100.0, 201
END_NM = PITCH_NM / 4 + N_SIDE * PITCH_NM


def half_width_nm(x_nm):
    """Plain device half-width: cavity |x| < P/4 at avg; right arm narrow-then-wide per period, left arm wide-then-narrow."""
    s = abs(x_nm) - PITCH_NM / 4
    if s < 0:
        return AVG_NM / 2
    narrow = ((s % PITCH_NM) < PITCH_NM / 2) == (x_nm > 0)
    return (AVG_NM - CORR_NM / 2 if narrow else AVG_NM + CORR_NM / 2) / 2


def sites(delta_nm):
    """Unit centres (x, y nm). Rect y-span = 2*PROT centred on the local wall -> outer edge = wall + PROT."""
    xs = []
    n_units = int(round(TOOTH_LEN_NM / UNIT_NM))
    for j in range(N_SIDE):
        for c in (PITCH_NM / 4 + (j + 0.25) * PITCH_NM + delta_nm,     # right arm: narrow half first
                  -(PITCH_NM / 4 + (j + 0.75) * PITCH_NM) + delta_nm):  # left arm: wide half first (outward)
            for k in range(n_units):
                x = c + (k - (n_units - 1) / 2) * UNIT_NM
                if PITCH_NM / 4 < abs(x) < END_NM and (x > 0) == (c > 0):
                    xs.append(x)
    return xs, [half_width_nm(x) for x in xs]


BASE = copy.deepcopy(BASE_FF)
BASE.scatterer.enabled = True
BASE.scatterer.shape = "rect"
BASE.scatterer.mirrored_y = True
BASE.farfield.short_run = True            # caps + time traces (Q_L); traces are exported only with far field + short_run on
BASE.farfield.farfield_freq_points = FF_POINTS
os.environ["TM_SIM_TIME_PS"] = str(SIM_PS)
LAMS, WINS = [LAM_NM] * len(DELTAS_NM), [WIN_NM] * len(DELTAS_NM)
XY = [sites(d) for d in DELTAS_NM]
n = len(DELTAS_NM)
SPEC = SweepSpec(
    polarization              = [POL] * n,
    pitch_nm                  = [PITCH_NM] * n,
    corrugation_depth_nm      = [CORR_NM] * n,
    avg_width_nm              = [AVG_NM] * n,
    cavity_width_nm           = [None] * n,
    width_narrow_per_tooth_nm = [None] * n,
    width_wide_per_tooth_nm   = [None] * n,
    n_periods_each_side       = [N_SIDE] * n,
    center_wavelength_nm      = LAMS,
    scan_width_nm             = WINS,
    y_span_um                 = [BOX_Y] * n,
    span_mult                 = [zmult(BOX_Z, lam) for lam in LAMS],
    scatterer_shape           = ["rect"] * n,
    scatterer_x_span_um       = [UNIT_NM * 1e-3] * n,
    scatterer_y_span_nm       = [2 * PROT_NM] * n,
    scatterer_index           = [1.97] * n,
    scatterer_mirrored_y      = [True] * n,
    scatterer_x_list_nm       = [xy[0] for xy in XY],
    scatterer_y_list_nm       = [xy[1] for xy in XY],
    mode  = "zipped",
    label = "plain_te_second_tooth",
)

if __name__ == "__main__":
    assert FF_ROWS[0][4] is None and FF_ROWS[0][5] is None and PITCH_NM == 500.0 and CORR_NM == 316.0, "not the plain TE device"
    for d, cfg in zip(DELTAS_NM, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        assert cfg.farfield.enabled and cfg.farfield.short_run and kw.get("short_run"), ("no time traces -> no Q_L", d)
        xs = [v * 1e9 for v in kw["scatterer_x_list_m"]]; ys = [v * 1e9 for v in kw["scatterer_y_list_m"]]
        span, ux = kw["scatterer_y_span_m"] * 1e9, kw["scatterer_x_span_m"] * 1e9
        for arm in (1, -1):                          # each arm: tooth centres sit at DELTA from that arm's narrow-half centres
            ref = 250.0 if arm > 0 else -500.0          # this arm's narrow-half centre of period 0
            offs = [((x - d - ref) + PITCH_NM / 2) % PITCH_NM - PITCH_NM / 2 for x in xs if x * arm > 0]
            assert all(min(abs(o - u) for u in (-UNIT_NM, 0.0, UNIT_NM)) < 1e-6 for o in offs), ("unit not in a DELTA tooth", d, arm)
            assert half_width_nm(ref + arm * PITCH_NM) < AVG_NM / 2, ("reference is not a narrow centre", arm)
            assert len(offs) >= 3 * (N_SIDE - 1), ("teeth missing", d, arm, len(offs))
        for x, y in zip(xs, ys):                     # property asserts: unit on one flat wall, edges on mesh lines, outer edge
            hw = half_width_nm(x)
            assert abs(half_width_nm(x - ux / 2 + 1e-3) - hw) < 1e-6 and abs(half_width_nm(x + ux / 2 - 1e-3) - hw) < 1e-6, ("straddles", d, x)
            assert abs(((x - ux / 2 - 25.0) / 50.0) - round((x - ux / 2 - 25.0) / 50.0)) < 1e-9, ("unit edge off the mesh lines", d, x)
            assert abs(y + span / 2 - (hw + PROT_NM)) < 1e-6, ("wrong protrusion", d, x)
        right = sorted(x for x in xs if x > 0)
        walls = "".join("N" if half_width_nm(x) < AVG_NM / 2 else "W" for x in right[3:6])
        print(f"DELTA {d:+6.0f}: {len(xs)} units x2 (+-y) | 2nd right tooth units {[round(x) for x in right[3:6]]} on {walls} | "
              f"unit {ux:.0f}x{span:.0f} nm, edge = wall + {PROT_NM:.0f} | window {cfg.spectral.center_wavelength_m*1e9:.2f}+-"
              f"{cfg.spectral.scan_width_nm/2} | ps {os.environ.get('TM_SIM_TIME_PS')} ff {cfg.farfield.enabled} short {cfg.farfield.short_run}")
