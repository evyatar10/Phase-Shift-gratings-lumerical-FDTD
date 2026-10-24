"""Mean-width RIPPLE probe on the 15.5-um Itai Nt80x overshoot (lab device): radiation response to one knob.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Purpose: same probe as farfield_sph_20um_ripple.py on the 15.5-um device whose loss is 90 % radiation
(265046: Q_i 5.25e5, Q_rad 5.86e5). 51 % of its leak is at grazing |u_x| > 0.8 (DERIVED from the 265046
tube far field) -> ripple period 4.66 um (= (beta - 0.85 k_clad) at lam 1619.5 nm, n_eff 1.575) sends the
+-beta guided waves to u_x = +-0.85 (its tooth-grid alias at 0.58 um covers the opposite lobe).
dw = 20 nm * g(teeth 4-40, 4-tooth raised-cosine ramps) * cos(2 pi x / 4.66 um + phi), added to BOTH the wide
and the narrow tooth (wide-narrow contrast unchanged). phi = 0 / 180 / 90 deg = three modules (one per
results dir, the tag ignores interior teeth): farfield_sph_15um_ripple, _ripple180, _ripple90.
Numerics = farfield_sph_15um ROUND C_NT80 PROD exactly (short run 340 ps, tube 100 um, caps, own +-0.25 nm
far-field window at 1619.455, box 6.8/6.81, pitch 514, N 96); control = the 265046 row. Analysis: scratch
ripple_response.py (two REAL knobs cos/-sin, Lorentzian-at-lam_ff phase, per-stored-energy amplitude),
ringdown_budget.py for Q_L / T / radiated fraction per row.
Dispatch (Athena):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_ripple --max-concurrent=1
"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS, BASE, N_SIDE, zmult, PITCH_ITAI_NM  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

RIPPLE_NM = 20.0
PERIOD_UM = 4.66
PHASE_DEG = float(os.environ.get("FF15_RIPPLE_DEG", "0"))
TOOTH_LO, TOOTH_HI, RAMP = 4, 40, 4


def window(d):
    if d < TOOTH_LO or d > TOOTH_HI:
        return 0.0
    if d < TOOTH_LO + RAMP:
        return 0.5 * (1 - math.cos(math.pi * (d - TOOTH_LO) / RAMP))
    if d > TOOTH_HI - RAMP:
        return 0.5 * (1 - math.cos(math.pi * (TOOTH_HI - d) / RAMP))
    return 1.0


def ripple(nar0, wid0):
    nar, wid = list(nar0), list(wid0)
    for d in range(len(nar)):
        dw = RIPPLE_NM * window(d) * math.cos(2 * math.pi * d * PITCH_ITAI_NM * 1e-3 / PERIOD_UM + math.radians(PHASE_DEG))
        nar[d] += dw
        wid[d] += dw
    return nar, wid


seed = ROWS[0]
ROW = seed[:5] + (ripple(*seed[5]),) + seed[6:]
SPEC = SweepSpec(
    polarization              = [ROW[0]],
    pitch_nm                  = [ROW[1]],
    corrugation_depth_nm      = [ROW[2]],
    avg_width_nm              = [ROW[3]],
    cavity_width_nm           = [ROW[4]],
    width_narrow_per_tooth_nm = [ROW[5][0]],
    width_wide_per_tooth_nm   = [ROW[5][1]],
    n_periods_each_side       = [N_SIDE],
    center_wavelength_nm      = [ROW[6]],
    scan_width_nm             = [ROW[7]],
    y_span_um                 = [ROW[8]],
    span_mult                 = [zmult(ROW[9], ROW[6])],
    mode  = "zipped",
    label = "farfield_sph_15um_ripple",
)

if __name__ == "__main__":
    nar0, wid0 = seed[5]; nar, wid = ROW[5]
    print(f"phase {PHASE_DEG:.0f} deg, period {PERIOD_UM} um, A {RIPPLE_NM} nm, pitch {PITCH_ITAI_NM}, N {N_SIDE}")
    for d in (0, 4, 6, 8, 10, 12, 20, 30, 38, 40, 41):
        print(f"d {d:2d} x {d*PITCH_ITAI_NM*1e-3:6.2f} um  dw {wid[d]-wid0[d]:+7.2f}  nar {nar[d]:7.1f} wid {wid[d]:7.1f}")
    assert all(abs((a - a0) - (b - b0)) < 1e-9 for a, a0, b, b0 in zip(nar, nar0, wid, wid0))
    assert nar[41:] == list(nar0[41:]) and wid[41:] == list(wid0[41:]) and nar[:4] == list(nar0[:4])
    print("DC sum (nm):", round(sum(wid[d] - wid0[d] for d in range(len(wid))), 2), "| max wide", round(max(wid), 1), "min narrow", round(min(nar), 1))
    for cfg in SPEC.expand(base=BASE):
        print(f"pitch {cfg.grating.pitch_m*1e9:.2f} N {cfg.grating.n_periods_each_side} window {cfg.spectral.center_wavelength_m*1e9:.3f} "
              f"+- {cfg.spectral.scan_width_nm/2:.1f} nm | box y {cfg.y_span*1e6:.2f} z {cfg.z_span*1e6:.2f} | short_run {cfg.farfield.short_run} "
              f"ffx {cfg.farfield.farfield_x_span_m*1e6:.0f} um | 2d window {cfg.farfield.use_2d_window} {cfg.monitors.monitor_2d_center_nm} +- {cfg.monitors.monitor_2d_span_nm/2} "
              f"| shutoff {cfg.mesh.auto_shutoff_min} | sim ps {os.environ.get('TM_SIM_TIME_PS')}")
