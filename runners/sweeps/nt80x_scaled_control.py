"""Width-matched CONTROL for nt80x_second_tooth DELTA -206: the Nt80x overshoot with its wide teeth enlarged, no second tooth.

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see dispatch note in the session; user approved ONE run)
Why: the second tooth at DELTA -206 (LOC 267814: lambda 1623.02, Q_L 2.18e5, W 14.69 um, T ~0.23 by gamma^2) has Q_i 4.2e5;
the overshoot scaled to that width by Q_i ~ W^2.5..3.6 gives 3.7..4.2e5 -> the tooth is 0..+13 % better in Q3dB (= 0.293 Q_i)
only if the exponent is right. This run measures the overshoot's own Q_i at that width instead.
Device: every period's wide segment raised to narrow + S * (wide - narrow), narrow and cavity untouched, S = 1 + 0.5 *
sin(0.3 pi) * cos(2 pi * 51.4 / 514) = 1.327 -- the same first Fourier (Bragg) component the tooth adds (a +c/2 bump over
154.2 nm = 0.3 pitch, centred 51.4 nm from the wide-half centre), and nearly the same mean-width increase (0.164 c vs
0.150 c), so lambda and W should land close to the tooth row. Period 0 is flat (c = 0) and stays flat.
Numerics = the stored control 265046 (C_NT80: production mesh, box 6.8/6.81, short_run caps + traces); window 10 nm /
4001 points (2.5 pm < the 7 pm linewidth) because lambda is EXPECTED, not measured; run 750 ps = ~2 decay times at Q 2.2e5.
Read: Q_L from the cavity-trace decay, T_port x gamma^2 (ringdown_budget), Q_i = Q_L/(1 - sqrt T); compare with the tooth row
at the measured width (residual width difference -> W^2.5..3.6 over a few % only).
Dispatch (Athena): SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.nt80x_scaled_control --gpu=a100
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

POL, PITCH_NM, CORR_NM, AVG_NM, CAV_W_NM, TEETH, _, _, BOX_Y, BOX_Z = FF_ROWS[0]
S = 1 + 0.5 * math.sin(0.3 * math.pi) * math.cos(2 * math.pi * 51.4 / 514.0)
NAR = list(TEETH[0])
WID = [n + S * (w - n) for n, w in zip(TEETH[0], TEETH[1])]
LAM_NM, WIN_NM, N_PTS, SIM_PS = 1623.5, 10.0, 4001, 750.0

os.environ["TM_SIM_TIME_PS"] = str(SIM_PS)
BASE = copy.deepcopy(BASE_FF)
BASE.farfield.short_run = True               # caps + time traces (Q_L); exported only with far field + short_run on
BASE.farfield.use_2d_window = False          # far field over the source window (not needed for Q_i / T)
BASE.spectral.n_wl_points = N_PTS
SPEC = SweepSpec(
    polarization              = [POL],
    pitch_nm                  = [PITCH_NM],
    corrugation_depth_nm      = [CORR_NM],
    avg_width_nm              = [AVG_NM],
    cavity_width_nm           = [CAV_W_NM],
    width_narrow_per_tooth_nm = [NAR],
    width_wide_per_tooth_nm   = [WID],
    n_periods_each_side       = [N_SIDE],
    center_wavelength_nm      = [LAM_NM],
    scan_width_nm             = [WIN_NM],
    y_span_um                 = [BOX_Y],
    span_mult                 = [zmult(BOX_Z, LAM_NM)],
    mode  = "zipped",
    label = "nt80x_scaled_control",
)

if __name__ == "__main__":
    assert abs(PITCH_NM - 514.0) < 1e-9 and abs(TEETH[0][1] - 919.5) < 1e-9, "not the Nt80x device"
    cfg = SPEC.expand(base=BASE)[0]; kw = cfg.to_device_kwargs()
    n_b = [v * 1e9 for v in kw["width_narrow_per_tooth_m"]]; w_b = [v * 1e9 for v in kw["width_wide_per_tooth_m"]]
    assert all(abs(a - b) < 1e-6 for a, b in zip(n_b, TEETH[0])), "narrow widths changed"
    assert all(abs((w - n) - S * (w0 - n0)) < 1e-6 for n, w, n0, w0 in zip(n_b, w_b, TEETH[0], TEETH[1])), "contrast not scaled by S"
    assert abs(w_b[0] - 950.3) < 1e-6 and not kw.get("scatterer_enabled", False), "period 0 not flat / scatterers on"
    print(f"S = {S:.4f} | periods 0-3 narrow {[round(v, 1) for v in n_b[:4]]} wide {[round(v, 1) for v in w_b[:4]]} (was {[round(v, 1) for v in TEETH[1][:4]]})")
    print(f"bulk narrow {n_b[-1]:.1f} wide {w_b[-1]:.1f} | max wide {max(w_b):.1f} | window {cfg.spectral.center_wavelength_m*1e9:.1f}+-{cfg.spectral.scan_width_nm/2}"
          f" x {cfg.spectral.n_wl_points} | sim ps {os.environ['TM_SIM_TIME_PS']} | short {cfg.farfield.short_run} | box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f}")
