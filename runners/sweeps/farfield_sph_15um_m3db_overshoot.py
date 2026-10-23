"""-3 dB LENGTH CALIBRATION of the bare TE overshoot (Nt80x): the stored control lengthened to 108 and 116 periods/side.

Study dir: runners/sweeps/ | Created 2026-10-23 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-23.md)
Why (user go 2026-10-23; GPT planning review 01a16881): Q at -3 dB is so far DERIVED as 0.293 Q_i at 96 periods (T 0.664 bare, 0.770 with
the 3-row post array). A direct -3 dB device needs the coupling Q to rise 1.22e5 -> ~3.6e5; the per-period growth of Q_c in the outer
(bulk 1250/750 nm) section is not measured, so two lengths bracket T = 0.5 and give Q_c(N). Reusable for ANY post-array winner (posts end at
|x| <= 30 um; the added periods sit at |x| > 49 um). Identical numerics to the control (C_NT80 PROD, box 6.8/6.81, production mesh, 4 nm
source, tube far field) except N (teeth padded with the bulk 750/1250 nm, exactly as the 96-period control pads its 81 apodized teeth) and
the run length (~2 amplitude decay times at the expected Q_L: 800 ps at N 108, 1100 ps at N 116; skill highq-short-run).
Reading: Q_L from the trace decay, T x gamma^2, Q_i, width (solve_v4.py --short --energy vs control_nt80x_T340.mat); Q_c(N) = 1/(1/Q_L - 1/Q_i).
Dispatch (Athena): SBATCH_MEM=250G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_m3db_overshoot \
                     --max-concurrent=2 --qos=4d_1g --gpu=a100
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, NT80X_NARROW_NM, NT80X_WIDE_NM, pad, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

POL, PITCH_NM, CORR_NM, AVG_NM, CAV_W_NM, TEETH96, LAM_NM, WIN_NM, BOX_Y, BOX_Z = FF_ROWS[0]
ROWS = [(108, 800.0), (116, 1100.0)]          # (periods per side, run length ps)
_TASK = int(os.environ.get("SWEEP_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID") or 0)
os.environ["TM_SIM_TIME_PS"] = str(ROWS[_TASK][1])   # read by bragg_device at build (per array task)

BASE = copy.deepcopy(BASE_FF)
n = len(ROWS)
SPEC = SweepSpec(
    polarization              = [POL] * n,
    pitch_nm                  = [PITCH_NM] * n,
    corrugation_depth_nm      = [CORR_NM] * n,
    avg_width_nm              = [AVG_NM] * n,
    cavity_width_nm           = [CAV_W_NM] * n,
    width_narrow_per_tooth_nm = [pad(NT80X_NARROW_NM, NT80X_WIDE_NM, N)[0] for N, _ in ROWS],
    width_wide_per_tooth_nm   = [pad(NT80X_NARROW_NM, NT80X_WIDE_NM, N)[1] for N, _ in ROWS],
    n_periods_each_side       = [N for N, _ in ROWS],
    center_wavelength_nm      = [LAM_NM] * n,
    scan_width_nm             = [WIN_NM] * n,
    y_span_um                 = [BOX_Y] * n,
    span_mult                 = [zmult(BOX_Z, LAM_NM)] * n,
    mode  = "zipped",
    label = "farfield_sph_15um_m3db_overshoot",
)

if __name__ == "__main__":
    for (N, ps), cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs(); nar = [v * 1e9 for v in kw["width_narrow_per_tooth_m"]]; wid = [v * 1e9 for v in kw["width_wide_per_tooth_m"]]
        # property asserts: same first 96 teeth as the control, bulk padding beyond, N, TE, box, short run, no scatterers
        assert kw["n_periods_each_side"] == N and len(nar) == len(wid) == N, ("N", N)
        assert all(abs(a - b) < 1e-6 for a, b in zip(nar[:96], TEETH96[0])) and all(abs(a - b) < 1e-6 for a, b in zip(wid[:96], TEETH96[1])), ("teeth 1-96 = control", N)
        assert set(round(v, 3) for v in nar[96:]) == {750.0} and set(round(v, 3) for v in wid[96:]) == {1250.0}, ("bulk padding", N)
        assert cfg.source.polarization == "TE" and abs(cfg.y_span * 1e6 - BOX_Y) < 1e-6 and cfg.farfield.short_run and not cfg.scatterer.enabled, ("numerics", N)
        print(f"N {N}: teeth {len(nar)} (81 apodized + bulk), last {nar[-1]:.0f}/{wid[-1]:.0f} nm, run {ps:.0f} ps, box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f}, "
              f"lam {cfg.spectral.center_wavelength_m*1e9:.3f}, window {cfg.spectral.scan_width_nm} nm, short {cfg.farfield.short_run}, TE")
    print("TM_SIM_TIME_PS for this process (task", _TASK, "):", os.environ["TM_SIM_TIME_PS"])
