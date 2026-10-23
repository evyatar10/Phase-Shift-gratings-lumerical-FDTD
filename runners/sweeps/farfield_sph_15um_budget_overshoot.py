"""CONVERGED LOSS BUDGET of the TE overshoot (Nt80x): dy 29 nm at two transverse box sizes.

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Why (spec = Q at -3 dB at fixed mode width, ~0.29 Q_i, so a cancellation gain is capped by radiation's TRUE share of 1/Q_i):
MEASURED so far: dy 50 (control, T340) Q_i 5.35e5; dy 37 (265195) Q_i 5.67e5, leak per stored energy 1.03 but the leak PATTERN
changes by |b|^2/|c|^2 0.36 (overlap 0.08); on the 20-um overshoot the leak fell to 78 % at dy 29 (265134). The "radiation = 90 %
of loss" figure is therefore an upper bound and the pattern is not converged. Rows: dy 29 (y_mesh_cells 26: 750/26 = 28.8 nm) at
the production box 6.8/6.81 um and at 8.8/8.8 um (PML/box effect on a grazing leak, Fable review a588520c). Same short-run numerics
(C_NT80 PROD, 340 ps, tube far field) otherwise. Reading: Q_L from the trace decay, Q_i = Q_L/(1-sqrt(T gamma^2)), far power per
FULL stored energy, l-split; compare dy 50 / 37 / 29 and the two boxes. Decides whether any radiation-cancellation route can reach
1.5x Q_3dB (needs radiation >= ~37 % of the converged 1/Q_i removed).
Dispatch (Athena): SBATCH_MEM=250G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_budget_overshoot \
                     --max-concurrent=2 --qos=4d_1g --gpu=a100
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, N_SIDE, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

POL, PITCH_NM, CORR_NM, AVG_NM, CAV_W_NM, TEETH, LAM_NM, WIN_NM, BOX_Y, BOX_Z = FF_ROWS[0]
Y_CELLS = 26
ROWS = [("dy29_box6p8", BOX_Y, BOX_Z), ("dy29_box8p8", 8.8, 8.8)]

BASE = copy.deepcopy(BASE_FF)
BASE.mesh.y_mesh_cells = Y_CELLS
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
    y_span_um                 = [r[1] for r in ROWS],
    span_mult                 = [zmult(r[2], LAM_NM) for r in ROWS],
    mode  = "zipped",
    label = "farfield_sph_15um_budget_overshoot",
)

if __name__ == "__main__":
    for r, cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        dy = kw["width_narrow"] * 1e9 / kw["y_mesh_cells"]
        # property asserts: dy ~29 nm, teeth = the original Nt80x (not the +20 parent), boxes as named, short-run numerics
        assert kw["y_mesh_cells"] == Y_CELLS and abs(dy - 28.85) < 0.5, ("mesh", r[0], dy)
        assert list(kw["width_narrow_per_tooth_m"])[:2] == [x * 1e-9 for x in TEETH[0][:2]], ("teeth", r[0])
        assert abs(cfg.y_span * 1e6 - r[1]) < 1e-6 and abs(cfg.z_span * 1e6 - r[2]) < 0.05, ("box", r[0], cfg.y_span, cfg.z_span)
        assert cfg.farfield.short_run, ("short run", r[0])
        print(f"{r[0]}: dy {dy:.2f} nm (cells {kw['y_mesh_cells']}) box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} | sim ps {os.environ.get('TM_SIM_TIME_PS')} "
              f"lam {cfg.spectral.center_wavelength_m*1e9:.3f} | teeth[0] {TEETH[0][0]}/{TEETH[1][0]}")
