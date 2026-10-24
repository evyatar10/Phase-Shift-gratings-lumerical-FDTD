"""Nt80x (15.5-um lab device) at a finer y mesh (dy ~37 nm): how much of its simulated radiation is physical.

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why (MEASURED on the 20-um overshoot, Athena 265134/265135 vs 182131): the l > 10 half of the dx-50 leak shrinks with the y mesh
(51 -> 45 -> 33 % at dy 50/37/29) and its pattern is uncorrelated between meshes, while l <= 10 is converged -> the Nt80x budget
"radiation = 90 % of loss" (265046, dy 50) is an upper bound. One row at y_mesh_cells 20 (dy = 750/20 = 37.5 nm) with the exact
C_NT80 PROD numerics (short run 340 ps, tube + caps, own +-0.25 nm window) gives T, Q_L (decay), Q_i, far power and the l-split.
Dispatch (Athena, outside the 24h_1g cap):
    SBATCH_MEM=200G bash athena/deploy_athena.sh --cards=runners.sweeps.farfield_sph_15um_meshcheck --max-concurrent=1 --qos=4d_1g --gpu=a100
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "C_NT80"
os.environ["FF15_C_STEP"] = "PROD"
from runners.sweeps.farfield_sph_15um_bumps import BASE as BASE_C, NAR0, WID0  # noqa: E402
from experiment_card import ExperimentCard  # noqa: E402

Y_CELLS = 20
BASE = copy.deepcopy(BASE_C)
BASE.mesh.y_mesh_cells = Y_CELLS
CARDS = [ExperimentCard(width_narrow_per_tooth_nm=list(NAR0), width_wide_per_tooth_nm=list(WID0), label=f"nt80x_ycells{Y_CELLS}")]

if __name__ == "__main__":
    cfg = CARDS[0].to_config(base=BASE); kw = cfg.to_device_kwargs()
    print(f"{CARDS[0].label}: dy {cfg.geometry.width_narrow_m*1e9/kw['y_mesh_cells']:.1f} nm | pitch {cfg.grating.pitch_m*1e9:.1f} N {cfg.grating.n_periods_each_side} "
          f"| short {cfg.farfield.short_run} 2dwin {cfg.monitors.monitor_2d_center_nm} | box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} | sim ps {os.environ.get('TM_SIM_TIME_PS')}")
