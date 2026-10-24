"""STAIRCASE-MOIRE kill test: the 20-um overshoot control at two finer y-meshes (dy 29 / 38 nm instead of 58 nm).

Study dir: runners/sweeps/ | Created 2026-10-20 | Job(s): IGUM (see memory project_farfield_sph_15um.md)
Why (agent analysis of the MEASURED 182131 far field, 2026-10-20): the l > 10 half of the leak is a comb of angular lines at
|u_x| ~ 0.62 / 0.72 / 0.86 emitted from teeth 6-12; a y-mesh staircase model (sidewall edges crossing mesh cells along the
apodization ramp, dy ~50 nm, edge motion 6-10 nm/period) reproduces the comb with corr 0.82 (0.68 on the 15.5-um Nt80x). If true,
~half the dx-50 "leak" is numerics and the knob responses are half mesh response. Kill test: same geometry, override-region
dy = width_narrow / y_mesh_cells with y_mesh_cells 26 (~29 nm) and 20 (~38 nm). Model prediction: the 0.56-0.76 band of the
l > 10 part falls from 53 % to < 15 % of that part and the comb reappears at |u_x| 0.08-0.28 (dy 29) / 0.40-0.52 (dy 38);
the l > 10 share drops. Comb unchanged => moire dead. Control = Athena 182131 / 265100 (dy 58 nm, identical otherwise;
cross-cluster identity proven 2026-08-10). Everything else = round C (150 ps, conformal, shutoff off, box 6.8/6.81, tube E/H).
NEW ENGINE KNOB (inert by default, snapshot gate 6/6 SAME): MeshConfig.y_mesh_cells (default 13) -> bragg_device dy_global.
Dispatch (IGUM; one module per mesh because the cards path takes one BASE per module):
    bash igum/deploy_igum.sh --cards=runners.sweeps.farfield_sph_20um_meshcheck --max-concurrent=2
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF_SPH_ROUND"] = "C"
from runners.sweeps.farfield_sph_20um_bumps import BASE as BASE_C, N_SIDE, hh  # noqa: E402  (expanded round-C seed)
from experiment_card import ExperimentCard  # noqa: E402

Y_CELLS = int(os.environ.get("FF_MESHCHECK_YCELLS", "26"))
BASE = copy.deepcopy(BASE_C)
BASE.mesh.y_mesh_cells = Y_CELLS
nar0, wid0 = hh.teeth(N_SIDE)
CARDS = [ExperimentCard(width_narrow_per_tooth_nm=list(nar0), width_wide_per_tooth_nm=list(wid0), label=f"seed_ycells{Y_CELLS}")]

if __name__ == "__main__":
    cfg = CARDS[0].to_config(base=BASE); kw = cfg.to_device_kwargs()
    print(f"{CARDS[0].label}: y_mesh_cells {kw['y_mesh_cells']} -> dy {cfg.geometry.width_narrow_m*1e9/kw['y_mesh_cells']:.1f} nm | pitch {cfg.grating.pitch_m*1e9:.2f} N {cfg.grating.n_periods_each_side} "
          f"| box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f} shutoff {cfg.mesh.auto_shutoff_min} mode {cfg.mesh.simulation_mode} sim ps {os.environ.get('TM_SIM_TIME_PS')} eh {cfg.farfield.save_surface_eh}")
