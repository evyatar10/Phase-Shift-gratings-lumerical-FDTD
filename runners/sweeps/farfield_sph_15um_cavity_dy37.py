"""farfield_sph_15um_cavity on the FINER y mesh (y_mesh_cells 20, dy 37.5 nm): mesh check of the notch result.

Study dir: runners/sweeps/ | Created 2026-10-21 | Job(s): Athena (see memory project_farfield_sph_15um.md)
Why: the notch set (row 4, 5 nm) cut the +20 parent's radiation to 0.896 at dy 50 (266486_4). GPT gate (01a15a66): accept only
against the stored dy-37 +20 parent (farfield_sph_15um_meshcheck_c5 card nt80x_ycells20_diff0001_p20nm, 265968), same mesh.
Dispatch (Athena): SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_cavity_dy37                      --array-tasks=4 --max-concurrent=1 --qos=4d_1g --gpu=a100
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runners.sweeps import farfield_sph_15um_cavity as cv  # noqa: E402
from runners.sweeps.farfield_sph_15um_meshcheck import Y_CELLS  # noqa: E402  (20, the stored dy-37 parent's value)

BASE = copy.deepcopy(cv.BASE)
BASE.mesh.y_mesh_cells = Y_CELLS
SPEC = copy.deepcopy(cv.SPEC)
SPEC.label = "farfield_sph_15um_cavity_dy37"

if __name__ == "__main__":
    cfg = list(SPEC.expand(base=BASE))[4]; kw = cfg.to_device_kwargs()
    assert kw["y_mesh_cells"] == 20 and abs(kw["scatterer_index"] - 1.444) < 1e-9, "not the dy-37 notch row"
    print(f"row 4: y_mesh_cells {kw['y_mesh_cells']} (dy {cfg.geometry.width_narrow_m*1e9/kw['y_mesh_cells']:.1f} nm) n {kw['scatterer_index']} "
          f"sites {[round(v*1e9) for v in kw['scatterer_x_list_m']]} | sim ps {os.environ.get('TM_SIM_TIME_PS')}")
