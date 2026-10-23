"""Born-designed SiN POST ARRAY on the PLAIN TE pi-shift grating (best rectangular cavity 1050 nm): does the overshoot result carry over?

Study dir: runners/sweeps/ | Created 2026-10-23 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-23.md)
Why (user 2026-10-23): on the TE overshoot a 3-row array gave Q_i +63 %; on the plain TE grating single combs never helped (cusp-born,
broad leak). The Born model (python_tools/radcancel/postarray_*.py, steady normalisation, control = round-A plain TE with the 1050-nm
cavity, c1050) predicts a 3-row array (rows y 0.8/1.1/1.4 um, 162 posts/side, r 50-100 nm, gap >= 50 nm) at leak 0.640.
Control = the stored cavsize_plain card cavW_1050nm (round A row 0: corr 316, pitch 500, N 96, 150 ps, 1558.8 +- 5 nm, box 6.8) at
identical numerics (asserted below). Reading: solve_v4.py (no --short) vs control_plain_c1050.mat; Q_i = Q_L/(1-sqrt T).
Dispatch (Athena): bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_postarray_plain --max-concurrent=1 --gpu=a100
"""
import copy
import json
import os
import sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "A"
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, N_SIDE, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

POL, PITCH_NM, CORR_NM, AVG_NM, _CAV, _TEETH, LAM_NM, WIN_NM, BOX_Y, BOX_Z = FF_ROWS[0]
CAV_W_NM = 1050.0
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "postarray_overshoot_designs.json")))["plain_pa3_r100"]

# variants (env FF15_PLAIN_SET, wrappers _dy29 / _dy29ctrl; user 2026-10-23): the same array and a NO-POST control at the converged
# core mesh (y_mesh_cells 26: dy 642/26 = 24.7 nm), because the overshoot pa3 gain fell from +63 % (production) to +33 % (dy29).
PLAIN_SET = os.environ.get("FF15_PLAIN_SET", "")
BASE = copy.deepcopy(BASE_FF)
BASE.scatterer.enabled = PLAIN_SET != "dy29ctrl"
BASE.scatterer.shape = "cylinder"
BASE.scatterer.mirrored_y = True
if PLAIN_SET in ("dy29", "dy29ctrl"):
    BASE.mesh.y_mesh_cells = 26
SPEC = SweepSpec(
    polarization=[POL], pitch_nm=[PITCH_NM], corrugation_depth_nm=[CORR_NM], avg_width_nm=[AVG_NM], cavity_width_nm=[CAV_W_NM],
    n_periods_each_side=[N_SIDE], center_wavelength_nm=[LAM_NM], scan_width_nm=[WIN_NM], y_span_um=[BOX_Y], span_mult=[zmult(BOX_Z, LAM_NM)],
    scatterer_shape=["cylinder"], scatterer_radius_nm=[75.0], scatterer_height_nm=[350.0], scatterer_index=[1.97], scatterer_mirrored_y=[True],
    scatterer_x_list_nm=[D["x_nm"]], scatterer_y_list_nm=[D["y_nm"]], scatterer_r_list_nm=[D["r_nm"]],
    mode="zipped", label="farfield_sph_15um_postarray_plain" + (f"_{PLAIN_SET}" if PLAIN_SET else ""),
)
if PLAIN_SET == "dy29ctrl":
    for f_ in ("scatterer_shape", "scatterer_radius_nm", "scatterer_height_nm", "scatterer_index", "scatterer_mirrored_y",
               "scatterer_x_list_nm", "scatterer_y_list_nm", "scatterer_r_list_nm"):
        setattr(SPEC, f_, None)

if __name__ == "__main__":
    from runners.sweeps.farfield_sph_15um_cavsize_plain import CARDS, BASE as BASE_CTRL
    ctrl = [c for c in CARDS if c.label == "cavW_1050nm"][0].to_config(base=BASE_CTRL).to_device_kwargs()
    cfg = SPEC.expand(base=BASE)[0]; kw = cfg.to_device_kwargs()
    if PLAIN_SET: ctrl["y_mesh_cells"] = 26        # the ONLY intended numerics change vs the stored control (converged core mesh)
    diff = {k: (ctrl.get(k), kw.get(k)) for k in set(ctrl) | set(kw) if not k.startswith("scatterer") and str(ctrl.get(k)) != str(kw.get(k))}
    assert not diff, ("non-post settings differ from the stored control", diff)
    if PLAIN_SET == "dy29ctrl":
        assert not cfg.scatterer.enabled and kw["y_mesh_cells"] == 26
        print(f"dy29ctrl: NO posts, cavity {kw['cavity_width_m']*1e9:.0f}, dy {kw['width_narrow']*1e9/kw['y_mesh_cells']:.1f} nm, sim ps {os.environ.get('TM_SIM_TIME_PS')} | identical to cavW_1050nm except y_mesh_cells 26")
        sys.exit(0)
    xb, yb, rb = (np.array(kw[k]) * 1e9 for k in ("scatterer_x_list_m", "scatterer_y_list_m", "scatterer_r_list_m"))
    wall = max(CAV_W_NM / 2, (AVG_NM + CORR_NM / 2) / 2)   # cavity 525 nm vs wide tooth (958 nm) half-width 479 nm
    gap = np.hypot(xb[:, None] - xb[None], yb[:, None] - yb[None]) - rb[:, None] - rb[None]; np.fill_diagonal(gap, 1e9)
    assert len(xb) == 162 and 49.9 <= rb.min() and rb.max() <= 100.1 and (yb - rb).min() - wall >= 150 and gap.min() >= 49.5, ("posts", len(xb), (yb - rb).min() - wall, gap.min())
    assert (yb + rb).max() <= cfg.y_span * 1e9 / 2 - 600 and np.abs(xb).max() + 100 <= cfg.farfield.farfield_x_span_m * 1e9 / 2, "box / far-field span"
    print(f"plain_pa3_r100: {len(xb)} posts/side rows {sorted(set(np.round(yb).astype(int)))} r {rb.min():.0f}-{rb.max():.0f} x {xb.min()/1e3:.1f}..{xb.max()/1e3:.1f} um | "
          f"wall {wall:.0f} clearance {(yb - rb).min() - wall:.0f} nm, min gap {gap.min():.0f} nm | cavity {kw['cavity_width_m']*1e9:.0f} | "
          f"sim ps {os.environ.get('TM_SIM_TIME_PS')} short {cfg.farfield.short_run} box {cfg.y_span*1e6:.2f} | identical to control cavW_1050nm except posts")
