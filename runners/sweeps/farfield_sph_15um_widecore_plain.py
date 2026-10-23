"""LIGHT-CONE MARGIN test on the PLAIN TE pi-shift device: wider mean core at the SAME Bragg contrast (planar, single layer).

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see memory project_radcancel_overnight_2026-10-22.md)
Why: both blind-spot reviewers (Fable a588520c, GPT 01a162bc, 2026-10-22) name the light-cone margin beta - k_ox (only 0.46/um on the
plain device, n_eff 1.5585 vs 1.444) as the lever every cancellation trick sits under; Fable argues my FDE step-mismatch kill of wide
cores counts non-radiating TE2/near-field conversion and predicts Q_i x2-4 from the margin. This run decides it on the cheap device.
Design (FDE, scratch fde_dark.py; TE0, n 1.97/1.444, 350 nm, 1558.5 nm): plain 642/958 has Bragg contrast dn = 0.0626; equal dn (equal
kappa -> ~equal envelope/mode width) at mean 1250 nm -> 921/1579 (n_avg ~1.608, margin +0.057 vs plain) and mean 1500 nm -> 1028/1972
(n_avg ~1.624, margin +0.073); pitch scaled by n_avg (500 x 1.5515/n_avg); cavity = mean width, pitch/2 long. Window widened to 20 nm
(the pitch estimate is first order). Control = the stored round-A plain far field (Q_i 2.02e4, W 15.29 um); this is a different
device, so the comparison is Q_i and radiation per stored energy at (nearly) equal mode width, not a row-minus-control response.
Pass: Q_i >= 1.5x the plain control with |dW| <= 10 %; else the margin story is dead for planar width.
Dispatch (Athena): bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_widecore_plain --gpu=a100   (1250)
                   bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.farfield_sph_15um_widecore_plain_w1500 --gpu=a100   (1500)
"""
import copy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["FF15_ROUND"] = "A"
from runners.sweeps.farfield_sph_15um import ROWS as FF_ROWS, BASE as BASE_FF, N_SIDE, zmult  # noqa: E402
from runners.sweeps.sweep_spec import SweepSpec  # noqa: E402

POL, _, _, _, _, _, LAM_NM, _, BOX_Y, BOX_Z = FF_ROWS[0]
N_AVG_PLAIN = (1.52013 + 1.58276) / 2                         # FDE n_eff of the 642 / 958 halves
WIN_NM = 20.0
# (name, mean width nm, narrow nm, wide nm, n_avg FDE, y_mesh_cells). MESH RULE (user 2026-10-22): dy = narrow / y_mesh_cells must match
# the plain control's 642/13 = 49.4 nm (job 267539 ran these rows at the default 13 cells = dy 70.8 / 79.1 nm -> cancelled, not comparable);
# dx = pitch/10 stays commensurate with the period automatically. One row per module (y_mesh_cells is a BASE knob): env FF15_WC selects.
ALL_ROWS = {"1250": ("wc1250", 1250.0, 921.0, 1579.0, 1.608, 19), "1500": ("wc1500", 1500.0, 1028.0, 1972.0, 1.624, 21)}
ROWS = [ALL_ROWS[os.environ.get("FF15_WC", "1250")]]
PITCH = {r[0]: round(500.0 * N_AVG_PLAIN / r[4], 1) for r in ROWS}

BASE = copy.deepcopy(BASE_FF)
BASE.mesh.y_mesh_cells = ROWS[0][5]
n = len(ROWS)
SPEC = SweepSpec(
    polarization              = [POL] * n,
    pitch_nm                  = [PITCH[r[0]] for r in ROWS],
    corrugation_depth_nm      = [r[3] - r[2] for r in ROWS],
    avg_width_nm              = [r[1] for r in ROWS],
    cavity_width_nm           = [r[1] for r in ROWS],
    width_narrow_per_tooth_nm = [None] * n,
    width_wide_per_tooth_nm   = [None] * n,
    n_periods_each_side       = [N_SIDE] * n,
    center_wavelength_nm      = [LAM_NM] * n,
    scan_width_nm             = [WIN_NM] * n,
    y_span_um                 = [BOX_Y] * n,
    span_mult                 = [zmult(BOX_Z, LAM_NM)] * n,
    mode  = "zipped",
    label = f"farfield_sph_15um_widecore_plain_{ROWS[0][0]}",
)

if __name__ == "__main__":
    for r, cfg in zip(ROWS, SPEC.expand(base=BASE)):
        kw = cfg.to_device_kwargs()
        # property asserts: narrow/wide as designed, pitch scaled, cavity = mean, guide inside the box with >= 2 um clearance
        wn, ww = kw["width_narrow"] * 1e9, kw["width_wide"] * 1e9
        assert abs(wn - r[2]) < 0.5 and abs(ww - r[3]) < 0.5, ("teeth", r[0], wn, ww)
        assert abs(kw["pitch"] * 1e9 - PITCH[r[0]]) < 0.05 and abs(kw["cavity_width_m"] * 1e9 - r[1]) < 0.5, ("pitch/cavity", r[0])
        assert cfg.y_span * 1e9 / 2 - ww / 2 >= 2000, ("box too tight", r[0])
        dy = wn / kw["y_mesh_cells"]; dx = kw["pitch"] * 1e9 / 2 / kw["cells_per_half_period"]
        assert abs(dy - 642.0 / 13) < 1.5, ("dy does not match the plain control 49.4 nm", r[0], dy)
        assert abs(kw["pitch"] * 1e9 / dx - 2 * kw["cells_per_half_period"]) < 1e-9, ("dx not commensurate with the pitch", r[0])
        print(f"{r[0]}: narrow {wn:.0f} wide {ww:.0f} mean {r[1]:.0f} pitch {kw['pitch']*1e9:.1f} nm (port {kw['width_port']*1e9:.0f}) dx {dx:.2f} dy {dy:.2f} | window {cfg.spectral.scan_width_nm} "
              f"centre {cfg.spectral.center_wavelength_m*1e9:.1f} | sim ps {os.environ.get('TM_SIM_TIME_PS')} box {cfg.y_span*1e6:.2f}/{cfg.z_span*1e6:.2f}")
