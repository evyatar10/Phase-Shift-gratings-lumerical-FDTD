"""Proposal figure: |E|^2 field maps, bare vs inverse-designed TM corr-325 grating, N=100.

Study dir: runners/sweeps/   |   Created 2026-09-30   |   Job(s): TBD (Athena)
Purpose (user 2026-09-30, research-proposal comment on paragraph 2): show the mode
radiating from the UNIFORM device and radiating less from the OPTIMIZED one, at the
same mode width. Short device (N=100/side) on purpose -- the figure says so.

Rows = the two STORED N=100 rows, rebuilt bit-for-bit, plus the 2D field planes:
  0  bare corr-325        tm_nladder_c325 row 3     (IGUM 51736)  T 0.9104  lam 1559.006  Q 1760   W 19.2448
  1  BEST_T9636 design    invdesign_q3db_20um row 0 (IGUM 63202_0) T 0.97228 lam 1560.947 Q 1818.6 W 19.1709
Why re-run (section 6): neither stored .mat holds 2D planes -- the planes are the new
observable. Identity is otherwise unchanged (R1.3, same builder, box, mesh, window,
4001 pts), and the file tags equal the stored filenames (asserted in __main__). GATE:
each row must reproduce its stored T / lambda / Q before its maps are used.

2D monitors: own 0.4 nm window centred on the MEASURED resonance, 5 points = 0.1 nm
spacing, so the nearest plane is <=50 pm off a ~860 pm line (Q~1800). Planes: XY
(z = core mid-height) and XZ (y = 0), both recorded.

Dispatch:
    SBATCH_MEM=160G bash athena/deploy_athena.sh \
        --option3 --spec=runners.sweeps.proposal_fieldmaps_n100 --max-concurrent=2
Output -> results/proposal_fieldmaps_n100/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.sweeps import invdesign_q3db_20um as D      # the design row, reused verbatim

N_PERIODS     = 100
LAM_RES_NM    = [1559.006, 1560.947]      # MEASURED, stored rows above
MON2D_SPAN_NM = 0.4
N_2D_POINTS   = 5

BASE = D.BASE                               # build_ports_base + y 8 um + 20 nm / 4001 pts
BASE.monitors.record_2d_fields = True
BASE.spectral.n_2d_monitor_points = N_2D_POINTS

SPEC = SweepSpec(
    n_periods_each_side       = [N_PERIODS, N_PERIODS],
    corrugation_depth_nm      = [D.CORR_NM, D.CORR_NM],
    width_narrow_per_tooth_nm = [None, D.W_NARROW],
    width_wide_per_tooth_nm   = [None, D.W_WIDE],
    inner_shift_list_nm       = [None, D.SHIFTS],
    n_free_inner_teeth        = [1, D.N_FREE],
    cavity_width_nm           = [None, D.CAVITY_W_NM],
    center_wavelength_nm      = [D.SCAN_CENTER_NM, D.SCAN_CENTER_NM],
    scan_width_nm             = [D.SCAN_WIDTH_NM, D.SCAN_WIDTH_NM],
    scatterer_radius_nm       = [0.0, D.COMB_R_NM],     # bare row: NO scatterer (default-ON trap)
    scatterer_x_list_nm       = [None, D.X_COMB],
    scatterer_y_list_nm       = [None, [D.COMB_D_NM] * len(D.X_COMB)],
    scatterer_height_nm       = [None, 350.0],
    monitor_2d_center_nm      = LAM_RES_NM,
    monitor_2d_span_nm        = [MON2D_SPAN_NM, MON2D_SPAN_NM],
    mode  = "zipped",
    label = "proposal_fieldmaps_n100",
)

if __name__ == "__main__":
    from bragg_device import PiShiftBraggFDTD
    from sim_helpers import generate_file_tag
    print(SPEC.describe())
    stored = ["N100_TM_avg_C325_Ybox8p0_Zbox8p8",
              "N100_TM_W961_C325_dsh25S66s3_ptw25W964to981_ptn25W640to620_Ybox8p0_Zbox8p8"
              "_scR80_arr57_X-14467to15269_Y1900to1900_C325_pair"]
    for cfg, want in zip(SPEC.expand(BASE), stored):
        tag = generate_file_tag(PiShiftBraggFDTD(**cfg.to_device_kwargs()))
        print(f"  tag {tag}  {'== stored' if tag == want else '!= stored ' + want}")
        assert tag == want, "row does not rebuild the stored device"
