"""Proposal figure, FULL -3 dB devices: field planes + far field, uniform vs inverse-designed.

Study dir: runners/sweeps/   |   Created 2026-09-30   |   Job(s): TBD (Athena)
Purpose (user 2026-09-30): the N=100 maps (proposal_fieldmaps_n100.py, job 165471) are a
short-device stand-in; this records the SAME observables on the two devices paragraph 2
of the proposal actually quotes, plus complex far fields.

Rows = the two STORED q3db rows rebuilt bit-for-bit (tags asserted in __main__):
  0  bare corr-325 N=165   (Athena 130458 ctrl)       lam 1559.0010  T 0.49058  Q 13930  W 19.970
  1  BEST_T9636 N=220      (IGUM invdesign row 4)     lam 1560.8508  T 0.49944  Q 88868  W 19.904
Windows exactly as stored: row 0 = 20 nm @1559.5, row 1 = 3 nm @1560.7, both 4001 pts
(5 pm vs lw 112 pm; 0.75 pm vs lw 17.6 pm). Box y 8.0 / z 8.8 (pinned, FF-independent).
GATE: each row must reproduce its stored T / lambda / Q before its maps are used.

Ring-down (memory project_highq_measurement_adequacy): tau = Q/omega = 73.6 ps at Q 88868,
16.1*tau = 1.19 ns < the 2 ns cap -> ends by auto-shutoff 1e-7 (confirm in the log).

Recorded:
  - 2D planes XY (z=0), XZ (y=0), YZ (x=+pitch/4): own window centred on the stored
    resonance, 9 points over ~2 linewidths (row 0: 0.2 nm, row 1: 0.04 nm).
  - Far field, top + side monitors, complex, 401^2, x-span 170 um (the whole N=165
    device; for N=220 the uncovered |x| > 85 um carries <1e-3 of the mode envelope),
    recorded in the SAME narrow window (farfield.use_2d_window) and projected at the
    point nearest the found resonance.

Dispatch:
    SBATCH_MEM=200G bash athena/deploy_athena.sh \
        --option3 --spec=runners.sweeps.proposal_fieldmaps_q3db --max-concurrent=2
Output -> results/proposal_fieldmaps_q3db/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.sweeps import invdesign_q3db_20um as D      # design + q3db base, reused verbatim

N_ROWS      = [165, 220]
LAM_RES_NM  = [1559.0010, 1560.8508]      # MEASURED, stored rows above
MON_SPAN_NM = [0.2, 0.04]                 # ~2 linewidths each
N_MON_PTS   = 9
SCAN_CENTER = [1559.5, 1560.7]            # stored windows
SCAN_WIDTH  = [20.0, 3.0]

BASE = D.BASE                               # build_ports_base + y 8 um + 4001 pts
BASE.monitors.record_2d_fields = True
BASE.spectral.n_2d_monitor_points = N_MON_PTS
BASE.farfield.enabled = True
BASE.farfield.save_complex = True
BASE.farfield.save_nearfield = False
BASE.farfield.farfield_x_span_m = 170e-6
BASE.farfield.ff_resolution = 401
BASE.farfield.farfield_freq_points = N_MON_PTS
BASE.farfield.use_2d_window = True

SPEC = SweepSpec(
    n_periods_each_side       = N_ROWS,
    corrugation_depth_nm      = [D.CORR_NM, D.CORR_NM],
    width_narrow_per_tooth_nm = [None, D.W_NARROW],
    width_wide_per_tooth_nm   = [None, D.W_WIDE],
    inner_shift_list_nm       = [None, D.SHIFTS],
    n_free_inner_teeth        = [1, D.N_FREE],
    cavity_width_nm           = [None, D.CAVITY_W_NM],
    center_wavelength_nm      = SCAN_CENTER,
    scan_width_nm             = SCAN_WIDTH,
    scatterer_radius_nm       = [0.0, D.COMB_R_NM],     # bare row: NO scatterer (default-ON trap)
    scatterer_x_list_nm       = [None, D.X_COMB],
    scatterer_y_list_nm       = [None, [D.COMB_D_NM] * len(D.X_COMB)],
    scatterer_height_nm       = [None, 350.0],
    monitor_2d_center_nm      = LAM_RES_NM,
    monitor_2d_span_nm        = MON_SPAN_NM,
    mode  = "zipped",
    label = "proposal_fieldmaps_q3db",
)

if __name__ == "__main__":
    from bragg_device import PiShiftBraggFDTD
    from sim_helpers import generate_file_tag
    print(SPEC.describe().split("width_narrow")[0])
    stored = ["N165_TM_avg_C325_Ybox8p0_Zbox8p8",
              "N220_TM_W961_C325_dsh25S66s3_ptw25W964to981_ptn25W640to620_Ybox8p0_Zbox8p8"
              "_scR80_arr57_X-14467to15269_Y1900to1900_C325_pair"]
    for cfg, want in zip(SPEC.expand(BASE), stored):
        sim = PiShiftBraggFDTD(**cfg.to_device_kwargs())
        tag = generate_file_tag(sim)
        sim.close()
        print(f"  tag {tag}  {'== stored' if tag == want else '!= stored ' + want}")
        assert tag == want, "row does not rebuild the stored device"
