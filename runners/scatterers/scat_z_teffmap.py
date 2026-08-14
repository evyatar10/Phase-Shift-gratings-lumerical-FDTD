"""Stage Z — TE far-field map: where does TE's 12% loss actually radiate?

Study dir: runners/scatterers/   |   Created 2026-08-10   |   Job(s): TBD
Purpose (user): first-ever TE far-field data, to decide whether a multi-post
row (comb) can work on TE and to resolve the pair's TE mechanism. Rows:
  0  TE ctrl + FF map     in-study baseline; TE loss vs angle
  1  TE + pair [0,270]    what the measured +0.0284 (job 124194) removes
REGISTERED: NO grazing needle on TE (n_eff ~1.56 sits ~7 sigma of envelope
k-broadening past the horizon, vs TM ~4.7 sigma; trench TE-null 124531 says
the loss is steep/vertical). Comb go/no-go = a coherent narrow lobe in the
row-0 map (then its angle -> Lambda, phase -> dx). Pair row expects
dT ~ +0.028 at its own map numerics.

Physics line (CLAUDE.md section 4): TE h350, pitch 500, corr 300, W800,
N=80/side; resonance ~1559.1 nm (124194), window 1549.0-1569.0 (20 nm /
1501 pts), box y=16, opt mesh. FF monitors 60 um, complex save.

Dispatch (IGUM, user pick 2026-08-10; Athena q3db 130458 in parallel —
license peak 1+1 well under 6):
    SBATCH_MEM=160G ARRAY_TIME=03:00:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.scatterers.scat_z_teffmap --max-concurrent=1
Output -> results/scat_z_teffmap/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

TE_PITCH_NM    = 500.0
TE_CORR_NM     = 300.0
BOX_Y_UM       = 16.0
SCAN_CENTER_NM = 1559.0
SCAN_WIDTH_NM  = 20.0
N_WL_POINTS    = 1501

# rows: (r_nm, x_list_nm, y_list_nm) — row 0 = no-scatterer control
ROWS = [(0.0, [0.0], [700.0]),
        (80.0, [0.0, 270.0], [700.0, 700.0])]   # the 124194 pair exactly

BASE = _common.build_ff_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

SPEC = SweepSpec(
    polarization         = ["TE"] * len(ROWS),
    pitch_nm             = [TE_PITCH_NM] * len(ROWS),
    corrugation_depth_nm = [TE_CORR_NM] * len(ROWS),
    center_wavelength_nm = [SCAN_CENTER_NM] * len(ROWS),
    scatterer_radius_nm  = [r for r, _, _ in ROWS],
    scatterer_x_list_nm  = [x for _, x, _ in ROWS],
    scatterer_y_list_nm  = [y for _, _, y in ROWS],
    mode  = "zipped",
    label = "scat_z_teffmap",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("anchors (124194, ybox4.8): TE ctrl T 0.8733 | pair T 0.9017 (+0.0284)")
