"""Full-size TM devices (N = 150/side) with/without air trenches + 2D field maps.

Study dir: runners/metal_mirror/   |   Created 2026-07-20   |   Job(s): TBD
Purpose (user, overnight): the air trench is measured on N=80 devices
(W800 +0.0159 / W1050 +0.0157, h4 +0.0036 more / apod-10 +0.0039). This is
the full-scale comparison the real devices will use: N = 150 periods/side,
six sims = {regular W800, wider W1050, apodized n=10} x {no trench, trench},
each pair sharing identical numerics, with 2D field-profile maps (XY
horizontal + XZ vertical + YZ cross) saved in the .mat.

Design decisions (rules stated per work-alone):
- Trench = measured best geometry: air, L 156 um (spans the full N=150
  grating, +/-77.7 um), w 800 nm, h 4000 nm (job 124538: h4 beats h2 by
  +0.0036), center d = 1800 nm, mirrored +/-y.
- Resonances are KNOWN from N=80 (per-row window centers below); window
  10 nm (user: 10-20 nm; smallest = least DFT memory, +/-5 nm margin vs
  expected <0.1 nm N-shift), 2001 points = 5 pm sampling (>=30 samples
  across the narrowest expected line).
- 2D maps: 101 frequency points across the 10 nm window = 0.1 nm plane
  spacing, so the nearest recorded plane sits within 50 pm of the true
  resonance. Server .mat will be GB-scale: extract resonance-lambda planes
  on Athena, download slices only (CLAUDE.md section 6).
- Convergence (user's explicit worry): builder sets max sim time 2000 ps
  (TM_SIM_TIME_PS) with auto shutoff 1e-7 — the GATE task must show in its
  log that it ended by SHUTOFF, not the time cap, before tasks 1-5 launch.
- Box y = 8 um (converged, trench-validated in job 124531), z-mult 5.42,
  optimization mesh dx = 50 nm — identical mesh to every N=80 anchor
  (user: do not change mesh).

Dispatch (gate first, then the rest — queue EMPTY between, section 6):
    SBATCH_MEM=180G ARRAY_TIME=08:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.trench_n150_full \
        --max-concurrent=3 --array-tasks=0-0        # convergence gate
    # gate PASS (resonance in window, ended by shutoff, MaxRSS < request):
    SBATCH_MEM=180G ARRAY_TIME=08:00:00 bash athena/deploy_athena.sh \
        --option3 --spec=runners.metal_mirror.trench_n150_full \
        --max-concurrent=3 --array-tasks=1-5
Output -> results/trench_n150_full/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

# ── Study knobs ─────────────────────────────────────────────────────────────────
N_PERIODS        = 150      # per side (~155 um grating; N=80 anchors were ~83 um)
TRENCH_LEN_UM    = 156.0    # spans the full grating (edges +/-78 um)
TRENCH_W_NM      = 800.0
TRENCH_H_NM      = 12000.0  # FULL-Z (user 2026-07-20): z domain is ~8.8 um, so
                            # 12 um passes clean through both z-PMLs — no
                            # artificial trench top/bottom inside the domain.
                            # (First batch 124551 ran h=4000; those trench-row
                            # .mat are preserved in results_h4000/ server-side.)
TRENCH_D_NM      = 1800.0   # measured optimum (job 124379)

BOX_Y_UM         = 8.0      # converged + trench-validated box
SCAN_WIDTH_NM    = 10.0     # user rule: 10-20 nm; resonance known per row
N_WL_POINTS      = 2001     # 5 pm sampling
N_2D_FREQ_POINTS = 41       # 0.25 nm plane spacing (<=125 pm off resonance).
                            # NOT more: 101 pts overflowed the MAT-5 4 GiB
                            # per-variable cap (gate job 124540: MatWriteError
                            # after a good solve); 41 keeps structs ~1.6 GB.

# 2D monitors get their OWN narrow window (per-row) so the recorded planes sit
# within ~30 pm of the true line (N=150 linewidths are only 60-110 pm; the old
# source-limits sampling put planes up to 125 pm off). Centers below:
# controls = MEASURED N=150 resonances (job 124551); trench rows = measured
# h=4um N=150 resonances (full-z drag margin covered by the 2.5 nm span).
MON2D_SPAN_NM = 2.5

# rows: (label, cavity_width_nm, apod_method, scan_center_nm [N=80 MEASURED],
#        mon2d_center_nm [N=150 MEASURED], trench)
ROWS = [
    ("W800 ctrl",      None,   "none",   1558.61, 1558.60, False),
    ("W800 +trench",   None,   "none",   1557.91, 1557.84, True),
    ("W1050 ctrl",     1050.0, "none",   1558.80, 1558.78, False),
    ("W1050 +trench",  1050.0, "none",   1558.11, 1558.03, True),
    ("apod10 ctrl",    None,   "linear", 1559.20, 1559.12, False),
    ("apod10 +trench", None,   "linear", 1558.52, 1558.45, True),
]

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM
BASE.spectral.n_2d_monitor_points = N_2D_FREQ_POINTS
BASE.monitors.record_2d_fields = True
BASE.scatterer.height_m = TRENCH_H_NM * 1e-9

_PML_CLEAR_NM = 1200.0
assert TRENCH_D_NM + 0.5 * TRENCH_W_NM + _PML_CLEAR_NM <= BOX_Y_UM * 1000.0 / 2.0
assert TRENCH_D_NM - 0.5 * TRENCH_W_NM >= 500.0 + (1050.0 - 800.0) / 2.0  # W1050 tips at 625

SPEC = SweepSpec(
    n_periods_each_side      = [N_PERIODS] * len(ROWS),
    cavity_width_nm          = [r[1] for r in ROWS],
    apod_method              = [r[2] for r in ROWS],
    n_apod_periods_each_side = [10] * len(ROWS),          # inert on 'none' rows
    center_wavelength_nm     = [r[3] for r in ROWS],
    monitor_2d_center_nm     = [r[4] for r in ROWS],
    monitor_2d_span_nm       = [MON2D_SPAN_NM] * len(ROWS),
    scatterer_shape          = ["rect"] * len(ROWS),
    scatterer_x_span_um      = [TRENCH_LEN_UM if r[5] else 0.0 for r in ROWS],
    scatterer_y_span_nm      = [TRENCH_W_NM if r[5] else 0.0 for r in ROWS],
    scatterer_y_nm           = [TRENCH_D_NM] * len(ROWS),
    scatterer_index          = [1.0] * len(ROWS),
    mode  = "zipped",
    label = "trench_n150_full",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, (name, *_rest) in enumerate(ROWS):
        print(f"  task {i}: {name}  (window center {ROWS[i][3]} nm)")
