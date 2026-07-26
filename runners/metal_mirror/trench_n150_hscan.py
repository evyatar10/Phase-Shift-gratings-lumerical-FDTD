"""Trench-height scan on the full N=150 W800 device (IGUM, parallel array).

Study dir: runners/metal_mirror/   |   Created 2026-07-25   |   Job(s): TBD
Purpose (user): the N=150 ladder has only h=350 (+0.0167, job 125289), h=4000
(+0.0461, 124551) and full-z (+0.0481, 124590) — all on Athena. Fill the curve
with 12 heights between 350 nm and full-z, launched IN PARALLEL on IGUM
(Athena login node is down 2026-07-25). Deliverable: peak T (dB) and Q vs
trench height.

IGUM runs native Lumerical 2026 R1.2 (Athena container is R1): T offsets
~0.004 and lambda offsets ~2 nm vs Athena were measured (igum/README.md
section 10), so the curve must be SELF-CONTAINED on IGUM — task 0 is the
in-study no-trench control and the ladder re-anchors 350 + full-z (12000)
alongside the 12 new heights. 15 tasks total, one array.

Heights are log-spaced (the knee is at low h: N=80 ladder 350:+0.0079 /
2000:+0.0159 / 4000:+0.0195), rounded, with 2000/4000 kept exactly for
cross-checks against the Athena anchors. Trench is z-centered on the core
(symmetric), like every trench so far.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 400, W800,
N = 150/side; expected resonances 1557.8-1558.6 nm on Athena numerics, plus
up to ~+-2 nm IGUM solver offset -> window 15 nm centered 1558.3
(1550.8-1565.8), 3001 pts = 5 pm sampling (stage-M density). Scalars only:
NO 2D field maps (stage-M already delivered maps; this study wants T and Q).

Dispatch (IGUM, split over both partitions with DISJOINT task ranges — same
byte-identical sweep_list, so no section-6 clobber; 2026-07-25 the A100 pool
was fully allocated with 69 pending, part-lumerical had 3/7 GPUs free):
    # guaranteed lane: coarse but complete ladder (ctrl+350+735+1200+2000+4000+12000)
    SBATCH_MEM=75G ARRAY_TIME=23:30:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.metal_mirror.trench_n150_hscan \
        --max-concurrent=4 --array-tasks=0,1,4,6,8,11,14
    # preempt-pool fill-in points (igum.conf switched to part-preempt/qos-preempt):
    SBATCH_MEM=110G ARRAY_TIME=23:30:00 bash igum/deploy_igum.sh \
        --option3 --spec=runners.metal_mirror.trench_n150_hscan \
        --max-concurrent=2 --array-tasks=2,3,5,7,9,10,12,13
LICENSE CEILING (MEASURED 2026-07-25, jobs 41767+42317+42325): one GPU solve
checks out SEVEN lum_fdtd_solve seats and only 42 of the 50 issued seats are
grantable (~8 reserved server-side) -> MAX 6 CONCURRENT SOLVES total across
all arrays/clusters; the 7th dies instantly with FlexNet -4 (bare "in run:").
Keep the sum of array throttles <= 6 and chain extra arrays with
--dependency=afterany. Preempt-node system libs come from WORK_DIR/scilibs
(see igum/README.md).
Output -> results/trench_n150_hscan/results/ (download to results_from_igum/).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

N_PERIODS     = 150
TRENCH_LEN_UM = 156.0        # full-grating span (stage-M geometry)
TRENCH_W_NM   = 800.0
TRENCH_D_NM   = 1800.0       # measured optimum (job 124379, re-confirmed full-z)

# THE knob. 0 = control (no trench); 350 + 12000 re-anchor the Athena endpoints
# on IGUM numerics; the 12 in between are the new points. 12000 = full-z
# (z domain ~8.8 um -> passes through both z-PMLs, stage-M convention).
HEIGHTS_NM = [0.0, 350.0,
              450.0, 575.0, 735.0, 940.0, 1200.0, 1550.0,
              2000.0, 2550.0, 3250.0, 4000.0, 5400.0, 6900.0,
              12000.0]

BOX_Y_UM       = 8.0         # stage-M numerics exactly
SCAN_WIDTH_NM  = 15.0        # widened from stage-M 10 to absorb the IGUM offset
N_WL_POINTS    = 2001        # 7.5 pm sampling (~12 pts across the ~90-100 pm
                             # N=150 FWHM); keeps port-DFT RAM ~60-70G so three
                             # tasks pack on a 230 GB part-lumerical node
SCAN_CENTER_NM = 1558.3

BASE = _common.build_ports_base()
BASE.y_span_override_m = BOX_Y_UM * 1e-6
BASE.spectral.n_wl_points = N_WL_POINTS
BASE.spectral.scan_width_nm = SCAN_WIDTH_NM

assert TRENCH_D_NM + 0.5 * TRENCH_W_NM + 1200.0 <= BOX_Y_UM * 1000.0 / 2.0
assert TRENCH_D_NM - 0.5 * TRENCH_W_NM >= 500.0

SPEC = SweepSpec(
    n_periods_each_side  = [N_PERIODS] * len(HEIGHTS_NM),
    center_wavelength_nm = [SCAN_CENTER_NM] * len(HEIGHTS_NM),
    scatterer_shape      = ["rect"] * len(HEIGHTS_NM),
    scatterer_x_span_um  = [TRENCH_LEN_UM if h > 0 else 0.0 for h in HEIGHTS_NM],
    scatterer_y_span_nm  = [TRENCH_W_NM if h > 0 else 0.0 for h in HEIGHTS_NM],
    scatterer_y_nm       = [TRENCH_D_NM] * len(HEIGHTS_NM),
    scatterer_index      = [1.0] * len(HEIGHTS_NM),
    scatterer_height_nm  = [h if h > 0 else 350.0 for h in HEIGHTS_NM],
    mode  = "zipped",
    label = "trench_n150_hscan",
)

if __name__ == "__main__":
    print(SPEC.describe())
    for i, h in enumerate(HEIGHTS_NM):
        name = "ctrl (no trench)" if h == 0 else f"trench h={h:.0f} nm"
        print(f"  task {i}: {name}")
