"""z-sym-OFF control for the flush-top q3db ladder — anomaly discriminator.

Study dir: runners/metal_mirror/   |   Created 2026-08-07   |   Job(s): TBD
Purpose: the trench_flush_q3db ladder (Athena 128925, N167-169 measured) came
out with resonance + stopband ~+1.6 nm RED of the stored no-trench ctrl
(1560.602 vs 1559.001) — impossible for pure dielectric removal, and the scene
dump diff vs ctrl/full-z is clean except the intended trench z + z-BC/mesh
flags. Two candidate causes with opposite implications: (a) z-sym-OFF /
force-symmetric-z-mesh numerics offset AT CORR 325 (the corr-400 null,
si_substrate_check row 0 dlam 5 pm, may not transfer), or (b) real physics of
the broken z-mirror (parity hybridization). This ONE rerun discriminates:
ctrl C325 N165 with z-sym OFF — reproduces 1559.001 => (b), ladder stands;
lands near 1560.6 => (a), numerics offset, re-anchor the family.
Rerun justification (add-study step 0): symmetry/BC numerics changed vs the
stored baseline (z-sym OFF + force symmetric z mesh 0) AND a §2 anomaly is
open — exactly the allowed control case. Stored point: IGUM 48458-era,
result_N165_TM_avg_C325_Ybox8p0_Zbox8p8.mat, T 0.4906 / lam 1559.001.

Physics line (CLAUDE.md section 4): TM h350, pitch 516.83, corr 325, N=165/side,
no trench; expected resonance 1559.0 (or ~1560.6 if numerics offset), window
20 nm centered 1559.5 (1549.5-1569.5, 4001 pts) — ladder numerics exactly.

Dispatch on IGUM (keeps Athena's sweep_list untouched while 129103 runs):
    SBATCH_MEM=160G bash igum/deploy_igum.sh \
        --option3 --spec=runners.metal_mirror.trench_flush_q3db_ctrl
Output -> results/trench_flush_q3db_ctrl/results/.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

BASE = _common.build_ports_base()
BASE.y_span_override_m = 8.0e-6
BASE.spectral.n_wl_points = 4001
BASE.spectral.scan_width_nm = 20.0
BASE.symmetry.use_z_symmetry = False   # THE knob under test (vs stored z-sym ON)

SPEC = SweepSpec(
    corrugation_depth_nm = [325.0],
    n_periods_each_side  = [165],
    center_wavelength_nm = [1559.5],
    scatterer_shape      = ["rect"],
    scatterer_x_span_um  = [0.0],      # spans 0 = no-scatterer control
    scatterer_y_span_nm  = [0.0],
    scatterer_y_nm       = [1800.0],
    scatterer_index      = [1.0],
    mode  = "zipped",
    label = "trench_flush_q3db_ctrl",
)

if __name__ == "__main__":
    print(SPEC.describe())
    print("ctrl C325 N165, z-sym OFF: discriminates numerics-offset vs real flush physics")
