"""Stage A — baseline: prelim resonance find -> lambda-locked complex-far-field run.

Study dir: runners/scatterers/   |   Created 2026-07-12   |   Job(s): TBD
Purpose: (1) prelim ports-only task measures lambda_res at the program numerics and
writes the shared sidecar (_common.LAMBDA_SIDECAR); (2) the chained main task re-runs
the no-scatterer device with the scan window CENTERED on that lambda_res and records
the COMPLEX far field (side+top) — the E_baseline of the response matrix, and the
end-to-end proof of the complex-FF save path before stage C spends ~100 tasks.

Dispatch (ONE command — deploy chains prelim -> main via --dependency=afterok):
    PRELIM_TIME=00:30:00 bash athena/deploy_athena.sh --option3 --spec=runners.scatterers.scat_a_baseline
Output -> results/scat_a_baseline/results/. Any device change ==> rerun this stage.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from runners.sweeps.sweep_spec import SweepSpec
from runners.scatterers import _common

# Prelim: ports-only resonance find at identical mesh/box numerics.
PRELIM_BASE = _common.build_ports_base()
PRELIM_SPEC = SweepSpec(
    scatterer_radius_nm = [0.0],          # radius 0 = no scatterer drawn
    label = "scat_a_prelim",
)

# Written by the prelim task, read by every lambda-locked task in stages A/B/C/E.
LOCKED_LAMBDA_FILE = _common.LAMBDA_SIDECAR

# Main: the lambda-locked baseline with complex far field.
BASE = _common.build_ff_base()
SPEC = SweepSpec(
    scatterer_radius_nm = [0.0],
    label = "scat_a_baseline",
)

if __name__ == "__main__":
    print(PRELIM_SPEC.describe())
    print(SPEC.describe())
