"""nt80x_second_tooth, STEP=PROD (own module = own results dir and sweep list): peak T, Q_L, far field at each row's LOC lambda.

Study dir: runners/sweeps/ | Created 2026-10-22 | Job(s): Athena (see nt80x_second_tooth.py)
Run per task = max(60, 2 * 2Q_L/omega) ps from the LOC table (DELTA -206: ~752 ps). Production mesh (dy 57.7 nm) = the
stored control 265046's numerics, so peak T x gamma^2 compares directly; the far-field PATTERN is not mesh-converged there.
Dispatch: SBATCH_MEM=200G bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.nt80x_second_tooth_prod --array-tasks=0-2 --max-concurrent=3 --gpu=a100
"""
import os
os.environ["NT2_STEP"] = "PROD"
from runners.sweeps.nt80x_second_tooth import SPEC, BASE  # noqa: E402,F401
