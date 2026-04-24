#!/bin/bash
#
# SLURM job: run the full Python/lumapi pipeline on Athena with GPU.
# This is the Athena analog of hpc/jobs/run_python_job.sh (Zeus/PBS).
#
# The lumapi process (fdtd-solutions) requires a virtual X11 display even in
# headless mode because the Qt framework links against X11 libs at init time.
# We use xvfb-run (pre-installed in the container) to satisfy this.
#
# GPU is enabled inside the simulation via fdtd.setresource() in athena_run.py,
# not via a command-line flag — this is the correct lumapi method.
#
# Usage:
#   sbatch --export=RUN_SCRIPT=single_sim run_python_gpu.sh
#   sbatch --export=RUN_SCRIPT=sweep_shift run_python_gpu.sh
#   sbatch --export=RUN_SCRIPT=sweep_inner_size run_python_gpu.sh
#
#SBATCH --job-name=lum_pipeline_gpu
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/lum_pipeline_gpu-%j.out
#SBATCH --error=logs/lum_pipeline_gpu-%j.out

ulimit -s unlimited
mkdir -p logs

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
WORK_DIR="/home/evyatarrubin/bragg_sim_gpu"
PROJECT_DIR="${WORK_DIR}/project"
SCRIPTS_DIR="${WORK_DIR}/scripts"

CONTAINER="$HOME/containers/lumerical-2026R1.sqsh"
LUM_HOME="/opt/lumerical/v261"
LICENSE="11055@dgx-master"

# RUN_SCRIPT can be set via --export at sbatch, or hardcoded here as fallback.
# Valid values: single_sim | sweep_shift | sweep_inner_size
RUN_SCRIPT="${RUN_SCRIPT:-single_sim}"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       ${SLURM_GPUS}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "RUN_SCRIPT:${RUN_SCRIPT}"
echo "Work dir:  ${WORK_DIR}"
echo "Container: ${CONTAINER}"
echo "============================================================"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container not found: ${CONTAINER}"
    echo "Build it with:  bash hpc_gpu/container/build.sh"
    exit 1
fi

# ── Run the Python pipeline inside the container ──────────────────────────────
# xvfb-run -a provides a virtual X11 display automatically (picks a free display
# number with -a). The Python script uses lumapi which loads the Qt-based
# fdtd-solutions process; without a display it will fail with:
#   "QXcbConnection: Could not connect to display"
#
# All simulation output goes to /scratch/$USER/bragg_sim_gpu/ inside the container.
# The scratch path is mounted read-write from the actual Athena scratch filesystem.


srun \
    --container-image="${CONTAINER}" \
    --container-mounts="${PROJECT_DIR}:/work/project,${SCRIPTS_DIR}:/work/scripts,${WORK_DIR}/data:/work/data,${WORK_DIR}/results:/work/results,${WORK_DIR}/logs:/work/logs" \
    --container-workdir=/work \
    bash -c "export ANSYSLMD_LICENSE_FILE='${LICENSE}' && export RUN_SCRIPT='${RUN_SCRIPT}' && xvfb-run -a python /work/scripts/athena_run.py"

EXIT_CODE=$?

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo ""
    echo "Pipeline failed. Check the log for the fallback sbatch command."
    echo "Tip: if lumapi failed to acquire a license, run:"
    echo "  nc -vz 132.68.48.51 1055"
    echo "to verify the license server is reachable from this compute node."
fi
echo "============================================================"

exit "${EXIT_CODE}"
