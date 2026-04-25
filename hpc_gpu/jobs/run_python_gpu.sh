#!/bin/bash
#
# SLURM job: run the full Python/lumapi pipeline on Athena with GPU.
# This is the Athena analog of hpc/jobs/run_python_job.sh (Zeus/PBS).
#
# Runtime: apptainer exec --nv (matches run_fsp_gpu.sh — single image format,
# no Pyxis/Enroot dependence). The CUDA forward-compat shim is activated by
# the LD_LIBRARY_PATH ordering baked into the container's %environment block.
#
# The lumapi process (fdtd-solutions) requires a virtual X11 display even in
# headless mode because the Qt framework links against X11 libs at init time.
# We use xvfb-run (pre-installed in the container) to satisfy this.
#
# GPU is enabled inside the simulation via fdtd.setresource() in athena_run.py,
# not via a command-line flag — this is the correct lumapi method.
#
# Usage:
#   sbatch --export=ALL,RUN_SCRIPT=single_sim run_python_gpu.sh
#   sbatch --export=ALL,RUN_SCRIPT=sweep_shift run_python_gpu.sh
#   sbatch --export=ALL,RUN_SCRIPT=sweep_inner_size run_python_gpu.sh
#
# To hard-fail if GPU isn't actually used (recommended for sweeps):
#   sbatch --export=ALL,RUN_SCRIPT=sweep_shift,REQUIRE_GPU=1 run_python_gpu.sh
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
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

CONTAINER="$HOME/containers/lumerical-2026R1.sif"
LUM_HOME="/opt/lumerical/v261"

# License values come from deploy_athena.sh via --export=ALL,ATHENA_LICENSE=...
# Defaults below let the script also be invoked manually with sbatch directly.
LICENSE="${ATHENA_LICENSE:-11055@dgx-master}"
INTERCONNECT="${ATHENA_INTERCONNECT:-12325@172.25.0.12}"

# RUN_SCRIPT can be set via --export at sbatch, or hardcoded here as fallback.
# Valid values: single_sim | sweep_shift | sweep_inner_size
RUN_SCRIPT="${RUN_SCRIPT:-single_sim}"

# REQUIRE_GPU=1 makes athena_run.py exit non-zero if setresource(...,"GPU",True)
# fails — prevents long sweeps silently CPU-falling-back when the license tier
# is missing fdtd_gpu.
REQUIRE_GPU="${REQUIRE_GPU:-0}"
# ─────────────────────────────────────────────────────────────────────────────

# Make sure remote scratch dirs exist
mkdir -p "${DATA_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "GPU:        ${SLURM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "CPUs:       ${SLURM_CPUS_PER_TASK}"
echo "RUN_SCRIPT: ${RUN_SCRIPT}"
echo "REQUIRE_GPU:${REQUIRE_GPU}"
echo "Work dir:   ${WORK_DIR}"
echo "Container:  ${CONTAINER}"
echo "============================================================"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container not found: ${CONTAINER}"
    echo "Build it with:  bash hpc_gpu/container/build.sh"
    exit 1
fi

# License-server hostname lumerical-lm.ece.technion.ac.il (132.68.48.51) is
# not in Athena's DNS. Bind a pre-built hosts file from home (NFS-shared).
# Identical mechanism to run_fsp_gpu.sh — needed because the FlexLM server
# replies with this hostname during the licensing handshake even when the
# client connects via the dgx-master forward.
HOSTS_FILE="${HOME}/hosts_lum"
if [[ ! -f "${HOSTS_FILE}" ]]; then
    cp /etc/hosts "${HOSTS_FILE}"
    echo "132.68.48.51 lumerical-lm.ece.technion.ac.il lumerical-lm" >> "${HOSTS_FILE}"
fi

# ── Run the Python pipeline inside the container ──────────────────────────────
# --nv          : auto-inject NVIDIA GPU devices + host driver libs.
#                 The container's %environment puts /usr/local/cuda/compat
#                 first on LD_LIBRARY_PATH, so the cuda-compat-12-2 shim is
#                 loaded ahead of the host's R470 libcuda.so.1.
# --bind        : project, scripts, data, results, logs, hosts file.
# --pwd /work   : matches the path layout athena_run.py assumes.
#
# xvfb-run -a   : provides a virtual X11 display automatically (picks a free
#                 display number with -a). Without it, lumapi's Qt init fails
#                 with "QXcbConnection: Could not connect to display".
#
# OpenMPI/UCX env vars: copied from run_fsp_gpu.sh — these solved real
# Athena cluster errors (EFA fork-safety, OFI selection, UCX device binding)
# during FSP-path bring-up. The Python pipeline drives the same engine
# binary internally via lumapi, so it needs the same fixes.

apptainer exec --nv \
    --bind "${PROJECT_DIR}:/work/project" \
    --bind "${SCRIPTS_DIR}:/work/scripts" \
    --bind "${DATA_DIR}:/work/data" \
    --bind "${RESULTS_DIR}:/work/results" \
    --bind "${LOGS_DIR}:/work/logs" \
    --bind "${HOSTS_FILE}:/etc/hosts" \
    --pwd /work \
    "${CONTAINER}" \
    bash -c "
export LANG=C
export LC_ALL=C
export ANSYSLMD_LICENSE_FILE='${LICENSE}'
export ANSYSLI_SERVERS='${INTERCONNECT}'
export ANSYS_APIP_DISABLE=1
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_PROVIDER='^efa'
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mtl='^ofi'
export UCX_TLS=self,sm,tcp
export UCX_NET_DEVICES=lo
export RUN_SCRIPT='${RUN_SCRIPT}'
export REQUIRE_GPU='${REQUIRE_GPU}'
xvfb-run -a python /work/scripts/athena_run.py"

EXIT_CODE=$?

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo ""
    echo "Pipeline failed. Useful next steps:"
    echo "  - License: nc -vz dgx-master 11055 (FlexLM)"
    echo "             nc -vz dgx-master 12325 (Ansys interconnect)"
    echo "  - GPU:     check that 'nvidia-smi' inside the container reports"
    echo "             CUDA Version 12.x (not 11.4); if 11.4, the LD_LIBRARY_PATH"
    echo "             prefix in lumerical.def %environment is being overridden."
    echo "  - REQUIRE_GPU=${REQUIRE_GPU}: exit code 2 from athena_run.py means"
    echo "             setresource('FDTD',1,'GPU',True) failed (no fdtd_gpu seat?)."
fi
echo "============================================================"

exit "${EXIT_CODE}"
