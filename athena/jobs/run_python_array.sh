#!/bin/bash
#
# SLURM job array: run the Python/lumapi pipeline as one task per sweep value.
# Each task gets its own GPU + CPUs and runs ONE simulation end-to-end
# (build → solve → analyze → save .mat).
#
# Designed for parameter sweeps where the value list is generated locally by
# athena/scripts/build_sweep_list.py and uploaded to /work/data/sweep_list.txt.
#
# Companion to run_python_gpu.sh (single sequential job). All container
# plumbing — Xvfb, LD_PRELOAD, license env, OpenMPI/UCX tunings — is
# identical; only the dispatch (athena_run_one.py vs athena_run.py)
# and per-task logging differ.
#
# Usage (submitted by deploy_athena.sh --option3):
#   sbatch --array=0-N-1%K \
#          --gpus=1 --cpus-per-task=8 \
#          --export=ALL,SWEEP_KIND=shift,ATHENA_LICENSE=...,ATHENA_INTERCONNECT=... \
#          jobs/run_python_array.sh
#
#SBATCH --job-name=lum_pipeline_array
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/lum_array-%A_%a.out
#SBATCH --error=logs/lum_array-%A_%a.out

ulimit -s unlimited
mkdir -p logs

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
WORK_DIR="/home/evyatarrubin/bragg_sim_athena"
PROJECT_DIR="${WORK_DIR}/project"
SCRIPTS_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

CONTAINER="$HOME/containers/lumerical-2026R1.sif"
LUM_HOME="/opt/lumerical/v261"

LICENSE="${ATHENA_LICENSE:-}"
INTERCONNECT="${ATHENA_INTERCONNECT:-}"

# REQUIRE_GPU=1 makes athena_run_one.py exit non-zero if GPU resource setup
# fails — strongly recommended for parallel sweeps so a silent CPU fallback
# doesn't waste GPU-task allocations.
REQUIRE_GPU="${REQUIRE_GPU:-1}"

# NVML trampoline (used on dgx/R470) is intentionally NOT used here.
# All Athena GPU partitions run R570+ drivers that already export every
# NVML symbol Lumerical 2026R1 needs. Mounting the trampoline corrupts
# CUDA init on the newer driver (verified empirically — job 76907,
# 2026-05-05, all GPUs failed with cudaGetDeviceCount).

# Scientific libs (libgfortran, libquadmath) needed by scipy/numpy.
# Container's base image (CUDA devel) lacks libgfortran. We supply it via a
# user-maintained directory bound into /scilibs and appended to LD_LIBRARY_PATH.
# This is INDEPENDENT of the (legacy) NVML trampoline mechanism.
SCILIBS="${HOME}/scilibs"

# Sweep dispatch — these env vars must be passed via sbatch --export by deploy_athena.sh.
SWEEP_KIND="${SWEEP_KIND:-}"
SWEEP_LIST="${SWEEP_LIST:-/work/data/sweep_list.txt}"
SWEEP_PARAM="${SWEEP_PARAM:-}"
SWEEP_FIXED_DZ="${SWEEP_FIXED_DZ:-}"
SWEEP_FIXED_CELLS="${SWEEP_FIXED_CELLS:-}"
SWEEP_SPEC_MODULE="${SWEEP_SPEC_MODULE:-}"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}" "${LOGS_DIR}"

echo "============================================================"
echo "Array:      ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "GPU:        ${SLURM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "CPUs:       ${SLURM_CPUS_PER_TASK}"
echo "SWEEP_KIND: ${SWEEP_KIND}"
echo "SWEEP_LIST: ${SWEEP_LIST}"
echo "SWEEP_INDEX:${SLURM_ARRAY_TASK_ID}"
echo "REQUIRE_GPU:${REQUIRE_GPU}"
echo "Container:  ${CONTAINER}"
echo "============================================================"
nvidia-smi || echo "(nvidia-smi not available)"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container not found: ${CONTAINER}"
    exit 1
fi

# Bind hosts file so container resolves lumerical-lm.ece.technion.ac.il (FlexLM
# returns this hostname during the licensing handshake even via dgx-master).
HOSTS_FILE="${HOME}/hosts_lum"
if [[ ! -f "${HOSTS_FILE}" ]]; then
    cp /etc/hosts "${HOSTS_FILE}"
    echo "132.68.48.51 lumerical-lm.ece.technion.ac.il lumerical-lm" >> "${HOSTS_FILE}"
fi

apptainer exec --nv \
    --bind "${PROJECT_DIR}:/work/project" \
    --bind "${SCRIPTS_DIR}:/work/scripts" \
    --bind "${DATA_DIR}:/work/data" \
    --bind "${RESULTS_DIR}:/work/results" \
    --bind "${LOGS_DIR}:/work/logs" \
    --bind "${HOSTS_FILE}:/etc/hosts" \
    --bind "${SCILIBS}:/scilibs" \
    --pwd /work \
    "${CONTAINER}" \
    bash -c "
export LANG=C
export LC_ALL=C
# Strip /usr/local/cuda/compat* from LD_LIBRARY_PATH. The container ships a
# CUDA 12.2 forward-compat shim (libcuda.so.1) intended for hosts running
# R470 drivers. On Athena the host driver is R570/R595 — newer than the
# compat shim — so loading the shim first causes:
#   cudaGetDeviceCount Failed: unsupported display driver / cuda driver combination
# With the shim removed, Apptainer's --nv injection of the host libcuda
# (via /.singularity.d/libs) is used directly. No version skew.
export LD_LIBRARY_PATH=\"\$(echo \"\${LD_LIBRARY_PATH}\" | tr ':' '\\n' | grep -v '^/usr/local/cuda/compat' | paste -sd: -)\"
export LUMERICAL_LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}\"
export LD_PRELOAD=\"/opt/lumerical/v261/lib/libtbbmalloc.so.2:/opt/lumerical/v261/lib/libtbbmalloc_proxy.so.2\"
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
export SWEEP_KIND='${SWEEP_KIND}'
export SWEEP_LIST='${SWEEP_LIST}'
export SWEEP_INDEX='${SLURM_ARRAY_TASK_ID}'
export SWEEP_PARAM='${SWEEP_PARAM}'
export SWEEP_FIXED_DZ='${SWEEP_FIXED_DZ}'
export SWEEP_FIXED_CELLS='${SWEEP_FIXED_CELLS}'
export SWEEP_SPEC_MODULE='${SWEEP_SPEC_MODULE}'
export REQUIRE_GPU='${REQUIRE_GPU}'
export KEEP_H5=\"\${KEEP_H5:-0}\"
# Append /scilibs (libgfortran + libquadmath) to LD_LIBRARY_PATH so scipy/numpy
# imports succeed. Use SUFFIX (not prefix) so host driver libs injected by --nv
# still win for libnvidia-ml/libcuda.
export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:/scilibs\"
# LOCKED_LAMBDA_FILE — when set by deploy_athena.sh (chained-prelim sweeps),
# athena_run_one._run_kind_spec reads this JSON sidecar and overrides
# cfg.spectral.center_wavelength_m before running the per-task sim.
export LOCKED_LAMBDA_FILE='${LOCKED_LAMBDA_FILE:-}'

Xvfb :99 -screen 0 1024x768x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
XVFB_PID=\$!
trap 'kill \$XVFB_PID 2>/dev/null; wait \$XVFB_PID 2>/dev/null' EXIT
export DISPLAY=:99
sleep 1

/opt/lumerical/v261/python/bin/python /work/scripts/athena_run_one.py
PY_RC=\$?
echo \"[wrapper] python exit code: \$PY_RC\"
exit \$PY_RC"

EXIT_CODE=$?

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"

if [[ -n "${NTFY_TOPIC}" ]]; then
    _MINS=$(( SECONDS / 60 )); _SECS=$(( SECONDS % 60 ))
    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        _MSG="✓ Array task ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}] done — ${SWEEP_KIND} — ${_MINS}m${_SECS}s"
    else
        _MSG="✗ Array task ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}] FAILED (exit ${EXIT_CODE}) — ${SWEEP_KIND}"
    fi
    curl -s -H "Title: Bragg FDTD" -d "${_MSG}" \
        "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
fi

exit "${EXIT_CODE}"
