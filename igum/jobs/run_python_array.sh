#!/bin/bash
#
# SLURM job array: run the Python/lumapi pipeline as one task per sweep value.
# Each task gets its own GPU + CPUs and runs ONE simulation end-to-end
# (build → solve → analyze → save .mat).
#
# Designed for parameter sweeps where the value list is generated locally by
# athena/scripts/build_sweep_list.py and uploaded to /work/data/sweep_list.txt.
#
# Companion to run_python_gpu.sh (single sequential job). Runs NATIVELY
# (no container): license env + Qt-offscreen + WORK_DIR-derived paths;
# only the dispatch (athena_run_one.py vs athena_run.py) and logging differ.
#
#
# Usage (submitted by deploy_athena.sh --option3):
#   sbatch --array=0-N-1%K \
#          --gpus=1 --cpus-per-task=8 \
#          --export=ALL,SWEEP_KIND=shift,ATHENA_LICENSE=...,ATHENA_INTERCONNECT=... \
#          jobs/run_python_array.sh
#
#SBATCH --job-name=lum_pipeline_array
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/lum_array-%A_%a.out
#SBATCH --error=logs/lum_array-%A_%a.out

ulimit -s unlimited
mkdir -p logs

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
WORK_DIR="${WORK_DIR:-/home/evyatarrubin/research/bragg_sim_igum}"
PROJECT_DIR="${WORK_DIR}/project"
SCRIPTS_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

# NATIVE Lumerical install on IGUM — containers are NOT supported here (no
# apptainer/singularity, docker daemon denied). 2026 R1.3 is OUR own
# RPM-extracted tree on the research volume; the admins' R1.2 stays at
# /apps/ansys/Lumerical-2026-R1.2/opt/lumerical/v261 as fallback.
# (numpy/scipy in the bundled python + system libgfortran verified working.)
LUM_HOME="/home/evyatarrubin/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261"

LICENSE="${ATHENA_LICENSE:-}"
INTERCONNECT="${ATHENA_INTERCONNECT:-}"

# REQUIRE_GPU=1 makes athena_run_one.py exit non-zero if GPU resource setup
# fails — strongly recommended for parallel sweeps so a silent CPU fallback
# doesn't waste GPU-task allocations.
REQUIRE_GPU="${REQUIRE_GPU:-1}"

# Sweep dispatch — these env vars must be passed via sbatch --export by deploy_igum.sh.
SWEEP_KIND="${SWEEP_KIND:-}"
SWEEP_LIST="${SWEEP_LIST:-${WORK_DIR}/data/sweep_list.txt}"
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
echo "Lumerical:  ${LUM_HOME} (native)"
echo "============================================================"
nvidia-smi || echo "(nvidia-smi not available)"

if [[ ! -d "${LUM_HOME}" ]]; then
    echo "ERROR: native Lumerical install not found: ${LUM_HOME}"
    exit 1
fi

# ── Native execution (no container) ──────────────────────────────────────────
export LANG=C
export LC_ALL=C
export ANSYSLMD_LICENSE_FILE="${LICENSE}"
export ANSYSLI_SERVERS="${INTERCONNECT}"
export ANSYS_APIP_DISABLE=1
# Headless CAD: IGUM nodes have no Xvfb; Qt offscreen verified working
# (lumapi session + geometry ops tested on igum-login1, 2026-07-05).
export QT_QPA_PLATFORM=offscreen
# athena_run_one.py derives all paths from WORK_DIR and LUMAPI_PATH.
export WORK_DIR
export LUMAPI_PATH="${LUM_HOME}/api/python/lumapi.py"
# scilibs: libgfortran.so.5 (+libquadmath) for scipy — present on igum-login1
# and ece-alecohen1 but MISSING on the part-preempt compute nodes (ece-ykasten1
# killed job 41776 in 2 s on `import scipy.io`, 2026-07-25). Staged from the
# login node into the shared research volume.
export LD_LIBRARY_PATH="${LUM_HOME}/lib:${WORK_DIR}/scilibs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# fdtd-solutions resets LD_LIBRARY_PATH to $FDTD_LD_LIBRARY_PATH:$LUMERICAL_LD_LIBRARY_PATH
# at relaunch — capture the current value so the reset restores it (same
# mechanism as run_python_gpu.sh; was missing here vs the Athena array script).
export LUMERICAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
# Force-load Lumerical's TBB malloc interceptor — prevents the intermittent
# "free(): invalid pointer" crash when lumapi spawns fdtd-solutions.
export LD_PRELOAD="${LUM_HOME}/lib/libtbbmalloc.so.2:${LUM_HOME}/lib/libtbbmalloc_proxy.so.2"
# Fabric autodetect guards (same values proven in the Athena container).
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_PROVIDER='^efa'
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mtl='^ofi'
export UCX_TLS=self,sm,tcp
export UCX_NET_DEVICES=lo
export SWEEP_KIND SWEEP_LIST SWEEP_PARAM SWEEP_FIXED_DZ SWEEP_FIXED_CELLS SWEEP_SPEC_MODULE
export SWEEP_INDEX="${SLURM_ARRAY_TASK_ID}"
export REQUIRE_GPU
export KEEP_H5="${KEEP_H5:-0}"
# LOCKED_LAMBDA_FILE — when set by deploy_igum.sh (chained-prelim sweeps),
# athena_run_one._run_kind_spec reads this JSON sidecar and overrides
# cfg.spectral.center_wavelength_m before running the per-task sim.
export LOCKED_LAMBDA_FILE="${LOCKED_LAMBDA_FILE:-}"

cd "${WORK_DIR}"
"${LUM_HOME}/python/bin/python" -u "${WORK_DIR}/scripts/athena_run_one.py"

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
