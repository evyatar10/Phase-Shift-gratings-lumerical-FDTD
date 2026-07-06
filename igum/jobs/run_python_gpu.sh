#!/bin/bash
#
# SLURM job: run the full Python/lumapi pipeline on the real Athena cluster with GPU.
# This is the real-Athena analog of athena/jobs/run_python_gpu.sh (dgx-master).
#
# Runtime: NATIVE Lumerical 2026 R1.2 under /apps/ansys (IGUM: no containers).
#
#
#
#
#
#
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
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/lum_pipeline_gpu-%j.out
#SBATCH --error=logs/lum_pipeline_gpu-%j.out

ulimit -s unlimited
mkdir -p logs

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
WORK_DIR="${WORK_DIR:-/home/evyatarrubin/research/bragg_sim_igum}"
PROJECT_DIR="${WORK_DIR}/project"
SCRIPTS_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

# NATIVE Lumerical install on IGUM — containers are NOT supported on this
# cluster; Lumerical 2026 R1.2 is provided by the admins under /apps/ansys.
LUM_HOME="/apps/ansys/Lumerical-2026-R1.2/opt/lumerical/v261"

# License values come from deploy_athena.sh via --export=ALL,ATHENA_LICENSE=...
# Defaults below let the script also be invoked manually with sbatch directly.
LICENSE="${ATHENA_LICENSE:-}"
INTERCONNECT="${ATHENA_INTERCONNECT:-}"

# RUN_SCRIPT can be set via --export at sbatch, or hardcoded here as fallback.
# Valid values: bare module name from runners/single/ or convergence_testing/
# (auto-discovered by athena_run.py — anything with a top-level `run` callable).
RUN_SCRIPT="${RUN_SCRIPT:-run_simulation}"

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
echo "Lumerical:  ${LUM_HOME} (native)"
echo "============================================================"
echo "--- nvidia-smi ---"
nvidia-smi || echo "(nvidia-smi not available on this node)"
echo "------------------"

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
# athena_run.py derives all paths from WORK_DIR and LUMAPI_PATH.
export WORK_DIR
export LUMAPI_PATH="${LUM_HOME}/api/python/lumapi.py"
export LD_LIBRARY_PATH="${LUM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# fdtd-solutions resets LD_LIBRARY_PATH to $FDTD_LD_LIBRARY_PATH:$LUMERICAL_LD_LIBRARY_PATH
# at relaunch — capture the current value so the reset restores it instead of
# wiping it (same mechanism documented in the Athena container job).
export LUMERICAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
# Force-load Lumerical's TBB malloc interceptor before any other library —
# prevents the intermittent "free(): invalid pointer" crash when lumapi spawns
# fdtd-solutions subprocesses (same fix proven on Athena; absolute paths so
# ld.so resolves them regardless of subprocess LD_LIBRARY_PATH at start time).
export LD_PRELOAD="${LUM_HOME}/lib/libtbbmalloc.so.2:${LUM_HOME}/lib/libtbbmalloc_proxy.so.2"
# Fabric autodetect guards (same values proven on Athena).
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_PROVIDER='^efa'
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mtl='^ofi'
export UCX_TLS=self,sm,tcp
export UCX_NET_DEVICES=lo
export RUN_SCRIPT REQUIRE_GPU
# KEEP_H5=1 disables the per-iteration .h5 scratch cleanup (default: cleanup on).
export KEEP_H5="${KEEP_H5:-0}"
# LOCKED_LAMBDA_FILE — used by RUN_SCRIPT=compare_3d_field_prelim (writes the
# resonance λ here for the array half to read). Empty for unrelated runs.
export LOCKED_LAMBDA_FILE="${LOCKED_LAMBDA_FILE:-}"

cd "${WORK_DIR}"
"${LUM_HOME}/python/bin/python" "${WORK_DIR}/scripts/athena_run.py"

EXIT_CODE=$?

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo ""
    echo "Pipeline failed. Useful next steps:"
    echo "  - License: check ATHENA_LICENSE is set correctly in athena.conf"
    echo "  - GPU:     check that 'nvidia-smi' inside the container reports"
    echo "             CUDA Version 12.x or 13.x (R570+ driver expected)"
    echo "  - REQUIRE_GPU=${REQUIRE_GPU}: exit code 2 from athena_run.py means"
    echo "             setresource('FDTD',1,'GPU',True) failed (no fdtd_gpu seat?)."
fi
echo "============================================================"

if [[ -n "${NTFY_TOPIC}" ]]; then
    _MINS=$(( SECONDS / 60 )); _SECS=$(( SECONDS % 60 ))
    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        _MSG="✓ Job ${SLURM_JOB_ID} done — ${RUN_SCRIPT} — ${_MINS}m${_SECS}s"
    else
        _MSG="✗ Job ${SLURM_JOB_ID} FAILED (exit ${EXIT_CODE}) — ${RUN_SCRIPT}"
    fi
    curl -s -H "Title: Bragg FDTD" -d "${_MSG}" \
        "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
fi

exit "${EXIT_CODE}"
