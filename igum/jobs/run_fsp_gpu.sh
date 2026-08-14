#!/bin/bash
#
# SLURM job: run an existing .fsp file with the Lumerical FDTD GPU engine.
# Runs the .fsp with the NATIVE Lumerical FDTD engine (IGUM: no containers).
#
#
# Usage — pass FSP_FILE as an environment variable at sbatch:
#   sbatch --export=FSP_FILE="layout_yourfile.fsp" run_fsp_gpu.sh
#
# Or edit FSP_FILE below and submit directly:
#   sbatch run_fsp_gpu.sh
#
#SBATCH --job-name=lum_fdtd_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/lum_fdtd_gpu-%j.out
#SBATCH --error=logs/lum_fdtd_gpu-%j.out

ulimit -s unlimited
mkdir -p logs

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
# Per-run layout: each .fsp lives in its own folder ${RESULTS_ROOT}/<stem>/<stem>.fsp.
# Engine writes its outputs (.h5 monitors, .log) into the same per-stem folder.
RESULTS_ROOT="${WORK_DIR:-/home/evyatarrubin/research/bragg_sim_igum}/results"

# FSP_FILE can be set via --export when submitting, or hardcoded here as fallback:
FSP_FILE="${FSP_FILE:-layout_REPLACE_ME.fsp}"
FSP_STEM="${FSP_FILE%.fsp}"
FSP_DIR="${RESULTS_ROOT}/${FSP_STEM}"

# NATIVE Lumerical install on IGUM — containers are NOT supported here (no
# apptainer/singularity, docker daemon denied). 2026 R1.3 is OUR own
# RPM-extracted tree on the research volume; the admins' R1.2 stays at
# /apps/ansys/Lumerical-2026-R1.2/opt/lumerical/v261 as fallback.
# NOTE: the -ompi-lcl engine variant needs libmpi.so.40 which the RPM-extracted
# install does NOT ship — use the plain single-process engine (fine for 1 GPU).
LUM_HOME="/home/evyatarrubin/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261"
ENGINE="${LUM_HOME}/bin/fdtd-engine"
LMUTIL="${LUM_HOME}/licensingclient/linx64/lmutil"
# License values come from deploy_igum.sh via --export=ALL,ATHENA_LICENSE=...
# Defaults below let the script also be invoked manually with sbatch directly.
LICENSE="${ATHENA_LICENSE:-}"
INTERCONNECT="${ATHENA_INTERCONNECT:-}"

# Threads — use all allocated CPUs (single rank).
NTHREADS="${SLURM_CPUS_PER_TASK}"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}  Threads: ${NTHREADS}"
echo "FSP dir:   ${FSP_DIR}"
echo "FSP file:  ${FSP_FILE}"
echo "Lumerical: ${LUM_HOME} (native)"
echo "============================================================"
echo "--- nvidia-smi ---"
nvidia-smi || echo "(nvidia-smi not available on this node)"
echo "------------------"

if [[ ! -d "${LUM_HOME}" ]]; then
    echo "ERROR: native Lumerical install not found: ${LUM_HOME}"
    exit 1
fi

# ── Run the FDTD engine natively (no container) ──────────────────────────────
# Engine flags:
#   -t N               : OpenMP threads
#   -logall            : verbose solver log
#   -use-gpu-resources : enable CUDA GPU offload

# REQUIRE_GPU=1 (default) runs the 120s nvidia-smi watchdog that kills the
# engine if GPU memory stays near-zero (catches silent CPU fallback).
REQUIRE_GPU="${REQUIRE_GPU:-1}"

export LANG=C
export LC_ALL=C
export ANSYSLMD_LICENSE_FILE="${LICENSE}"
export ANSYSLI_SERVERS="${INTERCONNECT}"
export ANSYS_APIP_DISABLE=1
export LD_LIBRARY_PATH="${LUM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_PROVIDER='^efa'
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mtl='^ofi'
export UCX_TLS=self,sm,tcp
export UCX_NET_DEVICES=lo

# Pre-flight: verify the license server is reachable and lum_fdtd_solve is licensed.
# NOTE the Athena lore: lmstat can report -96 even when the license WORKS.
# On IGUM the FQDN resolves natively so lmstat should be reliable — but keep the
# check advisory-friendly: only hard-fail when the feature list is truly absent.
if [ "${REQUIRE_GPU}" = "1" ]; then
    LMSTAT_OUT=$("${LMUTIL}" lmstat -a -c "${ANSYSLMD_LICENSE_FILE}" 2>&1)
    if ! echo "${LMSTAT_OUT}" | grep -q 'Users of lum_fdtd_solve'; then
        echo '============================================================'
        echo 'FATAL: lum_fdtd_solve not found in license pool.'
        echo 'Either the license server is unreachable or the pool is empty.'
        echo 'Available features:'
        echo "${LMSTAT_OUT}" | grep -E 'Users of' | head -20
        echo '============================================================'
        exit 2
    fi
    echo 'License pre-flight: lum_fdtd_solve confirmed present.'
fi

cd "${FSP_DIR}"
# Engine in background so we can attach an nvidia-smi watchdog
"${ENGINE}" -t "${NTHREADS}" -logall -use-gpu-resources "${FSP_DIR}/${FSP_FILE}" &
ENGINE_PID=$!

if [ "${REQUIRE_GPU}" = "1" ]; then
    (
        sleep 120
        if kill -0 ${ENGINE_PID} 2>/dev/null; then
            MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
            if [ -z "${MEM_USED}" ] || [ "${MEM_USED}" -lt 200 ]; then
                echo '============================================================'
                echo "FATAL: GPU appears unused 120s after engine start (mem=${MEM_USED}MiB)."
                echo 'Engine has silently fallen back to CPU. Killing it to avoid wasted hours.'
                echo 'Resubmit with REQUIRE_GPU=0 if a CPU run is acceptable.'
                echo '============================================================'
                kill -TERM ${ENGINE_PID} 2>/dev/null
                sleep 5
                kill -KILL ${ENGINE_PID} 2>/dev/null
            fi
        fi
    ) &
    WATCHDOG_PID=$!
fi

wait ${ENGINE_PID}
EXIT_CODE=$?
[ -n "${WATCHDOG_PID}" ] && kill ${WATCHDOG_PID} 2>/dev/null

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"

if [[ -n "${NTFY_TOPIC}" ]]; then
    _MINS=$(( SECONDS / 60 )); _SECS=$(( SECONDS % 60 ))
    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        _MSG="✓ Job ${SLURM_JOB_ID} done — ${FSP_FILE} — ${_MINS}m${_SECS}s"
    else
        _MSG="✗ Job ${SLURM_JOB_ID} FAILED (exit ${EXIT_CODE}) — ${FSP_FILE}"
    fi
    curl -s -H "Title: Bragg FDTD" -d "${_MSG}" \
        "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
fi

exit "${EXIT_CODE}"
