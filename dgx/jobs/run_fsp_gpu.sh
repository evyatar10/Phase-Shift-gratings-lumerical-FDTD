#!/bin/bash
#
# SLURM job: run an existing .fsp file with the Lumerical FDTD GPU engine.
# Uses apptainer exec --nv (not Pyxis/srun) so GPU devices and driver libs
# are injected automatically without manual mount hacks.
#
# Usage — pass FSP_FILE as an environment variable at sbatch:
#   sbatch --export=FSP_FILE="layout_yourfile.fsp" run_fsp_gpu.sh
#
# Or edit FSP_FILE below and submit directly:
#   sbatch run_fsp_gpu.sh
#
#SBATCH --job-name=lum_fdtd_gpu
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/lum_fdtd_gpu-%j.out
#SBATCH --error=logs/lum_fdtd_gpu-%j.out

ulimit -s unlimited
mkdir -p logs

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
# Per-run layout: each .fsp lives in its own folder ${RESULTS_ROOT}/<stem>/<stem>.fsp.
# Engine writes its outputs (.h5 monitors, .log) into the same per-stem folder.
RESULTS_ROOT="/home/evyatarrubin/bragg_sim_gpu/results"

# FSP_FILE can be set via --export when submitting, or hardcoded here as fallback:
FSP_FILE="${FSP_FILE:-layout_REPLACE_ME.fsp}"
FSP_STEM="${FSP_FILE%.fsp}"
FSP_DIR="${RESULTS_ROOT}/${FSP_STEM}"

CONTAINER="$HOME/containers/lumerical-2026R1.sif"
LUM_HOME="/opt/lumerical/v261"
ENGINE="${LUM_HOME}/bin/fdtd-engine-ompi-lcl"
# License values come from deploy_dgx.sh via --export=ALL,ATHENA_LICENSE=...
# Defaults below let the script also be invoked manually with sbatch directly.
LICENSE="${DGX_LICENSE:-11055@dgx-master}"
INTERCONNECT="${DGX_INTERCONNECT:-12325@172.25.0.12}"

# Threads — use all allocated CPUs (single rank).
NTHREADS="${SLURM_CPUS_PER_TASK}"

# NVML trampoline: shim that stubs three symbols absent from Athena's R470 NVML
# (nvmlDeviceGetPcieLinkMaxSpeed, nvmlDeviceGetBusType, nvmlDeviceGetMemoryBusWidth).
# Without it, the dynamic linker aborts when loading liblumcudaquery.so.
# Build once with: bash ~/bragg_sim_gpu/jobs/build_nvml_tramp.sh
# Skip (set to "") only on newer-driver nodes (R535+, e.g. L40S or H200 partition).
NVML_TRAMP="${HOME}/nvml_tramp"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}  Threads: ${NTHREADS}"
echo "FSP dir:   ${FSP_DIR}"
echo "FSP file:  ${FSP_FILE}"
echo "Container: ${CONTAINER}"
echo "============================================================"
echo "--- nvidia-smi ---"
nvidia-smi || echo "(nvidia-smi not available on this node)"
echo "------------------"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container not found: ${CONTAINER}"
    exit 1
fi

# The license server hostname lumerical-lm.ece.technion.ac.il (132.68.48.51)
# is not in Athena's DNS. Bind a pre-built hosts file from home (NFS-shared).
HOSTS_FILE="${HOME}/hosts_lum"
if [[ ! -f "${HOSTS_FILE}" ]]; then
    cp /etc/hosts "${HOSTS_FILE}"
    echo "132.68.48.51 lumerical-lm.ece.technion.ac.il lumerical-lm" >> "${HOSTS_FILE}"
fi

# ── Run the FDTD engine via Apptainer ────────────────────────────────────────
# --nv          : auto-inject NVIDIA GPU devices + host driver libs (libcuda.so)
# --bind        : mount FSP directory, custom hosts file, and NVML trampoline
# --pwd         : working directory inside container
#
# Engine flags:
#   -t N               : OpenMP threads
#   -logall            : verbose solver log
#   -use-gpu-resources : enable CUDA GPU offload

# REQUIRE_GPU=1 (default) runs the 120s nvidia-smi watchdog that kills the
# engine if GPU memory stays near-zero (catches silent CPU fallback).
# Set REQUIRE_GPU=0 only if you intentionally want a CPU run on Athena.
REQUIRE_GPU="${REQUIRE_GPU:-1}"

# Build the --bind flag for the NVML trampoline (omit if directory absent or empty).
TRAMP_BIND=""
if [[ -n "${NVML_TRAMP}" && -d "${NVML_TRAMP}" && -f "${NVML_TRAMP}/libnvidia-ml.so.1" ]]; then
    TRAMP_BIND="--bind ${NVML_TRAMP}:/nvml_tramp"
    echo "NVML trampoline: ${NVML_TRAMP}"
else
    echo "NVML trampoline not found at ${NVML_TRAMP} — skipping (safe on R535+ nodes)"
fi

apptainer exec --nv \
    --bind "${FSP_DIR}:/work/layouts" \
    --bind "${HOSTS_FILE}:/etc/hosts" \
    ${TRAMP_BIND} \
    --pwd /work/layouts \
    "${CONTAINER}" \
    bash -c "
export LANG=C
export LC_ALL=C
# NVML trampoline: prepend /nvml_tramp so our libnvidia-ml.so.1 is found before
# /.singularity.d/libs (Apptainer --nv injection). On R535+ nodes the trampoline
# directory won't exist, so LD_LIBRARY_PATH stays unchanged.
if [ -f /nvml_tramp/libnvidia-ml.so.1 ]; then
    export LD_LIBRARY_PATH=\"/nvml_tramp:\${LD_LIBRARY_PATH}\"
    echo '[nvml_tramp] activated: /nvml_tramp prepended to LD_LIBRARY_PATH'
fi
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

# Pre-flight: verify the license server is reachable and lum_fdtd_solve is licensed.
# Note: no separate GPU FDTD feature is required — lum_fdtd_solve covers GPU runs
# (GPU tasks just consume more seats based on SM count).
if [ \"\$REQUIRE_GPU\" = \"1\" ]; then
    LMUTIL=/ansys_inc/v261/licensingclient/linx64/lmutil
    LMSTAT_OUT=\$(\$LMUTIL lmstat -a -c \"\$ANSYSLMD_LICENSE_FILE\" 2>&1)
    if ! echo \"\$LMSTAT_OUT\" | grep -q 'Users of lum_fdtd_solve'; then
        echo '============================================================'
        echo 'FATAL: lum_fdtd_solve not found in license pool.'
        echo 'Either the license server is unreachable or the pool is empty.'
        echo 'Available features:'
        echo \"\$LMSTAT_OUT\" | grep -E 'Users of' | head -20
        echo '============================================================'
        exit 2
    fi
    echo 'License pre-flight: lum_fdtd_solve confirmed present.'
fi

# Engine in background so we can attach an nvidia-smi watchdog
${ENGINE} -t ${NTHREADS} -logall -use-gpu-resources /work/layouts/${FSP_FILE} &
ENGINE_PID=\$!

if [ \"\$REQUIRE_GPU\" = \"1\" ]; then
    (
        sleep 120
        if kill -0 \$ENGINE_PID 2>/dev/null; then
            MEM_USED=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
            if [ -z \"\$MEM_USED\" ] || [ \"\$MEM_USED\" -lt 200 ]; then
                echo '============================================================'
                echo \"FATAL: GPU appears unused 120s after engine start (mem=\${MEM_USED}MiB).\"
                echo 'Engine has silently fallen back to CPU. Killing it to avoid wasted hours.'
                echo 'Resubmit with REQUIRE_GPU=0 if a CPU run is acceptable.'
                echo '============================================================'
                kill -TERM \$ENGINE_PID 2>/dev/null
                sleep 5
                kill -KILL \$ENGINE_PID 2>/dev/null
            fi
        fi
    ) &
    WATCHDOG_PID=\$!
fi

wait \$ENGINE_PID
ENGINE_EXIT=\$?
[ -n \"\$WATCHDOG_PID\" ] && kill \$WATCHDOG_PID 2>/dev/null
exit \$ENGINE_EXIT
"

EXIT_CODE=$?

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
