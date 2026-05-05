#!/bin/bash
#
# SLURM job array: run a list of .fsp files in parallel on GPU, with
# license-throttling to avoid exhausting the ANSYS license server.
#
# Designed for parameter sweeps (innermost-shift, inner-size, etc.) where the
# .fsp files are generated locally by local_save_fsp.py and uploaded.
#
# Usage:
#   1. Generate .fsp files locally, collect their remote basenames in a file:
#        bash hpc_gpu/deploy_athena.sh --option1 --preset sweep_shift
#      This creates fsp_list.txt on Athena with one filename per line.
#
#   2. Submit the array:
#        sbatch --array=0-<N-1>%<K> run_fsp_gpu_array.sh
#      where N = total number of .fsp files, K = max concurrent jobs.
#
#      K should not exceed the number of FDTD engine seats on your license.
#      Run  lmutil lmstat -a -c 1055@132.68.48.51  to find the seat count.
#      A safe default is K=4 unless you know otherwise.
#
# Example (50 .fsp files, max 4 running at once):
#   sbatch --array=0-49%4 run_fsp_gpu_array.sh
#
#SBATCH --job-name=lum_sweep_gpu
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/lum_sweep_gpu-%A_%a.out
#SBATCH --error=logs/lum_sweep_gpu-%A_%a.out

ulimit -s unlimited
mkdir -p logs

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
# Per-run layout: each .fsp lives in its own folder ${RESULTS_ROOT}/<stem>/<stem>.fsp.
# fsp_list.txt at ${RESULTS_ROOT}/fsp_list.txt holds one stem per line.
RESULTS_ROOT="/home/evyatarrubin/bragg_sim_gpu/results"
FSP_LIST="${RESULTS_ROOT}/fsp_list.txt"

CONTAINER="$HOME/containers/lumerical-2026R1.sif"
LUM_HOME="/opt/lumerical/v261"
ENGINE="${LUM_HOME}/bin/fdtd-engine-ompi-lcl"
LICENSE="${ATHENA_LICENSE:-11055@dgx-master}"
INTERCONNECT="${ATHENA_INTERCONNECT:-12325@172.25.0.12}"

NTHREADS="${SLURM_CPUS_PER_TASK}"
NVML_TRAMP="${HOME}/nvml_tramp"
# ─────────────────────────────────────────────────────────────────────────────

# Select this task's .fsp file (0-indexed)
if [[ ! -f "${FSP_LIST}" ]]; then
    echo "ERROR: fsp_list.txt not found at ${FSP_LIST}"
    echo "Run deploy_athena.sh --option1 --preset <preset> to create it."
    exit 1
fi

FSP_STEM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${FSP_LIST}")
if [[ -z "${FSP_STEM}" ]]; then
    echo "ERROR: No entry at line $((SLURM_ARRAY_TASK_ID + 1)) of ${FSP_LIST}"
    exit 1
fi
FSP_FILE="${FSP_STEM}.fsp"
FSP_DIR="${RESULTS_ROOT}/${FSP_STEM}"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container not found: ${CONTAINER}"
    exit 1
fi

# Bind custom hosts file so container can resolve lumerical-lm.ece.technion.ac.il → 132.68.48.51
HOSTS_FILE="${HOME}/hosts_lum"
if [[ ! -f "${HOSTS_FILE}" ]]; then
    cp /etc/hosts "${HOSTS_FILE}"
    echo "132.68.48.51 lumerical-lm.ece.technion.ac.il lumerical-lm" >> "${HOSTS_FILE}"
fi

echo "============================================================"
echo "Array job: ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES}"
echo "Threads:   ${NTHREADS}"
echo "FSP file:  ${FSP_FILE}"
echo "Container: ${CONTAINER}"
echo "============================================================"

REQUIRE_GPU="${REQUIRE_GPU:-1}"

TRAMP_BIND=""
if [[ -n "${NVML_TRAMP}" && -d "${NVML_TRAMP}" && -f "${NVML_TRAMP}/libnvidia-ml.so.1" ]]; then
    TRAMP_BIND="--bind ${NVML_TRAMP}:/nvml_tramp"
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
if [ -f /nvml_tramp/libnvidia-ml.so.1 ]; then
    export LD_LIBRARY_PATH=\"/nvml_tramp:\${LD_LIBRARY_PATH}\"
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

if [ \"\$REQUIRE_GPU\" = \"1\" ]; then
    LMUTIL=/ansys_inc/v261/licensingclient/linx64/lmutil
    LMSTAT_OUT=\$(\$LMUTIL lmstat -a -c \"\$ANSYSLMD_LICENSE_FILE\" 2>&1)
    if ! echo \"\$LMSTAT_OUT\" | grep -q 'Users of lum_fdtd_solve'; then
        echo '============================================================'
        echo 'FATAL: lum_fdtd_solve not found — license server unreachable or pool empty.'
        echo 'Available features:'
        echo \"\$LMSTAT_OUT\" | grep -E 'Users of' | head -20
        echo '============================================================'
        exit 2
    fi
fi

${ENGINE} -t ${NTHREADS} -logall -use-gpu-resources /work/layouts/${FSP_FILE} &
ENGINE_PID=\$!

if [ \"\$REQUIRE_GPU\" = \"1\" ]; then
    (
        sleep 120
        if kill -0 \$ENGINE_PID 2>/dev/null; then
            MEM_USED=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
            if [ -z \"\$MEM_USED\" ] || [ \"\$MEM_USED\" -lt 200 ]; then
                echo \"FATAL: GPU unused 120s after engine start (mem=\${MEM_USED}MiB) — silent CPU fallback. Killing.\"
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
        _MSG="✓ Array task ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}] done — ${FSP_FILE} — ${_MINS}m${_SECS}s"
    else
        _MSG="✗ Array task ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}] FAILED (exit ${EXIT_CODE}) — ${FSP_FILE}"
    fi
    curl -s -H "Title: Bragg FDTD" -d "${_MSG}" \
        "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
fi

exit "${EXIT_CODE}"
