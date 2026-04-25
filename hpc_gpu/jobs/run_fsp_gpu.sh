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
FSP_DIR="/home/evyatarrubin/bragg_sim_gpu/results/layouts"

# FSP_FILE can be set via --export when submitting, or hardcoded here as fallback:
FSP_FILE="${FSP_FILE:-layout_REPLACE_ME.fsp}"

CONTAINER="$HOME/containers/lumerical-2026R1.sif"
LUM_HOME="/opt/lumerical/v261"
ENGINE="${LUM_HOME}/bin/fdtd-engine-ompi-lcl"
# License values come from deploy_athena.sh via --export=ALL,ATHENA_LICENSE=...
# Defaults below let the script also be invoked manually with sbatch directly.
LICENSE="${ATHENA_LICENSE:-11055@dgx-master}"
INTERCONNECT="${ATHENA_INTERCONNECT:-12325@172.25.0.12}"

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
echo "Container: ${CONTAINER}"
echo "============================================================"

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
# --bind        : mount FSP directory and custom hosts file into container
# --pwd         : working directory inside container
#
# Engine flags:
#   -t N               : OpenMP threads
#   -logall            : verbose solver log
#   -use-gpu-resources : enable CUDA GPU offload

# REQUIRE_GPU=1 (default) makes the script abort if the engine can't actually
# use the GPU — protects against silent CPU fallback when the FlexLM license
# pool lacks the GPU FDTD feature (lum_fdtd_solve_gpu). Set REQUIRE_GPU=0 to
# allow CPU runs (only useful if you specifically want a CPU benchmark).
REQUIRE_GPU="${REQUIRE_GPU:-1}"

apptainer exec --nv \
    --bind "${FSP_DIR}:/work/layouts" \
    --bind "${HOSTS_FILE}:/etc/hosts" \
    --pwd /work/layouts \
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

# Pre-flight: confirm the license pool actually has a GPU FDTD feature.
# If it doesn't, the engine would silently CPU-fall-back. Catch it now.
if [ \"\$REQUIRE_GPU\" = \"1\" ]; then
    LMUTIL=/ansys_inc/v261/licensingclient/linx64/lmutil
    LMSTAT_OUT=\$(\$LMUTIL lmstat -a -c \"\$ANSYSLMD_LICENSE_FILE\" 2>&1)
    if ! echo \"\$LMSTAT_OUT\" | grep -qE 'Users of (lum_fdtd_solve_gpu|lum_fdtd_gpu|fdtd_gpu)\b'; then
        echo '============================================================'
        echo 'FATAL: license pool has no GPU FDTD feature (lum_fdtd_solve_gpu).'
        echo 'The engine would run on CPU at ~5–10x slower despite --gpus=1.'
        echo 'Available Lumerical features in this pool:'
        echo \"\$LMSTAT_OUT\" | grep -E 'Users of lum_' | head -20
        echo
        echo 'Ask Technion CIS to add lum_fdtd_solve_gpu (Speos already has'
        echo 'speos_solver_gpu in this pool — same precedent).'
        echo 'To override and run on CPU anyway, resubmit with REQUIRE_GPU=0.'
        echo '============================================================'
        exit 2
    fi
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

exit "${EXIT_CODE}"
