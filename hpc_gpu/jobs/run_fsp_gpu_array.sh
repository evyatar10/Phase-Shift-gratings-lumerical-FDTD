#!/bin/bash
#
# SLURM job array: run a list of .fsp files in parallel on GPU, with
# license-throttling to avoid exhausting the ANSYS license server.
#
# Designed for parameter sweeps (ToothShift, inner-size, etc.) where the
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
FSP_DIR="/home/evyatarrubin/bragg_sim_gpu/results/layouts"
FSP_LIST="${FSP_DIR}/fsp_list.txt"   # one .fsp basename per line

CONTAINER="$HOME/containers/lumerical-2026R1.sqsh"
LUM_HOME="/opt/lumerical/v261"
ENGINE="${LUM_HOME}/bin/fdtd-engine-ompi-lcl"
LICENSE="11055@172.25.0.12"

NTHREADS="${SLURM_CPUS_PER_TASK}"
# ─────────────────────────────────────────────────────────────────────────────

# Select this task's .fsp file (0-indexed)
if [[ ! -f "${FSP_LIST}" ]]; then
    echo "ERROR: fsp_list.txt not found at ${FSP_LIST}"
    echo "Run deploy_athena.sh --option1 --preset <preset> to create it."
    exit 1
fi

FSP_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${FSP_LIST}")
if [[ -z "${FSP_FILE}" ]]; then
    echo "ERROR: No entry at line $((SLURM_ARRAY_TASK_ID + 1)) of ${FSP_LIST}"
    exit 1
fi

echo "============================================================"
echo "Array job: ${SLURM_ARRAY_JOB_ID}[${SLURM_ARRAY_TASK_ID}]"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       ${SLURM_GPUS}"
echo "Threads:   ${NTHREADS}"
echo "FSP file:  ${FSP_FILE}"
echo "============================================================"


srun \
    --container-image="${CONTAINER}" \
    --container-mounts="${FSP_DIR}:/work/layouts,/home/evyatarrubin/stub_apip.so:/stub_apip.so" \
    --container-workdir=/work/layouts \
    bash -c "export LANG=C && export LC_ALL=C && export ANSYSLMD_LICENSE_FILE='${LICENSE}' && export ANSYSLI_SERVERS='12325@172.25.0.12' && export LD_PRELOAD=/stub_apip.so && export OMPI_MCA_btl='^openib' && export OMPI_MCA_mtl='^ofi' && export UCX_TLS=tcp && echo 'ANSYSLMD_LICENSE_FILE='\$ANSYSLMD_LICENSE_FILE && echo 'ANSYSLI_SERVERS='\$ANSYSLI_SERVERS && ${ENGINE} -t ${NTHREADS} -logall -use-gpu-resources /work/layouts/${FSP_FILE}"

EXIT_CODE=$?

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"

exit "${EXIT_CODE}"
