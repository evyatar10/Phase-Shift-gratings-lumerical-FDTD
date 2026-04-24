#!/bin/bash
#
# SLURM job: run an existing .fsp file with the Lumerical FDTD GPU engine.
# This is the Athena analog of hpc/jobs/run_fsp_job.sh (Zeus/PBS).
#
# Usage — pass FSP_FILE as an environment variable at sbatch:
#   sbatch --export=FSP_FILE="layout_yourfile.fsp" run_fsp_gpu.sh
#
# Or edit FSP_FILE below and submit directly:
#   sbatch run_fsp_gpu.sh
#
# Scale GPUs:
#   --gpus=1   single A100 (start here — benchmark first)
#   --gpus=4   4× A100 on one DGX node
#   --gpus=8   full DGX node (8× A100)
#   For --gpus > 1, NUM_MPI_RANKS below should match GPU count.
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

# Expose GPU devices inside the Pyxis/Enroot container.
# SLURM sets CUDA_VISIBLE_DEVICES to the allocated GPU indices;
# Pyxis uses NVIDIA_VISIBLE_DEVICES to decide which /dev/nvidia* to inject.
export NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"

# Host NVIDIA driver library path (driver 470 on Athena DGX nodes).
# Mounted into the container below so libcuda.so is findable at runtime.
NVIDIALIB_HOST="/usr/lib/x86_64-linux-gnu"
NVIDIALIB_CTR="/usr/local/nv_host_libs"

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
FSP_DIR="/home/evyatarrubin/bragg_sim_gpu/results/layouts"

# FSP_FILE can be set via --export when submitting, or hardcoded here as fallback:
FSP_FILE="${FSP_FILE:-layout_REPLACE_ME.fsp}"

CONTAINER="$HOME/containers/lumerical-2026R1.sqsh"
LUM_HOME="/opt/lumerical/v261"
ENGINE="${LUM_HOME}/bin/fdtd-engine-ompi-lcl"
LICENSE="11055@dgx-master"

# One MPI rank per GPU; increase to match --gpus= when running multi-GPU.
NUM_MPI_RANKS=1
# Threads per rank — use all allocated CPUs divided by rank count.
NTHREADS=$(( SLURM_CPUS_PER_TASK / NUM_MPI_RANKS ))
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Job:       ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPUs:      ${SLURM_GPUS}"
echo "CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "MPI ranks: ${NUM_MPI_RANKS}  Threads/rank: ${NTHREADS}"
echo "FSP dir:   ${FSP_DIR}"
echo "FSP file:  ${FSP_FILE}"
echo "Container: ${CONTAINER}"
echo "============================================================"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container not found: ${CONTAINER}"
    echo "Build it with:  bash hpc_gpu/container/build.sh"
    exit 1
fi


# ── Run the FDTD engine inside the container ──────────────────────────────────
# Pyxis/Enroot flags:
#   --container-image   : .sqsh file to use
#   --container-mounts  : host_path:container_path (colon-separated pairs)
#   --container-env     : environment variables to pass through
#   --container-workdir : working directory inside the container
#
# Lumerical flags:
#   -t N          : N OpenMP threads per MPI rank
#   -logall       : verbose solver output
#   -use-gpu-resources : enable CUDA GPU offload for 3D FDTD

srun \
    --container-image="${CONTAINER}" \
    --container-mounts="${FSP_DIR}:/work/layouts,${NVIDIALIB_HOST}:${NVIDIALIB_CTR}" \
    --container-workdir=/work/layouts \
    --ntasks="${NUM_MPI_RANKS}" \
    bash -c "
export LANG=C
export LC_ALL=C
# Make host libcuda.so visible to the Lumerical engine (driver not in container image).
export LD_LIBRARY_PATH=${NVIDIALIB_CTR}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
export ANSYSLMD_LICENSE_FILE='${LICENSE}'
export ANSYSLI_SERVERS='12325@172.25.0.12'
export ANSYS_APIP_DISABLE=1
# IB/EFA/RDMA not needed for single-node, single-rank job.
# Restricting to loopback+shared-memory avoids UCX IB fork crash on cleanup.
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_PROVIDER='^efa'
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mtl='^ofi'
export UCX_TLS=self,sm,tcp
export UCX_NET_DEVICES=lo

${ENGINE} -t ${NTHREADS} -logall -use-gpu-resources /work/layouts/${FSP_FILE}"

EXIT_CODE=$?

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"

exit "${EXIT_CODE}"
