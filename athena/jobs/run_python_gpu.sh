#!/bin/bash
#
# SLURM job: run the full Python/lumapi pipeline on the real Athena cluster with GPU.
# This is the real-Athena analog of athena/jobs/run_python_gpu.sh (dgx-master).
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
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/lum_pipeline_gpu-%j.out
#SBATCH --error=logs/lum_pipeline_gpu-%j.out

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
echo "--- nvidia-smi ---"
nvidia-smi || echo "(nvidia-smi not available on this node)"
echo "------------------"

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
#                 loaded ahead of the host's driver libcuda.so.1.
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
# fdtd-solutions resets LD_LIBRARY_PATH to \$FDTD_LD_LIBRARY_PATH:\$LUMERICAL_LD_LIBRARY_PATH.
# Both are empty in the container, wiping all Apptainer-injected paths (NVML trampoline,
# CUDA compat shim, /.singularity.d/libs with host libcuda.so.1, Lumerical libs).
# Capture the full current LD_LIBRARY_PATH into LUMERICAL_LD_LIBRARY_PATH so fdtd-solutions
# effectively restores it instead of resetting to empty.
export LUMERICAL_LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}\"
# Force-load Intel TBB's malloc/free interceptor BEFORE any other library. Without
# this preload some library allocates with the system malloc before tbbmalloc_proxy
# is initialized; the later free goes through TBB -> 'free(): invalid pointer' crash.
# Both libs preloaded with absolute paths because lumapi spawns subprocesses
# (e.g. /bin/bash for fdtd-solutions) with an environment where /opt/lumerical/v261/lib
# is not on LD_LIBRARY_PATH at process-start time, so ld.so can't resolve the
# tbbmalloc_proxy -> tbbmalloc DT_NEEDED dependency by name alone.
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
export RUN_SCRIPT='${RUN_SCRIPT}'
export REQUIRE_GPU='${REQUIRE_GPU}'
# KEEP_H5=1 disables the per-iteration .h5 scratch cleanup (default: cleanup on).
export KEEP_H5=\"\${KEEP_H5:-0}\"
# Append /scilibs (libgfortran + libquadmath) to LD_LIBRARY_PATH so scipy/numpy
# imports succeed. Use SUFFIX (not prefix) so host driver libs injected by --nv
# still win for libnvidia-ml/libcuda.
export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:/scilibs\"
# LOCKED_LAMBDA_FILE — used by RUN_SCRIPT=compare_3d_field_prelim (writes the
# resonance λ here for the array half to read). Empty for unrelated runs.
export LOCKED_LAMBDA_FILE='${LOCKED_LAMBDA_FILE:-}'
# Manage Xvfb manually instead of via xvfb-run. The xvfb-run wrapper's
# Xvfb-shutdown logic returns non-zero on this host even when the inner
# Python process exited 0, which silently flips SLURM's job state to FAILED
# despite all results being saved. Direct Xvfb control preserves Python's
# exit code exactly.
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
XVFB_PID=\$!
trap 'kill \$XVFB_PID 2>/dev/null; wait \$XVFB_PID 2>/dev/null' EXIT
export DISPLAY=:99
sleep 1   # let Xvfb finish initialising before lumapi connects

/opt/lumerical/v261/python/bin/python /work/scripts/athena_run.py
PY_RC=\$?
echo \"[wrapper] python exit code: \$PY_RC\"
exit \$PY_RC"

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
