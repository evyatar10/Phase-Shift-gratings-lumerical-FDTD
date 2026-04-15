#!/bin/bash
#
# Local script: upload project files to Zeus and submit the Python pipeline job.
# Run this from your Windows machine (Git Bash or WSL).
#
# Usage:
#   bash hpc/deploy.sh                  # upload + submit pipeline job
#   bash hpc/deploy.sh --upload-only    # only sync files, don't submit
#   bash hpc/deploy.sh --results        # download results after job completes

# ── CONFIGURE ────────────────────────────────────────────────────────────────
ZEUS_USER="evyat"
ZEUS_HOST="zeus.technion.ac.il"          # update if hostname differs
REMOTE_BASE="/home/${ZEUS_USER}/bragg_sim"

LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"   # project root (one level up from hpc/)
LOCAL_NEFF="C:/Users/evyat/Lumerical/pi_shifts_FDTD_results/neff_vs_wl_new/FDE_sweep_results.mat"
LOCAL_RESULTS_DIR="./results_from_server"           # where to download results locally
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_ONLY=false
DOWNLOAD_RESULTS=false
for arg in "$@"; do
    case "$arg" in
        --upload-only)   UPLOAD_ONLY=true ;;
        --results)       DOWNLOAD_RESULTS=true ;;
    esac
done

SSH="${ZEUS_USER}@${ZEUS_HOST}"

echo "============================================================"
echo "Target: ${SSH}"
echo "Remote: ${REMOTE_BASE}"
echo "============================================================"

# ── Step 1: Create remote directory structure ─────────────────────────────────
echo ""
echo "=== Creating remote directories ==="
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/{project,data,results,jobs,scripts}"

# ── Step 2: Upload project .py files (no git history, no local result dirs) ──
echo ""
echo "=== Uploading project files ==="
rsync -avz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='hpc/' \
    --include='*.py' \
    --include='*/' \
    --exclude='*' \
    "${LOCAL_PROJECT}/" \
    "${SSH}:${REMOTE_BASE}/project/"

# ── Step 3: Upload neff data ──────────────────────────────────────────────────
echo ""
echo "=== Uploading neff data ==="
if [[ -f "${LOCAL_NEFF}" ]]; then
    rsync -avz "${LOCAL_NEFF}" "${SSH}:${REMOTE_BASE}/data/FDE_sweep_results.mat"
else
    echo "WARNING: neff data file not found at:"
    echo "  ${LOCAL_NEFF}"
    echo "Update LOCAL_NEFF in deploy.sh or upload it manually."
fi

# ── Step 4: Upload HPC wrapper scripts ───────────────────────────────────────
echo ""
echo "=== Uploading HPC scripts ==="
rsync -avz "${LOCAL_PROJECT}/hpc/jobs/"    "${SSH}:${REMOTE_BASE}/jobs/"
rsync -avz "${LOCAL_PROJECT}/hpc/scripts/" "${SSH}:${REMOTE_BASE}/scripts/"

# Make PBS scripts executable on the server
ssh "${SSH}" "chmod +x ${REMOTE_BASE}/jobs/*.sh"

echo ""
echo "=== Upload complete ==="

if [[ "${UPLOAD_ONLY}" == "true" ]]; then
    echo "Skipping job submission (--upload-only)."
    exit 0
fi

if [[ "${DOWNLOAD_RESULTS}" == "true" ]]; then
    echo ""
    echo "=== Downloading results ==="
    mkdir -p "${LOCAL_RESULTS_DIR}"
    rsync -avz "${SSH}:${REMOTE_BASE}/results/" "${LOCAL_RESULTS_DIR}/"
    echo "Results saved to: ${LOCAL_RESULTS_DIR}"
    exit 0
fi

# ── Step 5: Submit PBS job ────────────────────────────────────────────────────
echo ""
echo "=== Submitting PBS job ==="
JOB_ID=$(ssh "${SSH}" "cd ${REMOTE_BASE} && qsub jobs/run_python_job.sh")
if [[ $? -ne 0 ]]; then
    echo "ERROR: qsub failed. Check that you're connected to Zeus and PBS is available."
    exit 1
fi
echo "Submitted: ${JOB_ID}"

# ── Step 6: Show monitoring commands ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "Job submitted: ${JOB_ID}"
echo ""
echo "Monitor on Zeus:"
echo "  ssh ${SSH}"
echo "  qstat -u ${ZEUS_USER}         # queue status"
echo "  qstat -f ${JOB_ID}            # detailed info"
echo "  tail -f ${REMOTE_BASE}/jobs/bragg_pipeline.out  # live log"
echo ""
echo "Download results after job finishes:"
echo "  bash hpc/deploy.sh --results"
echo "============================================================"
