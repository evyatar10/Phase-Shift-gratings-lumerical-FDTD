#!/bin/bash
#
# Local script: upload project files to Zeus and submit a job.
# Run from Git Bash, WSL, or via VS Code tasks.
#
# Usage:
#   bash hpc/deploy.sh --option1    # generate .fsp locally → upload → run engine on Zeus (RECOMMENDED)
#   bash hpc/deploy.sh --option2    # upload Python code → run full lumapi pipeline on Zeus
#   bash hpc/deploy.sh --upload-only
#   bash hpc/deploy.sh --watch
#   bash hpc/deploy.sh --watch-only
#   bash hpc/deploy.sh --results

# ── CONFIGURE ────────────────────────────────────────────────────────────────
ZEUS_USER="evyatarrubin"
ZEUS_HOST="zeus.technion.ac.il"
REMOTE_BASE="/home/${ZEUS_USER}/bragg_sim"

LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_NEFF="C:/Users/evyat/Lumerical/pi_shifts_FDTD_results/neff_vs_wl_new/FDE_sweep_results.mat"
LOCAL_RESULTS_DIR="${LOCAL_PROJECT}/results_from_server"

POLL_INTERVAL=60   # seconds between qstat checks
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_ONLY=false
DOWNLOAD_RESULTS=false
WATCH=false
WATCH_ONLY=false
OPTION=""
RUN_SCRIPT=""
FSP_PRESET=""

for arg in "$@"; do
    case "$arg" in
        --option1)      OPTION="1" ;;
        --option2)      OPTION="2" ;;
        --upload-only)  UPLOAD_ONLY=true ;;
        --results)      DOWNLOAD_RESULTS=true ;;
        --watch)        WATCH=true ;;
        --watch-only)   WATCH_ONLY=true ;;
        --run)          shift; RUN_SCRIPT="$1" ;;
        --run=*)        RUN_SCRIPT="${arg#--run=}" ;;
        --preset=*)     FSP_PRESET="${arg#--preset=}" ;;
    esac
done

# ── Prompt for option if not specified ───────────────────────────────────────
if [[ -z "${OPTION}" && "${UPLOAD_ONLY}" == "false" && "${DOWNLOAD_RESULTS}" == "false" && "${WATCH_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Choose run mode:"
    echo "  1) Generate .fsp locally → upload → run FDTD engine on Zeus"
    echo "     (RECOMMENDED — no license issues on compute nodes)"
    echo ""
    echo "  2) Upload Python code → run full lumapi pipeline on Zeus"
    echo "     (may fail if Lumerical license is unavailable on node)"
    echo "============================================================"
    read -rp "Enter 1 or 2: " OPTION
    if [[ "${OPTION}" != "1" && "${OPTION}" != "2" ]]; then
        echo "Invalid option. Exiting."
        exit 1
    fi
fi

# ── Prompt for layout preset if option 1 and not specified ───────────────────
if [[ "${OPTION}" == "1" && -z "${FSP_PRESET}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Choose which layout to generate locally:"
    echo "  1) single          — one file, no sweep"
    echo "  2) sweep_shift     — innermost tooth shift sweep"
    echo "  3) sweep_inner_size — shift × inner-size 2D sweep"
    echo "============================================================"
    read -rp "Enter 1, 2, or 3: " _preset_choice
    case "${_preset_choice}" in
        1) FSP_PRESET="single" ;;
        2) FSP_PRESET="sweep_shift" ;;
        3) FSP_PRESET="sweep_inner_size" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

# ── Prompt for script if option 2 and not specified ──────────────────────────
if [[ "${OPTION}" == "2" && -z "${RUN_SCRIPT}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Choose which script to run on Zeus:"
    echo "  1) single_sim          — run_simulation.py (one simulation)"
    echo "  2) sweep_shift         — ToothShift/run_sweep_innermost_shift.py"
    echo "  3) sweep_inner_size    — ToothShift/run_sweep_inner_tooth_size.py"
    echo "============================================================"
    read -rp "Enter 1, 2, or 3: " _script_choice
    case "${_script_choice}" in
        1) RUN_SCRIPT="single_sim" ;;
        2) RUN_SCRIPT="sweep_shift" ;;
        3) RUN_SCRIPT="sweep_inner_size" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

SSH="${ZEUS_USER}@${ZEUS_HOST}"

# ── Helper: download results ──────────────────────────────────────────────────
download_results() {
    echo ""
    echo "=== Downloading results ==="
    mkdir -p "${LOCAL_RESULTS_DIR}"
    scp -r "${SSH}:${REMOTE_BASE}/results/." "${LOCAL_RESULTS_DIR}/"
    echo "Results saved to: ${LOCAL_RESULTS_DIR}"
}

# ── Helper: poll until job completes ─────────────────────────────────────────
watch_job() {
    local job_id="$1"
    echo ""
    echo "=== Watching job ${job_id} (checking every ${POLL_INTERVAL}s) ==="
    echo "Press Ctrl+C to stop watching (job will keep running on Zeus)."
    echo ""

    while true; do
        STATUS=$(ssh "${SSH}" "qstat ${job_id} 2>/dev/null | tail -1 | awk '{print \$5}'")
        TIMESTAMP=$(date '+%H:%M:%S')

        if [[ -z "${STATUS}" ]]; then
            echo "[${TIMESTAMP}] Job ${job_id} no longer in queue — finished (or failed)."
            break
        fi

        case "${STATUS}" in
            R) echo "[${TIMESTAMP}] Job ${job_id} is RUNNING..." ;;
            Q) echo "[${TIMESTAMP}] Job ${job_id} is QUEUED (waiting for resources)..." ;;
            H) echo "[${TIMESTAMP}] Job ${job_id} is HELD." ;;
            E) echo "[${TIMESTAMP}] Job ${job_id} is EXITING..." ;;
            *) echo "[${TIMESTAMP}] Job ${job_id} status: ${STATUS}" ;;
        esac

        sleep "${POLL_INTERVAL}"
    done

    # Print the last lines of the job output log
    echo ""
    echo "=== Last 30 lines of job log ==="
    ssh "${SSH}" "tail -30 ${REMOTE_BASE}/jobs/bragg_pipeline.out 2>/dev/null || echo '(log not found)'"
}

# ── --watch-only: find the most recent job by this user and watch it ──────────
if [[ "${WATCH_ONLY}" == "true" ]]; then
    echo "=== Looking for last submitted job on Zeus ==="
    LAST_JOB=$(ssh "${SSH}" "qstat -u ${ZEUS_USER} 2>/dev/null | grep ${ZEUS_USER} | tail -1 | awk '{print \$1}'")
    if [[ -z "${LAST_JOB}" ]]; then
        echo "No running jobs found for ${ZEUS_USER}."
        echo "Checking if a recent output log exists instead..."
        ssh "${SSH}" "tail -30 ${REMOTE_BASE}/jobs/bragg_pipeline.out 2>/dev/null || echo '(no log found)'"
        exit 0
    fi
    echo "Found job: ${LAST_JOB}"
    watch_job "${LAST_JOB}"
    download_results
    exit 0
fi

# ── --results: immediate download ────────────────────────────────────────────
if [[ "${DOWNLOAD_RESULTS}" == "true" ]]; then
    download_results
    exit 0
fi

# ── Normal flow: upload ───────────────────────────────────────────────────────
echo "============================================================"
echo "Target: ${SSH}"
echo "Remote: ${REMOTE_BASE}"
echo "============================================================"

echo ""
echo "=== Creating remote directories ==="
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/{project,data,results,jobs,scripts}"

echo ""
echo "=== Uploading project files ==="
# Core files required to run simulations on Zeus
for f in config.py simulation_config.py sim_helpers.py \
          bragg_device.py \
          run_simulation.py post_processing.py; do
    if [[ -f "${LOCAL_PROJECT}/${f}" ]]; then
        scp "${LOCAL_PROJECT}/${f}" "${SSH}:${REMOTE_BASE}/project/"
    fi
done
echo "  uploading ToothShift/"
scp -r "${LOCAL_PROJECT}/ToothShift" "${SSH}:${REMOTE_BASE}/project/"

echo ""
echo "=== Uploading neff data ==="
if [[ -f "${LOCAL_NEFF}" ]]; then
    scp "${LOCAL_NEFF}" "${SSH}:${REMOTE_BASE}/data/FDE_sweep_results.mat"
else
    echo "WARNING: neff data file not found at:"
    echo "  ${LOCAL_NEFF}"
    echo "  Update LOCAL_NEFF in deploy.sh or upload manually."
fi

echo ""
echo "=== Uploading HPC scripts ==="
scp "${LOCAL_PROJECT}/hpc/jobs/"*.sh    "${SSH}:${REMOTE_BASE}/jobs/"
scp "${LOCAL_PROJECT}/hpc/jobs/"*.c     "${SSH}:${REMOTE_BASE}/jobs/" 2>/dev/null || true
scp "${LOCAL_PROJECT}/hpc/scripts/"*.py "${SSH}:${REMOTE_BASE}/scripts/"
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/jobs/bin"
scp "${LOCAL_PROJECT}/hpc/jobs/bin/fdtd-solutions" "${SSH}:${REMOTE_BASE}/jobs/bin/fdtd-solutions"
ssh "${SSH}" "chmod +x ${REMOTE_BASE}/jobs/*.sh ${REMOTE_BASE}/jobs/bin/fdtd-solutions"

echo ""
echo "=== Upload complete ==="

if [[ "${UPLOAD_ONLY}" == "true" ]]; then
    echo "Skipping job submission (--upload-only)."
    exit 0
fi

# ── Submit job ────────────────────────────────────────────────────────────────
echo ""
echo "=== Submitting PBS job ==="

if [[ "${OPTION}" == "1" ]]; then
    # ── Option 1: generate .fsp locally, upload it, run engine on Zeus ───────
    echo "Option 1: generating .fsp locally..."
    LOCAL_PYTHON=$(which python 2>/dev/null || which python3 2>/dev/null)
    FSP_OUTPUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/hpc/scripts/local_save_fsp.py" --preset "${FSP_PRESET}" 2>&1)
    echo "${FSP_OUTPUT}"

    FSP_PATH=$(echo "${FSP_OUTPUT}" | grep "^FSP_SAVED:" | sed 's/FSP_SAVED://')
    if [[ -z "${FSP_PATH}" ]]; then
        echo "ERROR: Failed to generate .fsp file locally."
        exit 1
    fi
    FSP_NAME=$(basename "${FSP_PATH}")
    echo "Generated: ${FSP_PATH}"

    echo "Uploading .fsp to Zeus..."
    ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/results/layouts"
    scp "${FSP_PATH}" "${SSH}:${REMOTE_BASE}/results/layouts/${FSP_NAME}"

    JOB_ID=$(ssh "${SSH}" "cd ${REMOTE_BASE} && qsub -v FSP_FILE=\"${FSP_NAME}\" jobs/run_fsp_job.sh")
    if [[ $? -ne 0 ]]; then
        echo "ERROR: qsub failed."
        exit 1
    fi
else
    # ── Option 2: full Python pipeline on Zeus ────────────────────────────────
    JOB_ID=$(ssh "${SSH}" "cd ${REMOTE_BASE} && qsub -v RUN_SCRIPT=${RUN_SCRIPT} jobs/run_python_job.sh")
    if [[ $? -ne 0 ]]; then
        echo "ERROR: qsub failed. Check that you're connected to Zeus and PBS is available."
        exit 1
    fi
fi
echo "Submitted: ${JOB_ID}"

# ── Show monitoring commands ──────────────────────────────────────────────────
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

# ── --watch: poll until done then auto-download ───────────────────────────────
if [[ "${WATCH}" == "true" ]]; then
    watch_job "${JOB_ID}"
    download_results
fi
