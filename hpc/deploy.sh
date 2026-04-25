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

# ── CONFIGURE — edit hpc/zeus.conf, not this file ────────────────────────────
CONF="$(cd "$(dirname "$0")" && pwd)/zeus.conf"
if [[ ! -f "${CONF}" ]]; then
    echo "ERROR: zeus.conf not found at ${CONF}"
    exit 1
fi
source "${CONF}"

LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RESULTS_DIR="${LOCAL_PROJECT}/results_from_server"
LOCAL_NEFF=$(python -c "import sys; sys.path.insert(0,'${LOCAL_PROJECT}'); import config; print(config.NEFF_DATA_PATH)")
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_ONLY=false
DOWNLOAD_RESULTS=false
STATUS=false
OPTION=""
RUN_SCRIPT=""
FSP_PRESET=""

for arg in "$@"; do
    case "$arg" in
        --option1)      OPTION="1" ;;
        --option2)      OPTION="2" ;;
        --upload-only)  UPLOAD_ONLY=true ;;
        --results)      DOWNLOAD_RESULTS=true ;;
        --status)       STATUS=true ;;
        --run)          shift; RUN_SCRIPT="$1" ;;
        --run=*)        RUN_SCRIPT="${arg#--run=}" ;;
        --preset=*)     FSP_PRESET="${arg#--preset=}" ;;
    esac
done

# ── Prompt for option if not specified ───────────────────────────────────────
if [[ -z "${OPTION}" && "${UPLOAD_ONLY}" == "false" && "${DOWNLOAD_RESULTS}" == "false" && "${STATUS}" == "false" ]]; then
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

# ── Helper: one-shot status check ────────────────────────────────────────────
check_status() {
    echo ""
    echo "=== Jobs in queue (${ZEUS_USER}) ==="
    ssh "${SSH}" "qstat -u ${ZEUS_USER} 2>/dev/null || echo '(no jobs in queue)'"
    echo ""
    echo "=== Last 40 lines of most recent log ==="
    ssh "${SSH}" "ls -t ${REMOTE_BASE}/jobs/*.out 2>/dev/null | head -1 | xargs tail -40 2>/dev/null || echo '(no log found)'"
}

# ── --status: one-shot check then exit ───────────────────────────────────────
if [[ "${STATUS}" == "true" ]]; then
    check_status
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

    JOB_ID=$(ssh "${SSH}" "cd ${REMOTE_BASE} && qsub -l select=1:ncpus=${N_CPUS} -v FSP_FILE=\"${FSP_NAME}\",NTFY_TOPIC=${NTFY_TOPIC} jobs/run_fsp_job.sh")
    if [[ $? -ne 0 ]]; then
        echo "ERROR: qsub failed."
        exit 1
    fi
else
    # ── Option 2: full Python pipeline on Zeus ────────────────────────────────
    JOB_ID=$(ssh "${SSH}" "cd ${REMOTE_BASE} && qsub -l select=1:ncpus=${N_CPUS} -v RUN_SCRIPT=${RUN_SCRIPT},NTFY_TOPIC=${NTFY_TOPIC} jobs/run_python_job.sh")
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
echo "You will receive an email at job start, finish, and failure."
echo ""
echo "Check status at any time:"
echo "  bash hpc/deploy.sh --status"
echo ""
echo "Live log on Zeus:"
echo "  ssh ${SSH} tail -f ${REMOTE_BASE}/jobs/bragg_pipeline.out"
echo ""
echo "Download results after job finishes:"
echo "  bash hpc/deploy.sh --results"
echo "============================================================"
