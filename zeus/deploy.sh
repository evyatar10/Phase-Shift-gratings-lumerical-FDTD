#!/bin/bash
#
# Local script: upload project files to Zeus and submit a job.
# Run from Git Bash, WSL, or via VS Code tasks.
#
# Usage:
#   bash zeus/deploy.sh                                      # interactive: engine or pipeline
#   bash zeus/deploy.sh --option1                            # engine: generate .fsp locally → run engine on Zeus
#   bash zeus/deploy.sh --option2                            # pipeline; prompts: single or sweep
#   bash zeus/deploy.sh --option2 --run=single_sim           # pipeline single (no prompt)
#   bash zeus/deploy.sh --option2 --run=simple_bragg         # pipeline single, uniform Bragg
#   bash zeus/deploy.sh --option2 --spec=runners.sweeps.innermost_shift  # sweep (sequential, single PBS job)
#   bash zeus/deploy.sh --upload-only
#   bash zeus/deploy.sh --watch
#   bash zeus/deploy.sh --watch-only
#   bash zeus/deploy.sh --results            # prompts: data only or full (incl. .fsp)
#   bash zeus/deploy.sh --results-no-fsp    # download .mat / logs only (fast)
#   bash zeus/deploy.sh --results-full      # download everything incl. .fsp (heavy)

# ── CONFIGURE — edit zeus/zeus.conf, not this file ────────────────────────────
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
DOWNLOAD_NO_FSP=false
DOWNLOAD_MODE_SET=false
STATUS=false
OPTION=""
RUN_SCRIPT=""
FSP_PRESET=""
SPEC_MODULE=""

for arg in "$@"; do
    case "$arg" in
        --option1)          OPTION="1" ;;
        --option2)          OPTION="2" ;;
        --upload-only)      UPLOAD_ONLY=true ;;
        --results)          DOWNLOAD_RESULTS=true ;;
        --results-no-fsp)   DOWNLOAD_RESULTS=true; DOWNLOAD_NO_FSP=true; DOWNLOAD_MODE_SET=true ;;
        --results-full)     DOWNLOAD_RESULTS=true; DOWNLOAD_NO_FSP=false; DOWNLOAD_MODE_SET=true ;;
        --status)           STATUS=true ;;
        --run)              shift; RUN_SCRIPT="$1" ;;
        --run=*)            RUN_SCRIPT="${arg#--run=}" ;;
        --preset=*)         FSP_PRESET="${arg#--preset=}" ;;
        --spec=*)           RUN_SCRIPT="sweep_spec"; SPEC_MODULE="${arg#--spec=}" ;;
    esac
done

# ── Prompt for top-level mode if not specified ───────────────────────────────
# Two conceptual modes (matches Athena layout):
#   1) Engine    → generate .fsp locally, run FDTD engine on Zeus
#   2) Pipeline  → run lumapi Python pipeline on Zeus (single sim only)
# Sweeps are not supported on Zeus (no SLURM array) — use Athena for those.
if [[ -z "${OPTION}" && "${UPLOAD_ONLY}" == "false" && "${DOWNLOAD_RESULTS}" == "false" && "${STATUS}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Zeus — Choose run mode:"
    echo "  1) FSP job    — generate .fsp locally → run FDTD engine on Zeus"
    echo ""
    echo "  2) Python job — run lumapi Python pipeline on Zeus (single sim only)"
    echo "                  (for sweeps, use Athena: athena/deploy_athena.sh)"
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

# ── Pipeline sub-prompt: single or sweep ─────────────────────────────────────
# Mirrors the Athena layout. Zeus has no SLURM array support, so a "sweep"
# here is just a sequential loop inside a single PBS job (the same as running
# `run_sweep_spec(SPEC, target="local")` locally — no parallelism).
if [[ "${OPTION}" == "2" && -z "${RUN_SCRIPT}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Python pipeline mode:"
    echo "  1) Single simulation"
    echo "  2) Sweep (sequential — one config after another, single PBS job)"
    echo "============================================================"
    read -rp "Enter 1 or 2: " _pipeline_choice
    case "${_pipeline_choice}" in
        1) _PIPELINE_KIND="single" ;;
        2) _PIPELINE_KIND="sweep" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

# ── Single-runner picker (option 2 → single) ─────────────────────────────────
# Auto-discovers runners/single/run_*.py files. New runners appear here as
# soon as they expose a top-level callable matching _SCRIPTS in server_run.py.
if [[ "${OPTION}" == "2" && "${_PIPELINE_KIND:-}" == "single" && -z "${RUN_SCRIPT}" ]]; then
    echo ""
    echo "============================================================"
    echo "  Choose which single-sim script to run on Zeus:"
    echo "  1) single_sim     — Pi-Shift Bragg with cavity     (run_simulation.py)"
    echo "  2) simple_bragg   — uniform Bragg, no cavity        (run_simple_bragg.py)"
    echo "  3) run_experiment — ExperimentCard example          (run_experiment.py)"
    echo "============================================================"
    read -rp "Enter 1, 2, or 3: " _script_choice
    case "${_script_choice}" in
        1) RUN_SCRIPT="single_sim" ;;
        2) RUN_SCRIPT="simple_bragg" ;;
        3) RUN_SCRIPT="run_experiment" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

# ── Sweep study picker (option 2 → sweep) ────────────────────────────────────
# Auto-discovers any module under runners/sweeps/ that defines a top-level SPEC.
# New studies appear automatically the next time this script runs.
if [[ "${OPTION}" == "2" && "${_PIPELINE_KIND:-}" == "sweep" && -z "${SPEC_MODULE}" ]]; then
    mapfile -t _STUDIES < <(
        grep -l '^SPEC' "${LOCAL_PROJECT}/runners/sweeps/"*.py 2>/dev/null \
            | xargs -n1 basename | sed 's/\.py$//' | sort
    )
    if [[ ${#_STUDIES[@]} -eq 0 ]]; then
        echo "ERROR: no study modules found in runners/sweeps/ (need top-level SPEC = SweepSpec(...))"
        exit 1
    fi
    echo ""
    echo "============================================================"
    echo "  Choose sweep study to run (runners/sweeps/<name>.py):"
    for _i in "${!_STUDIES[@]}"; do
        printf "  %d) %s\n" "$((_i+1))" "${_STUDIES[$_i]}"
    done
    echo "============================================================"
    read -rp "Enter number: " _study_choice
    if [[ "${_study_choice}" =~ ^[0-9]+$ ]] && (( _study_choice >= 1 && _study_choice <= ${#_STUDIES[@]} )); then
        SPEC_MODULE="runners.sweeps.${_STUDIES[$((_study_choice-1))]}"
        RUN_SCRIPT="sweep_spec"
        echo "Selected: ${SPEC_MODULE}"
    else
        echo "Invalid choice. Exiting."; exit 1
    fi
fi

SSH="${ZEUS_USER}@${ZEUS_HOST}"

# ── Helper: download results ──────────────────────────────────────────────────
download_results() {
    local no_fsp="${1:-false}"
    echo ""
    mkdir -p "${LOCAL_RESULTS_DIR}"
    if [[ "${no_fsp}" == "true" ]]; then
        echo "=== Downloading results (data files only, skipping .fsp) ==="
        ssh "${SSH}" "tar --exclude='*.fsp' -czf - -C '${REMOTE_BASE}/results' ." \
            | tar -xzf - -C "${LOCAL_RESULTS_DIR}/"
    else
        echo "=== Downloading results (full, including .fsp) ==="
        scp -r "${SSH}:${REMOTE_BASE}/results/." "${LOCAL_RESULTS_DIR}/"
    fi
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
    if [[ "${DOWNLOAD_MODE_SET}" == "false" ]]; then
        echo ""
        echo "============================================================"
        echo "  Choose download mode:"
        echo "  1) Data files only — .mat and logs, no .fsp  (fast)"
        echo "  2) Full results    — includes .fsp layout files (heavy)"
        echo "============================================================"
        read -rp "Enter 1 or 2: " _dl_choice
        case "${_dl_choice}" in
            1) DOWNLOAD_NO_FSP=true ;;
            2) DOWNLOAD_NO_FSP=false ;;
            *) echo "Invalid choice. Exiting."; exit 1 ;;
        esac
    fi
    download_results "${DOWNLOAD_NO_FSP}"
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
          bragg_device.py analysis.py experiment_card.py \
          post_processing.py; do
    if [[ -f "${LOCAL_PROJECT}/${f}" ]]; then
        scp "${LOCAL_PROJECT}/${f}" "${SSH}:${REMOTE_BASE}/project/"
    fi
done
echo "  uploading runners/ (single + sweeps + studies)"
scp -r "${LOCAL_PROJECT}/runners" "${SSH}:${REMOTE_BASE}/project/"

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
scp "${LOCAL_PROJECT}/zeus/jobs/"*.sh    "${SSH}:${REMOTE_BASE}/jobs/"
scp "${LOCAL_PROJECT}/zeus/jobs/"*.c     "${SSH}:${REMOTE_BASE}/jobs/" 2>/dev/null || true
scp "${LOCAL_PROJECT}/zeus/scripts/"*.py "${SSH}:${REMOTE_BASE}/scripts/"
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/jobs/bin"
scp "${LOCAL_PROJECT}/zeus/jobs/bin/fdtd-solutions" "${SSH}:${REMOTE_BASE}/jobs/bin/fdtd-solutions"
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
    FSP_OUTPUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/zeus/scripts/local_save_fsp.py" --preset "${FSP_PRESET}" 2>&1)
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
    # SWEEP_SPEC_MODULE is only used when RUN_SCRIPT=sweep_spec (sweep mode);
    # for single runs it's empty and ignored by server_run.py.
    JOB_ID=$(ssh "${SSH}" "cd ${REMOTE_BASE} && qsub -l select=1:ncpus=${N_CPUS} -v RUN_SCRIPT=${RUN_SCRIPT},SWEEP_SPEC_MODULE=${SPEC_MODULE},NTFY_TOPIC=${NTFY_TOPIC} jobs/run_python_job.sh")
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
echo "  bash zeus/deploy.sh --status"
echo ""
echo "Live log on Zeus:"
echo "  ssh ${SSH} tail -f ${REMOTE_BASE}/jobs/bragg_pipeline.out"
echo ""
echo "Download results after job finishes:"
echo "  bash zeus/deploy.sh --results           # prompts for mode"
echo "  bash zeus/deploy.sh --results-no-fsp    # data files only (fast)"
echo "  bash zeus/deploy.sh --results-full      # everything incl. .fsp (heavy)"
echo "============================================================"
