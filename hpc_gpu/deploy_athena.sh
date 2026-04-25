#!/bin/bash
#
# Local script: upload project files to Athena and submit a SLURM GPU job.
# This is the Athena analog of hpc/deploy.sh (Zeus/PBS). Zeus is NOT touched.
#
# Run from Git Bash, WSL, or VS Code tasks on your local Windows machine.
#
# Usage:
#   bash hpc_gpu/deploy_athena.sh --option1 --preset single       # RECOMMENDED
#   bash hpc_gpu/deploy_athena.sh --option1 --preset sweep_shift
#   bash hpc_gpu/deploy_athena.sh --option1 --preset sweep_inner_size
#   bash hpc_gpu/deploy_athena.sh --option2 --run single_sim
#   bash hpc_gpu/deploy_athena.sh --option2 --run sweep_shift
#   bash hpc_gpu/deploy_athena.sh --upload-only
#   bash hpc_gpu/deploy_athena.sh --watch
#   bash hpc_gpu/deploy_athena.sh --watch-only
#   bash hpc_gpu/deploy_athena.sh --results

# ── CONFIGURE ─────────────────────────────────────────────────────────────────
ATHENA_USER="evyatarrubin"
ATHENA_HOST="dgx-master.technion.ac.il"

# Separate remote root from Zeus to guarantee zero collision.
# Zeus uses ~/bragg_sim — this uses ~/bragg_sim_gpu.
REMOTE_BASE="/home/${ATHENA_USER}/bragg_sim_gpu"

LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_NEFF="C:/Users/evyat/Lumerical/pi_shifts_FDTD_results/neff_vs_wl_new/FDE_sweep_results.mat"
LOCAL_RESULTS_DIR="${LOCAL_PROJECT}/results_from_athena"

POLL_INTERVAL=60   # seconds between squeue checks

# License — single source of truth. Job scripts read these via --export.
# (Verified 2026-04-25: 11055@dgx-master and 1055@132.68.48.51 both reach the
# same backend lmgrd; 11055@dgx-master is what the working FSP path uses.)
ATHENA_LICENSE="11055@dgx-master"
ATHENA_INTERCONNECT="12325@172.25.0.12"
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_ONLY=false
DOWNLOAD_RESULTS=false
WATCH=false
WATCH_ONLY=false
OPTION=""
RUN_SCRIPT=""
FSP_PRESET=""
FSP_EXPLICIT=""

for arg in "$@"; do
    case "${arg}" in
        --option1)      OPTION="1" ;;
        --option2)      OPTION="2" ;;
        --upload-only)  UPLOAD_ONLY=true ;;
        --results)      DOWNLOAD_RESULTS=true ;;
        --watch)        WATCH=true ;;
        --watch-only)   WATCH_ONLY=true ;;
        --run=*)        RUN_SCRIPT="${arg#--run=}" ;;
        --preset=*)     FSP_PRESET="${arg#--preset=}" ;;
        --fsp=*)        FSP_EXPLICIT="${arg#--fsp=}" ;;
    esac
done

SSH="${ATHENA_USER}@${ATHENA_HOST}"

# ── Prompt for option if not specified ────────────────────────────────────────
if [[ -z "${OPTION}" && "${UPLOAD_ONLY}" == "false" && "${DOWNLOAD_RESULTS}" == "false" && "${WATCH_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Athena GPU — Choose run mode:"
    echo "  1) Generate .fsp locally → upload → run FDTD engine on GPU"
    echo "     (RECOMMENDED — no license required on compute node at FSP build time)"
    echo ""
    echo "  2) Upload Python code → run full lumapi pipeline on GPU"
    echo "     (requires license at run time; enables USE_GPU=True)"
    echo "============================================================"
    read -rp "Enter 1 or 2: " OPTION
    if [[ "${OPTION}" != "1" && "${OPTION}" != "2" ]]; then
        echo "Invalid option. Exiting."
        exit 1
    fi
fi

# ── Prompt for preset if option 1 ─────────────────────────────────────────────
if [[ "${OPTION}" == "1" && -z "${FSP_PRESET}" && -z "${FSP_EXPLICIT}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Choose layout to generate locally:"
    echo "  1) single             — one simulation file"
    echo "  2) sweep_shift        — innermost tooth shift sweep"
    echo "  3) sweep_inner_size   — shift × inner-size 2D sweep"
    echo "============================================================"
    read -rp "Enter 1, 2, or 3: " _preset_choice
    case "${_preset_choice}" in
        1) FSP_PRESET="single" ;;
        2) FSP_PRESET="sweep_shift" ;;
        3) FSP_PRESET="sweep_inner_size" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

# ── Prompt for script if option 2 ─────────────────────────────────────────────
if [[ "${OPTION}" == "2" && -z "${RUN_SCRIPT}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Choose Python script to run on Athena:"
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

# ── Helper: download results ───────────────────────────────────────────────────
download_results() {
    echo ""
    echo "=== Downloading results from Athena ==="
    mkdir -p "${LOCAL_RESULTS_DIR}"
    scp -r "${SSH}:${REMOTE_BASE}/results/." "${LOCAL_RESULTS_DIR}/"
    echo "Results saved to: ${LOCAL_RESULTS_DIR}"
}

# ── Helper: poll until SLURM job completes ────────────────────────────────────
watch_job() {
    local job_id="$1"
    echo ""
    echo "=== Watching SLURM job ${job_id} (checking every ${POLL_INTERVAL}s) ==="
    echo "Press Ctrl+C to stop watching (job will keep running on Athena)."
    echo ""

    while true; do
        STATUS=$(ssh "${SSH}" "squeue -j ${job_id} -h -o '%T' 2>/dev/null")
        TIMESTAMP=$(date '+%H:%M:%S')

        if [[ -z "${STATUS}" ]]; then
            echo "[${TIMESTAMP}] Job ${job_id} no longer in queue — finished (or failed)."
            break
        fi

        case "${STATUS}" in
            RUNNING)    echo "[${TIMESTAMP}] Job ${job_id} is RUNNING..." ;;
            PENDING)    echo "[${TIMESTAMP}] Job ${job_id} is PENDING (waiting for resources)..." ;;
            COMPLETING) echo "[${TIMESTAMP}] Job ${job_id} is COMPLETING..." ;;
            FAILED)
                echo "[${TIMESTAMP}] Job ${job_id} FAILED."
                break
                ;;
            *) echo "[${TIMESTAMP}] Job ${job_id} status: ${STATUS}" ;;
        esac

        sleep "${POLL_INTERVAL}"
    done

    echo ""
    echo "=== Last 40 lines of job log ==="
    ssh "${SSH}" "ls ${REMOTE_BASE}/jobs/logs/*${job_id}* 2>/dev/null | \
        head -1 | xargs tail -40 2>/dev/null || echo '(log not found)'"
}

# ── --watch-only ───────────────────────────────────────────────────────────────
if [[ "${WATCH_ONLY}" == "true" ]]; then
    echo "=== Looking for last submitted job on Athena ==="
    LAST_JOB=$(ssh "${SSH}" "squeue -u ${ATHENA_USER} -h -o '%i' 2>/dev/null | tail -1")
    if [[ -z "${LAST_JOB}" ]]; then
        echo "No running jobs found for ${ATHENA_USER}."
        echo "Checking most recent log file..."
        ssh "${SSH}" "ls -t ${REMOTE_BASE}/jobs/logs/*.out 2>/dev/null | \
            head -1 | xargs tail -40 2>/dev/null || echo '(no log found)'"
        exit 0
    fi
    echo "Found job: ${LAST_JOB}"
    watch_job "${LAST_JOB}"
    download_results
    exit 0
fi

# ── --results: immediate download ─────────────────────────────────────────────
if [[ "${DOWNLOAD_RESULTS}" == "true" ]]; then
    download_results
    exit 0
fi

# ── Normal flow: create remote directories and upload ─────────────────────────
echo "============================================================"
echo "Target: ${SSH}"
echo "Remote: ${REMOTE_BASE}"
echo "============================================================"

echo ""
echo "=== Creating remote directories ==="
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/{project,data,results/layouts,jobs/logs,scripts}"

echo ""
echo "=== Uploading project files ==="
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
    echo "WARNING: neff data not found at: ${LOCAL_NEFF}"
    echo "  Update LOCAL_NEFF in deploy_athena.sh or upload manually."
fi

echo ""
echo "=== Uploading HPC GPU scripts ==="
scp "${LOCAL_PROJECT}/hpc_gpu/jobs/"*.sh      "${SSH}:${REMOTE_BASE}/jobs/"
scp "${LOCAL_PROJECT}/hpc_gpu/scripts/"*.py   "${SSH}:${REMOTE_BASE}/scripts/"
ssh "${SSH}" "chmod +x ${REMOTE_BASE}/jobs/*.sh"
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/jobs/logs"

echo ""
echo "=== Building NVML trampoline on Athena (R470 A100 nodes) ==="
# The trampoline shim stubs three NVML symbols absent from R470 that Lumerical
# 2026 R1's GPU plugin imports. On newer-driver nodes (L40S / H200, R535+) the
# tramp directory is not bound and has no effect.
scp "${LOCAL_PROJECT}/hpc_gpu/container/nvml_tramp.c" \
    "${SSH}:${REMOTE_BASE}/jobs/nvml_tramp.c"
ssh "${SSH}" "
    mkdir -p \${HOME}/nvml_tramp
    gcc -O2 -shared -fPIC -Wl,-soname,libnvidia-ml.so.1 \
        -o \${HOME}/nvml_tramp/libnvidia-ml.so.1 \
        ${REMOTE_BASE}/jobs/nvml_tramp.c -ldl && \
    echo '  NVML trampoline built: ~/nvml_tramp/libnvidia-ml.so.1' || \
    echo '  WARNING: gcc failed — GPU jobs on R470 nodes will skip the trampoline'
"

echo ""
echo "=== Upload complete ==="

if [[ "${UPLOAD_ONLY}" == "true" ]]; then
    echo "Skipping job submission (--upload-only)."
    exit 0
fi

# ── Submit SLURM job ───────────────────────────────────────────────────────────
echo ""
echo "=== Submitting SLURM job ==="

if [[ "${OPTION}" == "1" ]]; then
    # ── Option 1: generate .fsp locally, upload, run engine on GPU ───────────
    if [[ -n "${FSP_EXPLICIT}" ]]; then
        echo "Option 1: using explicit .fsp: ${FSP_EXPLICIT}"
        FSP_NAME="${FSP_EXPLICIT}"
    else
        echo "Option 1: generating .fsp locally (preset: ${FSP_PRESET})..."
        LOCAL_PYTHON=$(which python 2>/dev/null || which python3 2>/dev/null)
        FSP_OUTPUT=$("${LOCAL_PYTHON}" \
            "${LOCAL_PROJECT}/hpc/scripts/local_save_fsp.py" \
            --preset "${FSP_PRESET}" --gpu 2>&1)
        echo "${FSP_OUTPUT}"

        FSP_PATH=$(echo "${FSP_OUTPUT}" | grep "^FSP_SAVED:" | sed 's/FSP_SAVED://')
        if [[ -z "${FSP_PATH}" ]]; then
            echo "ERROR: Failed to generate .fsp file locally."
            exit 1
        fi
        FSP_NAME=$(basename "${FSP_PATH}")
        echo "Generated: ${FSP_PATH}"

        echo "Uploading .fsp to Athena..."
        scp "${FSP_PATH}" "${SSH}:${REMOTE_BASE}/results/layouts/${FSP_NAME}"
    fi

    # Detect if this is a sweep (multiple .fsp files → job array)
    FSP_COUNT=$(ssh "${SSH}" "ls ${REMOTE_BASE}/results/layouts/*.fsp 2>/dev/null | wc -l")
    echo "Found ${FSP_COUNT} .fsp file(s) in ${REMOTE_BASE}/results/layouts/"

    if [[ "${FSP_COUNT}" -gt 1 ]]; then
        echo ""
        echo "Multiple .fsp files detected — submitting as a SLURM job array."
        echo "Building fsp_list.txt on Athena..."
        ssh "${SSH}" "ls ${REMOTE_BASE}/results/layouts/*.fsp | xargs -n1 basename \
            > ${REMOTE_BASE}/results/layouts/fsp_list.txt"
        ARRAY_END=$(( FSP_COUNT - 1 ))

        # K = max concurrent jobs (license throttle). Adjust once you know your seat count.
        K=4
        echo "Array: 0-${ARRAY_END}%${K}  (max ${K} concurrent, ~${FSP_COUNT} engine seats used)"
        echo "Edit K in deploy_athena.sh after running: lmutil lmstat -a -c 1055@132.68.48.51"

        JOB_ID=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --array=0-${ARRAY_END}%${K} \
                --export=ALL,ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu_array.sh")
    else
        JOB_ID=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --export=ALL,FSP_FILE=\"${FSP_NAME}\",ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu.sh")
    fi

else
    # ── Option 2: full Python pipeline ───────────────────────────────────────
    JOB_ID=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch \
            --export=ALL,RUN_SCRIPT=${RUN_SCRIPT},ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1} \
            --chdir=${REMOTE_BASE}/jobs \
            jobs/run_python_gpu.sh")
fi

if [[ $? -ne 0 ]]; then
    echo "ERROR: sbatch failed."
    exit 1
fi

echo "Submitted: ${JOB_ID}"
# Extract numeric ID from sbatch output ("Submitted batch job 12345")
NUMERIC_JOB=$(echo "${JOB_ID}" | awk '{print $NF}')

# ── Show monitoring commands ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Job submitted: ${JOB_ID}"
echo ""
echo "Monitor on Athena:"
echo "  ssh ${SSH}"
echo "  squeue -u ${ATHENA_USER}              # queue status"
echo "  squeue -j ${NUMERIC_JOB} -l           # detailed info"
echo "  tail -f ${REMOTE_BASE}/jobs/logs/lum_*${NUMERIC_JOB}*.out  # live log"
echo ""
echo "Download results after job finishes:"
echo "  bash hpc_gpu/deploy_athena.sh --results"
echo "============================================================"

# ── --watch: poll until done then auto-download ───────────────────────────────
if [[ "${WATCH}" == "true" ]]; then
    watch_job "${NUMERIC_JOB}"
    download_results
fi
