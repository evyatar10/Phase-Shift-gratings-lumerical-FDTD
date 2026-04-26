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

# ── CONFIGURE — edit hpc_gpu/athena.conf, not this file ──────────────────────
CONF="$(cd "$(dirname "$0")" && pwd)/athena.conf"
if [[ ! -f "${CONF}" ]]; then
    echo "ERROR: athena.conf not found at ${CONF}"
    exit 1
fi
source "${CONF}"

LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RESULTS_DIR="${LOCAL_PROJECT}/results_from_athena"
LOCAL_NEFF=$(python -c "import sys; sys.path.insert(0,'${LOCAL_PROJECT}'); import config; print(config.NEFF_DATA_PATH)")
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_ONLY=false
DOWNLOAD_RESULTS=false
STATUS=false
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
        --status)       STATUS=true ;;
        --run=*)        RUN_SCRIPT="${arg#--run=}" ;;
        --preset=*)     FSP_PRESET="${arg#--preset=}" ;;
        --fsp=*)        FSP_EXPLICIT="${arg#--fsp=}" ;;
        --keep-h5)      KEEP_H5=1 ;;
    esac
done

SSH="${ATHENA_USER}@${ATHENA_HOST}"

# ── Prompt for option if not specified ────────────────────────────────────────
if [[ -z "${OPTION}" && "${UPLOAD_ONLY}" == "false" && "${DOWNLOAD_RESULTS}" == "false" && "${STATUS}" == "false" ]]; then
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

# ── Helper: one-shot status check ─────────────────────────────────────────────
check_status() {
    echo ""
    echo "=== Jobs in queue (${ATHENA_USER}) ==="
    ssh "${SSH}" "squeue -u ${ATHENA_USER} -o '%.10i %.12j %.8T %.10M %.6D %R' 2>/dev/null || echo '(no jobs in queue)'"
    echo ""
    echo "=== Last 40 lines of most recent log ==="
    ssh "${SSH}" "ls -t ${REMOTE_BASE}/jobs/logs/*.out 2>/dev/null | head -1 | xargs tail -40 2>/dev/null || echo '(no log found)'"
}

# ── --status: one-shot check then exit ────────────────────────────────────────
if [[ "${STATUS}" == "true" ]]; then
    check_status
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
scp "${LOCAL_PROJECT}"/*.py "${SSH}:${REMOTE_BASE}/project/"
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
                --gpus=${N_GPUS} --cpus-per-task=${N_CPUS} \
                --export=ALL,ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},NTFY_TOPIC=${NTFY_TOPIC} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu_array.sh")
    else
        JOB_ID=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --gpus=${N_GPUS} --cpus-per-task=${N_CPUS} \
                --export=ALL,FSP_FILE=\"${FSP_NAME}\",ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},NTFY_TOPIC=${NTFY_TOPIC} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu.sh")
    fi

else
    # ── Option 2: full Python pipeline ───────────────────────────────────────
    JOB_ID=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch \
            --gpus=${N_GPUS} --cpus-per-task=${N_CPUS} \
            --export=ALL,RUN_SCRIPT=${RUN_SCRIPT},ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC} \
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
echo "You will receive an email at job start, finish, and failure."
echo ""
echo "Check status at any time:"
echo "  bash hpc_gpu/deploy_athena.sh --status"
echo ""
echo "Live log on Athena:"
echo "  ssh ${SSH} tail -f ${REMOTE_BASE}/jobs/logs/lum_*${NUMERIC_JOB}*.out"
echo ""
echo "Download results after job finishes:"
echo "  bash hpc_gpu/deploy_athena.sh --results"
echo "============================================================"
