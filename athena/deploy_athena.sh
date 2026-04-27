#!/bin/bash
#
# Local script: upload project files to Athena and submit a SLURM GPU job.
# This is the Athena analog of zeus/deploy.sh (Zeus/PBS). Zeus is NOT touched.
#
# Run from Git Bash, WSL, or VS Code tasks on your local Windows machine.
#
# Usage:
#   bash athena/deploy_athena.sh                                 # interactive: engine or pipeline
#   bash athena/deploy_athena.sh --option1 --preset single       # engine, RECOMMENDED
#   bash athena/deploy_athena.sh --option1 --preset sweep_shift
#   bash athena/deploy_athena.sh --option1 --preset sweep_inner_size
#   bash athena/deploy_athena.sh --option2                       # pipeline, prompts: single or sweep
#   bash athena/deploy_athena.sh --option2 --run=single_sim      # pipeline, single (no prompt)
#   bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.innermost_shift  # pipeline, sweep (no prompt)
#   bash athena/deploy_athena.sh --upload-only
#   bash athena/deploy_athena.sh --watch
#   bash athena/deploy_athena.sh --watch-only
#   bash athena/deploy_athena.sh --results            # prompts: data only or full (incl. .fsp)
#   bash athena/deploy_athena.sh --results-no-fsp    # download .mat / logs only (fast)
#   bash athena/deploy_athena.sh --results-full      # download everything incl. .fsp (heavy)

# ── CONFIGURE — edit athena/athena.conf, not this file ───────────────────────
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
DOWNLOAD_NO_FSP=false
DOWNLOAD_MODE_SET=false
STATUS=false
LICENSE_PROBE=false
OPTION=""
RUN_SCRIPT=""
FSP_PRESET=""
FSP_EXPLICIT=""
SWEEP_KIND=""
SPEC_MODULE=""

for arg in "$@"; do
    case "${arg}" in
        --option1)            OPTION="1" ;;
        --option2)            OPTION="2" ;;
        --option3)            OPTION="3" ;;
        --upload-only)        UPLOAD_ONLY=true ;;
        --results)            DOWNLOAD_RESULTS=true ;;
        --results-no-fsp)     DOWNLOAD_RESULTS=true; DOWNLOAD_NO_FSP=true; DOWNLOAD_MODE_SET=true ;;
        --results-full)       DOWNLOAD_RESULTS=true; DOWNLOAD_NO_FSP=false; DOWNLOAD_MODE_SET=true ;;
        --status)             STATUS=true ;;
        --license-probe)      LICENSE_PROBE=true ;;
        --run=*)              RUN_SCRIPT="${arg#--run=}" ;;
        --preset=*)           FSP_PRESET="${arg#--preset=}" ;;
        --fsp=*)              FSP_EXPLICIT="${arg#--fsp=}" ;;
        --sweep=*)            SWEEP_KIND="${arg#--sweep=}" ;;
        --spec=*)             OPTION="3"; SWEEP_KIND="spec"; SPEC_MODULE="${arg#--spec=}" ;;
        --max-concurrent=*)   MAX_CONCURRENT="${arg#--max-concurrent=}" ;;
        --keep-h5)            KEEP_H5=1 ;;
    esac
done

SSH="${ATHENA_USER}@${ATHENA_HOST}"

# ── Prompt for top-level mode if not specified ────────────────────────────────
# Two conceptual modes presented to the user:
#   1) Engine    → generate .fsp locally, run FDTD engine on GPU (no Python on node)
#   2) Pipeline  → run lumapi Python pipeline on GPU (sub-prompt: single or sweep)
# Internally OPTION is still 1/2/3 (3 = pipeline-as-array). The single→sweep flip
# happens in the pipeline sub-prompt below.
if [[ -z "${OPTION}" && "${UPLOAD_ONLY}" == "false" && "${DOWNLOAD_RESULTS}" == "false" && "${STATUS}" == "false" && "${LICENSE_PROBE}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Athena GPU — Choose run mode:"
    echo "  1) FSP job    — generate .fsp locally → run FDTD engine on GPU"
    echo ""
    echo "  2) Python job — run lumapi Python pipeline on GPU"
    echo "                  (asks: single simulation or sweep)"
    echo "============================================================"
    read -rp "Enter 1 or 2: " _mode_choice
    case "${_mode_choice}" in
        1) OPTION="1" ;;
        2) OPTION="2" ;;  # pipeline; sub-prompt below may flip to OPTION=3 (sweep)
        *) echo "Invalid option. Exiting."; exit 1 ;;
    esac
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

# ── Pipeline sub-prompt: single or sweep ──────────────────────────────────────
# Fires only for OPTION=2 with no explicit script/sweep already passed on CLI.
# Picking "sweep" flips OPTION to 3 (SLURM job array) and routes to the study
# picker below.
if [[ "${OPTION}" == "2" && -z "${RUN_SCRIPT}" && -z "${SWEEP_KIND}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Python pipeline mode:"
    echo "  1) Single simulation"
    echo "  2) Sweep (parallel SLURM job array)"
    echo "============================================================"
    read -rp "Enter 1 or 2: " _pipeline_choice
    case "${_pipeline_choice}" in
        1) _PIPELINE_KIND="single" ;;
        2) OPTION="3"; SWEEP_KIND="spec" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

# ── Single-runner picker (option 2 → single) ──────────────────────────────────
# Mirrors zeus/deploy.sh. Available runners must be wired up in athena_run.py's
# _SCRIPTS dict. Add new runners there once and they're selectable from both servers.
if [[ "${OPTION}" == "2" && "${_PIPELINE_KIND:-}" == "single" && -z "${RUN_SCRIPT}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Choose which single-sim script to run on Athena:"
    echo "  1) single_sim               — Pi-Shift Bragg with cavity     (run_simulation.py)"
    echo "  2) simple_bragg             — uniform Bragg, no cavity        (run_simple_bragg.py)"
    echo "  3) run_experiment           — ExperimentCard example          (run_experiment.py)"
    echo "  4) compare_3d_field_default_vs_shift — 3D + far-field; default vs 100 nm shift"
    echo "============================================================"
    read -rp "Enter 1, 2, 3, or 4: " _script_choice
    case "${_script_choice}" in
        1) RUN_SCRIPT="single_sim" ;;
        2) RUN_SCRIPT="simple_bragg" ;;
        3) RUN_SCRIPT="run_experiment" ;;
        4) RUN_SCRIPT="compare_3d_field_default_vs_shift" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

# ── Sweep study picker (option 3, kind=spec) ──────────────────────────────────
# Auto-discovers any module under runners/sweeps/ that defines a top-level SPEC.
# New studies appear automatically the next time this script runs.
if [[ "${OPTION}" == "3" && "${SWEEP_KIND}" == "spec" && -z "${SPEC_MODULE}" && "${UPLOAD_ONLY}" == "false" ]]; then
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
        echo "Selected: ${SPEC_MODULE}"
    else
        echo "Invalid choice. Exiting."; exit 1
    fi
fi

# ── Sweep kind picker (only when --sweep= not given and not the spec path) ────
# Reachable only via explicit --option3 with no --sweep= and no --spec= (e.g.
# convergence-testing flows). Normal users go through the pipeline sub-prompt.
if [[ "${OPTION}" == "3" && -z "${SWEEP_KIND}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Choose sweep kind:"
    echo "  1) spec            — SweepSpec study (pass --spec=runners.sweeps.<study>)"
    echo "  2) mesh_conv_a     — convergence Phase A (cells_per_half_period)"
    echo "  3) mesh_conv_b     — convergence Phase B (dz_divisor); requires Phase A done"
    echo "============================================================"
    read -rp "Enter 1-3: " _sweep_choice
    case "${_sweep_choice}" in
        1) SWEEP_KIND="spec" ;;
        2) SWEEP_KIND="mesh_conv_a" ;;
        3) SWEEP_KIND="mesh_conv_b" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
    if [[ "${SWEEP_KIND}" == "spec" && -z "${SPEC_MODULE}" ]]; then
        read -rp "Enter dotted module path (e.g. runners.sweeps.innermost_shift): " SPEC_MODULE
        if [[ -z "${SPEC_MODULE}" ]]; then
            echo "ERROR: spec module is required."; exit 1
        fi
    fi
fi

# ── Helper: download results ───────────────────────────────────────────────────
download_results() {
    local no_fsp="${1:-false}"
    echo ""
    mkdir -p "${LOCAL_RESULTS_DIR}"
    if [[ "${no_fsp}" == "true" ]]; then
        echo "=== Downloading results from Athena (data files only, skipping .fsp) ==="
        ssh "${SSH}" "tar --exclude='*.fsp' -czf - -C '${REMOTE_BASE}/results' ." \
            | tar -xzf - -C "${LOCAL_RESULTS_DIR}/"
    else
        echo "=== Downloading results from Athena (full, including .fsp) ==="
        scp -r "${SSH}:${REMOTE_BASE}/results/." "${LOCAL_RESULTS_DIR}/"
    fi
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

# ── --license-probe: report FlexLM seat counts then exit ──────────────────────
if [[ "${LICENSE_PROBE}" == "true" ]]; then
    echo ""
    echo "=== Probing Lumerical license seats on Athena (server: ${ATHENA_LICENSE}) ==="
    # lmutil ships inside the Lumerical install on Athena. The path matches the one
    # used in run_fsp_gpu_array.sh. Keep both in sync if Lumerical's install path moves.
    ssh "${SSH}" "
        for lmutil in /ansys_inc/v261/licensingclient/linx64/lmutil \
                      /opt/lumerical/v261/lumerical/lib/lmutil \
                      /opt/lumerical/v261/bin/lmutil; do
            if [ -x \"\$lmutil\" ]; then
                echo \"Using: \$lmutil\"
                \"\$lmutil\" lmstat -a -c '${ATHENA_LICENSE}' \
                    | grep -E 'Users of (lum_fdtd|fdtd_)' \
                    || echo '  (no FDTD features found)'
                exit 0
            fi
        done
        echo 'ERROR: lmutil not found on Athena. Try inside the container instead:'
        echo '  apptainer exec ~/containers/lumerical-2026R1.sif /ansys_inc/v261/licensingclient/linx64/lmutil lmstat -a -c ${ATHENA_LICENSE}'
    "
    echo ""
    echo "Set MAX_CONCURRENT in athena/athena.conf to (free seats), capped at 8."
    exit 0
fi

# ── --results: immediate download ─────────────────────────────────────────────
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
echo "  uploading runners/ (single + sweeps + studies)"
scp -r "${LOCAL_PROJECT}/runners" "${SSH}:${REMOTE_BASE}/project/"

echo ""
echo "=== Uploading neff data ==="
if [[ -f "${LOCAL_NEFF}" ]]; then
    scp "${LOCAL_NEFF}" "${SSH}:${REMOTE_BASE}/data/FDE_sweep_results.mat"
else
    echo "WARNING: neff data not found at: ${LOCAL_NEFF}"
    echo "  Update LOCAL_NEFF in deploy_athena.sh or upload manually."
fi

echo ""
echo "=== Uploading Athena scripts ==="
scp "${LOCAL_PROJECT}/athena/jobs/"*.sh      "${SSH}:${REMOTE_BASE}/jobs/"
scp "${LOCAL_PROJECT}/athena/scripts/"*.py   "${SSH}:${REMOTE_BASE}/scripts/"
ssh "${SSH}" "chmod +x ${REMOTE_BASE}/jobs/*.sh"
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/jobs/logs"

echo ""
echo "=== Building NVML trampoline on Athena (R470 A100 nodes) ==="
# The trampoline shim stubs three NVML symbols absent from R470 that Lumerical
# 2026 R1's GPU plugin imports. On newer-driver nodes (L40S / H200, R535+) the
# tramp directory is not bound and has no effect.
scp "${LOCAL_PROJECT}/athena/container/nvml_tramp.c" \
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

        # K = max concurrent jobs (license throttle). Set in athena.conf; override
        # for one run via --max-concurrent N. Discover the real seat count with
        # `bash athena/deploy_athena.sh --license-probe`.
        K="${MAX_CONCURRENT:-4}"
        echo "Array: 0-${ARRAY_END}%${K}  (max ${K} concurrent, ~${FSP_COUNT} engine seats used)"

        JOB_ID=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --array=0-${ARRAY_END}%${K} \
                --gpus=1 --cpus-per-task=${N_CPUS} \
                --export=ALL,ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},NTFY_TOPIC=${NTFY_TOPIC} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu_array.sh")
    else
        JOB_ID=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --gpus=1 --cpus-per-task=${N_CPUS} \
                --export=ALL,FSP_FILE=\"${FSP_NAME}\",ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},NTFY_TOPIC=${NTFY_TOPIC} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu.sh")
    fi

elif [[ "${OPTION}" == "2" ]]; then
    # ── Option 2: full Python pipeline (single sequential job) ───────────────
    JOB_ID=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch \
            --gpus=1 --cpus-per-task=${N_CPUS} \
            --export=ALL,RUN_SCRIPT=${RUN_SCRIPT},ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC} \
            --chdir=${REMOTE_BASE}/jobs \
            jobs/run_python_gpu.sh")

else
    # ── Option 3: Python pipeline as a SLURM job array (parallel sweep) ──────
    echo ""
    echo "Option 3: building sweep_list.txt locally for kind='${SWEEP_KIND}'..."
    LOCAL_SWEEP_LIST="${LOCAL_PROJECT}/results_from_athena/_sweep_list.txt"
    LOCAL_PYTHON=$(which python 2>/dev/null || which python3 2>/dev/null)
    if [[ -z "${LOCAL_PYTHON}" ]]; then
        echo "ERROR: no local python found on PATH."
        exit 1
    fi
    if [[ "${SWEEP_KIND}" == "spec" ]]; then
        if [[ -z "${SPEC_MODULE}" ]]; then
            echo "ERROR: --spec=<module> is required for kind=spec."
            exit 1
        fi
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/athena/scripts/build_sweep_list.py" \
            --kind spec --module "${SPEC_MODULE}" --output "${LOCAL_SWEEP_LIST}" 2>&1)
    else
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/athena/scripts/build_sweep_list.py" \
            --kind "${SWEEP_KIND}" --output "${LOCAL_SWEEP_LIST}" 2>&1)
    fi
    echo "${BUILD_OUT}"
    if [[ $? -ne 0 ]]; then
        echo "ERROR: build_sweep_list.py failed."
        exit 1
    fi

    # Capture optional metadata (SWEEP_META: key=value lines) into env vars to
    # forward via sbatch --export. Lines look like "SWEEP_META: param=foo.bar".
    SWEEP_PARAM=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: param=//p')
    SWEEP_FIXED_DZ=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: fixed_dz=//p')
    SWEEP_FIXED_CELLS=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: fixed_cells=//p')
    SWEEP_SPEC_MODULE=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: spec_module=//p')
    # Optional prelim chaining (kind=spec only): when the spec module declares
    # PRELIM_RUN_SCRIPT, submit that single sim first, then queue the array with
    # --dependency=afterok so the array starts only if the prelim succeeds.
    SWEEP_PRELIM_RUN_SCRIPT=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: prelim_run_script=//p')
    SWEEP_LOCKED_LAMBDA_FILE=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: locked_lambda_file=//p')

    N_TASKS=$(grep -c . "${LOCAL_SWEEP_LIST}")
    if [[ "${N_TASKS}" -lt 1 ]]; then
        echo "ERROR: sweep_list.txt is empty."
        exit 1
    fi
    ARRAY_END=$(( N_TASKS - 1 ))
    K="${MAX_CONCURRENT:-4}"

    echo "Uploading sweep_list.txt to Athena..."
    ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/data"
    scp "${LOCAL_SWEEP_LIST}" "${SSH}:${REMOTE_BASE}/data/sweep_list.txt"

    echo "Array: 0-${ARRAY_END}%${K}  (${N_TASKS} tasks, max ${K} concurrent)"
    if [[ -n "${SWEEP_PARAM}" ]];      then echo "  SWEEP_PARAM=${SWEEP_PARAM}"; fi
    if [[ -n "${SWEEP_FIXED_DZ}" ]];   then echo "  SWEEP_FIXED_DZ=${SWEEP_FIXED_DZ}"; fi
    if [[ -n "${SWEEP_FIXED_CELLS}" ]];then echo "  SWEEP_FIXED_CELLS=${SWEEP_FIXED_CELLS}"; fi

    # ── Optional prelim chain ────────────────────────────────────────────────
    # If the spec module declared PRELIM_RUN_SCRIPT, submit a single-sim
    # prerequisite job first. The array then waits with --dependency=afterok.
    # Both jobs receive LOCKED_LAMBDA_FILE in --export so the prelim writes
    # the JSON sidecar and each array task reads it.
    DEP_FLAG=""
    EXTRA_EXPORT=""
    if [[ -n "${SWEEP_PRELIM_RUN_SCRIPT}" ]]; then
        echo ""
        echo "Spec declares prelim: RUN_SCRIPT=${SWEEP_PRELIM_RUN_SCRIPT}"
        echo "  locked-lambda sidecar: ${SWEEP_LOCKED_LAMBDA_FILE:-(spec default)}"
        echo "Submitting prelim (single sim, GPU)..."
        PRELIM_RAW=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --gpus=1 --cpus-per-task=${N_CPUS} \
                --export=ALL,RUN_SCRIPT=${SWEEP_PRELIM_RUN_SCRIPT},LOCKED_LAMBDA_FILE=${SWEEP_LOCKED_LAMBDA_FILE},ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_python_gpu.sh")
        if [[ $? -ne 0 ]]; then
            echo "ERROR: prelim sbatch failed."
            exit 1
        fi
        echo "Submitted prelim: ${PRELIM_RAW}"
        PRELIM_ID=$(echo "${PRELIM_RAW}" | awk '{print $NF}')
        DEP_FLAG="--dependency=afterok:${PRELIM_ID}"
        EXTRA_EXPORT=",LOCKED_LAMBDA_FILE=${SWEEP_LOCKED_LAMBDA_FILE}"
        echo "Array will start after prelim job ${PRELIM_ID} succeeds."
    fi

    JOB_ID=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch \
            --array=0-${ARRAY_END}%${K} ${DEP_FLAG} \
            --gpus=1 --cpus-per-task=${N_CPUS} \
            --export=ALL,SWEEP_KIND=${SWEEP_KIND},SWEEP_LIST=/work/data/sweep_list.txt,SWEEP_PARAM=${SWEEP_PARAM},SWEEP_FIXED_DZ=${SWEEP_FIXED_DZ},SWEEP_FIXED_CELLS=${SWEEP_FIXED_CELLS},SWEEP_SPEC_MODULE=${SWEEP_SPEC_MODULE},ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC}${EXTRA_EXPORT} \
            --chdir=${REMOTE_BASE}/jobs \
            jobs/run_python_array.sh")
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
echo "  bash athena/deploy_athena.sh --status"
echo ""
echo "Live log on Athena:"
echo "  ssh ${SSH} tail -f ${REMOTE_BASE}/jobs/logs/lum_*${NUMERIC_JOB}*.out"
echo ""
echo "Download results after job finishes:"
echo "  bash athena/deploy_athena.sh --results           # prompts for mode"
echo "  bash athena/deploy_athena.sh --results-no-fsp    # data files only (fast)"
echo "  bash athena/deploy_athena.sh --results-full      # everything incl. .fsp (heavy)"
echo "============================================================"
