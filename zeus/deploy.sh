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
#   bash zeus/deploy.sh --results-files     # interactive: pick subfolder, then files
#   bash zeus/deploy.sh --results-files=path1,path2,...  # download specific files (rel to results/)

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
# rsync misreads Windows-style "C:\..." as a remote "host:path"; normalize to
# POSIX form. cygpath ships with Git Bash; on Linux/WSL the path is already POSIX.
if command -v cygpath >/dev/null 2>&1; then
    LOCAL_NEFF=$(cygpath -u "${LOCAL_NEFF}")
fi
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_ONLY=false
DOWNLOAD_RESULTS=false
DOWNLOAD_NO_FSP=false
DOWNLOAD_MODE_SET=false
DOWNLOAD_FILES_MODE=false
DOWNLOAD_FILES_PATHS=""
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
        --results-files)    DOWNLOAD_RESULTS=true; DOWNLOAD_FILES_MODE=true ;;
        --results-files=*)  DOWNLOAD_RESULTS=true; DOWNLOAD_FILES_MODE=true; DOWNLOAD_FILES_PATHS="${arg#--results-files=}" ;;
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

# ── Helper: download specific files (two-step picker or by comma-separated paths) ──
# Step 1 (interactive): list immediate subfolders of ${REMOTE_BASE}/results/, plus a
#                       sentinel "(files at root)" if the directory itself has files.
# Step 2 (interactive): recursively list files inside the picked subfolder, accept
#                       multi-select via "1,3,5", "1-3,7", or "all".
# Non-interactive form: --results-files=path1,path2,... (paths relative to results/).
# Transfer uses a single `tar` over SSH so multi-file selections need only one
# round trip; the local extraction recreates parent directories automatically.
download_files() {
    local _paths=()
    echo ""
    if [[ -n "${DOWNLOAD_FILES_PATHS}" ]]; then
        IFS=',' read -ra _paths <<< "${DOWNLOAD_FILES_PATHS}"
    else
        echo "=== Listing subfolders under ${REMOTE_BASE}/results/ ==="
        local _step1
        _step1=$(ssh "${SSH}" "
            cd '${REMOTE_BASE}/results' 2>/dev/null || exit 1
            for d in */; do [ -d \"\$d\" ] && echo \"DIR \${d%/}\"; done
            for f in *; do [ -f \"\$f\" ] && { echo 'ROOT'; break; }; done
        ")
        local _dirs=() _has_root=false
        while IFS= read -r _line; do
            case "${_line}" in
                'DIR '*) _dirs+=("${_line#DIR }") ;;
                'ROOT')  _has_root=true ;;
            esac
        done <<< "${_step1}"
        if [[ ${#_dirs[@]} -eq 0 && "${_has_root}" == "false" ]]; then
            echo "ERROR: ${REMOTE_BASE}/results/ is empty."
            exit 1
        fi

        local _options=()
        if [[ "${_has_root}" == "true" ]]; then
            _options+=("__ROOT__")
        fi
        for _d in "${_dirs[@]}"; do
            _options+=("${_d}")
        done

        echo ""
        echo "============================================================"
        for _i in "${!_options[@]}"; do
            if [[ "${_options[$_i]}" == "__ROOT__" ]]; then
                printf "  %d) (files at root)\n" "$((_i+1))"
            else
                printf "  %d) %s/\n" "$((_i+1))" "${_options[$_i]}"
            fi
        done
        echo "============================================================"
        read -rp "Pick a subfolder (number): " _dir_choice
        if [[ ! "${_dir_choice}" =~ ^[0-9]+$ ]] || \
           (( _dir_choice < 1 || _dir_choice > ${#_options[@]} )); then
            echo "Invalid choice. Exiting."
            exit 1
        fi

        local _picked="${_options[$((_dir_choice-1))]}"
        local _scope_prefix="" _find_target _maxdepth_arg=""
        if [[ "${_picked}" == "__ROOT__" ]]; then
            _find_target="${REMOTE_BASE}/results"
            _maxdepth_arg="-maxdepth 1"
        else
            _scope_prefix="${_picked}/"
            _find_target="${REMOTE_BASE}/results/${_picked}"
        fi

        echo ""
        echo "=== Files under ${_picked//__ROOT__/(root)} (newest first) ==="
        local _listing
        _listing=$(ssh "${SSH}" \
            "find '${_find_target}' ${_maxdepth_arg} -type f -printf '%T@ %P\n' 2>/dev/null \
                | sort -rn | cut -d' ' -f2-")
        if [[ -z "${_listing}" ]]; then
            echo "ERROR: no files found in selection."
            exit 1
        fi
        local _files=()
        while IFS= read -r _line; do
            [[ -n "${_line}" ]] && _files+=("${_line}")
        done <<< "${_listing}"

        echo ""
        echo "============================================================"
        for _i in "${!_files[@]}"; do
            printf "  %d) %s\n" "$((_i+1))" "${_files[$_i]}"
        done
        echo "============================================================"
        echo "Enter file numbers (e.g. '1,3,5' or '1-3,7' or 'all'):"
        read -rp "> " _file_input
        if [[ -z "${_file_input}" ]]; then
            echo "No selection. Exiting."
            exit 1
        fi

        local _indices=()
        if [[ "${_file_input}" == "all" ]]; then
            for _i in "${!_files[@]}"; do _indices+=("$((_i+1))"); done
        else
            local _toks
            IFS=',' read -ra _toks <<< "${_file_input}"
            for _tok in "${_toks[@]}"; do
                _tok="${_tok// /}"
                if [[ "${_tok}" =~ ^[0-9]+$ ]]; then
                    _indices+=("${_tok}")
                elif [[ "${_tok}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                    local _a="${BASH_REMATCH[1]}" _b="${BASH_REMATCH[2]}"
                    while (( _a <= _b )); do _indices+=("${_a}"); _a=$((_a+1)); done
                else
                    echo "Invalid token: '${_tok}'. Exiting."
                    exit 1
                fi
            done
        fi

        for _idx in "${_indices[@]}"; do
            if (( _idx < 1 || _idx > ${#_files[@]} )); then
                echo "Index out of range: ${_idx}. Exiting."
                exit 1
            fi
            _paths+=("${_scope_prefix}${_files[$((_idx-1))]}")
        done
    fi

    if [[ ${#_paths[@]} -eq 0 ]]; then
        echo "No files selected. Exiting."
        exit 1
    fi

    mkdir -p "${LOCAL_RESULTS_DIR}"
    echo ""
    echo "=== Downloading ${#_paths[@]} file(s) from Zeus ==="
    for _p in "${_paths[@]}"; do echo "  ${_p}"; done
    local _tar_args=""
    for _p in "${_paths[@]}"; do _tar_args+=" '${_p}'"; done
    ssh "${SSH}" "tar -czf - -C '${REMOTE_BASE}/results' ${_tar_args}" \
        | tar -xzf - -C "${LOCAL_RESULTS_DIR}/"
    echo "Saved under: ${LOCAL_RESULTS_DIR}/"
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
    if [[ "${DOWNLOAD_FILES_MODE}" == "true" ]]; then
        download_files
        exit 0
    fi
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
# Core files required to run simulations on Zeus.
# rsync transfers only files whose size or mtime differs from the remote copy;
# repeated deploys are near-instant when nothing has changed.
core_files=()
for f in config.py simulation_config.py sim_helpers.py \
          bragg_device.py analysis.py experiment_card.py \
          post_processing.py; do
    if [[ -f "${LOCAL_PROJECT}/${f}" ]]; then
        core_files+=("${LOCAL_PROJECT}/${f}")
    fi
done
if (( ${#core_files[@]} > 0 )); then
    rsync -av --itemize-changes "${core_files[@]}" "${SSH}:${REMOTE_BASE}/project/"
fi
echo "  syncing runners/ (single + sweeps + studies)"
rsync -av --delete --itemize-changes \
    "${LOCAL_PROJECT}/runners/" "${SSH}:${REMOTE_BASE}/project/runners/"

echo ""
echo "=== Uploading neff data ==="
if [[ -f "${LOCAL_NEFF}" ]]; then
    rsync -av --itemize-changes "${LOCAL_NEFF}" "${SSH}:${REMOTE_BASE}/data/FDE_sweep_results.mat"
else
    echo "WARNING: neff data file not found at:"
    echo "  ${LOCAL_NEFF}"
    echo "  Update LOCAL_NEFF in deploy.sh or upload manually."
fi

echo ""
echo "=== Uploading HPC scripts ==="
rsync -av --itemize-changes "${LOCAL_PROJECT}/zeus/jobs/"*.sh "${SSH}:${REMOTE_BASE}/jobs/"
shopt -s nullglob
zeus_c_files=( "${LOCAL_PROJECT}/zeus/jobs/"*.c )
shopt -u nullglob
if (( ${#zeus_c_files[@]} > 0 )); then
    rsync -av --itemize-changes "${zeus_c_files[@]}" "${SSH}:${REMOTE_BASE}/jobs/"
fi
rsync -av --itemize-changes "${LOCAL_PROJECT}/zeus/scripts/"*.py "${SSH}:${REMOTE_BASE}/scripts/"
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/jobs/bin"
rsync -av --itemize-changes "${LOCAL_PROJECT}/zeus/jobs/bin/fdtd-solutions" "${SSH}:${REMOTE_BASE}/jobs/bin/fdtd-solutions"
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

    mapfile -t FSP_PATHS < <(echo "${FSP_OUTPUT}" | sed -n 's/^FSP_SAVED://p')
    if [[ ${#FSP_PATHS[@]} -eq 0 ]]; then
        echo "ERROR: Failed to generate .fsp file locally."
        exit 1
    fi

    # Upload each .fsp into its own per-stem folder under ${REMOTE_BASE}/results/<stem>/.
    # Engine writes outputs alongside the input. Zeus runs single jobs only — if
    # the preset produced multiple .fsps, only the first is submitted; warn loudly.
    echo "Uploading ${#FSP_PATHS[@]} .fsp file(s) to Zeus (per-stem folders)..."
    for _fsp in "${FSP_PATHS[@]}"; do
        _name=$(basename "${_fsp}")
        _stem="${_name%.fsp}"
        ssh "${SSH}" "mkdir -p '${REMOTE_BASE}/results/${_stem}'"
        scp "${_fsp}" "${SSH}:${REMOTE_BASE}/results/${_stem}/${_name}"
    done

    if [[ ${#FSP_PATHS[@]} -gt 1 ]]; then
        echo "WARNING: Zeus does not support sweep arrays — only the first .fsp will be submitted."
        echo "         Use Athena (athena/deploy_athena.sh) for multi-FSP sweeps."
    fi
    FSP_NAME=$(basename "${FSP_PATHS[0]}")
    FSP_STEM="${FSP_NAME%.fsp}"

    JOB_ID=$(ssh "${SSH}" "cd ${REMOTE_BASE} && qsub -l select=1:ncpus=${N_CPUS} -v FSP_FILE=\"${FSP_NAME}\",FSP_STEM=\"${FSP_STEM}\",NTFY_TOPIC=${NTFY_TOPIC} jobs/run_fsp_job.sh")
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
