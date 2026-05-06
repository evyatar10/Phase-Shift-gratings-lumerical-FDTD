#!/bin/bash
#
# Local script: upload project files to the DGX cluster and submit a SLURM GPU job.
# This is the DGX analog of zeus/deploy.sh (Zeus/PBS). Zeus is NOT touched.
# Target: dgx-master.technion.ac.il (legacy cluster, R470 A100, needs nvml_tramp).
#
# Run from Git Bash, WSL, or VS Code tasks on your local Windows machine.
#
# Usage:
#   bash dgx/deploy_dgx.sh                                 # interactive: engine or pipeline
#   bash dgx/deploy_dgx.sh --option1 --preset single       # engine, RECOMMENDED
#   bash dgx/deploy_dgx.sh --option1 --preset sweep_shift
#   bash dgx/deploy_dgx.sh --option1 --preset sweep_inner_size
#   bash dgx/deploy_dgx.sh --option2                       # pipeline, prompts: single or sweep
#   bash dgx/deploy_dgx.sh --option2 --run=single_sim      # pipeline, single (no prompt)
#   bash dgx/deploy_dgx.sh --option3 --spec=runners.sweeps.innermost_shift  # pipeline, sweep (no prompt)
#   bash dgx/deploy_dgx.sh --upload-only
#   bash dgx/deploy_dgx.sh --watch
#   bash dgx/deploy_dgx.sh --watch-only
#   bash dgx/deploy_dgx.sh --results            # prompts: data only or full (incl. .fsp)
#   bash dgx/deploy_dgx.sh --results-no-fsp    # download .mat / logs only (fast)
#   bash dgx/deploy_dgx.sh --results-full      # download everything incl. .fsp (heavy)
#   bash dgx/deploy_dgx.sh --results-files     # interactive: pick subfolder, then files
#   bash dgx/deploy_dgx.sh --results-files=path1,path2,...  # download specific files (rel to results/)

# ── CONFIGURE — edit dgx/dgx.conf, not this file ─────────────────────────────
CONF="$(cd "$(dirname "$0")" && pwd)/dgx.conf"
if [[ ! -f "${CONF}" ]]; then
    echo "ERROR: dgx.conf not found at ${CONF}"
    exit 1
fi
source "${CONF}"

LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RESULTS_DIR="${LOCAL_PROJECT}/results_from_dgx"
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
        --results-files)      DOWNLOAD_RESULTS=true; DOWNLOAD_FILES_MODE=true ;;
        --results-files=*)    DOWNLOAD_RESULTS=true; DOWNLOAD_FILES_MODE=true; DOWNLOAD_FILES_PATHS="${arg#--results-files=}" ;;
        --status)             STATUS=true ;;
        --license-probe)      LICENSE_PROBE=true ;;
        --run=*)              RUN_SCRIPT="${arg#--run=}" ;;
        --preset=*)           FSP_PRESET="${arg#--preset=}" ;;
        --fsp=*)              FSP_EXPLICIT="${arg#--fsp=}" ;;
        --sweep=*)            SWEEP_KIND="${arg#--sweep=}" ;;
        --spec=*)             OPTION="3"; SWEEP_KIND="spec"; SPEC_MODULE="${arg#--spec=}" ;;
        --inverse-design=*)   OPTION="3"; SWEEP_KIND="inverse_design"; SPEC_MODULE="${arg#--inverse-design=}" ;;
        --max-concurrent=*)   MAX_CONCURRENT="${arg#--max-concurrent=}" ;;
        --keep-h5)            KEEP_H5=1 ;;
    esac
done

SSH="${DGX_USER}@${DGX_HOST}"

# ── Prompt for top-level mode if not specified ────────────────────────────────
# Two conceptual modes presented to the user:
#   1) Engine    → generate .fsp locally, run FDTD engine on GPU (no Python on node)
#   2) Pipeline  → run lumapi Python pipeline on GPU (sub-prompt: single or sweep)
# Internally OPTION is still 1/2/3 (3 = pipeline-as-array). The single→sweep flip
# happens in the pipeline sub-prompt below.
if [[ -z "${OPTION}" && "${UPLOAD_ONLY}" == "false" && "${DOWNLOAD_RESULTS}" == "false" && "${STATUS}" == "false" && "${LICENSE_PROBE}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  DGX GPU — Choose run mode:"
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

# ── Pipeline sub-prompt: single / sweep / convergence ────────────────────────
# Fires only for OPTION=2 with no explicit script/sweep already passed on CLI.
# Each branch dispatches to its OWN folder picker — no cross-mixing.
#   1) Single       → runners/single/*.py  (OPTION=2, sequential job)
#   2) Sweep        → runners/sweeps/*.py  (OPTION=3, kind=spec, one entry per study)
#   3) Convergence  → convergence_testing/*.py
#                     (files with PHASES = [...] expand to one entry per phase
#                      and submit as SLURM arrays; bare files run sequentially)
if [[ "${OPTION}" == "2" && -z "${RUN_SCRIPT}" && -z "${SWEEP_KIND}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  Python pipeline mode:"
    echo "  1) Single       (one node, sequential — runners/single/)"
    echo "  2) Sweep        (parallel SLURM job array — runners/sweeps/)"
    echo "  3) Convergence  (convergence_testing/ — incl. mesh_conv X/YZ)"
    echo "============================================================"
    read -rp "Enter 1, 2, or 3: " _pipeline_choice
    case "${_pipeline_choice}" in
        1) _PIPELINE_KIND="single" ;;
        2) OPTION="3"; _PIPELINE_KIND="sweep" ;;
        3) _PIPELINE_KIND="convergence" ;;
        *) echo "Invalid choice. Exiting."; exit 1 ;;
    esac
fi

# ── Single-runner picker (mode=single) — AUTO-DISCOVERED ─────────────────────
# Scans runners/single/*.py only. Any module with a top-level `run` callable
# appears; drop a new file in, it shows up next time.
# Files starting with "_" or named __init__.py are skipped.
if [[ "${_PIPELINE_KIND:-}" == "single" && -z "${RUN_SCRIPT}" && "${UPLOAD_ONLY}" == "false" ]]; then
    # Static scan (grep, not import) — local Python doesn't have lumapi.
    mapfile -t _RUNNERS < <(
        for _f in "${LOCAL_PROJECT}/runners/single"/*.py; do
            [[ -e "${_f}" ]] || continue
            _name=$(basename "${_f}" .py)
            [[ "${_name}" == _* || "${_name}" == "__init__" ]] && continue
            grep -qE '^(def[[:space:]]+run[[:space:]]*\(|run[[:space:]]*=)' "${_f}" 2>/dev/null || continue
            grep -qE '^IS_HELPER[[:space:]]*=[[:space:]]*True' "${_f}" 2>/dev/null && continue
            echo "${_name}"
        done | sort
    )
    if [[ ${#_RUNNERS[@]} -eq 0 ]]; then
        echo "ERROR: no runners discovered in runners/single/."
        echo "       Each module needs a top-level callable 'run'."
        exit 1
    fi
    echo ""
    echo "============================================================"
    echo "  Choose which script to run on DGX (runners/single/):"
    for _i in "${!_RUNNERS[@]}"; do
        printf "  %d) %s\n" "$((_i+1))" "${_RUNNERS[$_i]}"
    done
    echo "============================================================"
    read -rp "Enter number: " _script_choice
    if [[ "${_script_choice}" =~ ^[0-9]+$ ]] && (( _script_choice >= 1 && _script_choice <= ${#_RUNNERS[@]} )); then
        RUN_SCRIPT="${_RUNNERS[$((_script_choice-1))]}"
        echo "Selected: ${RUN_SCRIPT}"
    else
        echo "Invalid choice. Exiting."; exit 1
    fi
fi

# ── Sweep study picker (mode=sweep) ──────────────────────────────────────────
# Auto-discovers any module under runners/sweeps/ that defines a top-level SPEC.
# Selecting a study sets SWEEP_KIND=spec and SPEC_MODULE — no separate prompt.
if [[ "${_PIPELINE_KIND:-}" == "sweep" && -z "${SPEC_MODULE}" && "${UPLOAD_ONLY}" == "false" ]]; then
    SWEEP_KIND="spec"
    # Hide prelim-only helpers (IS_PRELIM = True) — internal to chained workflows.
    mapfile -t _STUDIES < <(
        for _f in $(grep -rl '[[:space:]]*SPEC[[:space:]]*=' "${LOCAL_PROJECT}/runners/sweeps/"*.py 2>/dev/null \
                        | grep -v 'sweep_spec\.py'); do
            grep -q '^IS_PRELIM[[:space:]]*=[[:space:]]*True' "${_f}" || basename "${_f}" .py
        done | sort
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

# ── Convergence picker (mode=convergence) — AUTO-DISCOVERED w/ phase expansion ─
# Scans convergence_testing/*.py. Two flavors per file:
#   • Has PHASES = [...] and KIND_PREFIX = "..." → one entry per phase,
#     each runs as a SLURM array with SWEEP_KIND=<prefix>_<phase_lower>.
#   • Otherwise → one sequential entry that runs as a normal pipeline job.
# Drop a new file in convergence_testing/ → it appears automatically.
if [[ "${_PIPELINE_KIND:-}" == "convergence" && -z "${RUN_SCRIPT}" && -z "${SWEEP_KIND}" && "${UPLOAD_ONLY}" == "false" ]]; then
    # Each row: "<label>|<routing>|<payload>"
    #   routing=array  → payload is SWEEP_KIND
    #   routing=single → payload is RUN_SCRIPT (module bare name)
    mapfile -t _CONV < <(
        for _f in "${LOCAL_PROJECT}/convergence_testing"/*.py; do
            [[ -e "${_f}" ]] || continue
            _name=$(basename "${_f}" .py)
            [[ "${_name}" == _* || "${_name}" == "__init__" ]] && continue
            grep -qE '^(def[[:space:]]+run[[:space:]]*\(|run[[:space:]]*=)' "${_f}" 2>/dev/null || continue
            _kind_prefix=$(grep -E '^KIND_PREFIX[[:space:]]*=' "${_f}" 2>/dev/null \
                | sed -E 's/^KIND_PREFIX[[:space:]]*=[[:space:]]*"([^"]+)".*/\1/' | head -1)
            _phases_raw=$(grep -E '^PHASES[[:space:]]*=' "${_f}" 2>/dev/null | head -1)
            _phases=$(echo "${_phases_raw}" | grep -oE '"[A-Za-z0-9_]+"' | tr -d '"')
            if [[ -n "${_phases}" && -n "${_kind_prefix}" ]]; then
                for _ph in ${_phases}; do
                    _ph_lower=$(echo "${_ph}" | tr '[:upper:]' '[:lower:]')
                    printf "%s %s (array)|array|%s_%s\n" "${_name}" "${_ph}" "${_kind_prefix}" "${_ph_lower}"
                done
            else
                printf "%s (sequential)|single|%s\n" "${_name}" "${_name}"
            fi
        done | sort
    )
    if [[ ${#_CONV[@]} -eq 0 ]]; then
        echo "ERROR: no convergence modules discovered in convergence_testing/."
        exit 1
    fi
    echo ""
    echo "============================================================"
    echo "  Choose convergence run (convergence_testing/):"
    for _i in "${!_CONV[@]}"; do
        IFS='|' read -r _label _ _ <<<"${_CONV[$_i]}"
        printf "  %d) %s\n" "$((_i+1))" "${_label}"
    done
    echo "============================================================"
    read -rp "Enter number: " _conv_choice
    if [[ "${_conv_choice}" =~ ^[0-9]+$ ]] && (( _conv_choice >= 1 && _conv_choice <= ${#_CONV[@]} )); then
        IFS='|' read -r _label _routing _payload <<<"${_CONV[$((_conv_choice-1))]}"
        if [[ "${_routing}" == "array" ]]; then
            OPTION="3"
            SWEEP_KIND="${_payload}"
            echo "Selected: ${_label} → SWEEP_KIND=${SWEEP_KIND}"
        else
            OPTION="2"
            RUN_SCRIPT="${_payload}"
            echo "Selected: ${_label} → RUN_SCRIPT=${RUN_SCRIPT}"
        fi
    else
        echo "Invalid choice. Exiting."; exit 1
    fi
fi

# ── Sweep kind picker (defensive fallback) ────────────────────────────────────
# Only reachable via explicit --option3 with no --sweep= and no --spec= on CLI.
# Interactive flow always sets SWEEP_KIND in the mode pickers above.
if [[ "${OPTION}" == "3" && -z "${SWEEP_KIND}" && "${UPLOAD_ONLY}" == "false" ]]; then
    echo "ERROR: --option3 requires --sweep=<kind> or --spec=<module>."
    echo "       Use the interactive flow (no flags) for the menu."
    exit 1
fi

# ── Helper: download results ───────────────────────────────────────────────────
download_results() {
    local no_fsp="${1:-false}"
    echo ""
    mkdir -p "${LOCAL_RESULTS_DIR}"
    if [[ "${no_fsp}" == "true" ]]; then
        echo "=== Downloading results from DGX (data files only, skipping .fsp) ==="
        ssh "${SSH}" "tar --exclude='*.fsp' -czf - -C '${REMOTE_BASE}/results' ." \
            | tar -xzf - -C "${LOCAL_RESULTS_DIR}/"
    else
        echo "=== Downloading results from DGX (full, including .fsp) ==="
        scp -r "${SSH}:${REMOTE_BASE}/results/." "${LOCAL_RESULTS_DIR}/"
    fi
    echo "Results saved to: ${LOCAL_RESULTS_DIR}"
}

# ── Helper: download specific files (categorized picker or by relative paths) ──
# Three-step interactive picker:
#   1) Category — convergence / runners/single / runners/sweeps
#   2) Run folder — discovered remote dirs whose name matches a module under
#                   convergence_testing/, runners/single/, or runners/sweeps/
#   3) Files     — multi-select via "1,3,5" | "1-3,7" | "all"
# Categories auto-update: every invocation re-scans the local source tree, so
# new modules and new run folders appear without code changes here.
# Non-interactive form: --results-files=path1,path2,... (paths relative to results/).
# Transfer uses a single tar stream so multi-file selections need only one
# round-trip; the local extraction recreates parent directories automatically.
download_files() {
    local _paths=()
    echo ""
    if [[ -n "${DOWNLOAD_FILES_PATHS}" ]]; then
        IFS=',' read -ra _paths <<< "${DOWNLOAD_FILES_PATHS}"
    else
        # ── Auto-discover categories from the local source tree ─────────────
        # Categories = convergence_testing/ + every subdirectory of runners/.
        # Adding a new runners/<name>/ later (or a new module file inside any
        # category) makes it appear automatically — no code changes here.
        # Parallel arrays:  _cat_dirs[i] / _cat_mods[i] / _cat_runs[i]
        local _cat_dirs=() _cat_mods=() _cat_runs=()
        _add_category() {
            # $1 = category label (e.g. "convergence_testing", "runners/single")
            # $2 = absolute filesystem path to the directory holding the .py modules
            local _label="$1" _path="$2" _mods="" _f _n
            [[ -d "${_path}" ]] || return 0
            for _f in "${_path}"/*.py; do
                [[ -e "${_f}" ]] || continue
                _n=$(basename "${_f}" .py)
                [[ "${_n}" == _* || "${_n}" == "__init__" || "${_n}" == "sweep_spec" ]] && continue
                _mods+=" ${_n}"
                # Strip a "run_" prefix too: run_mesh_convergence.py writes to
                # CONV_DIR="mesh_convergence" and run_auto_shutoff_convergence
                # follows the same pattern.
                [[ "${_n}" == run_* ]] && _mods+=" ${_n#run_}"
            done
            _cat_dirs+=("${_label}")
            _cat_mods+=("${_mods}")
            _cat_runs+=("")
        }
        _add_category "convergence_testing" "${LOCAL_PROJECT}/convergence_testing"
        local _runner_dir
        for _runner_dir in "${LOCAL_PROJECT}"/runners/*/; do
            [[ -d "${_runner_dir}" ]] || continue
            local _rname=$(basename "${_runner_dir}")
            [[ "${_rname}" == _* || "${_rname}" == "__pycache__" ]] && continue
            _add_category "runners/${_rname}" "${_runner_dir%/}"
        done

        echo "=== Discovering run folders under ${REMOTE_BASE}/results/ on DGX ==="
        local _remote_dirs
        _remote_dirs=$(ssh "${SSH}" \
            "cd '${REMOTE_BASE}/results' 2>/dev/null && \
             for d in */; do [ -d \"\$d\" ] && echo \"\${d%/}\"; done")

        # Bucket each remote dir into the first category whose module list contains it.
        local _d _i
        while IFS= read -r _d; do
            [[ -z "${_d}" ]] && continue
            for _i in "${!_cat_dirs[@]}"; do
                if [[ " ${_cat_mods[$_i]} " == *" ${_d} "* ]]; then
                    _cat_runs[$_i]+=" ${_d}"
                    break
                fi
            done
        done <<< "${_remote_dirs}"

        # Always show every discovered category, even if it has zero runs on the
        # remote — this lets the user see the project structure and confirms
        # which subdir of runners/ a future run will land in. Empty categories
        # are flagged "(empty)" and rejected at step 2.
        local _menu_labels=()
        for _i in "${!_cat_dirs[@]}"; do
            local _trim="${_cat_runs[$_i]## }"
            local _count=0
            [[ -n "${_trim}" ]] && _count=$(echo "${_trim}" | wc -w)
            local _suffix
            if (( _count == 0 )); then
                _suffix="(empty)"
            else
                _suffix="(${_count} run(s))"
            fi
            _menu_labels+=("$(printf '%-25s %s' "${_cat_dirs[$_i]}" "${_suffix}")")
        done

        if [[ ${#_cat_dirs[@]} -eq 0 ]]; then
            echo "ERROR: no categories discovered. convergence_testing/ and runners/ are missing locally."
            exit 1
        fi

        echo ""
        echo "============================================================"
        echo "  Step 1: choose a category"
        echo "============================================================"
        for _i in "${!_menu_labels[@]}"; do
            printf "  %d) %s\n" "$((_i+1))" "${_menu_labels[$_i]}"
        done
        echo "============================================================"
        read -rp "Pick a category (number): " _cat_choice
        if [[ ! "${_cat_choice}" =~ ^[0-9]+$ ]] || \
           (( _cat_choice < 1 || _cat_choice > ${#_cat_dirs[@]} )); then
            echo "Invalid choice. Exiting."
            exit 1
        fi

        local _cat_pick_idx=$((_cat_choice-1))
        local _runs=()
        read -ra _runs <<< "${_cat_runs[$_cat_pick_idx]}"
        if [[ ${#_runs[@]} -eq 0 ]]; then
            echo ""
            echo "No runs in '${_cat_dirs[$_cat_pick_idx]}' yet on ${REMOTE_BASE}/results/."
            echo "Once a script in this category runs on the server, its output folder will appear here."
            exit 0
        fi

        echo ""
        echo "============================================================"
        echo "  Step 2: choose a run"
        echo "============================================================"
        for _i in "${!_runs[@]}"; do
            printf "  %d) %s/\n" "$((_i+1))" "${_runs[$_i]}"
        done
        echo "============================================================"
        read -rp "Pick a run (number): " _run_choice
        if [[ ! "${_run_choice}" =~ ^[0-9]+$ ]] || \
           (( _run_choice < 1 || _run_choice > ${#_runs[@]} )); then
            echo "Invalid choice. Exiting."
            exit 1
        fi
        local _picked="${_runs[$((_run_choice-1))]}"

        echo ""
        echo "=== Files under ${_picked}/ (newest first) ==="
        local _listing
        _listing=$(ssh "${SSH}" \
            "find '${REMOTE_BASE}/results/${_picked}' -type f -printf '%T@ %P\n' 2>/dev/null \
                | sort -rn | cut -d' ' -f2-")
        if [[ -z "${_listing}" ]]; then
            echo "ERROR: no files in ${_picked}/."
            exit 1
        fi
        local _files=()
        while IFS= read -r _line; do
            [[ -n "${_line}" ]] && _files+=("${_line}")
        done <<< "${_listing}"

        echo ""
        echo "============================================================"
        echo "  Step 3: pick files (multi-select)"
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
            _paths+=("${_picked}/${_files[$((_idx-1))]}")
        done
    fi

    if [[ ${#_paths[@]} -eq 0 ]]; then
        echo "No files selected. Exiting."
        exit 1
    fi

    mkdir -p "${LOCAL_RESULTS_DIR}"
    echo ""
    echo "=== Downloading ${#_paths[@]} file(s) from DGX ==="
    for _p in "${_paths[@]}"; do echo "  ${_p}"; done
    local _tar_args=""
    for _p in "${_paths[@]}"; do _tar_args+=" '${_p}'"; done
    ssh "${SSH}" "tar -czf - -C '${REMOTE_BASE}/results' ${_tar_args}" \
        | tar -xzf - -C "${LOCAL_RESULTS_DIR}/"
    echo "Saved under: ${LOCAL_RESULTS_DIR}/"
}

# ── Helper: one-shot status check ─────────────────────────────────────────────
check_status() {
    echo ""
    echo "=== Jobs in queue (${DGX_USER}) ==="
    ssh "${SSH}" "squeue -u ${DGX_USER} -o '%.10i %.12j %.8T %.10M %.6D %R' 2>/dev/null || echo '(no jobs in queue)'"
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
    echo "=== Probing Lumerical license seats on DGX (server: ${DGX_LICENSE}) ==="
    # lmutil ships inside the Lumerical install on DGX. The path matches the one
    # used in run_fsp_gpu_array.sh. Keep both in sync if Lumerical's install path moves.
    ssh "${SSH}" "
        for lmutil in /ansys_inc/v261/licensingclient/linx64/lmutil \
                      /opt/lumerical/v261/lumerical/lib/lmutil \
                      /opt/lumerical/v261/bin/lmutil; do
            if [ -x \"\$lmutil\" ]; then
                echo \"Using: \$lmutil\"
                \"\$lmutil\" lmstat -a -c '${DGX_LICENSE}' \
                    | grep -E 'Users of (lum_fdtd|fdtd_)' \
                    || echo '  (no FDTD features found)'
                exit 0
            fi
        done
        echo 'ERROR: lmutil not found on DGX. Try inside the container instead:'
        echo '  apptainer exec ~/containers/lumerical-2026R1.sif /ansys_inc/v261/licensingclient/linx64/lmutil lmstat -a -c ${DGX_LICENSE}'
    "
    echo ""
    echo "Set MAX_CONCURRENT in dgx/dgx.conf to (free seats), capped at 8."
    exit 0
fi

# ── --results: immediate download ─────────────────────────────────────────────
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
        echo "  3) Specific files  — pick subfolder, then files (interactive)"
        echo "============================================================"
        read -rp "Enter 1, 2, or 3: " _dl_choice
        case "${_dl_choice}" in
            1) DOWNLOAD_NO_FSP=true ;;
            2) DOWNLOAD_NO_FSP=false ;;
            3) download_files; exit 0 ;;
            *) echo "Invalid choice. Exiting."; exit 1 ;;
        esac
    fi
    download_results "${DOWNLOAD_NO_FSP}"
    exit 0
fi

# ── Normal flow: create remote directories and upload ─────────────────────────
echo "============================================================"
echo "Target: ${SSH}  (DGX cluster — dgx-master.technion.ac.il)"
echo "Remote: ${REMOTE_BASE}"
echo "============================================================"

echo ""
echo "=== Creating remote directories ==="
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/{project,data,results/layouts,jobs/logs,scripts}"

echo ""
echo "=== Uploading project files ==="
# rsync transfers only files whose size or mtime differs from the remote copy;
# repeated deploys are near-instant when nothing has changed.
rsync -av --itemize-changes "${LOCAL_PROJECT}"/*.py "${SSH}:${REMOTE_BASE}/project/"
echo "  syncing runners/ (single + sweeps + studies)"
rsync -av --delete --itemize-changes \
    "${LOCAL_PROJECT}/runners/" "${SSH}:${REMOTE_BASE}/project/runners/"
echo "  syncing convergence_testing/"
rsync -av --delete --itemize-changes \
    "${LOCAL_PROJECT}/convergence_testing/" "${SSH}:${REMOTE_BASE}/project/convergence_testing/"

echo ""
echo "=== Uploading neff data ==="
if [[ -f "${LOCAL_NEFF}" ]]; then
    rsync -av --itemize-changes "${LOCAL_NEFF}" "${SSH}:${REMOTE_BASE}/data/FDE_sweep_results.mat"
else
    echo "WARNING: neff data not found at: ${LOCAL_NEFF}"
    echo "  Update LOCAL_NEFF in deploy_athena.sh or upload manually."
fi

echo ""
echo "=== Uploading DGX scripts ==="
rsync -av --itemize-changes "${LOCAL_PROJECT}/dgx/jobs/"*.sh    "${SSH}:${REMOTE_BASE}/jobs/"
rsync -av --itemize-changes "${LOCAL_PROJECT}/dgx/scripts/"*.py "${SSH}:${REMOTE_BASE}/scripts/"
ssh "${SSH}" "chmod +x ${REMOTE_BASE}/jobs/*.sh"
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/jobs/logs"

echo ""
echo "=== Building NVML trampoline on DGX (R470 A100 nodes) ==="
# The trampoline shim stubs three NVML symbols absent from R470 that Lumerical
# 2026 R1's GPU plugin imports. On newer-driver nodes (L40S / H200, R535+) the
# tramp directory is not bound and has no effect.
rsync -av --itemize-changes "${LOCAL_PROJECT}/container/nvml_tramp.c" \
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
    # Each .fsp lands in its own per-stem folder: ${REMOTE_BASE}/results/<stem>/<stem>.fsp
    # Engine writes its outputs (.h5, .log) into the same folder. fsp_list.txt
    # for arrays sits at ${REMOTE_BASE}/results/fsp_list.txt and stores stems
    # (one per line, no extension, no parent dir).
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

        mapfile -t FSP_PATHS < <(echo "${FSP_OUTPUT}" | sed -n 's/^FSP_SAVED://p')
        if [[ ${#FSP_PATHS[@]} -eq 0 ]]; then
            echo "ERROR: Failed to generate .fsp file locally."
            exit 1
        fi

        echo "Uploading ${#FSP_PATHS[@]} .fsp file(s) to DGX (per-stem folders)..."
        for _fsp in "${FSP_PATHS[@]}"; do
            _name=$(basename "${_fsp}")
            _stem="${_name%.fsp}"
            ssh "${SSH}" "mkdir -p '${REMOTE_BASE}/results/${_stem}'"
            scp "${_fsp}" "${SSH}:${REMOTE_BASE}/results/${_stem}/${_name}"
        done
        # FSP_NAME is only consulted in the single-file branch below.
        FSP_NAME=$(basename "${FSP_PATHS[0]}")
    fi

    # Detect sweep based on the freshly uploaded .fsp set, not a remote glob —
    # legacy files in ${REMOTE_BASE}/results/layouts/ from earlier runs must not
    # influence the submission mode of the current deploy.
    if [[ -n "${FSP_EXPLICIT}" ]]; then
        FSP_COUNT=1
    else
        FSP_COUNT=${#FSP_PATHS[@]}
    fi
    echo "This deploy uploaded ${FSP_COUNT} .fsp file(s)."

    if [[ "${FSP_COUNT}" -gt 1 ]]; then
        echo ""
        echo "Multiple .fsp files detected — submitting as a SLURM job array."
        # Build fsp_list.txt locally (one stem per line) and upload to
        # ${REMOTE_BASE}/results/fsp_list.txt. The array job reads line N and
        # derives FSP_FILE and FSP_DIR from the stem.
        LOCAL_FSP_LIST="${LOCAL_PROJECT}/results_from_dgx/_fsp_list.txt"
        : > "${LOCAL_FSP_LIST}"
        for _fsp in "${FSP_PATHS[@]}"; do
            _name=$(basename "${_fsp}")
            echo "${_name%.fsp}" >> "${LOCAL_FSP_LIST}"
        done
        scp "${LOCAL_FSP_LIST}" "${SSH}:${REMOTE_BASE}/results/fsp_list.txt"
        ARRAY_END=$(( FSP_COUNT - 1 ))

        # K = max concurrent jobs (license throttle). Set in dgx.conf; override
        # for one run via --max-concurrent N. Discover the real seat count with
        # `bash athena/deploy_athena.sh --license-probe`.
        K="${MAX_CONCURRENT:-4}"
        echo "Array: 0-${ARRAY_END}%${K}  (max ${K} concurrent, ~${FSP_COUNT} engine seats used)"

        JOB_ID=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --array=0-${ARRAY_END}%${K} \
                --gpus=1 --cpus-per-task=${N_CPUS} \
                --export=ALL,DGX_LICENSE=${DGX_LICENSE},DGX_INTERCONNECT=${DGX_INTERCONNECT},NTFY_TOPIC=${NTFY_TOPIC} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu_array.sh")
    else
        JOB_ID=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --gpus=1 --cpus-per-task=${N_CPUS} \
                --export=ALL,FSP_FILE=\"${FSP_NAME}\",DGX_LICENSE=${DGX_LICENSE},DGX_INTERCONNECT=${DGX_INTERCONNECT},NTFY_TOPIC=${NTFY_TOPIC} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu.sh")
    fi

elif [[ "${OPTION}" == "2" ]]; then
    # ── Option 2: full Python pipeline (single sequential job) ───────────────
    JOB_ID=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch \
            --gpus=1 --cpus-per-task=${N_CPUS} \
            --export=ALL,RUN_SCRIPT=${RUN_SCRIPT},DGX_LICENSE=${DGX_LICENSE},DGX_INTERCONNECT=${DGX_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC} \
            --chdir=${REMOTE_BASE}/jobs \
            jobs/run_python_gpu.sh")

else
    # ── Option 3: Python pipeline as a SLURM job array (parallel sweep) ──────
    echo ""
    echo "Option 3: building sweep_list.txt locally for kind='${SWEEP_KIND}'..."
    LOCAL_SWEEP_LIST="${LOCAL_PROJECT}/results_from_dgx/_sweep_list.txt"
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
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/dgx/scripts/build_sweep_list.py" \
            --kind spec --module "${SPEC_MODULE}" --output "${LOCAL_SWEEP_LIST}" 2>&1)
    elif [[ "${SWEEP_KIND}" == "inverse_design" ]]; then
        if [[ -z "${SPEC_MODULE}" ]]; then
            echo "ERROR: --inverse-design=<module> is required for kind=inverse_design."
            exit 1
        fi
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/dgx/scripts/build_sweep_list.py" \
            --kind inverse_design --module "${SPEC_MODULE}" --output "${LOCAL_SWEEP_LIST}" 2>&1)
    else
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/dgx/scripts/build_sweep_list.py" \
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
    SWEEP_FIXED_DYZ_NM=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: fixed_dyz_nm=//p')
    SWEEP_FIXED_CELLS=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: fixed_cells=//p')
    SWEEP_SPEC_MODULE=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: spec_module=//p')
    # Optional prelim chaining (kind=spec only): when the spec module declares
    # PRELIM_RUN_SCRIPT, submit that single sim first, then queue the array with
    # --dependency=afterok so the array starts only if the prelim succeeds.
    SWEEP_PRELIM_RUN_SCRIPT=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: prelim_run_script=//p')
    SWEEP_HAS_PRELIM_SPEC=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: has_prelim_spec=//p')
    SWEEP_LOCKED_LAMBDA_FILE=$(echo "${BUILD_OUT}" | sed -n 's/^SWEEP_META: locked_lambda_file=//p')

    N_TASKS=$(grep -c . "${LOCAL_SWEEP_LIST}")
    if [[ "${N_TASKS}" -lt 1 ]]; then
        echo "ERROR: sweep_list.txt is empty."
        exit 1
    fi
    ARRAY_END=$(( N_TASKS - 1 ))
    K="${MAX_CONCURRENT:-4}"

    echo "Uploading sweep_list.txt to DGX..."
    ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/data"
    scp "${LOCAL_SWEEP_LIST}" "${SSH}:${REMOTE_BASE}/data/sweep_list.txt"

    echo "Array: 0-${ARRAY_END}%${K}  (${N_TASKS} tasks, max ${K} concurrent)"
    if [[ -n "${SWEEP_PARAM}" ]];        then echo "  SWEEP_PARAM=${SWEEP_PARAM}"; fi
    if [[ -n "${SWEEP_FIXED_DYZ_NM}" ]]; then echo "  SWEEP_FIXED_DYZ_NM=${SWEEP_FIXED_DYZ_NM}"; fi
    if [[ -n "${SWEEP_FIXED_CELLS}" ]];  then echo "  SWEEP_FIXED_CELLS=${SWEEP_FIXED_CELLS}"; fi

    # ── Optional prelim chain ────────────────────────────────────────────────
    # If the spec module declared PRELIM_RUN_SCRIPT, submit a single-sim
    # prerequisite job first. The array then waits with --dependency=afterok.
    # Both jobs receive LOCKED_LAMBDA_FILE in --export so the prelim writes
    # the JSON sidecar and each array task reads it.
    DEP_FLAG=""
    EXTRA_EXPORT=""
    # Always forward LOCKED_LAMBDA_FILE when the spec module declares one. The
    # value can be a path or a template containing {idx} (expanded per task by
    # athena_run_one). Used for both prelim sweeps (IS_PRELIM=True ⇒ task
    # WRITES the sidecar) and main sweeps (task READS it). No chaining here —
    # prelim and main are independent deploys submitted in sequence by hand.
    if [[ -n "${SWEEP_LOCKED_LAMBDA_FILE}" ]]; then
        EXTRA_EXPORT=",LOCKED_LAMBDA_FILE=${SWEEP_LOCKED_LAMBDA_FILE}"
        echo "Spec declares LOCKED_LAMBDA_FILE: ${SWEEP_LOCKED_LAMBDA_FILE}"
    fi
    if [[ "${SWEEP_HAS_PRELIM_SPEC}" == "true" ]]; then
        # ── Array-prelim chain ───────────────────────────────────────────────
        # Module defines PRELIM_SPEC alongside SPEC. Build a separate sweep_list
        # for the prelim using --prelim flag, submit it as its own array, then
        # queue the main array with --dependency=afterok.
        echo ""
        echo "Module declares PRELIM_SPEC — submitting prelim array first."
        echo "  locked-lambda sidecar template: ${SWEEP_LOCKED_LAMBDA_FILE}"
        LOCAL_PRELIM_LIST="${LOCAL_PROJECT}/results_from_dgx/_prelim_sweep_list.txt"
        PRELIM_BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/dgx/scripts/build_sweep_list.py" \
            --kind spec --module "${SWEEP_SPEC_MODULE}" --prelim --output "${LOCAL_PRELIM_LIST}" 2>&1)
        echo "${PRELIM_BUILD_OUT}"
        PRELIM_N=$(grep -c . "${LOCAL_PRELIM_LIST}")
        if [[ "${PRELIM_N}" -lt 1 ]]; then
            echo "ERROR: prelim sweep_list is empty."
            exit 1
        fi
        PRELIM_END=$(( PRELIM_N - 1 ))
        echo "Uploading prelim_sweep_list.txt to DGX..."
        scp "${LOCAL_PRELIM_LIST}" "${SSH}:${REMOTE_BASE}/data/prelim_sweep_list.txt"
        echo "Prelim array: 0-${PRELIM_END}%${K}  (${PRELIM_N} tasks)"
        echo "Submitting prelim array..."
        PRELIM_RAW=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --array=0-${PRELIM_END}%${K} \
                --gpus=1 --cpus-per-task=${N_CPUS} \
                --time=${PRELIM_TIME:-00:10:00} \
                --export=ALL,SWEEP_KIND=spec,SWEEP_LIST=/work/data/prelim_sweep_list.txt,SWEEP_SPEC_MODULE=${SWEEP_SPEC_MODULE},SWEEP_IS_PRELIM=1,LOCKED_LAMBDA_FILE=${SWEEP_LOCKED_LAMBDA_FILE},DGX_LICENSE=${DGX_LICENSE},DGX_INTERCONNECT=${DGX_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_python_array.sh")
        if [[ $? -ne 0 ]]; then
            echo "ERROR: prelim sbatch failed."
            exit 1
        fi
        echo "Submitted prelim: ${PRELIM_RAW}"
        PRELIM_ID=$(echo "${PRELIM_RAW}" | awk '{print $NF}')
        DEP_FLAG="--dependency=afterok:${PRELIM_ID}"
        echo "Main array will start after prelim array ${PRELIM_ID} succeeds (all tasks)."
    elif [[ -n "${SWEEP_PRELIM_RUN_SCRIPT}" ]]; then
        echo ""
        echo "Spec declares prelim: RUN_SCRIPT=${SWEEP_PRELIM_RUN_SCRIPT}"
        echo "  locked-lambda sidecar: ${SWEEP_LOCKED_LAMBDA_FILE:-(spec default)}"
        echo "Submitting prelim (single sim, GPU)..."
        PRELIM_RAW=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --gpus=1 --cpus-per-task=${N_CPUS} \
                --time=${PRELIM_TIME:-00:10:00} \
                --export=ALL,RUN_SCRIPT=${SWEEP_PRELIM_RUN_SCRIPT},LOCKED_LAMBDA_FILE=${SWEEP_LOCKED_LAMBDA_FILE},DGX_LICENSE=${DGX_LICENSE},DGX_INTERCONNECT=${DGX_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC} \
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
            --time=${ARRAY_TIME:-00:45:00} \
            --export=ALL,SWEEP_KIND=${SWEEP_KIND},SWEEP_LIST=/work/data/sweep_list.txt,SWEEP_PARAM=${SWEEP_PARAM},SWEEP_FIXED_DYZ_NM=${SWEEP_FIXED_DYZ_NM},SWEEP_FIXED_CELLS=${SWEEP_FIXED_CELLS},SWEEP_SPEC_MODULE=${SWEEP_SPEC_MODULE},DGX_LICENSE=${DGX_LICENSE},DGX_INTERCONNECT=${DGX_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC}${EXTRA_EXPORT} \
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

# ── Auto-queue server-side aggregator after a mesh-convergence array ─────────
# Submitted with --dependency=afterok so it only runs if every array task
# succeeded. Writes the converged best_cells / best_dyz into the checkpoint at
# /work/results/mesh_convergence/checkpoint_*.json — picked up by the next
# `--results` download with no extra local step. Fully replaces the old local
# `python ... --aggregate {X|YZ}` flow.
AGG_PHASE=""
if [[ "${SWEEP_KIND}" == "mesh_conv_x"  ]]; then AGG_PHASE="X";  fi
if [[ "${SWEEP_KIND}" == "mesh_conv_yz" ]]; then AGG_PHASE="YZ"; fi
if [[ -n "${AGG_PHASE}" ]]; then
    echo ""
    echo "=== Queueing server-side aggregator for Phase ${AGG_PHASE} ==="
    AGG_RAW=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch \
            --dependency=afterok:${NUMERIC_JOB} \
            --export=ALL,PHASE=${AGG_PHASE} \
            --chdir=${REMOTE_BASE}/jobs \
            jobs/run_mesh_aggregate.sh")
    if [[ $? -eq 0 ]]; then
        AGG_ID=$(echo "${AGG_RAW}" | awk '{print $NF}')
        echo "Aggregator queued: ${AGG_RAW}"
        echo "  Will run after array ${NUMERIC_JOB} completes (afterok)."
        echo "  Aggregator log: ${REMOTE_BASE}/jobs/logs/mesh_aggregate-${AGG_ID}.out"
    else
        echo "WARNING: aggregator sbatch failed — fall back to local"
        echo "  python convergence_testing/run_mesh_convergence.py --aggregate ${AGG_PHASE}"
        echo "  (after downloading results)."
    fi
fi

# ── Show monitoring commands ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Job submitted: ${JOB_ID}"
echo ""
echo "You will receive an email at job start, finish, and failure."
echo ""
echo "Check status at any time:"
echo "  bash dgx/deploy_dgx.sh --status"
echo ""
echo "Live log on DGX:"
echo "  ssh ${SSH} tail -f ${REMOTE_BASE}/jobs/logs/lum_*${NUMERIC_JOB}*.out"
echo ""
echo "Download results after job finishes:"
echo "  bash dgx/deploy_dgx.sh --results           # prompts for mode"
echo "  bash dgx/deploy_dgx.sh --results-no-fsp    # data files only (fast)"
echo "  bash dgx/deploy_dgx.sh --results-full      # everything incl. .fsp (heavy)"
echo "============================================================"
