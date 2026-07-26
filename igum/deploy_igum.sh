#!/bin/bash
#
# Local script: upload project files to the IGUM cluster (Technion ECE faculty) and
# submit a SLURM GPU job. Target: 132.68.58.101 (igum-login1; FQDN not resolvable
# off-cluster). Requires Technion VPN. This is SEPARATE from athena/deploy_athena.sh —
# Athena remains the default cluster; IGUM is an additional coexisting option.
# License seats are SHARED with Athena (same FlexLM server).
#
# Run from Git Bash, WSL, or VS Code tasks on your local Windows machine.
#
# Usage:
#   bash igum/deploy_igum.sh                                 # interactive: engine or pipeline
#   bash igum/deploy_igum.sh --option1 --preset single       # engine, RECOMMENDED
#   bash igum/deploy_igum.sh --option1 --preset sweep_shift
#   bash igum/deploy_igum.sh --option1 --preset sweep_inner_size
#   bash igum/deploy_igum.sh --option2                       # pipeline, prompts: single or sweep
#   bash igum/deploy_igum.sh --option2 --run=single_sim      # pipeline, single (no prompt)
#   bash igum/deploy_igum.sh --option3 --spec=runners.sweeps.innermost_shift  # pipeline, sweep (no prompt)
#   bash igum/deploy_igum.sh --upload-only
#   bash igum/deploy_igum.sh --watch
#   bash igum/deploy_igum.sh --watch-only
#   bash igum/deploy_igum.sh --results            # prompts: data only or full (incl. .fsp)
#   bash igum/deploy_igum.sh --results-no-fsp    # download .mat / logs only (fast)
#   bash igum/deploy_igum.sh --results-full      # download everything incl. .fsp (heavy)
#   bash igum/deploy_igum.sh --results-files     # interactive: pick subfolder, then files
#   bash igum/deploy_igum.sh --results-files=path1,path2,...  # download specific files (rel to results/)
#   bash igum/deploy_igum.sh --gpu=h200           # pin to a specific GPU type

# ── CONFIGURE — edit igum/igum.conf, not this file ──────────────────
CONF="$(cd "$(dirname "$0")" && pwd)/igum.conf"
if [[ ! -f "${CONF}" ]]; then
    echo "ERROR: igum.conf not found at ${CONF}"
    exit 1
fi
source "${CONF}"

LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_RESULTS_DIR="${LOCAL_PROJECT}/results_from_igum"
LOCAL_NEFF=$(python -c "import sys; sys.path.insert(0,'${LOCAL_PROJECT}'); import config; print(config.NEFF_DATA_PATH)")
# rsync misreads Windows-style "C:\..." as a remote "host:path"; normalize to
# POSIX form. cygpath ships with Git Bash; on Linux/WSL the path is already POSIX.
if command -v cygpath >/dev/null 2>&1; then
    LOCAL_NEFF=$(cygpath -u "${LOCAL_NEFF}")
fi
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_ONLY=false
POL_ARRAY=false
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
GPU_TYPE=""
ARRAY_TASKS=""

for arg in "$@"; do
    case "${arg}" in
        --option1)            OPTION="1" ;;
        --option2)            OPTION="2" ;;
        --option3)            OPTION="3" ;;
        --pol-array)          POL_ARRAY=true ;;
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
        --cards=*)            OPTION="3"; SWEEP_KIND="cards"; SPEC_MODULE="${arg#--cards=}" ;;
        --inverse-design=*)   OPTION="3"; SWEEP_KIND="inverse_design"; SPEC_MODULE="${arg#--inverse-design=}" ;;
        --gradient-free-design=*) OPTION="3"; SWEEP_KIND="gradient_free_design"; SPEC_MODULE="${arg#--gradient-free-design=}" ;;
        --lumerical-native=*) OPTION="3"; SWEEP_KIND="lumerical_native_optimization"; SPEC_MODULE="${arg#--lumerical-native=}" ;;
        --fd-gradient-design=*) OPTION="3"; SWEEP_KIND="fd_gradient_design"; SPEC_MODULE="${arg#--fd-gradient-design=}" ;;
        --max-concurrent=*)   MAX_CONCURRENT="${arg#--max-concurrent=}" ;;
        --keep-h5)            KEEP_H5=1 ;;
        --gpu=*)              GPU_TYPE="${arg#--gpu=}" ;;
        --gpu-status)         GPU_STATUS=true ;;
        --array-tasks=*)      ARRAY_TASKS="${arg#--array-tasks=}" ;;
    esac
done

# ── --gpu=<type>: override ARRAY_PARTITIONS with a single partition ───────────
# Sourced after the conf so ARRAY_PARTITIONS is defined before we override it.
if [[ -n "${GPU_TYPE:-}" ]]; then
    # IGUM exposes partitions (part-preempt / part-<researcher>), not per-GPU-type
    # queues like Athena. Until recon maps GPU-type features, --gpu is not supported.
    echo "ERROR: --gpu=<type> is not supported on igum (single partition list: ${ARRAY_PARTITIONS})."
    echo "       Edit ARRAY_PARTITIONS in igum/igum.conf instead."
    exit 1
fi

SSH="${IGUM_USER}@${IGUM_HOST}"

# ── Prompt for top-level mode if not specified ────────────────────────────────
# Two conceptual modes presented to the user:
#   1) Engine    → generate .fsp locally, run FDTD engine on GPU (no Python on node)
#   2) Pipeline  → run lumapi Python pipeline on GPU (sub-prompt: single or sweep)
# Internally OPTION is still 1/2/3 (3 = pipeline-as-array). The single→sweep flip
# happens in the pipeline sub-prompt below.
if [[ -z "${OPTION}" && "${UPLOAD_ONLY}" == "false" && "${DOWNLOAD_RESULTS}" == "false" && "${STATUS}" == "false" && "${LICENSE_PROBE}" == "false" && "${GPU_STATUS:-false}" == "false" ]]; then
    echo ""
    echo "============================================================"
    echo "  IGUM GPU (Technion ECE, 132.68.58.101) — Choose run mode:"
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
    echo "  1) Single                 (one node, sequential — runners/single/)"
    echo "  2) Sweep                  (parallel SLURM job array — runners/sweeps/)"
    echo "  3) Inverse design         (lumopt adjoint — runners/inverse_design/)"
    echo "  4) Gradient-free design   (Lumerical PSO — runners/gradient_free_design/)"
    echo "  5) Lumerical-native opt   (addsweep('Optimization') — runners/lumerical_native_optimization/)"
    echo "  6) FD-gradient design     (scipy L-BFGS-B + central-diff jac — runners/fd_gradient_design/)"
    echo "  7) Convergence            (convergence_testing/ — incl. mesh_conv X/YZ)"
    echo "  8) TM studies             (TM single-run studies — runners/tm/)"
    echo "============================================================"
    read -rp "Enter 1, 2, 3, 4, 5, 6, 7, or 8: " _pipeline_choice
    case "${_pipeline_choice}" in
        1) _PIPELINE_KIND="single" ;;
        2) OPTION="3"; _PIPELINE_KIND="sweep" ;;
        3) OPTION="3"; _PIPELINE_KIND="inverse_design" ;;
        4) OPTION="3"; _PIPELINE_KIND="gradient_free_design" ;;
        5) OPTION="3"; _PIPELINE_KIND="lumerical_native_optimization" ;;
        6) OPTION="3"; _PIPELINE_KIND="fd_gradient_design" ;;
        7) _PIPELINE_KIND="convergence" ;;
        8) _PIPELINE_KIND="tm" ;;
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
    echo "  Choose which script to run on Athena (runners/single/):"
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

# ── TM-studies picker (mode=tm) — AUTO-DISCOVERED ────────────────────────────
# Same contract as the Single picker, but scans runners/tm/*.py. TM single-run
# studies (polarization=TM) live here so they don't crowd runners/single/.
# Dispatched as an OPTION=2 sequential job; athena_run.py resolves the bare
# module name via its runners.tm auto-discovery dir.
if [[ "${_PIPELINE_KIND:-}" == "tm" && -z "${RUN_SCRIPT}" && "${UPLOAD_ONLY}" == "false" ]]; then
    mapfile -t _RUNNERS < <(
        for _f in "${LOCAL_PROJECT}/runners/tm"/*.py; do
            [[ -e "${_f}" ]] || continue
            _name=$(basename "${_f}" .py)
            [[ "${_name}" == _* || "${_name}" == "__init__" ]] && continue
            grep -qE '^(def[[:space:]]+run[[:space:]]*\(|run[[:space:]]*=)' "${_f}" 2>/dev/null || continue
            grep -qE '^IS_HELPER[[:space:]]*=[[:space:]]*True' "${_f}" 2>/dev/null && continue
            echo "${_name}"
        done | sort
    )
    if [[ ${#_RUNNERS[@]} -eq 0 ]]; then
        echo "ERROR: no runners discovered in runners/tm/."
        echo "       Each module needs a top-level callable 'run'."
        exit 1
    fi
    echo ""
    echo "============================================================"
    echo "  Choose which script to run on Athena (runners/tm/):"
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

# ── Inverse-design study picker (mode=inverse_design) ────────────────────────
# Auto-discovers any module under runners/inverse_design/ that defines a
# top-level SPEC: InverseDesignSpec. Selecting a study sets
# SWEEP_KIND=inverse_design and SPEC_MODULE — no separate prompt.
if [[ "${_PIPELINE_KIND:-}" == "inverse_design" && -z "${SPEC_MODULE}" && "${UPLOAD_ONLY}" == "false" ]]; then
    SWEEP_KIND="inverse_design"
    mapfile -t _ID_STUDIES < <(
        for _f in $(grep -rl '[[:space:]]*SPEC[[:space:]]*=' "${LOCAL_PROJECT}/runners/inverse_design/"*.py 2>/dev/null \
                        | grep -v 'inverse_design\.py' \
                        | grep -v 'test_geometry\.py'); do
            basename "${_f}" .py
        done | sort
    )
    if [[ ${#_ID_STUDIES[@]} -eq 0 ]]; then
        echo "ERROR: no inverse-design studies found in runners/inverse_design/ (need top-level SPEC = InverseDesignSpec(...))"
        exit 1
    fi
    echo ""
    echo "============================================================"
    echo "  Choose inverse-design study (runners/inverse_design/<name>.py):"
    for _i in "${!_ID_STUDIES[@]}"; do
        printf "  %d) %s\n" "$((_i+1))" "${_ID_STUDIES[$_i]}"
    done
    echo "============================================================"
    read -rp "Enter number: " _id_choice
    if [[ "${_id_choice}" =~ ^[0-9]+$ ]] && (( _id_choice >= 1 && _id_choice <= ${#_ID_STUDIES[@]} )); then
        SPEC_MODULE="runners.inverse_design.${_ID_STUDIES[$((_id_choice-1))]}"
        echo "Selected: ${SPEC_MODULE}"
    else
        echo "Invalid choice. Exiting."; exit 1
    fi
fi

# ── Gradient-free-design study picker (mode=gradient_free_design) ───────────
# Auto-discovers any module under runners/gradient_free_design/ that defines
# a top-level SPEC: GradientFreeDesignSpec.
if [[ "${_PIPELINE_KIND:-}" == "gradient_free_design" && -z "${SPEC_MODULE}" && "${UPLOAD_ONLY}" == "false" ]]; then
    SWEEP_KIND="gradient_free_design"
    mapfile -t _GFD_STUDIES < <(
        for _f in $(grep -rl '[[:space:]]*SPEC[[:space:]]*=' "${LOCAL_PROJECT}/runners/gradient_free_design/"*.py 2>/dev/null \
                        | grep -v 'gradient_free_design\.py' \
                        | grep -v 'test_geometry\.py'); do
            basename "${_f}" .py
        done | sort
    )
    if [[ ${#_GFD_STUDIES[@]} -eq 0 ]]; then
        echo "ERROR: no gradient-free-design studies found in runners/gradient_free_design/ (need top-level SPEC = GradientFreeDesignSpec(...))"
        exit 1
    fi
    echo ""
    echo "============================================================"
    echo "  Choose gradient-free-design study (runners/gradient_free_design/<name>.py):"
    for _i in "${!_GFD_STUDIES[@]}"; do
        printf "  %d) %s\n" "$((_i+1))" "${_GFD_STUDIES[$_i]}"
    done
    echo "============================================================"
    read -rp "Enter number: " _gfd_choice
    if [[ "${_gfd_choice}" =~ ^[0-9]+$ ]] && (( _gfd_choice >= 1 && _gfd_choice <= ${#_GFD_STUDIES[@]} )); then
        SPEC_MODULE="runners.gradient_free_design.${_GFD_STUDIES[$((_gfd_choice-1))]}"
        echo "Selected: ${SPEC_MODULE}"
    else
        echo "Invalid choice. Exiting."; exit 1
    fi
fi

# ── Lumerical-native optimization picker (mode=lumerical_native_optimization) ─
# Auto-discovers any module under runners/lumerical_native_optimization/ that
# defines a top-level SPEC: LumericalNativeSpec.
if [[ "${_PIPELINE_KIND:-}" == "lumerical_native_optimization" && -z "${SPEC_MODULE}" && "${UPLOAD_ONLY}" == "false" ]]; then
    SWEEP_KIND="lumerical_native_optimization"
    mapfile -t _LNO_STUDIES < <(
        for _f in $(grep -rl '[[:space:]]*SPEC[[:space:]]*=' "${LOCAL_PROJECT}/runners/lumerical_native_optimization/"*.py 2>/dev/null \
                        | grep -v 'lumerical_native_optimization\.py'); do
            basename "${_f}" .py
        done | sort
    )
    if [[ ${#_LNO_STUDIES[@]} -eq 0 ]]; then
        echo "ERROR: no lumerical-native studies found in runners/lumerical_native_optimization/ (need top-level SPEC = LumericalNativeSpec(...))"
        exit 1
    fi
    echo ""
    echo "============================================================"
    echo "  Choose lumerical-native study (runners/lumerical_native_optimization/<name>.py):"
    for _i in "${!_LNO_STUDIES[@]}"; do
        printf "  %d) %s\n" "$((_i+1))" "${_LNO_STUDIES[$_i]}"
    done
    echo "============================================================"
    read -rp "Enter number: " _lno_choice
    if [[ "${_lno_choice}" =~ ^[0-9]+$ ]] && (( _lno_choice >= 1 && _lno_choice <= ${#_LNO_STUDIES[@]} )); then
        SPEC_MODULE="runners.lumerical_native_optimization.${_LNO_STUDIES[$((_lno_choice-1))]}"
        echo "Selected: ${SPEC_MODULE}"
    else
        echo "Invalid choice. Exiting."; exit 1
    fi
fi

# ── FD-gradient-design study picker (mode=fd_gradient_design) ───────────────
# Auto-discovers any module under runners/fd_gradient_design/ that defines
# a top-level SPEC: FDGradientSpec.
if [[ "${_PIPELINE_KIND:-}" == "fd_gradient_design" && -z "${SPEC_MODULE}" && "${UPLOAD_ONLY}" == "false" ]]; then
    SWEEP_KIND="fd_gradient_design"
    mapfile -t _FDG_STUDIES < <(
        for _f in $(grep -rl '[[:space:]]*SPEC[[:space:]]*=' "${LOCAL_PROJECT}/runners/fd_gradient_design/"*.py 2>/dev/null \
                        | grep -v 'fd_gradient_design\.py'); do
            basename "${_f}" .py
        done | sort
    )
    if [[ ${#_FDG_STUDIES[@]} -eq 0 ]]; then
        echo "ERROR: no fd-gradient-design studies found in runners/fd_gradient_design/ (need top-level SPEC = FDGradientSpec(...))"
        exit 1
    fi
    echo ""
    echo "============================================================"
    echo "  Choose fd-gradient-design study (runners/fd_gradient_design/<name>.py):"
    for _i in "${!_FDG_STUDIES[@]}"; do
        printf "  %d) %s\n" "$((_i+1))" "${_FDG_STUDIES[$_i]}"
    done
    echo "============================================================"
    read -rp "Enter number: " _fdg_choice
    if [[ "${_fdg_choice}" =~ ^[0-9]+$ ]] && (( _fdg_choice >= 1 && _fdg_choice <= ${#_FDG_STUDIES[@]} )); then
        SPEC_MODULE="runners.fd_gradient_design.${_FDG_STUDIES[$((_fdg_choice-1))]}"
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
        echo "=== Downloading results from IGUM cluster (data files only, skipping .fsp) ==="
        ssh "${SSH}" "tar --exclude='*.fsp' -czf - -C '${REMOTE_BASE}/results' ." \
            | tar -xzf - -C "${LOCAL_RESULTS_DIR}/"
    else
        echo "=== Downloading results from IGUM cluster (full, including .fsp) ==="
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

        echo "=== Discovering run folders under ${REMOTE_BASE}/results/ ==="
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
    echo "=== Downloading ${#_paths[@]} file(s) from Athena cluster ==="
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
    echo "=== Jobs in queue (${IGUM_USER} @ ${IGUM_HOST}) ==="
    ssh "${SSH}" "squeue -u ${IGUM_USER} -o '%.10i %.12j %.8T %.10M %.6D %R' 2>/dev/null || echo '(no jobs in queue)'"
    echo ""
    echo "=== Last 40 lines of most recent log ==="
    ssh "${SSH}" "ls -t ${REMOTE_BASE}/jobs/logs/*.out 2>/dev/null | head -1 | xargs tail -40 2>/dev/null || echo '(no log found)'"
}

# ── --status: one-shot check then exit ────────────────────────────────────────
if [[ "${STATUS}" == "true" ]]; then
    check_status
    exit 0
fi

# ── --gpu-status: show free GPU capacity per partition then exit ──────────────
if [[ "${GPU_STATUS:-false}" == "true" ]]; then
    echo ""
    echo "=== Athena GPU availability (per-partition) ==="
    echo "  public  = won't be preempted (use for long/important jobs)"
    echo "  shared  = may be preempted by contributor jobs (more capacity)"
    echo ""
    ssh "${SSH}" "
        printf '%-16s %-10s %-22s %-12s %-12s %s\n' PARTITION TYPE NODE STATE GPUS-FREE PENDING
        printf '%-16s %-10s %-22s %-12s %-12s %s\n' --------- ---- ---- ----- --------- -------
        for p in h200-shared a100-public l40s-public l40s-shared rtx6k-shared; do
            ptype='shared'
            [[ \"\$p\" == *-public ]] && ptype='public'
            # sinfo: per-node GPU resource (allocated/total) and state
            sinfo -h -p \"\$p\" -N -o '%P|%n|%t|%G' | while IFS='|' read part node state gres; do
                # Parse 'gpu:type:8(S:...)' → total
                total=\$(echo \"\$gres\" | sed -nE 's/.*:([0-9]+)\\(.*/\\1/p')
                [[ -z \"\$total\" ]] && total=\$(echo \"\$gres\" | sed -nE 's/gpu:[^:]+:([0-9]+).*/\\1/p')
                # Allocated GPUs on this node from scontrol
                alloc=\$(scontrol show node \"\$node\" 2>/dev/null | grep -oE 'AllocTRES=.*gres/gpu=[0-9]+' | grep -oE 'gres/gpu=[0-9]+' | grep -oE '[0-9]+')
                [[ -z \"\$alloc\" ]] && alloc=0
                free=\$((total - alloc))
                pend=\$(squeue -h -p \"\$p\" -t PD -o '%i' | wc -l)
                printf '%-16s %-10s %-22s %-12s %-12s %s\n' \"\$part\" \"\$ptype\" \"\$node\" \"\$state\" \"\${free}/\${total}\" \"\$pend\"
            done
        done
    "
    echo ""
    echo "Tip: pass --gpu=<h200|a100|l40s-public|l40s-shared|rtx6k> with --option2/3 to pin."
    exit 0
fi

# ── --license-probe: report FlexLM seat counts then exit ──────────────────────
if [[ "${LICENSE_PROBE}" == "true" ]]; then
    echo ""
    echo "=== Probing Lumerical license seats from IGUM (server: ${ATHENA_LICENSE}) ==="
    # lmutil ships inside the NATIVE Lumerical install on IGUM (no container).
    # NOTE: seats are SHARED with Athena — check athena --status before big runs.
    ssh "${SSH}" "
        lmutil=/apps/ansys/Lumerical-2026-R1.2/opt/lumerical/v261/licensingclient/linx64/lmutil
        if [ -x \"\$lmutil\" ]; then
            echo \"Using: \$lmutil\"
            \"\$lmutil\" lmstat -a -c '${ATHENA_LICENSE}' \
                | grep -E 'Users of (lum_fdtd|fdtd_)' \
                || echo '  (no FDTD features found)'
        else
            echo 'ERROR: lmutil not found at native install path on IGUM.'
        fi
    "
    echo ""
    echo "Set MAX_CONCURRENT in igum/igum.conf to (free seats), capped at 8."
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
echo "Target: ${SSH}  (IGUM cluster — Technion ECE, igum-login1)"
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
    echo "  Update LOCAL_NEFF in deploy_igum.sh or upload manually."
fi

echo ""
echo "=== Uploading Athena job scripts and helpers ==="
rsync -av --itemize-changes "${LOCAL_PROJECT}/igum/jobs/"*.sh "${SSH}:${REMOTE_BASE}/jobs/"
rsync -av --itemize-changes "${LOCAL_PROJECT}/igum/scripts/"*.py "${SSH}:${REMOTE_BASE}/scripts/"
ssh "${SSH}" "chmod +x ${REMOTE_BASE}/jobs/*.sh"
ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/jobs/logs"

# NVML trampoline not needed: athena cluster uses R570+ drivers.

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

        echo "Uploading ${#FSP_PATHS[@]} .fsp file(s) to Athena cluster (per-stem folders)..."
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
        LOCAL_FSP_LIST="${LOCAL_PROJECT}/results_from_igum/_fsp_list.txt"
        : > "${LOCAL_FSP_LIST}"
        for _fsp in "${FSP_PATHS[@]}"; do
            _name=$(basename "${_fsp}")
            echo "${_name%.fsp}" >> "${LOCAL_FSP_LIST}"
        done
        scp "${LOCAL_FSP_LIST}" "${SSH}:${REMOTE_BASE}/results/fsp_list.txt"
        ARRAY_END=$(( FSP_COUNT - 1 ))

        # K = max concurrent jobs (license throttle). Set in igum.conf; override
        # for one run via --max-concurrent N. Discover the real seat count with
        # `bash igum/deploy_igum.sh --license-probe`.
        K="${MAX_CONCURRENT:-4}"
        echo "Array: 0-${ARRAY_END}%${K}  (max ${K} concurrent, ~${FSP_COUNT} engine seats used)"

        JOB_ID=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --array=0-${ARRAY_END}%${K} \
                ${GPU_REQ} --cpus-per-task=${N_CPUS} \
                --time=${ARRAY_TIME:-00:50:00} \
                --qos=${ARRAY_QOS} --account=${SLURM_ACCOUNT} \
                --partition=${ARRAY_PARTITIONS} \
                --export=ALL,ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},NTFY_TOPIC=${NTFY_TOPIC},WORK_DIR=${REMOTE_BASE} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu_array.sh")
    else
        JOB_ID=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                ${GPU_REQ} --cpus-per-task=${N_CPUS} \
                --time=${ARRAY_TIME:-00:50:00} \
                --qos=${ARRAY_QOS} --account=${SLURM_ACCOUNT} \
                --partition=${ARRAY_PARTITIONS} \
                --export=ALL,FSP_FILE=\"${FSP_NAME}\",ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},NTFY_TOPIC=${NTFY_TOPIC},WORK_DIR=${REMOTE_BASE} \
                --chdir=${REMOTE_BASE}/jobs \
                jobs/run_fsp_gpu.sh")
    fi

elif [[ "${OPTION}" == "2" ]]; then
    # ── Option 2: full Python pipeline ───────────────────────────────────────
    # Default: one sequential job. With --pol-array: a 2-task GPU array — task 0
    # and task 1 run on separate GPUs in parallel. The runner reads
    # SLURM_ARRAY_TASK_ID to pick its unit of work (e.g. run_tm_vs_te: 0=TE, 1=TM).
    POL_ARRAY_OPT=""
    if [[ "${POL_ARRAY}" == "true" ]]; then
        POL_ARRAY_OPT="--array=0-1%2"
        echo "--pol-array: submitting ${RUN_SCRIPT} as a 2-task GPU array (task 0 + task 1 in parallel)."
    fi
    # Optional host-RAM request. Default (unset) → cluster default ~64 GB. Set
    # SBATCH_MEM=256G for big-domain runs (far-field / large SPAN_MULT) — the
    # GPU nodes have 1–2.3 TB physical, so the 64 GB default is the only OOM cause.
    MEM_OPT=""
    if [[ -n "${SBATCH_MEM:-}" ]]; then
        MEM_OPT="--mem=${SBATCH_MEM}"
        echo "[--mem] requesting ${SBATCH_MEM} host RAM for this job"
    fi
    JOB_ID=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch ${POL_ARRAY_OPT} ${MEM_OPT} \
            ${GPU_REQ} --cpus-per-task=${N_CPUS} \
            --time=${ARRAY_TIME:-00:50:00} \
            --qos=${ARRAY_QOS} --account=${SLURM_ACCOUNT} \
            --partition=${ARRAY_PARTITIONS} \
            --export=ALL,RUN_SCRIPT=${RUN_SCRIPT},ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},TM_PITCH_NM=${TM_PITCH_NM:-500},TM_WIDE_PITCH_NM=${TM_WIDE_PITCH_NM:-},TM_SIM_TIME_PS=${TM_SIM_TIME_PS:-},TM_WIDE_SINGLE_CORR_NM=${TM_WIDE_SINGLE_CORR_NM:-},TM_HEIGHT_NM=${TM_HEIGHT_NM:-350},TM_SCAN_CENTER_NM=${TM_SCAN_CENTER_NM:-},TM_SCAN_WIDTH_NM=${TM_SCAN_WIDTH_NM:-},TM_SCAN_NPTS=${TM_SCAN_NPTS:-},TM_WIDE_CENTER_NM=${TM_WIDE_CENTER_NM:-},TM_WIDE_WIDTH_NM=${TM_WIDE_WIDTH_NM:-},TM_WIDE_NPTS=${TM_WIDE_NPTS:-},TM_WIDE_SEEDS_NM=${TM_WIDE_SEEDS_NM:-},TM_MESH=${TM_MESH:-optimization},TM_FARFIELD=${TM_FARFIELD:-0},TM_RECORD_2D=${TM_RECORD_2D:-0},SPAN_MULT=${SPAN_MULT:-},TM_CONST_MODE=${TM_CONST_MODE:-sampled},CONV_POL=${CONV_POL:-TE},NTFY_TOPIC=${NTFY_TOPIC},WORK_DIR=${REMOTE_BASE} \
            --chdir=${REMOTE_BASE}/jobs \
            jobs/run_python_gpu.sh")

else
    # ── Option 3: Python pipeline as a SLURM job array (parallel sweep) ──────
    echo ""
    echo "Option 3: building sweep_list.txt locally for kind='${SWEEP_KIND}'..."
    LOCAL_SWEEP_LIST="${LOCAL_PROJECT}/results_from_igum/_sweep_list.txt"
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
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/igum/scripts/build_sweep_list.py" \
            --kind spec --module "${SPEC_MODULE}" --output "${LOCAL_SWEEP_LIST}" 2>&1)
    elif [[ "${SWEEP_KIND}" == "cards" ]]; then
        if [[ -z "${SPEC_MODULE}" ]]; then
            echo "ERROR: --cards=<module> is required for kind=cards."
            exit 1
        fi
        LOCAL_CARDS_MANIFEST="${LOCAL_PROJECT}/results_from_igum/_cards_manifest.json"
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/igum/scripts/build_sweep_list.py" \
            --kind cards --module "${SPEC_MODULE}" \
            --cards-manifest "${LOCAL_CARDS_MANIFEST}" \
            --output "${LOCAL_SWEEP_LIST}" 2>&1)
    elif [[ "${SWEEP_KIND}" == "inverse_design" ]]; then
        if [[ -z "${SPEC_MODULE}" ]]; then
            echo "ERROR: --inverse-design=<module> is required for kind=inverse_design."
            exit 1
        fi
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/igum/scripts/build_sweep_list.py" \
            --kind inverse_design --module "${SPEC_MODULE}" --output "${LOCAL_SWEEP_LIST}" 2>&1)
    elif [[ "${SWEEP_KIND}" == "gradient_free_design" ]]; then
        if [[ -z "${SPEC_MODULE}" ]]; then
            echo "ERROR: --gradient-free-design=<module> is required for kind=gradient_free_design."
            exit 1
        fi
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/igum/scripts/build_sweep_list.py" \
            --kind gradient_free_design --module "${SPEC_MODULE}" --output "${LOCAL_SWEEP_LIST}" 2>&1)
    elif [[ "${SWEEP_KIND}" == "lumerical_native_optimization" ]]; then
        if [[ -z "${SPEC_MODULE}" ]]; then
            echo "ERROR: --lumerical-native=<module> is required for kind=lumerical_native_optimization."
            exit 1
        fi
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/igum/scripts/build_sweep_list.py" \
            --kind lumerical_native_optimization --module "${SPEC_MODULE}" --output "${LOCAL_SWEEP_LIST}" 2>&1)
    elif [[ "${SWEEP_KIND}" == "fd_gradient_design" ]]; then
        if [[ -z "${SPEC_MODULE}" ]]; then
            echo "ERROR: --fd-gradient-design=<module> is required for kind=fd_gradient_design."
            exit 1
        fi
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/igum/scripts/build_sweep_list.py" \
            --kind fd_gradient_design --module "${SPEC_MODULE}" --output "${LOCAL_SWEEP_LIST}" 2>&1)
    else
        BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/igum/scripts/build_sweep_list.py" \
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
    # --array-tasks=<spec> overrides the default 0-N range. Use for smoke tests
    # ("--array-tasks=0" runs only the first task). Concurrency suffix %K is
    # appended automatically.
    if [[ -n "${ARRAY_TASKS}" ]]; then
        ARRAY_RANGE="${ARRAY_TASKS}"
        echo "[--array-tasks] overriding array range to: ${ARRAY_RANGE}"
    else
        ARRAY_RANGE="0-${ARRAY_END}"
    fi

    echo "Uploading sweep_list.txt to Athena cluster..."
    ssh "${SSH}" "mkdir -p ${REMOTE_BASE}/data"
    scp "${LOCAL_SWEEP_LIST}" "${SSH}:${REMOTE_BASE}/data/sweep_list.txt"

    echo "Array: ${ARRAY_RANGE}%${K}  (${N_TASKS} tasks total, max ${K} concurrent)"
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
        LOCAL_PRELIM_LIST="${LOCAL_PROJECT}/results_from_igum/_prelim_sweep_list.txt"
        PRELIM_BUILD_OUT=$("${LOCAL_PYTHON}" "${LOCAL_PROJECT}/igum/scripts/build_sweep_list.py" \
            --kind spec --module "${SWEEP_SPEC_MODULE}" --prelim --output "${LOCAL_PRELIM_LIST}" 2>&1)
        echo "${PRELIM_BUILD_OUT}"
        PRELIM_N=$(grep -c . "${LOCAL_PRELIM_LIST}")
        if [[ "${PRELIM_N}" -lt 1 ]]; then
            echo "ERROR: prelim sweep_list is empty."
            exit 1
        fi
        PRELIM_END=$(( PRELIM_N - 1 ))
        echo "Uploading prelim_sweep_list.txt to Athena cluster..."
        scp "${LOCAL_PRELIM_LIST}" "${SSH}:${REMOTE_BASE}/data/prelim_sweep_list.txt"
        echo "Prelim array: 0-${PRELIM_END}%${K}  (${PRELIM_N} tasks)"
        echo "Submitting prelim array..."
        PRELIM_RAW=$(ssh "${SSH}" \
            "cd ${REMOTE_BASE} && sbatch \
                --array=0-${PRELIM_END}%${K} \
                ${GPU_REQ} --cpus-per-task=${N_CPUS} \
                --time=${PRELIM_TIME:-00:10:00} \
                --qos=${ARRAY_QOS} --account=${SLURM_ACCOUNT} \
                --partition=${ARRAY_PARTITIONS} \
                --export=ALL,SWEEP_KIND=spec,SWEEP_LIST=${REMOTE_BASE}/data/prelim_sweep_list.txt,SWEEP_SPEC_MODULE=${SWEEP_SPEC_MODULE},SWEEP_IS_PRELIM=1,LOCKED_LAMBDA_FILE=${SWEEP_LOCKED_LAMBDA_FILE},ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC},WORK_DIR=${REMOTE_BASE} \
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
                ${GPU_REQ} --cpus-per-task=${N_CPUS} \
                --time=${PRELIM_TIME:-00:10:00} \
                --qos=${ARRAY_QOS} --account=${SLURM_ACCOUNT} \
                --partition=${ARRAY_PARTITIONS} \
                --export=ALL,RUN_SCRIPT=${SWEEP_PRELIM_RUN_SCRIPT},LOCKED_LAMBDA_FILE=${SWEEP_LOCKED_LAMBDA_FILE},ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},NTFY_TOPIC=${NTFY_TOPIC},WORK_DIR=${REMOTE_BASE} \
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

    # SBATCH_MEM=<size> overrides the array script's 128G default — mirrored
    # from athena/deploy_athena.sh (the forks had drifted: igum only honored it
    # for --option2).
    MEM_OPT=""
    if [[ -n "${SBATCH_MEM:-}" ]]; then
        MEM_OPT="--mem=${SBATCH_MEM}"
        echo "[--mem] requesting ${SBATCH_MEM} host RAM for this array"
    fi
    JOB_ID=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch ${MEM_OPT} \
            --array=${ARRAY_RANGE}%${K} ${DEP_FLAG} \
            ${GPU_REQ} --cpus-per-task=${N_CPUS} \
            --time=${ARRAY_TIME:-00:50:00} \
            --qos=${ARRAY_QOS} --account=${SLURM_ACCOUNT} \
            --partition=${ARRAY_PARTITIONS} \
            --export=ALL,SWEEP_KIND=${SWEEP_KIND},SWEEP_LIST=${REMOTE_BASE}/data/sweep_list.txt,SWEEP_PARAM=${SWEEP_PARAM},SWEEP_FIXED_DYZ_NM=${SWEEP_FIXED_DYZ_NM},SWEEP_FIXED_CELLS=${SWEEP_FIXED_CELLS},SWEEP_SPEC_MODULE=${SWEEP_SPEC_MODULE},ATHENA_LICENSE=${ATHENA_LICENSE},ATHENA_INTERCONNECT=${ATHENA_INTERCONNECT},REQUIRE_GPU=${REQUIRE_GPU:-1},KEEP_H5=${KEEP_H5:-0},CONV_POL=${CONV_POL:-TE},NTFY_TOPIC=${NTFY_TOPIC}${EXTRA_EXPORT},WORK_DIR=${REMOTE_BASE} \
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
            --qos=${AGG_QOS} --account=${SLURM_ACCOUNT} \
            --partition=${AGG_PARTITION} \
            --export=ALL,PHASE=${AGG_PHASE},WORK_DIR=${REMOTE_BASE} \
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

# ── Tooth-shift sweep: seed shift=0 baseline + queue server-side summary ─────
# We do NOT re-run shift=0 — the baseline already exists. Copy the existing
# run_tm_vs_te baseline result files into this study's results folder, where they
# read as the shift=0 point (no `_S` tag). Then queue a CPU-only afterok job
# (same idiom as the mesh aggregator above) that runs plot_tm_te_shift.py once the
# array succeeds, writing transmission_vs_shift.png / modewidth_vs_shift.png /
# shift_summary.csv into the same folder. Those download with `--results-no-fsp`.
if [[ "${SWEEP_SPEC_MODULE}" == "runners.sweeps.tm_te_shift" ]]; then
    REMOTE_SHIFT_RESULTS="${REMOTE_BASE}/results/tm_te_shift/results"
    BASELINE_SRC="${LOCAL_PROJECT}/results_from_igum/run_tm_vs_te/results"
    echo ""
    echo "=== Seeding shift=0 baseline into ${REMOTE_SHIFT_RESULTS} ==="
    ssh "${SSH}" "mkdir -p ${REMOTE_SHIFT_RESULTS}"
    _seeded=0
    for _bf in result_N80_avg_te.mat result_N80_TM_avg_tm.mat; do
        if [[ -f "${BASELINE_SRC}/${_bf}" ]]; then
            scp "${BASELINE_SRC}/${_bf}" "${SSH}:${REMOTE_SHIFT_RESULTS}/${_bf}" \
                && _seeded=$((_seeded + 1))
        else
            echo "  WARNING: baseline file not found locally: ${BASELINE_SRC}/${_bf}"
        fi
    done
    echo "  Seeded ${_seeded}/2 baseline (shift=0) file(s)."
    if [[ "${_seeded}" -lt 2 ]]; then
        echo "  (summary will still plot the swept points; the 0-shift point is"
        echo "   only included for the polarizations whose baseline was seeded.)"
    fi

    echo ""
    echo "=== Queueing server-side tooth-shift summary plotter ==="
    SUM_RAW=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch \
            --dependency=afterok:${NUMERIC_JOB} \
            --qos=${AGG_QOS} --account=${SLURM_ACCOUNT} \
            --partition=${AGG_PARTITION} \
            --export=ALL,WORK_DIR=${REMOTE_BASE} \
            --chdir=${REMOTE_BASE}/jobs \
            jobs/run_shift_summary.sh")
    if [[ $? -eq 0 ]]; then
        SUM_ID=$(echo "${SUM_RAW}" | awk '{print $NF}')
        echo "Summary plotter queued: ${SUM_RAW}"
        echo "  Will run after array ${NUMERIC_JOB} completes (afterok)."
        echo "  Summary log: ${REMOTE_BASE}/jobs/logs/shift_summary-${SUM_ID}.out"
    else
        echo "WARNING: summary sbatch failed — fall back to local after download:"
        echo "  python runners/sweeps/plot_tm_te_shift.py --results-dir results_from_igum/tm_te_shift/results"
    fi
fi

# ── Pitch-matched TM shift rerun: build the CORRECTED TE-vs-TM comparison ─────
# tm_shift_p518 reruns ONLY TM at pitch 518.3. The summary image must compare the
# unchanged TE (pitch 500) points with the new TM (pitch 518.3) points, so we seed
# this study's results folder with: (a) the corrected shift=0 TM baseline
# (result_N80_TM_avg_tm_P518p3.mat, uploaded from local), and (b) the existing
# pitch-500 TE points (shift 0/50/100/150/200) copied server-side from the
# tm_te_shift results. Then plot_tm_te_shift.py over this folder yields the
# corrected transmission/mode-width-vs-shift images (TE@500 + TM@518.3).
if [[ "${SWEEP_SPEC_MODULE}" == "runners.sweeps.tm_shift_p518" ]]; then
    P518_RESULTS="${REMOTE_BASE}/results/tm_shift_p518/results"
    TE_SRC="${REMOTE_BASE}/results/tm_te_shift/results"
    TM0_LOCAL="${LOCAL_PROJECT}/results_from_igum/tm_te_pitch_matched/result_N80_TM_avg_tm_P518p3.mat"
    echo ""
    echo "=== Seeding corrected baseline + TE points into ${P518_RESULTS} ==="
    ssh "${SSH}" "mkdir -p ${P518_RESULTS}"
    # (a) corrected shift=0 TM baseline (pitch 518.3) — read as TM, shift 0 (no _S)
    if [[ -f "${TM0_LOCAL}" ]]; then
        scp "${TM0_LOCAL}" "${SSH}:${P518_RESULTS}/result_N80_TM_avg_tm_P518p3.mat" \
            && echo "  seeded corrected TM shift=0 baseline" \
            || echo "  WARNING: failed to upload corrected TM baseline"
    else
        echo "  WARNING: corrected TM baseline not found locally: ${TM0_LOCAL}"
    fi
    # (b) pitch-500 TE points (every result_*.mat in tm_te_shift WITHOUT _TM) —
    #     copied server-side so no re-run and no download round-trip.
    ssh "${SSH}" "if [[ -d '${TE_SRC}' ]]; then \
        for f in ${TE_SRC}/result_*.mat; do \
            b=\$(basename \"\$f\"); \
            case \"\$b\" in *_TM*|*summary*) continue;; esac; \
            cp \"\$f\" '${P518_RESULTS}/'; \
        done; \
        echo \"  seeded TE points: \$(ls ${P518_RESULTS}/ | grep -v _TM | grep -c '\.mat')\"; \
    else echo '  WARNING: TE source folder missing on server: ${TE_SRC}'; fi"

    echo ""
    echo "=== Queueing server-side corrected summary plotter ==="
    SUM_RAW=$(ssh "${SSH}" \
        "cd ${REMOTE_BASE} && sbatch \
            --dependency=afterok:${NUMERIC_JOB} \
            --qos=${AGG_QOS} --account=${SLURM_ACCOUNT} \
            --partition=${AGG_PARTITION} \
            --export=ALL,SUMMARY_DIR=${REMOTE_BASE}/results/tm_shift_p518/results,X_AXIS=relative,WORK_DIR=${REMOTE_BASE} \
            --chdir=${REMOTE_BASE}/jobs \
            jobs/run_shift_summary.sh")
    if [[ $? -eq 0 ]]; then
        SUM_ID=$(echo "${SUM_RAW}" | awk '{print $NF}')
        echo "Summary plotter queued: ${SUM_RAW}"
        echo "  Will run after array ${NUMERIC_JOB} completes (afterok)."
        echo "  Summary log: ${REMOTE_BASE}/jobs/logs/shift_summary-${SUM_ID}.out"
    else
        echo "WARNING: summary sbatch failed — fall back to local after download:"
        echo "  python runners/sweeps/plot_tm_te_shift.py --results-dir results_from_igum/tm_shift_p518/results"
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
echo "  bash igum/deploy_igum.sh --status"
echo ""
echo "Live log on Athena cluster:"
echo "  ssh ${SSH} tail -f ${REMOTE_BASE}/jobs/logs/lum_*${NUMERIC_JOB}*.out"
echo ""
echo "Download results after job finishes:"
echo "  bash igum/deploy_igum.sh --results           # prompts for mode"
echo "  bash igum/deploy_igum.sh --results-no-fsp    # data files only (fast)"
echo "  bash igum/deploy_igum.sh --results-full      # everything incl. .fsp (heavy)"
echo "============================================================"
