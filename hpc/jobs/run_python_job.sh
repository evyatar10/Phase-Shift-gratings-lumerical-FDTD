#!/bin/bash
#
# PBS job: run the full Python pipeline on Zeus.
# Generates .fsp, runs FDTD via lumapi, and post-processes — all in one job.
#
# If the simulation step fails, server_run.py will print a fallback message
# with the exact qsub command to run the .fsp file via run_fsp_job.sh.
#
#PBS -N bragg_pipeline
#PBS -q zeus_all_q
#PBS -l select=1:ncpus=80
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -S /bin/bash
#PBS -M evyatar10.rubin@gmail.com
#PBS -j oe
#PBS -k eod
#PBS -o bragg_pipeline.out

ulimit -s unlimited

WORK_DIR="/home/evyatarrubin/bragg_sim"

echo "============================================================"
echo "Job:     ${PBS_JOBID}"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Workdir: ${WORK_DIR}"
echo "============================================================"

cd "${WORK_DIR}" || { echo "ERROR: Cannot cd to ${WORK_DIR}"; exit 1; }

# ── Load Python via module system ────────────────────────────────────────────
module load EasyBuild/4.9.4
module load Python/3.11.3-GCCcore-12.3.0
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# ── Set Python path to project files ─────────────────────────────────────────
export PYTHONPATH="${WORK_DIR}/project:${PYTHONPATH}"

# ── Start virtual display for lumapi (no root required) ──────────────────────
# Uses user-extracted Xvfb + overlayfs to provide /usr/bin/xkbcomp.
# One-time setup: these files were extracted from Rocky8 RPMs to ~/xvfb_*.
DISPLAY_NUM=99
XVFB_BIN="${HOME}/xvfb_extracted/usr/bin/Xvfb"
XVFB_LIBS="${HOME}/xvfb_libs:${HOME}/xvfb_libs/usr/lib64"
XKBCOMP_SRC="${HOME}/xvfb_libs/usr/bin/xkbcomp"
XVFB_LOG="/tmp/xvfb_${PBS_JOBID}.log"

if [[ -x "${XVFB_BIN}" ]]; then
    # Run Xvfb inside a user namespace so we can overlay xkbcomp onto /usr/bin
    UPPER_DIR="/tmp/xkb_upper_${PBS_JOBID}"
    WORK_DIR_XKB="/tmp/xkb_work_${PBS_JOBID}"
    mkdir -p "${UPPER_DIR}" "${WORK_DIR_XKB}"
    cp "${XKBCOMP_SRC}" "${UPPER_DIR}/xkbcomp"

    unshare --user --map-root-user --mount bash -c "
        mount -t overlay overlay \
            -o lowerdir=/usr/bin,upperdir=${UPPER_DIR},workdir=${WORK_DIR_XKB} \
            /usr/bin
        LD_LIBRARY_PATH=${XVFB_LIBS} ${XVFB_BIN} :${DISPLAY_NUM} \
            -screen 0 1024x768x24 &>${XVFB_LOG}
    " &
    XVFB_NS_PID=$!
    export DISPLAY=:${DISPLAY_NUM}
    echo "Xvfb namespace PID: ${XVFB_NS_PID}  DISPLAY=${DISPLAY}"
    # Poll for the X11 socket instead of a fixed sleep — avoids race condition
    # on busy nodes where Xvfb takes longer than 4 seconds to start.
    XVFB_READY=0
    for _i in $(seq 1 10); do
        if [[ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]]; then
            echo "Xvfb socket confirmed after ${_i}s: /tmp/.X11-unix/X${DISPLAY_NUM}"
            XVFB_READY=1
            break
        fi
        sleep 1
    done
    if [[ "${XVFB_READY}" -eq 0 ]]; then
        echo "ERROR: Xvfb socket /tmp/.X11-unix/X${DISPLAY_NUM} NOT FOUND after 10s"
        echo "Xvfb log follows:"
        cat "${XVFB_LOG}" 2>/dev/null || echo "(log empty or missing)"
        kill "${XVFB_NS_PID}" 2>/dev/null
        exit 1
    fi
else
    echo "WARNING: ~/xvfb_extracted/usr/bin/Xvfb not found."
    echo "Run setup: see hpc/scripts/setup_xvfb.sh"
fi

# ── Compile ompt stub on the compute node ───────────────────────────────────
# Compiling here (rather than relying on a pre-built binary) ensures the stub
# works on whatever node PBS assigns.
OMPT_MAP="/tmp/ompt_version_${PBS_JOBID}.map"
echo 'VERSION { global: ompt_start_tool; };' > "${OMPT_MAP}"
if gcc -shared -fPIC -Wl,--version-script="${OMPT_MAP}" \
       -o "${HOME}/ompt_stub_v.so" "${WORK_DIR}/jobs/ompt_stub.c" 2>&1; then
    echo "ompt stub compiled OK on $(hostname)"
else
    echo "ERROR: ompt stub compilation failed — LD_PRELOAD fix will not work"
fi
rm -f "${OMPT_MAP}"

# ── Lumerical library environment ────────────────────────────────────────────
# fdtd-solutions wrapper resets LD_LIBRARY_PATH, so provide the Lumerical
# lib path via these variables so fdtd-solutions-app sees them.
export FDTD_LD_LIBRARY_PATH=/usr/local/lumerical-2021R2.5/lib
export LUMERICAL_LD_LIBRARY_PATH=/usr/local/lumerical-2021R2.5/lib
export LUMERICAL_QT_PLUGIN_PATH=/usr/local/lumerical-2021R2.5/bin
# Also keep it in the Python process's path for libinterop-api.so.1 (ctypes).
export LD_LIBRARY_PATH=/usr/local/lumerical-2021R2.5/lib:${LD_LIBRARY_PATH}
# LD_PRELOAD fix: libiomp5.so (Intel OpenMP bundled with Lumerical 2021R2.5)
# was compiled for glibc 2.17 (RHEL 7). On RHEL 8 / Rocky 8 (glibc 2.28+)
# its PLT entry for ompt_start_tool@@VERSION fails to self-resolve the weak
# symbol, crashing with "undefined symbol: ompt_start_tool (fatal)" during
# init. Preloading this stub provides a strong global definition before
# libiomp5.so loads. Returning NULL disables OMPT (harmless for simulations).
export LD_PRELOAD="${HOME}/ompt_stub_v.so${LD_PRELOAD:+:${LD_PRELOAD}}"

# ── Pre-flight: verify stub preloads and Lumerical libs are reachable ────────
echo "LD_PRELOAD=${LD_PRELOAD}"
echo "FDTD_LD_LIBRARY_PATH=${FDTD_LD_LIBRARY_PATH}"
python -c "
import ctypes, sys
try:
    ctypes.CDLL('/usr/local/lumerical-2021R2.5/lib/libiomp5.so')
    print('[preflight] libiomp5.so loaded OK')
except OSError as e:
    print(f'[preflight] libiomp5.so FAILED: {e}', file=sys.stderr)
    sys.exit(1)
" || { echo "Pre-flight failed — aborting"; exit 1; }

# ── Run pipeline ─────────────────────────────────────────────────────────────
python scripts/server_run.py
EXIT_CODE=$?

# ── Cleanup virtual display ───────────────────────────────────────────────────
if [[ -n "${XVFB_NS_PID}" ]]; then
    kill "${XVFB_NS_PID}" 2>/dev/null
    rm -rf "/tmp/xkb_upper_${PBS_JOBID}" "/tmp/xkb_work_${PBS_JOBID}"
fi

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo ""
    echo "Pipeline failed. Check the output above for the fallback command."
fi
echo "============================================================"

if [[ -n "${NTFY_TOPIC}" ]]; then
    _MINS=$(( SECONDS / 60 )); _SECS=$(( SECONDS % 60 ))
    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        _MSG="✓ Job ${PBS_JOBID} done — ${RUN_SCRIPT} — ${_MINS}m${_SECS}s"
    else
        _MSG="✗ Job ${PBS_JOBID} FAILED (exit ${EXIT_CODE}) — ${RUN_SCRIPT}"
    fi
    curl -s -H "Title: Bragg FDTD" -d "${_MSG}" \
        "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
fi

exit "${EXIT_CODE}"
