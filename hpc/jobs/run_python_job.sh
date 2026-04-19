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
    sleep 4
    export DISPLAY=:${DISPLAY_NUM}
    echo "Xvfb started (namespace PID ${XVFB_NS_PID}, DISPLAY=${DISPLAY})"
    echo "Xvfb log: ${XVFB_LOG}"
else
    echo "WARNING: ~/xvfb_extracted/usr/bin/Xvfb not found."
    echo "Run setup: see hpc/scripts/setup_xvfb.sh"
fi

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

exit "${EXIT_CODE}"
