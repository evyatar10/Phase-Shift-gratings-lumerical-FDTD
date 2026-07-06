#!/bin/bash
#
# Server-side summary plotter for the tooth-shift sweep (runners/sweeps/tm_te_shift.py).
#
# Submitted by deploy_athena.sh with --dependency=afterok:<array_id> right after
# the tm_te_shift array. Runs once on CPU cores (no GPU), scans
# /work/results/tm_te_shift/result_*.mat, and writes transmission_vs_shift.png,
# modewidth_vs_shift.png and shift_summary.csv into that folder. After this job
# completes, `--results-no-fsp` downloads the summary images with no extra step.
#
# The plotter (plot_tm_te_shift.py) imports only glob/scipy/matplotlib — no lumapi
# — so this job needs no GPU. Uses the native Lumerical python for scipy/matplotlib.
#
#
#SBATCH --job-name=shift_summary
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/shift_summary-%j.out
#SBATCH --error=logs/shift_summary-%j.out

mkdir -p logs

WORK_DIR="${WORK_DIR:-/home/evyatarrubin/research/bragg_sim_igum}"
PROJECT_DIR="${WORK_DIR}/project"
SCRIPTS_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

# NATIVE Lumerical install on IGUM — containers are NOT supported on this
# cluster; only the bundled python (scipy/matplotlib) is needed here.
LUM_HOME="/apps/ansys/Lumerical-2026-R1.2/opt/lumerical/v261"

# Sweep results subfolder. config puts run output under <RUN_NAME>/results, so
# the .mat (and the PNGs this job writes) live in ${RESULTS_DIR}/tm_te_shift/results.
SUMMARY_DIR="${SUMMARY_DIR:-${RESULTS_DIR}/tm_te_shift/results}"

# X_AXIS=relative plots shift as % of half-pitch (so different pitches compare);
# default 'absolute' plots shift in nm. Forwarded to plot_tm_te_shift.py --x.
X_AXIS="${X_AXIS:-absolute}"

echo "============================================================"
echo "Tooth-shift summary job — SUMMARY_DIR=${SUMMARY_DIR}"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "============================================================"

if [[ ! -d "${LUM_HOME}" ]]; then
    echo "ERROR: native Lumerical install not found: ${LUM_HOME}"
    exit 1
fi

export LANG=C
export LC_ALL=C
export QT_QPA_PLATFORM=offscreen
export LD_LIBRARY_PATH="${LUM_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${LUM_HOME}/python/bin/python" "${PROJECT_DIR}/runners/sweeps/plot_tm_te_shift.py" \
    --results-dir "${SUMMARY_DIR}" --x "${X_AXIS}"

EXIT_CODE=$?
echo "============================================================"
echo "Summary plotter finished — exit code ${EXIT_CODE}  ($(date))"
echo "============================================================"
exit ${EXIT_CODE}
