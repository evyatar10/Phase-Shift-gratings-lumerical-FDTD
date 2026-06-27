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
# — so this job needs no GPU and no Xvfb. We still run inside the container to get
# the container python's scipy/matplotlib (+ /scilibs for libgfortran/libquadmath).
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

WORK_DIR="/home/evyatarrubin/bragg_sim_athena"
PROJECT_DIR="${WORK_DIR}/project"
SCRIPTS_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

CONTAINER="$HOME/containers/lumerical-2026R1.sif"

# Scientific libs (libgfortran, libquadmath) needed by scipy/numpy.
# Container's base image (CUDA devel) lacks libgfortran. We supply it via a
# user-maintained directory bound into /scilibs and appended to LD_LIBRARY_PATH.
SCILIBS="${HOME}/scilibs"

# Sweep results subfolder (container path). RESULTS_DIR is bound to /work/results,
# and config puts run output under <RUN_NAME>/results, so the .mat (and the PNGs
# this job writes) live in /work/results/tm_te_shift/results.
SUMMARY_DIR="${SUMMARY_DIR:-/work/results/tm_te_shift/results}"

# X_AXIS=relative plots shift as % of half-pitch (so different pitches compare);
# default 'absolute' plots shift in nm. Forwarded to plot_tm_te_shift.py --x.
X_AXIS="${X_AXIS:-absolute}"

echo "============================================================"
echo "Tooth-shift summary job — SUMMARY_DIR=${SUMMARY_DIR}"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "============================================================"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container not found: ${CONTAINER}"
    exit 1
fi

apptainer exec \
    --bind "${PROJECT_DIR}:/work/project" \
    --bind "${SCRIPTS_DIR}:/work/scripts" \
    --bind "${DATA_DIR}:/work/data" \
    --bind "${RESULTS_DIR}:/work/results" \
    --bind "${LOGS_DIR}:/work/logs" \
    --bind "${SCILIBS}:/scilibs" \
    --pwd /work \
    "${CONTAINER}" \
    bash -c "
export LANG=C
export LC_ALL=C
# Strip /usr/local/cuda/compat* from LD_LIBRARY_PATH (CUDA 12.2 forward-compat
# shim intended for R470 hosts; Athena runs R570+ so the shim must not lead).
export LD_LIBRARY_PATH=\"\$(echo \"\${LD_LIBRARY_PATH}\" | tr ':' '\\n' | grep -v '^/usr/local/cuda/compat' | paste -sd: -)\"
export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:/scilibs\"
/opt/lumerical/v261/python/bin/python /work/project/runners/sweeps/plot_tm_te_shift.py --results-dir '${SUMMARY_DIR}' --x '${X_AXIS}'
"

EXIT_CODE=$?
echo "============================================================"
echo "Summary plotter finished — exit code ${EXIT_CODE}  ($(date))"
echo "============================================================"
exit ${EXIT_CODE}
