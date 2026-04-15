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

WORK_DIR="/home/evyat/bragg_sim"

echo "============================================================"
echo "Job:     ${PBS_JOBID}"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Workdir: ${WORK_DIR}"
echo "============================================================"

cd "${WORK_DIR}" || { echo "ERROR: Cannot cd to ${WORK_DIR}"; exit 1; }

# ── Activate conda environment ───────────────────────────────────────────────
# Adjust the path below if conda is installed elsewhere (e.g. ~/anaconda3)
CONDA_BASE="${HOME}/miniconda3"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="${HOME}/anaconda3"
fi

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda not found. Searched:"
    echo "  ~/miniconda3/etc/profile.d/conda.sh"
    echo "  ~/anaconda3/etc/profile.d/conda.sh"
    echo "Run 'which conda' on the login node and update CONDA_BASE in this script."
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate lumerical
echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV}"

# ── Run pipeline ─────────────────────────────────────────────────────────────
python scripts/server_run.py
EXIT_CODE=$?

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo ""
    echo "Pipeline failed. Check the output above for the fallback command."
fi
echo "============================================================"

exit "${EXIT_CODE}"
