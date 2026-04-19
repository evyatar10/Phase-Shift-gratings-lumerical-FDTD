#!/bin/bash
#
# PBS job: run an existing .fsp file with the Lumerical FDTD engine (MPI).
#
# Usage — pass FSP_FILE as a PBS variable:
#   qsub -v FSP_FILE="layout_yourfile.fsp" run_fsp_job.sh
#
# Or edit FSP_FILE directly below and submit with:
#   qsub run_fsp_job.sh
#
#PBS -N bragg_fsp
#PBS -q zeus_all_q
#PBS -l select=1:ncpus=80
#PBS -l walltime=24:00:00
#PBS -m abe
#PBS -S /bin/bash
#PBS -M evyatar10.rubin@gmail.com
#PBS -j oe
#PBS -k eod
#PBS -o bragg_fsp.out

ulimit -s unlimited

# ── CONFIGURE ────────────────────────────────────────────────────────────────
FSP_DIR="/home/evyatarrubin/bragg_sim/results/layouts"

# FSP_FILE can be set via -v when submitting, or hardcoded here as a fallback:
FSP_FILE="${FSP_FILE:-layout_REPLACE_ME.fsp}"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "Job:       ${PBS_JOBID}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "FSP dir:   ${FSP_DIR}"
echo "FSP file:  ${FSP_FILE}"
echo "============================================================"

cd "${FSP_DIR}" || { echo "ERROR: Cannot cd to ${FSP_DIR}"; exit 1; }

if [[ ! -f "${FSP_FILE}" ]]; then
    echo "ERROR: ${FSP_FILE} not found in ${FSP_DIR}"
    exit 1
fi

APP="/usr/local/lumerical/bin/fdtd-engine-mpich2nem"
MPI="/usr/local/lumerical/mpich2/nemesis/bin/mpiexec.hydra"
NCPUS=80   # must match #PBS -l select=1:ncpus=XX above

"${MPI}" -n "${NCPUS}" "${APP}" -t 1 "./${FSP_FILE}"
EXIT_CODE=$?

echo "============================================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================================"

exit "${EXIT_CODE}"
