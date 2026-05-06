#!/bin/bash
#
# Server-side aggregator for mesh-convergence array results.
#
# Submitted by deploy_dgx.sh with --dependency=afterok:<array_id> right
# after a mesh_conv_x or mesh_conv_yz array. Runs once on a CPU core, scans
# /work/results/mesh_convergence/array_part_ph${PHASE}_*.json, and folds them
# into the shared checkpoint JSON. After this job completes, `--results`
# downloads a fully-populated checkpoint with no extra local step.
#
# Required env (passed via sbatch --export):
#   PHASE   "X" or "YZ" — selects which array_part_*.json files to aggregate
#
# Why container: run_mesh_convergence.py imports bragg_device which imports
# lumapi at module load. Running inside the container avoids that ImportError
# without us having to refactor the aggregator out into a separate file.
#
#SBATCH --job-name=mesh_aggregate
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/mesh_aggregate-%j.out
#SBATCH --error=logs/mesh_aggregate-%j.out

mkdir -p logs

WORK_DIR="/home/evyatarrubin/bragg_sim_gpu"
PROJECT_DIR="${WORK_DIR}/project"
SCRIPTS_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

CONTAINER="$HOME/containers/lumerical-2026R1.sif"

PHASE="${PHASE:?PHASE env var required (X or YZ)}"

echo "============================================================"
echo "Mesh aggregator job — PHASE=${PHASE}"
echo "Job:        ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "============================================================"

if [[ ! -f "${CONTAINER}" ]]; then
    echo "ERROR: container not found: ${CONTAINER}"
    exit 1
fi

# Patch config.BASE_SAVE_DIR before run_mesh_convergence.py loads — it
# defaults to a Windows path that's unreachable on Athena and would crash
# at module-load time when os.makedirs(LAYOUTS_DIR) runs.
apptainer exec \
    --bind "${PROJECT_DIR}:/work/project" \
    --bind "${SCRIPTS_DIR}:/work/scripts" \
    --bind "${DATA_DIR}:/work/data" \
    --bind "${RESULTS_DIR}:/work/results" \
    --bind "${LOGS_DIR}:/work/logs" \
    --pwd /work \
    "${CONTAINER}" \
    /opt/lumerical/v261/python/bin/python -c "
import sys, runpy
sys.path.insert(0, '/work/project')
import config
config.BASE_SAVE_DIR  = '/work/results'
config.NEFF_DATA_PATH = '/work/data/FDE_sweep_results.mat'
sys.argv = ['run_mesh_convergence.py', '--aggregate', '${PHASE}']
runpy.run_path('/work/project/convergence_testing/run_mesh_convergence.py',
               run_name='__main__')
"

EXIT_CODE=$?
echo "============================================================"
echo "Aggregator finished — exit code ${EXIT_CODE}  ($(date))"
echo "============================================================"
exit ${EXIT_CODE}
