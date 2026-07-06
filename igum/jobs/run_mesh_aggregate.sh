#!/bin/bash
#
# Server-side aggregator for mesh-convergence array results.
#
# Submitted by deploy_athena.sh with --dependency=afterok:<array_id> right
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
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=evyatar10.rubin@gmail.com
#SBATCH --output=logs/mesh_aggregate-%j.out
#SBATCH --error=logs/mesh_aggregate-%j.out

mkdir -p logs

WORK_DIR="${WORK_DIR:-/home/evyatarrubin/research/bragg_sim_igum}"
PROJECT_DIR="${WORK_DIR}/project"
SCRIPTS_DIR="${WORK_DIR}/scripts"
DATA_DIR="${WORK_DIR}/data"
RESULTS_DIR="${WORK_DIR}/results"
LOGS_DIR="${WORK_DIR}/logs"

# NATIVE Lumerical install on IGUM — containers are NOT supported on this
# cluster. The aggregator only needs the bundled python (bragg_device imports
# lumapi at module load, so we still use Lumerical's python + LUMAPI_PATH).
LUM_HOME="/apps/ansys/Lumerical-2026-R1.2/opt/lumerical/v261"

PHASE="${PHASE:?PHASE env var required (X or YZ)}"

echo "============================================================"
echo "Mesh aggregator job — PHASE=${PHASE}"
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

# Patch config.BASE_SAVE_DIR before run_mesh_convergence.py loads — it
# defaults to a Windows path that's unreachable on the cluster and would crash
# at module-load time when os.makedirs(LAYOUTS_DIR) runs.
"${LUM_HOME}/python/bin/python" - <<PYEOF
import sys, runpy
sys.path.insert(0, '${PROJECT_DIR}')
import config
config.BASE_SAVE_DIR  = '${RESULTS_DIR}'
config.NEFF_DATA_PATH = '${DATA_DIR}/FDE_sweep_results.mat'
config.LUMAPI_PATH    = '${LUM_HOME}/api/python/lumapi.py'
sys.argv = ['run_mesh_convergence.py', '--aggregate', '${PHASE}']
runpy.run_path('${PROJECT_DIR}/convergence_testing/run_mesh_convergence.py',
               run_name='__main__')
PYEOF

EXIT_CODE=$?
echo "============================================================"
echo "Aggregator finished — exit code ${EXIT_CODE}  ($(date))"
echo "============================================================"
exit ${EXIT_CODE}
