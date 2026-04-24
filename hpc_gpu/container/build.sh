#!/bin/bash
#
# Build the Lumerical Apptainer image, convert to Enroot .sqsh, and upload
# to Athena. Run this from the hpc_gpu/container/ directory inside WSL2.
#
# Prerequisites:
#   1. Apptainer installed in WSL2  (sudo add-apt-repository ppa:apptainer/ppa
#                                    sudo apt install apptainer)
#   2. Lumerical 2026 R1.1 already installed at ~/ansys_incS/v261/Lumerical/
#      (done — the %files section of lumerical.def copies it from there)
#   3. Your Athena SSH key is loaded  (ssh-add ~/.ssh/id_rsa)
#
# Usage:
#   bash build.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ATHENA_USER="evyatarrubin"
ATHENA_HOST="dgx-master.technion.ac.il"
REMOTE_CONTAINER_DIR="/home/${ATHENA_USER}/containers"

IMAGE_NAME="lumerical-2026R1"
SIF="${IMAGE_NAME}.sif"
SQSH="${IMAGE_NAME}.sqsh"
DEF="lumerical.def"

LUM_SRC="${HOME}/ansys_incS/v261/Lumerical"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "  Lumerical container build + deploy"
echo "  Source dir : ${LUM_SRC}"
echo "  Image      : ${SIF}"
echo "  Squash     : ${SQSH}"
echo "  Target     : ${ATHENA_USER}@${ATHENA_HOST}:${REMOTE_CONTAINER_DIR}/"
echo "============================================================"

# ── 1. Verify Lumerical source ────────────────────────────────────────────────
if [[ ! -x "${LUM_SRC}/bin/fdtd-engine-ompi-lcl" ]]; then
    echo ""
    echo "ERROR: Lumerical not found or engine missing at:"
    echo "  ${LUM_SRC}/bin/fdtd-engine-ompi-lcl"
    echo ""
    echo "Install Lumerical 2026 R1.1 to ~/ansys_incS/v261/ first."
    exit 1
fi
echo "Source check OK: fdtd-engine-ompi-lcl found."

# ── 2. Build .sif ─────────────────────────────────────────────────────────────
echo ""
echo "=== Building Apptainer image (this takes ~5–15 minutes) ==="
echo "    (copying 3.7 GB Lumerical tree into the container)"
apptainer build --force "${SIF}" "${DEF}"

# ── 3. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test: fdtd-engine-ompi-lcl ==="
apptainer run "${SIF}" /opt/lumerical/v261/bin/fdtd-engine-ompi-lcl -v 2>&1 || true

# ── 4. Convert .sif → .sqsh for Enroot/Pyxis ─────────────────────────────────
echo ""
echo "=== Converting .sif → .sqsh for Enroot ==="
if command -v enroot &>/dev/null; then
    enroot import --output "${SQSH}" "apptainer://${SCRIPT_DIR}/${SIF}"
    echo "Created: ${SQSH}"
    UPLOAD_FILE="${SQSH}"
else
    echo "WARNING: enroot not found locally — uploading .sif and converting on Athena."
    UPLOAD_FILE="${SIF}"
fi

# ── 5. Upload to Athena ───────────────────────────────────────────────────────
echo ""
echo "=== Uploading ${UPLOAD_FILE} to Athena (this may take a while) ==="
SSH="${ATHENA_USER}@${ATHENA_HOST}"
ssh "${SSH}" "mkdir -p ${REMOTE_CONTAINER_DIR}"
scp "${UPLOAD_FILE}" "${SSH}:${REMOTE_CONTAINER_DIR}/${UPLOAD_FILE}"

echo ""
echo "============================================================"
echo "Upload complete: ${REMOTE_CONTAINER_DIR}/${UPLOAD_FILE}"
echo ""
if [[ "${UPLOAD_FILE}" == *.sif ]]; then
    echo "Convert .sif → .sqsh on Athena login node:"
    echo "  enroot import --output ${REMOTE_CONTAINER_DIR}/${SQSH} \\"
    echo "      apptainer://${REMOTE_CONTAINER_DIR}/${SIF}"
    echo ""
fi
echo "Verify license reachability:"
echo "  nc -vz 132.68.48.51 1055"
echo ""
echo "Test the container:"
echo "  srun --gpus=1 \\"
echo "    --container-image=${REMOTE_CONTAINER_DIR}/${IMAGE_NAME}.sqsh \\"
echo "    --container-env=ANSYSLMD_LICENSE_FILE=1055@132.68.48.51 \\"
echo "    fdtd-engine-ompi-lcl -v"
echo "============================================================"
