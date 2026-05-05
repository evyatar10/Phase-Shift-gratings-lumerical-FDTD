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
DEF="lumerical.def"

LUM_SRC="${HOME}/ansys_incS/v261/Lumerical"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "  Lumerical container build + deploy"
echo "  Source dir : ${LUM_SRC}"
echo "  Image      : ${SIF}"
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
# --squashfs-comp gzip: Athena runs Apptainer 1.1.6 which only supports gzip/lzo/lz4.
# Apptainer 1.2+ defaults to zstd, producing SIFs that 1.1.x cannot open.
APPTAINER_SQUASHFS_COMP=gzip apptainer build --force "${SIF}" "${DEF}"

# ── 3. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "=== Smoke test: fdtd-engine-ompi-lcl ==="
apptainer run "${SIF}" /opt/lumerical/v261/bin/fdtd-engine-ompi-lcl -v 2>&1 || true

# ── 4. Upload .sif to Athena ──────────────────────────────────────────────────
# We use Apptainer's `--nv` runtime exclusively (no Pyxis/Enroot). The .sqsh
# conversion that used to live here was dropped: it produced a redundant second
# image format and Pyxis NVIDIA injection on Athena does not handle the CUDA
# forward-compat shim ordering we need for the R470 host driver.
echo ""
echo "=== Uploading ${SIF} to Athena (this may take a while; ~5.5 GB) ==="
SSH="${ATHENA_USER}@${ATHENA_HOST}"
ssh "${SSH}" "mkdir -p ${REMOTE_CONTAINER_DIR}"
scp "${SIF}" "${SSH}:${REMOTE_CONTAINER_DIR}/${SIF}"

echo ""
echo "============================================================"
echo "Upload complete: ${REMOTE_CONTAINER_DIR}/${SIF}"
echo ""
echo "CUDA forward-compat sanity test (should print 'CUDA Version: 12.2'):"
echo "  ssh ${SSH}"
echo "  srun --gpus=1 --time=00:02:00 apptainer exec --nv \\"
echo "       ${REMOTE_CONTAINER_DIR}/${SIF} nvidia-smi | head -3"
echo ""
echo "Engine smoke test:"
echo "  srun --gpus=1 apptainer exec --nv \\"
echo "       ${REMOTE_CONTAINER_DIR}/${SIF} fdtd-engine-ompi-lcl -v"
echo "============================================================"
