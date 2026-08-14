#!/bin/bash
#
# Build the Lumerical Apptainer image and upload it to Athena. Run this from
# the container/ directory inside WSL2.
#
# ⚠ NOT THE ROUTINE UPGRADE PATH — this ends in a 5.5 GB upload over the VPN.
# To put a new Lumerical version into the EXISTING image, use the
# `update-container` skill (on-Athena sandbox surgery; the .sif never moves).
# Use this script only to recreate the image from scratch.
#
# Prerequisites:
#   1. Apptainer installed in WSL2  (sudo add-apt-repository ppa:apptainer/ppa
#                                    sudo apt install apptainer)
#   2. Lumerical 2026 R1.3 tree staged at ~/ansys_incS_R13/v261/Lumerical/
#      (source: the LINX64 package's single RPM,
#       <pkg>/rpm_install_files/Lumerical-2026R1-3-*.el8.x86_64.rpm, extracted
#       with `rpm2cpio <rpm> | cpio -idm` → opt/lumerical/v261;
#       the %files section of lumerical.def copies it from there)
#   3. Windows ssh config has the `athena` alias (upload uses Windows
#      ssh.exe/scp.exe — Athena's hostname does not resolve from WSL)
#
# Usage:
#   bash build.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ATHENA_ALIAS="athena"            # Windows ~/.ssh/config alias
REMOTE_CONTAINER_DIR="containers"

IMAGE_NAME="lumerical-2026R1"
SIF="${IMAGE_NAME}.sif"
DEF="lumerical.def"

LUM_SRC="${HOME}/ansys_incS_R13/v261/Lumerical"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "  Lumerical container build + deploy"
echo "  Source dir : ${LUM_SRC}"
echo "  Image      : ${SIF}"
echo "  Target     : ${ATHENA_ALIAS}:${REMOTE_CONTAINER_DIR}/${SIF}.new"
echo "============================================================"

# ── 1. Verify Lumerical source ────────────────────────────────────────────────
if [[ ! -x "${LUM_SRC}/bin/fdtd-engine-ompi-lcl" ]]; then
    echo ""
    echo "ERROR: Lumerical not found or engine missing at:"
    echo "  ${LUM_SRC}/bin/fdtd-engine-ompi-lcl"
    echo ""
    echo "Stage the Lumerical 2026 R1.3 tree at ~/ansys_incS_R13/v261/ first."
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
# Uploaded as ${SIF}.new — the live image is swapped deliberately afterwards
# (queue must be checked first; old image kept as .bak until the canary passes).
# Windows ssh.exe/scp.exe are used because Athena's hostname only resolves via
# the pinned alias in the Windows ~/.ssh/config.
echo ""
echo "=== Uploading ${SIF} to Athena as ${SIF}.new (~5.5 GB, slow link) ==="
SIF_WIN="$(wslpath -w "${SCRIPT_DIR}/${SIF}")"
ssh.exe "${ATHENA_ALIAS}" "mkdir -p ${REMOTE_CONTAINER_DIR}"
scp.exe "${SIF_WIN}" "${ATHENA_ALIAS}:${REMOTE_CONTAINER_DIR}/${SIF}.new"

echo ""
echo "============================================================"
echo "Upload complete: ${REMOTE_CONTAINER_DIR}/${SIF}.new"
echo ""
echo "To activate (AFTER checking squeue is empty of container jobs):"
echo "  ssh ${ATHENA_ALIAS} \"cd ${REMOTE_CONTAINER_DIR} && \\"
echo "      mv ${SIF} ${IMAGE_NAME}.prev.sif && mv ${SIF}.new ${SIF}\""
echo ""
echo "Then send ONE canary task and compare to a stored control before"
echo "dispatching anything else (engine change = named numerics change)."
echo "============================================================"
