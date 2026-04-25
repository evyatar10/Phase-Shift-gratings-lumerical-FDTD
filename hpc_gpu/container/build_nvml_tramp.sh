#!/bin/bash
#
# Build the NVML trampoline shim on the Athena login node.
# Run this ONCE after uploading nvml_tramp.c (deploy_athena.sh does it automatically).
#
# The resulting libnvidia-ml.so.1 is bind-mounted into the container at runtime
# to intercept NVML symbol lookups before Apptainer's host-lib injection directory.
#
# Usage (on Athena login node):
#   bash build_nvml_tramp.sh
#
# Or from your laptop:
#   ssh evyatarrubin@dgx-master.technion.ac.il "bash ~/bragg_sim_gpu/jobs/build_nvml_tramp.sh"

set -e

TRAMP_SRC="${HOME}/bragg_sim_gpu/jobs/nvml_tramp.c"
TRAMP_OUT="${HOME}/nvml_tramp/libnvidia-ml.so.1"

if [[ ! -f "${TRAMP_SRC}" ]]; then
    echo "ERROR: source not found: ${TRAMP_SRC}"
    echo "Run deploy_athena.sh first to upload nvml_tramp.c"
    exit 1
fi

echo "=== Building NVML trampoline ==="
mkdir -p "${HOME}/nvml_tramp"

gcc -O2 -shared -fPIC \
    -Wl,-soname,libnvidia-ml.so.1 \
    -o "${TRAMP_OUT}" \
    "${TRAMP_SRC}" \
    -ldl

echo "Built: ${TRAMP_OUT}"
echo ""

echo "=== Verifying exports ==="
MISSING=0
for sym in nvmlDeviceGetPcieLinkMaxSpeed nvmlDeviceGetBusType nvmlDeviceGetMemoryBusWidth \
           nvmlInit_v2 nvmlShutdown nvmlDeviceGetCount_v2 nvmlDeviceGetHandleByIndex_v2 \
           nvmlDeviceGetMemoryInfo nvmlDeviceGetPciInfo_v3 nvmlDeviceGetCudaComputeCapability; do
    if nm -D "${TRAMP_OUT}" | grep -q " T ${sym}$"; then
        echo "  EXPORTED: ${sym}"
    else
        echo "  MISSING:  ${sym}   <-- PROBLEM"
        MISSING=1
    fi
done

if [[ "${MISSING}" -eq 1 ]]; then
    echo ""
    echo "ERROR: one or more required exports are missing. Check nvml_tramp.c."
    exit 1
fi

echo ""
echo "=== Verifying real NVML is locatable at runtime ==="
# Quick test: load the shim in an interactive compute shell
# (run the apptainer test manually after building)
echo "To test inside the container on a compute node:"
echo "  srun --gpus=1 --time=00:02:00 apptainer exec --nv \\"
echo "      --bind \${HOME}/nvml_tramp:/nvml_tramp \\"
echo "      ~/containers/lumerical-2026R1.sif \\"
echo "      bash -c 'export LD_LIBRARY_PATH=/nvml_tramp:\$LD_LIBRARY_PATH; \\"
echo "               nm -D /nvml_tramp/libnvidia-ml.so.1 | grep -c \" T nvml\"'"
echo ""
echo "Build complete. The trampoline is ready for use."
