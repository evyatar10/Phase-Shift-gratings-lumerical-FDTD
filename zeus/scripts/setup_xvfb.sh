#!/bin/bash
#
# One-time setup: extract Xvfb + dependencies from Rocky8 RPMs (no root needed).
# Run this ONCE on the Zeus login node before submitting Option 2 jobs.
#
# Usage:
#   bash hpc/scripts/setup_xvfb.sh
#

set -e

RPM_BASE="/usr/local/rocky8.10"
XVFB_DIR="${HOME}/xvfb_extracted"
LIBS_DIR="${HOME}/xvfb_libs"

echo "=== Setting up user-space Xvfb on Zeus ==="
echo "Target dirs:"
echo "  Xvfb binary : ${XVFB_DIR}"
echo "  Libraries   : ${LIBS_DIR}"
echo ""

mkdir -p "${XVFB_DIR}" "${LIBS_DIR}"

echo "--- Extracting Xvfb binary ---"
cd "${XVFB_DIR}"
rpm2cpio "${RPM_BASE}/appstream/Packages/x/xorg-x11-server-Xvfb-1.20.11-25.el8_10.x86_64.rpm" | cpio -idmv 2>/dev/null
echo "  Xvfb: $(ls ${XVFB_DIR}/usr/bin/Xvfb)"

echo "--- Extracting xkbcomp (keyboard compiler) ---"
cd "${LIBS_DIR}"
rpm2cpio "${RPM_BASE}/appstream/Packages/x/xorg-x11-xkb-utils-7.7-28.el8.x86_64.rpm" | cpio -idmv 2>/dev/null
echo "  xkbcomp: $(ls ${LIBS_DIR}/usr/bin/xkbcomp)"

echo "--- Extracting libxkbfile ---"
rpm2cpio "${RPM_BASE}/appstream/Packages/l/libxkbfile-1.1.0-1.el8.x86_64.rpm" | cpio -idmv 2>/dev/null
echo "  libxkbfile: $(ls ${LIBS_DIR}/usr/lib64/libxkbfile.so.1)"

echo "--- Extracting libXdmcp ---"
rpm2cpio "${RPM_BASE}/appstream/Packages/l/libXdmcp-1.1.3-1.el8.x86_64.rpm" | cpio -idmv 2>/dev/null
echo "  libXdmcp: $(ls ${LIBS_DIR}/usr/lib64/libXdmcp.so.6)"

echo "--- Copying libXfont2 from EasyBuild ---"
cp /usr/local/eb/software/X11/20231019-GCCcore-13.2.0/lib/libXfont2.so.2* "${LIBS_DIR}/"
echo "  libXfont2: $(ls ${LIBS_DIR}/libXfont2.so.2)"

echo "--- Copying libbz2 from EasyBuild ---"
cp /usr/local/eb/software/bzip2/1.0.8-GCCcore-12.3.0/lib/libbz2.so.1.0 "${LIBS_DIR}/"
echo "  libbz2: $(ls ${LIBS_DIR}/libbz2.so.1.0)"

echo ""
echo "=== Verifying Xvfb can start ==="
UPPER="/tmp/xkb_setup_upper_$$"
WORK="/tmp/xkb_setup_work_$$"
mkdir -p "${UPPER}" "${WORK}"
cp "${LIBS_DIR}/usr/bin/xkbcomp" "${UPPER}/"

unshare --user --map-root-user --mount bash -c "
    mount -t overlay overlay \
        -o lowerdir=/usr/bin,upperdir=${UPPER},workdir=${WORK} \
        /usr/bin
    LD_LIBRARY_PATH=${LIBS_DIR}:${LIBS_DIR}/usr/lib64 \
        ${XVFB_DIR}/usr/bin/Xvfb :99 -screen 0 1024x768x24 \
        &>/tmp/xvfb_setup_test.log &
    sleep 4
    kill -0 \$! 2>/dev/null && echo 'Xvfb OK' || { echo 'Xvfb FAILED'; cat /tmp/xvfb_setup_test.log; exit 1; }
    kill \$!
" && rm -rf "${UPPER}" "${WORK}"

echo ""
echo "=== Setup complete. Option 2 PBS jobs will now work. ==="
