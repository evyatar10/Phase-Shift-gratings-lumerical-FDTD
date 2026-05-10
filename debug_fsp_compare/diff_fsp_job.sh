#!/bin/bash
#SBATCH --job-name=fsp_diff
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=l40s-public
#SBATCH --output=/home/evyatarrubin/diag/fsp_diff-%j.out
#SBATCH --error=/home/evyatarrubin/diag/fsp_diff-%j.out

DIAG_DIR=/home/evyatarrubin/diag
CONTAINER=$HOME/containers/lumerical-2026R1.sif
SCILIBS="${HOME}/scilibs"

LICENSE="${ATHENA_LICENSE:-1055@132.68.48.51}"
INTERCONNECT="${ATHENA_INTERCONNECT:-2325@132.68.48.51}"

HOSTS_FILE="${HOME}/hosts_lum"
if [[ ! -f "${HOSTS_FILE}" ]]; then
    cp /etc/hosts "${HOSTS_FILE}"
    echo "132.68.48.51 lumerical-lm.ece.technion.ac.il lumerical-lm" >> "${HOSTS_FILE}"
fi

apptainer exec --nv \
    --bind "${DIAG_DIR}:/work" \
    --bind "${HOSTS_FILE}:/etc/hosts" \
    --bind "${SCILIBS}:/scilibs" \
    --pwd /work \
    "${CONTAINER}" \
    bash -c "
export LANG=C
export LC_ALL=C
export LD_LIBRARY_PATH=\"\$(echo \"\${LD_LIBRARY_PATH}\" | tr ':' '\\n' | grep -v '^/usr/local/cuda/compat' | paste -sd: -)\"
export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}:/scilibs\"
export LUMERICAL_LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}\"
export LD_PRELOAD=\"/opt/lumerical/v261/lib/libtbbmalloc.so.2:/opt/lumerical/v261/lib/libtbbmalloc_proxy.so.2\"
export ANSYSLMD_LICENSE_FILE='${LICENSE}'
export ANSYSLI_SERVERS='${INTERCONNECT}'
export ANSYS_APIP_DISABLE=1
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_PROVIDER='^efa'
export OMPI_MCA_btl=self,tcp
export OMPI_MCA_mtl='^ofi'
export UCX_TLS=self,sm,tcp
export UCX_NET_DEVICES=lo
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
XVFB_PID=\$!
trap 'kill \$XVFB_PID 2>/dev/null; wait \$XVFB_PID 2>/dev/null' EXIT
export DISPLAY=:99
sleep 1
/opt/lumerical/v261/python/bin/python /work/diff_fsp.py
"
