# Athena GPU Workflow

Parallel GPU pipeline for running Lumerical FDTD on Technion's **Athena** cluster (9× DGX A100 nodes). This workflow is **fully independent** of the Zeus CPU workflow in `hpc/` — no files there are modified or shared.

---

## Architecture

```
hpc_gpu/
├── deploy_athena.sh          ← run this on your laptop to submit jobs
├── container/
│   ├── lumerical.def         ← Apptainer recipe (Lumerical 2026 R1.1 + CUDA 12.2)
│   └── build.sh              ← build .sif image and upload to Athena
├── jobs/
│   ├── run_fsp_gpu.sh        ← Phase 1: single .fsp → GPU engine (SLURM)
│   ├── run_fsp_gpu_array.sh  ← Phase 1b: sweep .fsp list → GPU array (SLURM)
│   └── run_python_gpu.sh     ← Phase 2: full lumapi pipeline via Python (SLURM)
└── scripts/
    └── athena_run.py         ← server-side dispatcher (GPU-aware, Athena paths)
```

**Zeus** (`hpc/`) uses PBS, `fdtd-engine-mpich2nem`, Lumerical 2021R2.5. Untouched.
**Athena** (`hpc_gpu/`) uses SLURM, `fdtd-engine-ompi-lcl`, Lumerical 2026 R1.1, GPU enabled.

### Container & CUDA pin

- Single image format: `.sif` (Apptainer). Both Option 1 and Option 2 use
  `apptainer exec --nv` — no Pyxis/Enroot involvement.
- Base image: `nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04`.
- Athena DGX hosts ship NVIDIA driver R470 LTS (native cap CUDA 11.4).
  CUDA 12 runs on R470 only via the **forward-compatibility shim** at
  `/usr/local/cuda/compat/`. The container's `%environment` block prepends
  that path to `LD_LIBRARY_PATH` so the shim is loaded ahead of the host's
  R470 `libcuda.so.1` that `--nv` injects. CUDA 12.2 is the smallest forward
  bridge over R470 of any CUDA 12.x — the rationale for this minor-version
  pin lives in the planning notes; bump it if/when the Athena driver is
  upgraded to R535+.
- CUDA-aware MPI: out of scope. Lumerical's engine is a closed binary linked
  against its own MPI; an external CUDA-aware OpenMPI build cannot be
  injected. Within a DGX A100 node, the engine already uses CUDA IPC over
  NVLink for inter-GPU traffic.

---

## Before you start — Step 0 checklist

SSH into Athena (`ssh evyatarrubin@dgx-master.technion.ac.il`) and run:

```bash
# 1. License server reachable?
nc -vz 132.68.48.51 1055          # must succeed from compute node too

# 2. License tier (gates multi-node multi-GPU)
# Copy lmutil from the Lumerical installer tarball first, then:
./lmutil lmstat -a -c 1055@132.68.48.51
# Look for: fdtd_gpu, fdtd_solutions_business/enterprise

# 3. Apptainer available for building?
apptainer --version               # if missing, build on WSL2 locally

# 4. Scratch filesystem path
df -h                              # ask CIS if /scratch/$USER doesn't exist

# 5. Billing account
# Most DGX setups require #SBATCH --account=<code>
# Add it to the job scripts if prompted by sbatch
```

If any of steps 1–2 fail, stop and contact Technion CIS before proceeding.

---

## Phase 1: Engine-only GPU runs

### Step 1 — Build and upload the container (once)

```bash
# Lumerical 2026 R1.1 must be installed in WSL2 at ~/ansys_incS/v261/Lumerical/
# (already done via AnsysInstaller.sh — no re-download needed)
cd hpc_gpu/container
bash build.sh
```

This builds `lumerical-2026R1.sif` and uploads it to `~/containers/` on Athena.

After the upload completes, run the **CUDA forward-compat sanity test** once:

```bash
ssh evyatarrubin@dgx-master.technion.ac.il
srun --gpus=1 --time=00:02:00 apptainer exec --nv \
     ~/containers/lumerical-2026R1.sif nvidia-smi | head -3
```

The output must show `CUDA Version: 12.2`. If it shows `11.4`, the
`LD_LIBRARY_PATH` prefix in `lumerical.def %environment` is being clobbered
by something downstream — the GPU engine will silently CPU-fallback until
that is fixed.

### Step 2 — Single simulation

```bash
bash hpc_gpu/deploy_athena.sh --option1 --preset single
```

Generates the `.fsp` locally (same `local_save_fsp.py` as Zeus), uploads it, and submits `run_fsp_gpu.sh` via `sbatch`.

### Step 3 — Parameter sweep (job array)

```bash
bash hpc_gpu/deploy_athena.sh --option1 --preset sweep_shift
```

When multiple `.fsp` files are detected, `deploy_athena.sh` automatically submits them as a SLURM job array throttled to max 4 concurrent (adjust `K` in `deploy_athena.sh` after checking your license seat count).

### Monitor and retrieve

```bash
bash hpc_gpu/deploy_athena.sh --watch-only   # poll latest job + auto-download
bash hpc_gpu/deploy_athena.sh --results       # immediate download only
```

Results land in `results_from_athena/` — separate from Zeus's `results_from_server/`.

---

## Phase 2: Full Python / lumapi pipeline

```bash
bash hpc_gpu/deploy_athena.sh --option2 --run single_sim
bash hpc_gpu/deploy_athena.sh --option2 --run sweep_shift
```

Uploads the project + `athena_run.py`, submits `run_python_gpu.sh`. The dispatcher at `hpc_gpu/scripts/athena_run.py` sets `USE_GPU=True` and enables GPU on the FDTD resource via `fdtd.setresource("FDTD", 1, "GPU", True)` — all without modifying any simulation module.

---

## GPU vs CPU guidance

| Use Athena GPU when | Use Zeus CPU when |
|---|---|
| Single 3D FDTD sim > ~50M Yee cells | varFDTD, MODE, FDE, RCWA, DEVICE (no GPU support) |
| Large convergence sweeps that would take days on Zeus | Small simulations (<50M cells) |
| Scaling tests needing 2–8 GPUs | License-constrained sweeps where Zeus runs in parallel for free |

**Measured speedup**: 1× A100 ≈ 5× vs 63-core CPU for 0.85B Yee cells.  
Peak GPU memory: ~32 GB for 0.85B cells (A100 80GB handles up to ~2B cells).

---

## Scaling GPUs

Edit `#SBATCH --gpus=N` in `run_fsp_gpu.sh` and set `NUM_MPI_RANKS=N` accordingly.

```
--gpus=1   1× A100, 1 MPI rank  (start here, benchmark first)
--gpus=4   4× A100, 4 MPI ranks (one rank per GPU)
--gpus=8   full DGX node, 8× A100
```

Multi-node (>8 GPUs) requires **Business or Enterprise license** — verify with `lmutil lmstat`.

---

## License

Single source of truth: `ATHENA_LICENSE` in
[deploy_athena.sh](deploy_athena.sh) (currently `11055@dgx-master`,
backed by Technion's ECE FlexLM at `132.68.48.51` via internal forwarding).
The deploy script `--export`s it to every sbatch; job scripts pick it up
with a sensible fallback default so manual `sbatch` invocations also work.

The license server's hostname (`lumerical-lm.ece.technion.ac.il`) is not in
Athena's DNS, so each job script binds a small `~/hosts_lum` file into the
container's `/etc/hosts` to make the FlexLM handshake succeed.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to display` | `xvfb-run` missing from container — rebuild with `bash container/build.sh` |
| `License server not responding` | Run `nc -vz 132.68.48.51 1055` from the compute node |
| `fdtd-engine-ompi-lcl: command not found` | Container not built / wrong path — check `$LUMERICAL_HOME/bin/` |
| `Out of GPU memory` | Reduce mesh size, or increase `--gpus` to spread across multiple A100s |
| `setresource GPU error` | Verify license includes `fdtd_gpu` feature via `lmutil lmstat` |
| `sbatch: error: Batch job submission failed` | Add `#SBATCH --account=<your_account>` to job scripts |
| `nvidia-smi` shows `CUDA Version: 11.4` inside the container | The forward-compat shim isn't winning the `LD_LIBRARY_PATH` race. Confirm `lumerical.def %environment` prepends `/usr/local/cuda/compat`. |
| Sweep ran for hours but logs say `WARNING: could not enable GPU` | Submit with `REQUIRE_GPU=1` (Option 2) so future runs hard-exit instead of silently CPU-falling-back. |
