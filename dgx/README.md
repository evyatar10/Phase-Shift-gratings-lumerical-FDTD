# DGX GPU Workflow

> **FROZEN — LEGACY, DO NOT MAINTAIN (2026-07-11).** The DGX FDTD GPU path is
> silently broken with Lumerical 2026R1 on the R470 driver (~3 s no-op solve,
> then port-expansion crash) and this tree no longer receives athena/ fixes —
> it is already behind (missing the `run_mesh_convergence` alias and the
> `STUDY_DIR_NAME` override in `scripts/athena_run.py`). Dispatch to **Athena**
> (default) or **IGUM** instead; keep this directory for reference only.

Parallel GPU pipeline for running Lumerical FDTD on Technion's **DGX** cluster (dgx-master, 9× DGX A100 nodes). This workflow is **fully independent** of the Zeus CPU workflow (`zeus/`) — no files there are modified or shared.

---

## Architecture

```
dgx/
├── deploy_dgx.sh          ← run this on your laptop to submit jobs
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
**Athena** (`dgx/`) uses SLURM, `fdtd-engine-ompi-lcl`, Lumerical 2026 R1.1, GPU enabled.

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

SSH into DGX (`ssh evyatarrubin@dgx-master.technion.ac.il`) and run:

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
cd container
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
bash dgx/deploy_dgx.sh --option1 --preset single
```

Generates the `.fsp` locally (same `local_save_fsp.py` as Zeus), uploads it, and submits `dgx/jobs/run_fsp_gpu.sh` via `sbatch`.

### Step 3 — Parameter sweep (job array)

```bash
bash dgx/deploy_dgx.sh --option1 --preset sweep_shift
```

When multiple `.fsp` files are detected, `deploy_dgx.sh` automatically submits them as a SLURM job array throttled to max 4 concurrent (adjust `K` in `deploy_dgx.sh` after checking your license seat count).

### Monitor and retrieve

```bash
bash dgx/deploy_dgx.sh --watch-only   # poll latest job + auto-download
bash dgx/deploy_dgx.sh --results       # immediate download only
```

Results land in `results_from_dgx/` — separate from Zeus's `results_from_server/`.

---

## Phase 2: Full Python / lumapi pipeline

```bash
bash dgx/deploy_dgx.sh --option2 --run single_sim
bash dgx/deploy_dgx.sh --option2 --run sweep_shift
```

Uploads the project + `athena_run.py`, submits `run_python_gpu.sh`. The dispatcher at `dgx/scripts/athena_run.py` sets `USE_GPU=True` and enables GPU on the FDTD resource via `fdtd.setresource("FDTD", 1, "GPU", True)` — all without modifying any simulation module.

The sweep loop inside this option runs **sequentially** — one simulation at a time inside one SLURM job. Use Option 3 below for parallel sweeps.

---

## Phase 3: Python sweep as a SLURM job array (parallel)

For parameter sweeps where each value is independent, submit one SLURM array task per value. Each task gets its own GPU + CPUs and runs end-to-end (build, solve, analyze, save `.mat`).

```bash
bash athena/deploy_dgx.sh --option3 --sweep=shift
bash athena/deploy_dgx.sh --option3 --sweep=inner_size
bash athena/deploy_dgx.sh --option3 --sweep=generic
bash athena/deploy_dgx.sh --option3 --sweep=mesh_conv_a
bash athena/deploy_dgx.sh --option3 --sweep=mesh_conv_b

# Or supply a project-level SweepSpec study file (one task per cartesian point):
bash athena/deploy_dgx.sh --spec=runners.sweeps.apod_and_shift
```

Flow:
1. `build_sweep_list.py` runs **locally**, emitting one task line per sweep value (one line per `SPEC.expand()` config).
2. The list is uploaded to `${REMOTE_BASE}/data/sweep_list.txt`.
3. `sbatch --array=0-N-1%K jobs/run_python_array.sh` submits the array. Each task reads its line by `$SLURM_ARRAY_TASK_ID` and dispatches via `athena_run_one.py`.

`K` (max concurrent tasks) is `MAX_CONCURRENT` from `dgx.conf` — override per run with `--max-concurrent=N`. Bounded by `lum_fdtd_solve` license seats; probe with:

```bash
bash athena/deploy_dgx.sh --license-probe
```

Available sweep kinds:
- `spec`            → any module in `runners/sweeps/` exposing `SPEC: SweepSpec`. Pass with `--spec=runners.sweeps.<study>`.
- `mesh_conv_a/b`   → `convergence_testing/run_mesh_convergence.py::PHASE_A_VALUES / PHASE_B_VALUES`

Per-task logs land in `${REMOTE_BASE}/jobs/logs/lum_array-<JOBID>_<TASKID>.out`.

---

## GPU vs CPU guidance

| Use DGX GPU when | Use Zeus CPU when |
|---|---|
| Single 3D FDTD sim > ~50M Yee cells | varFDTD, MODE, FDE, RCWA, DEVICE (no GPU support) |
| Large convergence sweeps that would take days on Zeus | Small simulations (<50M cells) |
| Scaling tests needing 2–8 GPUs | License-constrained sweeps where Zeus runs in parallel for free |

**Measured speedup**: 1× A100 ≈ 5× vs 63-core CPU for 0.85B Yee cells.  
Peak GPU memory: ~32 GB for 0.85B cells (A100 80GB handles up to ~2B cells).

---

## Scaling GPUs

Edit `#SBATCH --gpus=N` in `dgx/jobs/run_fsp_gpu.sh` and set `NUM_MPI_RANKS=N` accordingly.

```
--gpus=1   1× A100, 1 MPI rank  (start here, benchmark first)
--gpus=4   4× A100, 4 MPI ranks (one rank per GPU)
--gpus=8   full DGX node, 8× A100
```

Multi-node (>8 GPUs) requires **Business or Enterprise license** — verify with `lmutil lmstat`.

---

## License

Single source of truth: `DGX_LICENSE` in
[deploy_dgx.sh](deploy_dgx.sh) (currently `11055@dgx-master`,
backed by Technion's ECE FlexLM at `132.68.48.51` via internal forwarding).
The deploy script `--export`s it to every sbatch; job scripts pick it up
with a sensible fallback default so manual `sbatch` invocations also work.

The license server's hostname (`lumerical-lm.ece.technion.ac.il`) is not in
Athena's DNS, so each job script binds a small `~/hosts_lum` file into the
container's `/etc/hosts` to make the FlexLM handshake succeed.

### GPU FDTD feature is currently MISSING from the pool — confirmed 2026-04-25

The Technion ECE Lumerical pool does **not** carry `lum_fdtd_solve_gpu`. CPU
features (`lum_fdtd_solve`, `lum_fdtd_gui`, etc.) are present, but no GPU
variant. Compare to Speos in the same pool which *does* carry
`speos_solver_gpu` alongside `speos_solver`.

Symptoms when this is the case:
- Jobs run successfully on the assigned compute node — they just run on CPU.
- `nvidia-smi` on the assigned node shows 0% util, 0 MiB memory, no compute apps.
- Engine output shows `Required licenses (tasks) determined by total core and thread count: <CPU thread count>` and "*Max time remaining: 100+ hrs*" pacing.
- This is a **silent CPU fallback** — no error appears in the engine log.

How to verify the feature pool yourself:
```bash
ssh evyatarrubin@dgx-master.technion.ac.il
srun --gpus=1 --time=00:03:00 apptainer exec --nv \
  --bind $HOME/hosts_lum:/etc/hosts \
  ~/containers/lumerical-2026R1.sif \
  /ansys_inc/v261/licensingclient/linx64/lmutil lmstat -a -c 1055@132.68.48.51 \
  | grep -E 'Users of (lum_|speos_solver)'
```
Look for any line beginning `Users of lum_fdtd_solve_gpu` (or
`lum_fdtd_gpu` / `fdtd_gpu`). If absent, GPU runs cannot proceed.

What to ask Technion CIS for:
> "Please add the `lum_fdtd_solve_gpu` feature to the Lumerical entries
> in the campus FlexLM file `/ansyslm/shared_files/licensing/license_files/ansyslmd.lic`.
> The Speos product line in the same file already carries
> `speos_solver_gpu`, so the convention is established. Reference: Ansys
> Lumerical FDTD GPU support has been included in Standard since 2023 R2."

The job scripts now pre-flight check this feature before launching the
engine and a runtime watchdog kills the engine if no GPU memory is in use
120s after start (`REQUIRE_GPU=1` is the default — set `REQUIRE_GPU=0` only
if you specifically want a CPU run on Athena). Without these guards, a CPU
fallback can waste 100+ hours of cluster time per `.fsp`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to display` | `xvfb-run` missing from container — rebuild with `bash container/build.sh` |
| `License server not responding` | Run `nc -vz 132.68.48.51 1055` from the compute node |
| `fdtd-engine-ompi-lcl: command not found` | Container not built / wrong path — check `$LUMERICAL_HOME/bin/` |
| `Out of GPU memory` | Reduce mesh size, or increase `--gpus` to spread across multiple A100s |
| `setresource GPU error` | Verify license includes `lum_fdtd_solve_gpu` (or `fdtd_gpu`) feature — see "GPU FDTD feature missing" above. |
| Job runs but `nvidia-smi` shows 0% util, "Max time remaining: 100+ hrs" | Silent CPU fallback. License pool lacks GPU FDTD feature. The pre-flight check should now catch this in <10 s; if it doesn't, your `lmutil` query path is broken. |
| `sbatch: error: Batch job submission failed` | Add `#SBATCH --account=<your_account>` to job scripts |
| `nvidia-smi` shows `CUDA Version: 11.4` inside the container | The forward-compat shim isn't winning the `LD_LIBRARY_PATH` race. Confirm `lumerical.def %environment` prepends `/usr/local/cuda/compat`. |
| Sweep ran for hours but logs say `WARNING: could not enable GPU` | Submit with `REQUIRE_GPU=1` (Option 2) so future runs hard-exit instead of silently CPU-falling-back. |
