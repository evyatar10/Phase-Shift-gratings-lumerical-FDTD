# IGUM — Technion ECE faculty cluster (second dispatch target)

**Status: machinery ready and validated up to the SLURM gate.** Job submission is
blocked ONLY by a missing SLURM account association for `evyatarrubin` — one admin
action fixes it (email draft at the bottom). Everything else is proven working,
including a native lumapi FDTD session with a real license checkout.

**IGUM is an ADDITIONAL option, not a migration. Athena remains the default
cluster.** Nothing here touches `athena/`.

---

## 1. Cluster identity (recon 2026-07-05)

| Item | Value |
|---|---|
| Login/dev nodes | `132.68.58.101` / `.102` / `.103` (igum-login1/2; **FQDN does not resolve off-cluster — use the IP**) |
| Access | Technion **VPN required**; SSH alias `igum` (in `~/.ssh/config`), key installed 2026-07-05 |
| Username | `evyatarrubin` (same as Athena; Technion directory password) |
| OS / scheduler | Ubuntu 24.04, SLURM (ClusterName=`ece-igum`) |
| Docs | https://igum.ece.technion.ac.il/guides/getting-started (VPN-gated) |
| Monitoring | Zabbix: `igum-zabbix.ece.technion.ac.il:3000` |
| Support | ece.igum.ad@technion.ac.il |
| Dev node (login1) | 16 cores, 31 GB RAM, **RTX 3090 24 GB**, driver R595 — fine for smoke tests/interactive |

## 2. Storage

- `$HOME = /home/evyatarrubin` — small NFS volume (14 TB shared, ~92% full). Keep only dotfiles here.
- **`~/research` → `/research/amir.r/evyatarrubin`** (symlink, created by admin) — the
  research volume (28 TB, ~12 TB free). **All work goes here.**
- `REMOTE_BASE = /home/evyatarrubin/research/bragg_sim_igum` (via the symlink) with the
  standard `project/ data/ results/ jobs/logs scripts/` layout.
- Results download to **`results_from_igum/`** locally (never mixed with `results_from_athena/`).
- No `quota` command; watch `df -h ~/research`.

## 3. SLURM facts (verified via sinfo/scontrol)

**GPU inventory** (all "mixed" state = in active use by others):

| Partition | Nodes | GPUs | Notes |
|---|---|---|---|
| `part-preempt` (default) | ece-efrats[3-5], ece-silbmark[1-2], ece-ykasten1 | **40× A100 + 8× RTX PRO 6000** | PreemptMode=**REQUEUE**; requires `--qos=qos-preempt`; AllowAccounts=ALL |
| `part-interactive` | ece-efrats5, ece-ykasten1 | subset of above | acct-preempt + qos-interactive |
| `part-efrats` / `part-silbmark` / `part-ykasten` | lab-owned | — | not ours |
| `debug` / `part-ugproj` | ece-efrats2, ece-ugproj[1-2] | A100:8 / A5000:6 / A6000:7 | debug needs `--qos=normal`; ugproj = undergrad |

- **Submission recipe:** `--account=acct-preempt --partition=part-preempt --qos=qos-preempt --gres=gpu:N`
  (QOS must match the partition; `--account` is mandatory — Athena needs neither).
- MaxTime=UNLIMITED on part-preempt (no wall cap found); MaxArraySize=1001; MaxJobCount=10000.
- **Preemption = REQUEUE**: preempted jobs go back in queue and rerun. Fine for
  independent array sweep tasks; risky for long stateful optimizations (those stay on Athena).
- TRES billing: gpu weight 16 (fairshare accounting exists).
- `sacctmgr` cannot reach slurmdbd from the login node (connection refused) — use
  `sshare -U` and `scontrol show assoc_mgr` instead.

### CURRENT BLOCKER (2026-07-05)
`evyatarrubin` has **zero SLURM associations** — `sshare -U` empty; every
srun/sbatch under every account/partition/QOS combination returns
"Invalid account or account/partition combination". The admin must add the user
(presumably to `acct-preempt` with `qos-preempt` + `qos-interactive`). Email draft in §8.

## 4. Lumerical: NATIVE (containers are NOT supported on IGUM)

- Install: **`/apps/ansys/Lumerical-2026-R1.2/opt/lumerical/v261`** (2026 R1.2 —
  one patch newer than Athena's 2026 R1 container).
- Lmod modulefile exists (`module load lumerical` on nodes with Lmod init), but our job
  scripts set the env explicitly — no module dependency.
- **Verified working on igum-login1 (2026-07-05):**
  - `fdtd-engine -v` → "FDTD Solver Version 8.35.4522" (plain engine).
  - lumapi headless: `QT_QPA_PLATFORM=offscreen` + bundled python 3.13 →
    **full FDTD session opened, license checked out, `addfdtd()` OK**. No Xvfb on
    IGUM and none needed (Athena's container used Xvfb; here Qt-offscreen replaces it).
  - numpy 2.2.2 / scipy 1.14.1 in the bundled python; system libgfortran present
    (no `~/scilibs` needed).
- **Gotcha:** `fdtd-engine-ompi-lcl` (the MPI-wrapped engine Athena uses) fails with
  missing `libmpi.so.40` — the RPM-extracted install doesn't ship OpenMPI. The job
  scripts use the **plain `fdtd-engine`** instead (fine for single-GPU runs).
- lmutil: `/apps/ansys/Lumerical-2026-R1.2/opt/lumerical/v261/licensingclient/linx64/lmutil`.

## 5. License — SHARED with Athena

- Same FlexLM server both clusters use: `1055@132.68.48.51` / `2325@132.68.48.51`
  (= `lumerical-lm.ef.technion.ac.il`, alias `.ece.`). **DNS resolves natively on IGUM**
  (unlike Athena) — no `/etc/hosts` hack needed, and `lmstat` is reliable here
  (probe 2026-07-05: 50 `lum_fdtd_solve` seats, 10 in use).
- **Seats are one pool across Athena + IGUM.** Before a big IGUM run, check what
  Athena is consuming (`bash athena/deploy_athena.sh --status` / `--license-probe`).
  `MAX_CONCURRENT=4` in `igum.conf` is deliberately conservative for this reason.
- **One GPU solve = 7 `lum_fdtd_solve` seats, and the EFFECTIVE pool is 42 of
  the 50 issued (MEASURED 2026-07-25: 6 solves = 42/50 in use, and a 7th solve
  still died with `FlexNet -4` — ~8 seats are evidently reserved server-side).
  Hard ceiling: 6 concurrent solves total** across ALL arrays and clusters; the
  7th dies instantly with a bare `LumApiError: 'in run:'` (real error in the
  layout's `*_p0.log`). Keep the SUM of all array throttles ≤ 6, and stagger
  new arrays with `--dependency` instead of launching into a full house (jobs
  42317/42325 lost 12 tasks this way).
- **part-preempt compute nodes are bare** (no libgfortran, no X11/GL client
  libs — ece-ykasten1 at least; part-lumerical + login nodes have everything).
  Fix (2026-07-25, permanent): `REMOTE_BASE/scilibs/` carries libgfortran,
  libquadmath, the whole libxcb*/libX*/libGL* family and an apt-extracted
  libglut, and both job scripts put it on `LD_LIBRARY_PATH`. Some of these are
  dlopen'ed at runtime — `ldd` showing clean does NOT prove a node is OK; the
  real test is an `srun` lumapi `FDTD(hide=True)` probe.

## 6. How to use (mirrors the Athena workflow)

| Task | Command |
|---|---|
| Dispatch python pipeline | `bash igum/deploy_igum.sh --option2 --run=<module>` |
| Dispatch sweep array | `bash igum/deploy_igum.sh --option3 --spec=<runners.sweeps.module>` |
| Upload only | `bash igum/deploy_igum.sh --upload-only` |
| Queue status | `bash igum/deploy_igum.sh --status` (or `ssh igum squeue -u evyatarrubin`) |
| License seats | `bash igum/deploy_igum.sh --license-probe` |
| Download results | `bash igum/deploy_igum.sh --results-no-fsp` → `results_from_igum/` |
| Stop a job | `ssh igum "scancel <jobid>"` (confirm-first policy applies, same as Athena) |
| Logs | `ssh igum "tail -60 ~/research/bragg_sim_igum/jobs/logs/lum_*.out"` |

The `athena-*` skills are NOT forked; use the table above as the IGUM equivalent.
`--gpu=<type>` is not supported on IGUM (single partition list; edit `igum.conf`).

## 7. Differences vs Athena (operational)

| | Athena | IGUM |
|---|---|---|
| GPU request | `--gpus=1` | `--gres=gpu:1` (+ mandatory `--account`) |
| QOS | `24h_1g` (100 submit / 4 run caps) | `qos-preempt` (must match partition; no wall cap found) |
| Partitions | per-GPU-type queues, incl. non-preemptible `-public` | one `part-preempt` pool (A100/RTXPRO6000), **preemptible (REQUEUE)** |
| Runtime | apptainer container `lumerical-2026R1.sif` | **native** `/apps/ansys/Lumerical-2026-R1.2` |
| Headless display | Xvfb inside container | `QT_QPA_PLATFORM=offscreen` |
| License DNS | hosts-file hack required | resolves natively; lmstat reliable (no `-96` lore) |
| Engine binary | `fdtd-engine-ompi-lcl` | plain `fdtd-engine` (no libmpi.so.40 on IGUM) |
| Work dir | `/home/.../bragg_sim_athena` | `~/research/bragg_sim_igum` (research volume) |
| Local results | `results_from_athena/` | `results_from_igum/` |

## 8. Admin email draft (send to ece.igum.ad@technion.ac.il)

> **Subject:** SLURM account association missing for user evyatarrubin
>
> Hi, I'm a research student (supervisor: Amir Rosenthal; my research storage is
> already set up at /research/amir.r/evyatarrubin). I can SSH into igum-login1, but my
> user has no SLURM association: `sshare -U` returns nothing, and every
> srun/sbatch attempt fails with "Invalid account or account/partition combination
> specified" for every combination I tried (acct-preempt / qos-preempt /
> part-preempt, part-interactive, debug+normal). Could you please add my user to
> the appropriate account — I believe acct-preempt with qos-preempt (and
> qos-interactive), so I can run on part-preempt?
> My workload is Ansys Lumerical FDTD (GPU, using the faculty license server).
> Thanks! Evyatar Rubin

## 9. Cleanup candidates on IGUM (pending user approval — do NOT delete silently)

- `~/containers/lumerical-2026R1.sif` (5.4 GB) — transferred before we learned
  containers are unsupported; unused by the native pipeline.
- `~/scilibs/` — unused natively (system libgfortran exists).

## 10. Validation record

- 2026-07-05: upload-only deploy ✓ (full layout at `~/research/bragg_sim_igum`);
  license probe from IGUM ✓ (50 seats / 10 in use); native lumapi session +
  license checkout + geometry ops ✓ (igum-login1); entrypoint dry-run bootstrap ✓.
- **Real FDTD baseline (`run_simulation`) — PASSED, 2026-07-05**, natively on
  igum-login1's RTX 3090 (GPU solve confirmed at 99% util; license checked out).
  Result `result_N80_D13p76_avg.mat`, fetched to `results_from_igum/run_simulation/results/`:
  | | IGUM (2026 R1.2, RTX 3090) | Athena (2026 R1 container) |
  |---|---|---|
  | resonance_wavelength_nm | 1579.44 | 1577.30 |
  | resonance_transmission | 0.8686 | 0.8650 |
  | \|spectral_fwhm_nm\| | 1.211 | 1.183 |
  | fwhm_m (spatial) | 16.47 µm | 16.34 µm |
  Same physics; Δλ≈2.1 nm / ΔT≈0.004 consistent with the solver-version + hardware
  difference — per project policy, never mix IGUM and Athena numbers inside one
  study's comparison (each study stays on one cluster with its own control).
- First SLURM job: PENDING admin association fix (email in §8).
