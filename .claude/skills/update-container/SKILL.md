---
name: update-container
description: Update the Lumerical version inside the Athena container (lumerical-2026R1.sif) without moving gigabytes over the slow VPN — on-Athena sandbox surgery, verified by checksums and a physics canary. Use when a new Lumerical patch/release should go into the Athena container, or to bring Athena in sync with IGUM's native version.
---

# update-container

Battle-tested 2026-08-11 (R1.1 → R1.2, build job 130912, canary job 130913).
Core principle: **the 5 GB .sif never crosses the VPN** (link measured ~0.2 MB/s —
a WSL rebuild + upload is ~13 h; this procedure is ~1 h). All heavy work happens
on Athena; only small text (scripts, manifests) crosses the VPN.

**Never delete any old-version artifact** (user rule 2026-08-11): old sifs are
renamed (`lumerical-2026R1.1.sif`), replaced trees are parked, not removed.

## 1. Get the new-version Linux tree onto Athena

Two routes — pick whichever source exists:

- **From IGUM** (when IGUM's native install already has the target version):
  agent-forwarded tar stream over the Technion LAN (4.2 GB ≈ minutes). No
  authorized_keys edits — `ForwardAgent yes` is already in `~/.ssh/config`:
  ```bash
  eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519 && \
  ssh -A athena 'mkdir -p ~/lum_stage && ssh evyatarrubin@132.68.58.101 \
    "tar cf - -C /apps/ansys/Lumerical-<VER>/opt/lumerical v261" | tar xf - -C ~/lum_stage'
  ssh-agent -k
  ```
  Gotcha: if Athena's known_hosts has a stale IGUM key (IGUM rotated keys once
  already), verify the fingerprint out-of-band from local (`ssh-keyscan -t ed25519
  132.68.58.101 | ssh-keygen -lf -` must match what Athena is offered), then
  `ssh-keygen -R 132.68.58.101` on Athena.

- **From a PC download** (when the version exists nowhere on Technion servers) —
  DONE this way for R1.3 on 2026-08-12, and it is cheaper than the old estimate:
  the `LUMERICAL_<VER>_LINX64` package is **one ~1.1 GB RPM**, not 4–5 GB
  (`<pkg>/rpm_install_files/Lumerical-<VER>.el8.x86_64.rpm`; the rest is tiny
  Dockerfiles + scripts). Only the user can download it (Ansys portal login).
  `rsync -a --partial --inplace <rpm> evyatarrubin@athena.technion.ac.il:lum_r13_pkg/`
  measured **10.6 MB/s** — under 2 minutes, NOT the 0.19 MB/s in the old note, so
  don't plan an overnight push without measuring first. Gate on md5 (local
  `Get-FileHash -Algorithm MD5` vs remote `md5sum`), then extract ON ATHENA
  (Rocky 9 has `rpm2cpio`): `cd ~/lum_r13_stage && rpm2cpio <rpm> | cpio -idm
  --quiet && mv opt/lumerical/v261 ./v261`. R1.3 = 12,898 files / 4.0 GB, and
  `v261/VERSION` states MAJORRELEASE/MINORRELEASE/BUILDNUMBER — check it.

## 2. Verify the staged tree (gate — do not skip)

md5 manifest generated at the source, checked on Athena:
```bash
# at source:  cd <prefix>/opt/lumerical && find v261 -type f -print0 | sort -z | xargs -0 md5sum > manifest.txt
# on Athena:  cd ~/lum_stage && md5sum -c manifest.txt   # expect 100% OK (R1.2 run: 12893/12893)
```
Also check quota first: surgery needs ~19 GB headroom under the 300 GB soft cap
(`quota -s`); jobs hang at "--writable-tmpfs" when over.

## 3. Sandbox surgery — run as a SLURM CPU job, NEVER on the login node

★ Athena's login node **kills all user processes at ssh logout** (nohup and tmux
both die — measured). ★ `sbatch --wrap` is **forbidden** by the cli_filter.
So: a build script file + plain sbatch:
```bash
sbatch --job-name=lum_build --time=02:00:00 --cpus-per-task=8 --mem=32G \
       --output=/home/evyatarrubin/lum_build.log /home/evyatarrubin/lum_r12_build.sh
```
`~/lum_r12_build.sh` is kept on Athena from the R1.2 run — edit paths/version and
reuse. What it does (took ~13 min on a compute node):
1. `apptainer build --force --sandbox ~/lum_sb ~/containers/lumerical-2026R1.sif`
2. `mv` OLD `/opt/lumerical/v261` and `/ansys_inc/v261/licensingclient` OUT of the
   sandbox (parked in `~`, kept), `mv` the staged v261 in, `cp -a` its inner
   `licensingclient` to `/ansys_inc/v261/licensingclient` (engine hardcodes that path)
3. `chmod -R a+rX` + `chmod +x` the five `bin/fdtd-*` entries and
   `licensingclient/linx64/{ansyscl,lmutil,ansysli_util}`
4. sed the version in `.singularity.d/labels.json` + `.singularity.d/runscript.help`
5. `APPTAINER_SQUASHFS_COMP=gzip apptainer build --force
   ~/containers/lumerical-2026R1.sif.new ~/lum_sb`
6. In-sif verify: `fdtd-engine-ompi-lcl -v` prints the target version (OpenMPI
   "help file not found" chatter is cosmetic); engine md5 == manifest md5;
   `/ansys_inc/v261/licensingclient/linx64/ansyscl` present; env vars intact.

## 4. Swap (deliberate, never inside the build script)

```bash
squeue -r -u evyatarrubin        # must be EMPTY of container jobs
cd ~/containers && mv lumerical-2026R1.sif lumerical-<OLDVER>.sif \
                && mv lumerical-2026R1.sif.new lumerical-2026R1.sif
```
The live filename stays `lumerical-2026R1.sif` — ~6 athena job scripts hardcode it.
The old image stays under its version name (never deleted).

## 5. Canary gate (§2/§6: engine bump = named numerics change)

Dispatch ONE task re-running an in-family stored control at identical numerics —
template: `runners/metal_mirror/engine_canary.py` (comb_q3db ctrl row, corr-325 N165;
keep its version-bump log up to date):
```bash
SBATCH_MEM=160G ARRAY_TIME=08:00:00 bash athena/deploy_athena.sh \
    --option3 --spec=runners.metal_mirror.engine_canary --max-concurrent=1
```
PASS = stored anchor reproduced (job 130458 row 0: T 0.4906 / −3.09 dB, Q 13930,
λ 1558.3–1559.0; patch-level agreement should be EXACT — proven for R1.1↔R1.2).
Check the log's "Simulation time" is a real solve (~1 s = silent license no-op).
**Mismatch ⇒ swap back to the old sif and stop.** Only after PASS: delete the
sandbox + stage (new-version scaffolding only — never old-version artifacts).

## 6. Put the SAME version on IGUM (no container is possible there)

IGUM has **no apptainer and no singularity**, no module system, and although
`docker-ce` is installed and the user is in the `docker` group, `docker info` is
denied (verified 2026-08-12). So IGUM stays native, and a version bump there means
an **extracted RPM tree owned by the user** — the same thing the admins do under
`/apps/ansys`. Two gotchas: IGUM is Ubuntu and has **no `rpm2cpio`** (only `cpio`),
and `$HOME` is small — the tree goes on the research volume.

Do the extraction on Athena (step 1) and tar-stream the tree over the Technion
LAN, carrying the md5 manifest with it so the destination self-verifies. Run this
BEFORE the build job, which `mv`s the stage into the sandbox:
```bash
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519 && \
ssh -A evyatarrubin@athena.technion.ac.il '
  ssh evyatarrubin@132.68.58.101 "mkdir -p ~/research/lumerical/Lumerical-<VER>/opt/lumerical"
  tar cf - -C ~/lum_r13_stage v261 lum_r13_md5.txt | ssh evyatarrubin@132.68.58.101 \
    "tar xf - -C ~/research/lumerical/Lumerical-<VER>/opt/lumerical"
  ssh evyatarrubin@132.68.58.101 "cd ~/research/lumerical/Lumerical-<VER>/opt/lumerical \
    && md5sum -c lum_r13_md5.txt | grep -c \": OK\""'
ssh-agent -k
```
4.0 GB took ~5 min (LAN, many small files); R1.3 verified 12898/12898 OK.
Then repoint **`LUM_HOME` in all 6 `igum/jobs/*.sh`** plus the `lmutil` path in
`deploy_igum.sh --license-probe`; the admins' old `/apps/ansys/...` tree stays as
fallback (never deleted). Smoke-test on igum-login1 with the job scripts' own env
(`QT_QPA_PLATFORM=offscreen`, `LD_LIBRARY_PATH=$LUM_HOME/lib:$WORK_DIR/scilibs`,
the `libtbbmalloc` LD_PRELOAD) — **a bare `fdtd-engine -v` without `scilibs` fails
on `libglut.so.3` and is NOT a real failure.** Expect `fdtd-engine -v` = the new
version, a `lumapi.FDTD(hide=True)` session, and numpy/scipy importable.

## Related

- Cluster lockstep: prefer Athena == IGUM version (cross-cluster reproducibility
  is proven and load-bearing). Bump BOTH in the same session and run the canary on
  each — that is how R1.3 was done (2026-08-12).
- Local Windows: only the `LUMERICAL_<VER>_WINX64` installer (user-downloaded);
  installs into `C:\Program Files\Lumerical\v261`, no config change needed.
- Canonical from-scratch container build (WSL, needs installer or staged tree):
  `container/lumerical.def` + `container/build.sh` — updated 2026-08-11, staging
  path `~/ansys_incS_R12/v261/Lumerical` in WSL (not populated by default).
- Memory: `project_athena_container_rebuild_pipeline`,
  `project_lumerical_versions_and_athena_ansys_gate`.
