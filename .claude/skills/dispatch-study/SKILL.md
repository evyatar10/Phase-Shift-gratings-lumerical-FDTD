---
name: dispatch-study
description: Dispatch an FDTD study to Athena the right way — pick the correct deploy flag for the study type, run preflight, smoke-test when required, and report job ID + expected outputs. Use when the user asks to run/submit/dispatch/deploy a simulation, sweep, or optimization ("run this on the server", "send it to Athena").
---

# dispatch-study

All production FDTD goes to Athena via `bash athena/deploy_athena.sh` (CLAUDE.md §1).
This skill is the dispatch checklist; it exists because wrong dispatches are the most
expensive mistake class in this project's history.

## 1. Before dispatching (gate, in order)

0. **Think first, run second** (user rule 2026-08-07). State in one line what this run
   will DECIDE and why stored results can't answer it; prefer the smallest
   discriminating experiment. If the previous run of this program produced ANY
   anomaly (off-family λ/T/fwhm, instant crash, implausible timing), no new dispatch
   until the cause is understood from free diagnostics (stored .mat, scene diffs,
   job + solver `_p0.log` logs, local silent rebuilds). Verify the new run's effective
   numerics (incl. the solver's actual mesh — z-grid count in `_p0.log`) match the
   stored family it will be compared against.

0b. **Ask which cluster** (user rule 2026-08-07): both Athena and IGUM work — ask a
   plain one-line question ("Athena or IGUM?") before dispatching, unless the user
   already named the cluster for this task.

1. **Scope is confirmed** — an exploratory question is NOT authorization to dispatch
   (CLAUDE.md §8). For TM work, confirm height + pitch + corrugation first (§4).
2. **One line stating target resonance λ and scan-window width**, sanity-checked
   against the study (§4). If they conflict with anything the user said — ask.
3. **Echo the *built* config, not the intent**: pitch, n_core, N periods, and which
   monitors are ON (2D fields / far-field), read from the SPEC/runner file or the
   smoke output. Three full resubmissions happened because the dispatched config
   silently used the wrong pitch; three more reruns because far-field/2D monitors were
   off. TM studies copy anchors from `runners/tm/run_tm.py` — never raw
   `SimulationConfig` defaults. TE/TM comparison pairs must record **identical
   monitor sets**.
4. **Preflight**: run the `athena-preflight` skill (license ports, queue, quota).
   Never launch a second `--option3` sweep while one has pending tasks; serialize
   jobs that share `data/sweep_list.txt` / `results/` (§6). **No exceptions** — a
   different study or a tiny 4-task job still rewrites the shared sweep_list.txt and
   kills every pending task whose index exceeds the new length ("SWEEP_INDEX out of
   range"; 2026-07-02 hole-scan incident). Check pending with `squeue -r` (plain
   `squeue` collapses a pending array to one line). QOS `24h_1g` caps 100 submitted /
   4 running tasks per user → chunk big arrays with `--array-tasks=`.
5. **Smoke-test rule (§5)**: if the change touches geometry, a new builder/scaffold,
   gradients, or source/BC setup → smoke first. For the four optimization families
   that means dispatching `smoke_test.py` (~15–30 min) before `optimize_transmission.py`.
6. **Unique outputs**: new/parallel studies need distinct `generate_file_tag()` names
   and their own `STUDY_DIR` — shared `.h5`/`.mat` filenames have raced before.

## 2. Pick the dispatch form (from runners/README.md)

| Study type | Command |
|---|---|
| Single run (`runners/single/`) | `bash athena/deploy_athena.sh --option2 --run=<module_name>` |
| TM study (`runners/tm/`) | same contract as single, via the TM menu / `--run=` |
| TE-vs-TM parallel pair | add `--pol-array` (task 0 = TE, 1 = TM); after download, stitch: `python -m runners.tm.run_tm_vs_te --stitch <results_dir>` |
| Sweep (`runners/sweeps/*.py` with `SPEC =`) | `bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.<study>` |
| Experiment cards | `bash athena/deploy_athena.sh --cards=runners.experiment_comparison.<file>` |
| Optimization families | `--inverse-design=` / `--gradient-free-design=` / `--fd-gradient-design=` / `--lumerical-native=` with the module path |
| Upload code only | `--upload-only` |

- GPU/partition: use the default auto-pick (don't ask). Only heads-up case: a long
  stateful optimization on a `*-shared` partition can be preempted — mention
  `--gpu=a100`, don't block on it.
- Deploy does `rsync --delete` of the source tree — locally deleted/renamed files
  vanish from the server copy on dispatch.

## 3. After submitting

- Capture the job/array ID from the sbatch output and state it, plus the expected
  number of tasks (= sweep-list length) and rough walltime.
- Field-profile-monitor runs need `SBATCH_MEM` far above the 64 G default (monitors,
  not domain size, drive RAM — see memory `project_athena_job_memory_footprint.md`).
- Don't poll in a loop; check on demand with the `athena-status` skill. If the job
  ends implausibly fast / results are empty, apply §6 (license silent no-op) before
  re-dispatching.
- When it finishes: `fetch-results` skill.

## Dispatch machinery updates (2026-08-15 — general, all study kinds)

- **Per-study sweep lists**: deploys now upload `data/sweep_list_<study>.txt`
  (study = spec module basename) and export that path. One study's deploy can
  no longer kill/corrupt another study's pending or preemption-REQUEUEd tasks.
  Parallel deploys are allowed IFF both studies use per-study lists AND the
  new deploy touches only its own study's files (check rsync itemized output;
  shared engine/builder edits still serialize — CLAUDE.md §6 amendment).
- **`--after=<jobid>`**: chains the new array behind an in-flight job
  (afterok) — queue whole stage-sequences in one sitting; stages start
  automatically server-side even with the laptop off.
- **Walltime/QOS**: every Athena partition is PreemptMode=REQUEUE; the QOS is
  the walltime cap (default 24h_1g = 23:30; association also has 4d_1g etc.).
  ★`ARRAY_TIME=...` as an env override is silently IGNORED (athena.conf
  plain-assigns it); `SBATCH_MEM` works. lumopt2 campaigns use the
  LUMOPT2_QOS/LUMOPT2_TIME env knobs (read at submit time). Verify with
  `sacct --format=TimeLimit` after submitting.
- Long stateful drivers must cold-start-resume from their own persisted logs
  (REQUEUE can restart them anytime); array sim tasks are naturally idempotent.
- Slurm works INSIDE the Athena container when needed (submission proven):
  recipe in `memory/project_slurm_container_fixes.md`.

## QOS lane selection (2026-08-15, measured caps — Athena)

| QOS | MaxWall | GPUs/user | Jobs/user | Priority |
|---|---|---|---|---|
| 2h_2g | 2 h | 2 | 3 | **1000** |
| 12h_4g | 12 h | 4 | 3 | 500 |
| 24h_1g (default) | 24 h | — | 4 | 300 |
| 24h_4g | 24 h | 4 | 3 | 250 |
| 72h_8g | 72 h | 8 | 1 | 50 |
| 4d_1g | 4 d | — | 8 | 50 |

Priority is INVERSE to walltime, and GPU caps are PER-QOS lanes that stack.
Rules: pick the smallest QOS whose MaxWall covers the task and request an
honest `--time=` (backfill loves short honest requests): canaries ≤2 h →
`--qos=2h_2g` (3.3× default priority); validation ~3-12 h → `--qos=12h_4g`;
default arrays → 24h_1g; multi-day drivers → 4d_1g. Both `--qos=` and
`--time=` are per-dispatch deploy flags (mirrored athena+igum). Running short
tasks on a second lane raises total concurrent GPUs beyond the single-lane 4.
★afterok trap: if the dependency job FAILS, the dependent array pends forever
(DependencyNeverSatisfied) — release with
`scontrol update job <id> dependency=''` (also the tool for re-ordering a
chain, used live 2026-08-15).

## Job-placement policy by TYPE (user tiers, 2026-08-16) + measured preemption
## mechanics — memorize, this decides how every job is dispatched

Preemption on Athena is QOS-based (`preempt/qos`): the `contrib` QOS
(priority 10000, 7-day) PREYS ON EVERY lane we have (12h_4g, 24h_1g, 24h_4g,
4d_1g, 4h_0g, 72h_8g) — **no preempt-proof lane exists for us**; preempted
jobs get a 10-minute grace window after the signal, then REQUEUE. Therefore
protection comes from job design, not lane choice:

| Tier | Examples | Protection required | Lane advice |
|---|---|---|---|
| Stateless array tasks | sweeps, canaries, confirm rows | none — idempotent, requeue = harmless re-run (loss ≤ 1 solve) | smallest adequate QOS, highest priority (2h_2g / 12h_4g) |
| Long single solves | accurate-mesh rows, big-domain runs (1-3 h/task) | none needed, but budget the re-run risk on multi-hour tasks | honest --time, short-QOS lane |
| STATEFUL DRIVERS | inverse-design campaigns, optimizations, anything accumulating state | ★MANDATORY incremental persistence + cold-start resume (loss ≤ 1 eval); status/progress logging is part of the job's deliverable | any lane (resume makes preemption cheap); 4d_1g for walltime, NOT for safety |

Importance scales the care: a throwaway sweep row that dies is noise; an
inverse-design driver's state and STATUS VISIBILITY are part of the result.
When unsure which tier a job is: if a requeue-from-zero would make you angry,
it is tier 3 and needs resume before dispatch.

### Tier-2 refinement (user, 2026-08-16): LONG single solves — duration, not
### mesh mode, is the criterion

An optimization-mesh solve can still run hours-to-days (measured extreme: the
N=1300 production run, 71.5 h) — losing one mid-solve is harmful regardless of
mesh. Facts that bound the options:
- **No engine-level checkpoint/resume exists** (fdtd-engine CLI help has no
  checkpoint/restart option — a killed solve restarts from zero, only the
  10-min QOS grace exists). A single solve is atomic; it cannot be protected
  by logging.
- Therefore for solves expected >≈3 h: (1) PREFER IGUM — its partitions show
  partition-level PreemptMode=OFF and our job history there has zero
  preemptions (though the cluster config is preempt/qos, so treat IGUM as
  "empirically calm", NOT proven immune); (2) request honest --time;
  (3) accept and STATE the re-run budget at dispatch ("this task re-runs from
  zero if preempted, cost X h"); (4) remember MaxBatchRequeue=5 — SLURM
  auto-retries up to 5 times, so the job eventually completes unless
  contention is pathological — but each retry is from zero.
- If a future Lumerical version adds engine checkpointing, this tier changes —
  check the release notes on every version bump.

## Trouble-finder = standard post-dispatch step (user rule 2026-08-16, EVERY run)

After EVERY dispatch, arm a monitor proportionate to the run (template in the
work-alone skill; the point is trouble seen live, not at drain).
★Change-key rule (2026-08-16): the monitor's state key holds ONLY
decision-relevant fields (job+state, counts, errors, seat bands) — never
elapsed time/timestamps, or it wakes a costly model turn every sweep for
nothing (measured ~90 no-op wakes/day; details in work-alone):
- Small array (<~1 h total): one T+5-10 min log peek suffices (catches build/
  import/geometry crashes before GPU-hours burn). No standing monitor needed.
- Standard array: live event monitor — per-job STATE/NODE diffs (requeues!),
  new log error signatures (Traceback|TASK FAILED|LumApiError|Unable to
  checkout|dead device|DIVERGED), drain. Poll ~4-5 min.
- Fan-outs / campaigns: add license-seat bands (IGUM lmstat, ≥35/50 HIGH,
  ≥45/50 CRITICAL) and, for multi-day drivers, a quota sample (~300 GB hang
  trap) + per-study _files size.
- ★Implausibly-FAST completion is an event too (solve task ending in minutes
  = license silent-no-op suspect — check the log's "Simulation time" first).
- After the deploy itself: verify the rsync itemized output actually shipped
  the files you edited (the stale-server-code trap — perms can make rsync
  skip root *.py silently; --inplace is the known fix).
