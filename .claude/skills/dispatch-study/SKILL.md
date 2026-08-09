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
