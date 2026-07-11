# runners/archive/ — closed studies (kept runnable, not maintained)

Runner and analysis scripts from studies that are **finished**. They are moved
here unedited so the live `runners/` menus stay short; conclusions live in the
session memory and `docs/`. Nothing in the deploy menus scans this directory.

- `side_by_side/` — two parallel pi-shift devices coupling/detune program
  (closed 2026-06/07; results under `results_from_athena/side_by_side_*`).
  Includes its `analysis/` plot scripts.
- `sweeps/` — local viewers/analysis one-offs for the TM period-match study
  (`view_tm_match_*`, `plot_tm_periods_match_te`).
- `experiment_comparison/` — `pull_by_subname.py` result-puller utility.

To rerun an archived module use its new path, e.g.
`python -m runners.archive.side_by_side.side_by_side_coupling`, or dispatch with
an explicit `--spec=runners.archive.side_by_side.<name>`. Note: archived files
are removed from the cluster's `project/` copy on the next deploy
(`rsync --delete`) — redeploy after moving a study back if you revive it.
