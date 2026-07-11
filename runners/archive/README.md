# runners/archive/ — closed studies (kept runnable, not maintained)

Runner and analysis scripts from studies that are **finished**, moved here
unedited so the live `runners/` menus stay short. Conclusions live in the
session memory and `docs/`. The deploy menus never scan this directory.

## Contents (by program)

- `side_by_side/` — two parallel pi-shift devices coupling/detune program
  (closed 2026-06/07). Includes its `analysis/` plot scripts.
- `sweeps/scatterers/` — lateral-scatterer / in-core-hole recycling program
  (COMPLETE, ceiling ΔT≈+0.003; includes `tm_scatterer_scan.py`, whose
  `build_base` now lives in live `runners/sweeps/_tm_base.py`).
- `sweeps/cavity_shapes/` — cavity/tooth shape + width program (CLOSED
  2026-07-05; winner = plain rect cavity 1050 nm).
- `sweeps/loss_program/` — TM loss round 2: center completion, shift frontier,
  Pareto stack-vs-apod, derived profile, polarimetry, strip reflector (CLOSED
  2026-07-06; final best = W1050 + gap-pair + see-saw stack).
- `sweeps/bic_kerker_trench/` — BIC/Kerker/counterdiabatic batch + lateral
  air-trench studies (ALL DONE 2026-07-07).
- `sweeps/apodization/` — TE/TM apodization sweeps incl. pitch-518.3 variants.
- `sweeps/convergence/` — transverse-domain convergence ladders (box size
  decided: y=6.8/z=8.8 µm for true TM loss; results are keep-forever data).
- `sweeps/` (root) — `tm_periods_match_te.py` + its viewers,
  `compare_3d_field_shift.py`, `plot_tm_periods_match_te.py`.
- `experiment_comparison/` — `pull_by_subname.py` result-puller utility.

## Reviving an archived study

Run by its new module path, e.g.
`python -m runners.archive.sweeps.scatterers.tm_scatterer_scan`, or dispatch
with an explicit `--spec=runners.archive.sweeps.<family>.<name>`. Note:
archived files are removed from the cluster's `project/` copy on the next
deploy (`rsync --delete`); shared helpers they import
(`runners/sweeps/sweep_spec.py`, `runners/sweeps/_tm_base.py`,
`runners/tm/_tm_vs_te_common.py`) stay live, so imports keep working.
