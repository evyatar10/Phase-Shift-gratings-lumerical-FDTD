---
name: add-study
description: Create a new study/runner file (sweep, single-run, TM study, or optimization variant) that actually shows up in the Athena deploy menus and doesn't clobber other studies. Use when asked to add/create a new sweep, scan, study, or runner script.
---

# add-study

New studies are created by **copying the closest existing file** and editing the
config lines — never by scaffolding new infrastructure. The traps below are all from
`runners/README.md` ("the deploy-menu contract") and real incidents; violating them
produces a study that silently doesn't appear in the menu, or clobbers another study's
outputs.

## Recipe

1. **Copy the closest sibling** in the right directory:
   - parameter sweep → `runners/sweeps/<closest>.py`, edit the `SPEC = SweepSpec(...)`
     field lists (sweepable fields = `experiment_card._CARD_FIELD_MAP`; add there once
     to make a new field sweepable).
   - one-shot run → `runners/single/` (top-level `run` callable), TM work → `runners/tm/`.
   - optimization variant → copy `optimize_transmission.py`/`smoke_test.py` in the
     family directory; base config comes from `make_optimization_base()`.
2. **Set a unique label/STUDY_DIR** so outputs land in their own
   `results/<study>/results/` and filenames from `generate_file_tag()` don't collide
   with a concurrently running study (shared-filename races are a real incident).
3. **State the physics line**: target resonance λ + scan-window width + key geometry
   (height/pitch/corrugation for TM) — sanity-check per CLAUDE.md §4.
4. **Smoke locally before dispatch** when the study touches geometry/builders/sources
   (§5): build-only `save_fsp` < 1 min, eyeball geometry. Then `dispatch-study`.

## Deploy-menu contract (violate → invisible or broken study)

- **Menu discovery**: sweeps menu = any file in the family dir containing the literal
  text `SPEC =` (unanchored grep — even in a comment/docstring!). Single/TM menus =
  files with a top-level `run` callable at column 0. `_`-prefixed files and
  `IS_HELPER = True` modules are skipped.
- **Corollary**: shared helper modules must contain neither a top-level `run` nor the
  literal `SPEC =`, or they pollute the menus — put helpers at the `runners/` root
  (never scanned) or `_`-prefix them.
- **Never rename** `single/`, `tm/`, `sweeps/`, or the four optimization directories —
  hardcoded in `deploy_athena.sh`. A new category needs three edits: picker block +
  menu entry in `deploy_athena.sh`, and `_AUTO_DIRS` in BOTH `athena/scripts/athena_run.py`
  and `dgx/scripts/athena_run.py`.
- **rsync `--delete`**: moving/deleting a local file removes it from the server's
  `project/runners/` on the next deploy (server `results/` are safe).

## Known scripting gotchas (cost real debugging time)

- **`sbatch --export` truncates comma-separated values** (a seed list `1,2,3` arrives
  as `1`). Pass lists via a file or repeated env vars, never a comma string.
- Make `STUDY_DIR` **parameter-aware** (e.g. pitch in the name) when the same study
  runs at multiple anchor points, or later runs overwrite earlier ones.
- Forward any new window/geometry env vars through the whole chain (deploy →
  job script → `athena_run.py` → runner) — a var set only locally silently uses the
  server-side default.
- Filename suffixes matter to tooling: a stray tag (e.g. `_smp`) breaks the
  auto-anchor matching in downstream analysis scripts.
