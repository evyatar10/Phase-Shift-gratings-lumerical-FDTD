---
name: add-study
description: Create a new study/runner file (sweep, single-run, TM study, or optimization variant) that actually shows up in the Athena deploy menus and doesn't clobber other studies. Use when asked to add/create a new sweep, scan, study, or runner script.
---

# add-study

Read [project context](../project-context.md) and `runners/README.md`.
Copy the closest current runner in the same study family; keep its established
SPEC/main(task_idx)/N_TASKS interface rather than adding another framework.

1. Inventory matching stored results. Reuse identical-numerics controls; a
   new solve needs a stated question that those results cannot answer.
2. Give the study a distinct module, label and output directory. Check every
   generated filename for uniqueness across rows and concurrent studies.
3. State polarization, N, pitch, height, corrugation, resonance/window,
   monitor set and mesher. Preserve the user's retained device choices.
4. Check discovery in `build_sweep_list.py` and both deploy parsers. Helpers
   must not accidentally advertise a study; use `IS_HELPER = True` where
   supported. A new family needs both cluster launch paths updated.
5. Run offline checks, and hidden CAD for geometry/source/monitor changes.
   Verify the generated task list and exact supported task IDs before dispatch.

Deployment uses per-study manifests, but common source files remain shared.
Do not upload changed fingerprinted code while existing tasks depend on it.
Deploy now preserves remote-only files; matching files are still replaced.
Use dispatch-study for an authorized launch. Existing data is never a cleanup
opportunity. Original detailed history: [Claude source](references/claude-original.md).

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
