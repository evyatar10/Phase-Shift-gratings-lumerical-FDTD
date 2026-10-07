---
name: dispatch-study
description: Dispatch an FDTD study to Athena the right way — pick the correct deploy flag for the study type, run preflight, smoke-test when required, and report job ID + expected outputs. Use when the user asks to run/submit/dispatch/deploy a simulation, sweep, or optimization ("run this on the server", "send it to Athena").
---

# dispatch-study

Read [project context](../project-context.md), current cluster-ops guidance,
and the relevant live runner before preparing a launch.

1. Identify the already-authorized study, cluster, question and smallest useful
   task set. If the cluster was already chosen, do not ask again.
2. Reuse stored controls at matching numerics. Resolve prior anomalies before
   another costly run. State exact geometry, polarization, resonance/window,
   monitors and source version. Check task membership, not just N_TASKS.
3. Run the relevant offline/build/gradient gates and athena-preflight.
   A calibrated objective is required before optimizer smoke/campaign stages.
4. Use the current deploy parser: inverse design is
   `--lumopt2-design=runners.lumopt2_design.<runner>` with
   `--array-tasks=<ids> --max-concurrent=<count> --keep-h5`.
   Athena supports `--gpu=a100`; IGUM does not. Use LUMOPT2_QOS/LUMOPT2_TIME
   and SBATCH_MEM where implemented. ARRAY_TIME env can be overwritten.
5. Preserve prior matching source versions before an authorized update; inspect
   the upload list and resulting hashes. Per-study manifests prevent list
   collisions, but shared engine edits still require serialization.
6. Verify returned IDs, actual resources, dependency, task count and outputs.
   Long drivers must cold-resume with incremental state before launch.

For user-authorized parallel Athena/IGUM diagnostics, use the installed
cluster-ops workflow: both queues, one shared license budget, disjoint tasks,
preserved reference data and verified source/runtime parity. IGUM native paths
do not substitute automatically for Athena's `/work/` paths. Compare numerical
families before combining cross-server derivative data. Do not duplicate a
running leg, upload frozen dependencies, or bypass a failed scientific gate.
Report new FDTD solves separately from stored-field reconstruction and fitting.

Both Athena and IGUM QOS can requeue; partition-level OFF does not prove immunity.
Afterok chains wait indefinitely if an upstream stage fails. Diagnose before
altering dependencies or resubmitting only genuinely failed/missing work.
Arm proportionate monitoring, inspect an early new-code log, and carry through
result review/next authorized stages. No cancellation or cleanup is implied.
Historical recipes: [Claude source](references/claude-original.md).

Before dispatching a new module, exercise the real driver call convention in an
offline guard/reuse test. The current Athena driver calls main(task_idx=idx);
a runner that only accepts main(task) is incompatible despite passing positional
unit tests. Check the live IGUM driver too when that cluster is targeted.

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
