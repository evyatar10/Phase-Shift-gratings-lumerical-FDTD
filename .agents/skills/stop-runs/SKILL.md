---
name: stop-runs
description: Safely stop/cancel/pause specifically authorized Athena SLURM jobs — resolve the exact job IDs, state them back, cancel only those jobs, and verify. Use whenever the user says stop / kill / cancel / pause a run or "cancel and resubmit". Codex does not inherit Claude's cancellation prompt.
---

# stop-runs

Read [project context](../project-context.md). This skill requires a specific
user request to stop/cancel the relevant jobs; reading a failure or a historical
runbook is not cancellation authorization. Codex does not inherit Claude's
scancel prompt or remote_destructive_guard hook.

1. Read the current queue/accounting and map the requested study to exact
   job/task IDs. Resolve genuine ambiguity before acting; do not ask again
   when the user already authorized those exact jobs.
2. State the resolved IDs and study names. Cancel only those IDs; never use
   a blanket user-wide cancellation. Array root IDs stop every task; use a
   task ID when only that task was requested.
3. Verify resulting states and report any unrelated jobs still running.
   Preserve partial results, manifests, logs and companion HDF5.

If cancel-and-resubmit was authorized, diagnose/preserve first and preflight
before the targeted replacement. Do not resubmit completed matching tasks.
Never test a cancellation/deletion guard against a live server.
Original incident history: [Claude source](references/claude-original.md).

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
