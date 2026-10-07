---
name: safe-compact
description: Checkpoint the session so compaction/handoff loses nothing — snapshot server job state, persist program state + next steps to memory, refresh todos. Invoke when the user says "make it safe to compact", before/after long autonomous stretches, after any dispatch or verdict, and proactively whenever context is getting long. Do NOT wait to be asked.
---

# safe-compact

Read [project context](../project-context.md). Checkpoint so a fresh session
can resume from files alone; compaction is not a reason to abandon the task.

Save a dated repository continuation with current authorization, fixed design
choices, completed work, open issues, exact next actions and evidence paths.
Include actual job/task IDs, dependencies, expected outputs, timestamps,
watcher PID/command/liveness, frozen hashes and isolated candidate locations.
Label measured/derived/expected values and link their original artifacts.
Use a recent verified watcher snapshot or one bounded folded status call;
state explicitly if live access is unavailable rather than inventing status.

Preserve source/result history and note uncommitted changes. Stage or commit
only when the user has authorized it; use focused reviewed paths and preserve
unrelated edits. Save patches and manifests when important work exists only in
scratch space. Update AGENTS.md's current-state pointer; keep historical docs.
Claude memory can be read through the explicit path in project-context.md,
but the handoff must not depend on Claude-specific memory or TodoWrite tools.

Check the original task against the persisted next steps, including user
corrections and decisions. Continue the authorized workflow after checkpointing.
Report the checkpoint path briefly; do not present checkpointing as completion.
Original checklist: [Claude source](references/claude-original.md).

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
