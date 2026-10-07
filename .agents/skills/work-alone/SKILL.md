---
name: work-alone
description: Autonomous-session mode — the user is away for hours and wants the multi-stage pipeline driven to completion without questions. Invoke when the user says they're unavailable / "continue with the stages" / "keep working while I'm gone". Defines what to decide alone vs park, watcher discipline, and periodic safe-compact checkpoints.
---

# work-alone

Read [project context](../project-context.md). State the authorized stages,
quantitative decision gates and material choices that would need the user.
Continue routine reversible work and already-authorized launches without
repeated permission requests. A failed scientific gate blocks dependent
solves, but independent diagnostics/documentation can continue.

Use an existing durable watcher or afterok chain where appropriate. A shell
watcher only collects data; it does not automatically wake Codex. Only claim
future model follow-up when an actual supported automation/wakeup is configured.
Remain active or use an authorized supported follow-up mechanism; do not end
merely because the next stage is waiting. The historical Monitor/run_in_background
APIs and Fable/Opus routing are not assumed to exist.

Budget roughly3-6 SSH connections/hour/host, including manual checks. Fold
queue/accounting/log/quota/seats checks. Change keys contain decision-relevant
states/counts/errors, not timestamps. Distinguish SSH_UNREACHABLE from
SLURM_CTL_DOWN and real job completion. Stop automated retries on authentication
refusal; jobs can continue while visibility is unavailable.

Inspect new-code jobs early. Diagnose task failures from logs and preserved
results before any targeted recovery. No duplicate running jobs, no blanket
retries, cancellation, cleanup, destructive Git operations or unapproved physics changes.
Focused local commits are explicitly authorized in AGENTS.md; inspect the index,
stage reviewed explicit paths and preserve unrelated edits. No push is implied.
Repeated failures or ambiguous scientific choices require a clear blocker
report, not a bypass of the gate. Preserve pending work and known limits.

Checkpoint significant discoveries and milestones with safe-compact. Keep the
current report focused on jobs, measured evidence, decisions and next steps.
At validated milestones, maintain relevant rules/skills with confirmed reusable
findings as the user requested. Store volatile job state in dated records.
Coordinate authorized independent jobs across both clusters using one shared
license budget; parallelism does not remove scientific dependencies.
For long model-assisted work, follow the project's budget-aware rules in
AGENTS.md: check actual limits at stage boundaries, use focused Sol tasks,
and preserve a restart packet before capacity becomes critical. A supported
heartbeat is a wakeup mechanism, not an exemption from usage limits. Use the
installed ask-claude helper only within its verified text-only boundaries.
The retained `campaign_monitor_template.sh` is a historical template, not an
executable default; inspect/update its paths and failure semantics before use.
[Original history](../../../.claude/skills/work-alone/SKILL.md).
