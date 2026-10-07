---
name: athena-status
description: One-shot answer to "how is the run doing?" — Athena queue state, latest job-log tail, and freshly produced result files. Use whenever the user asks about job progress, whether a run finished, or whether results exist ("how doing", "is it running", "did we get results", "now?").
---

# athena-status

Read [project context](../project-context.md). Use one bounded, read-only SSH
round trip for the named study's queue, accounting, relevant job-log tail and
expected result/checkpoint files. Use the current cluster runbook for host,
base path and quoting. Do not pick an unrelated newest log.

Check command exit codes and stderr. Distinguish SSH_UNREACHABLE,
SLURM_CTL_DOWN, pending, running, failed and completed. Completion needs
successful accounting and expected outputs, followed by scientific inspection;
queue drain alone is insufficient. DependencyNeverSatisfied is a blocked chain,
not normal progress. A source-initialization phase can be CPU-heavy with no GPU
activity; inspect phase/thread count before declaring a license hang.

For the S2 recovery, reuse the existing read-only watcher's snapshots under
`results_from_athena/recover_te_s2_20261006/watch/` before opening another SSH
connection. Confirm snapshot age and watcher health. Keep total monitoring
within roughly 3-6 connections/hour/host; stop retrying on authentication refusal.

Keep accounting-database failures separate from SSH and scheduler-controller
failures. For example, a live `squeue` can succeed while `sacct` cannot reach
slurmdbd. Report which evidence is unavailable; do not infer job completion or
VPN loss from the accounting failure, and do not submit a replacement job.

Report job/task IDs, running/pending/completed counts, latest meaningful solver
phase, new valid results versus expected, and any explicit access failure.
If result inspection/fetching is already authorized, continue it automatically.
Original history: [Claude source](../../../.claude/skills/athena-status/SKILL.md).
