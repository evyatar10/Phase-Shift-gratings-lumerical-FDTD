---
name: athena-preflight
description: Pre-dispatch safety check for Athena — license seats, home-disk quota, and the job queue — before submitting an FDTD run. Use before deploying/dispatching to Athena, or when a job hangs / finishes implausibly fast / returns empty results.
---

# athena-preflight

Read [project context](../project-context.md) and the current cluster-ops skill.
Run a bounded preflight before an authorized Athena dispatch or when a solve
finishes suspiciously quickly. Fold checks into as few SSH calls as practical.

- Queue/accounting: identify active jobs, dependencies, resources and shared
  source/manifest consumers. An empty failed scheduler query is not an idle queue.
- Quota: use `quota -s`; the current home soft limit is 300G, hard 330G.
  Reserve space for retained FSP/HDF5, failed attempts and new results. No cleanup.
- Seats: query the shared `lum_fdtd_solve` pool via IGUM's native lmutil recipe
  in the cluster runbook. Open license ports prove reachability, not available
  seats. Athena's historical DNS-related lmstat -96 alone is not an outage.
  Include running and queued commitments on BOTH Athena and IGUM; queued
  tasks may start together. Recheck before release and preserve headroom.
  An unavailable seat query is not evidence of free seats.
- Hardware: verify QOS memory/time limits and actual allocation. Current TE
  field smoke needs 256G host memory; do not lower it to fit the 2h_2g lane.
  Check actual solver thread count against allocated CPUs. A100 normally uses
  seven standard GPU seats; L40S nine. Recheck current availability.
  QOS GPU limits are distinct from shared-seat budgets; query the applicable
  per-user/account constraints. See cluster-ops for dated verified limits.

Before a fan-out, hold at HIGH shared usage (>=35/50) and do not dispatch at
CRITICAL (>=45/50); these are project operating bands, not seat guarantees.
Report concrete go/no-go evidence and the intended resource request. Diagnose
resource failures without deleting HDF5 or cancelling unrelated jobs.
Original incidents: [Claude source](references/claude-original.md).

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
