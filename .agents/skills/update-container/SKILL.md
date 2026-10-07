---
name: update-container
description: Update the Lumerical version inside the Athena container (lumerical-2026R1.sif) without moving gigabytes over the slow VPN — on-Athena sandbox surgery, verified by checksums and a physics canary. Use when a new Lumerical patch/release should go into the Athena container, or to bring Athena in sync with IGUM's native version.
---

# update-container

Read [project context](../project-context.md) and the current cluster runbook.
Container replacement needs explicit scope/authorization and is unrelated to
ordinary calibration recovery. Do not execute an old build script verbatim.

Inventory the current engine/API build, active jobs, source installer, hashes,
quota and exact affected paths. Prefer verified on-cluster staging over moving
a multi-GB image across the VPN. Preserve every old image and installation.
Use trusted existing SSH host verification; never remove a host key merely
because a historical recipe says to refresh it.

Build a new versioned image on a scheduled CPU node using fresh staging paths.
Inspect the existing container recipe and job environment, including the
engine's hardcoded licensing-client path, scientific libraries and permissions.
Validate installer/tree checksums and inner version/API imports. No --force
against an existing valuable path, no automatic stage/sandbox cleanup.

Only after dependent jobs are drained and the authorized replacement is ready,
preserve the old image under a noncolliding path and publish the new one.
A version change justifies one matching-numerics stored-control canary. Define
its tolerance from actual repeatability; do not promise universal bitwise
agreement. Treat failures as a blocked deployment and preserve diagnostics.

IGUM uses a native user-owned installation, not Athena's container. Update its
job/config references only if that cluster was included in the request, then
validate its actual environment and compute path separately. Do not infer
permission to update both clusters from a request about one.
Detailed historical staging/build recipes:
[Claude source](references/claude-original.md).

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
