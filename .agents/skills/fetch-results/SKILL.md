---
name: fetch-results
description: Download finished FDTD results from Athena into results_from_athena/, render the study's MATLAB plot headlessly, and reply with full local file paths. Use when the user asks to download results, get/see a plot or figure from a finished run, or says "give me the graph/image of ...".
---

# fetch-results

Read [project context](../project-context.md). Identify the exact study/job and
expected outputs before transfer. Prefer a targeted download over fetching
multi-GB fields for a scalar question. Keep FSP and companion HDF5 on the server.

Download into `results_from_athena/<study>/` or the IGUM counterpart. Preserve
existing conflicting/partial local files; use a new staging path, verify hash
or size plus successful parsing, then publish without erasing prior evidence.
The deploy `--results-no-fsp` mode exists but can be broader than the request.
Run check-result before making scientific plots.

Use the existing study plotter. Discover the installed MATLAB executable,
then run `-batch` with the study/result path explicitly configured. Capture the
native exit code and output with subprocess; PowerShell redirection of benign
stderr can otherwise report a misleading wrapper failure. Inspect the rendered
image. Keep generated figures beside their data, preserve originals, and do not
stage them in Git. Use Interpreter=none for filename titles.

Project field-map labels are XZ='Top view', XY='Side view', propagation x
horizontal. Titles should identify physical dimensions, resonance and peak T;
keep legends compact and do not add an unexplained 'zoomed' label.
Deliver clickable full local paths to figures and results, with validation limits.
Original plotting notes: [Claude source](references/claude-original.md).

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
