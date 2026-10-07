---
name: lock-target
description: Tune a device to hit EXACT target values (spatial mode width, peak transmission / -3 dB point, resonance wavelength, Q via a loss knob) using the knob table + linearizing-coordinate ladder method. Use whenever the user asks for "exactly X µm mode", "peak T at -Y dB", "land the resonance at λ", "max Q given a T budget", or any find-parameter-for-target request — for any device or polarization.
---

# lock-target

Read [project context](../project-context.md). Use measured knobs and matched
numerics to hit a stated physical target; do not change the user's fixed
geometry or tolerances silently. First check whether predict-q3db already has
a calibrated family that answers the question with one confirmation.

Typical solve order: corrugation for spatial width, N for transmission, pitch
for resonance trim. Q at fixed T needs a separate loss/design knob and cannot
be treated as an independent free target without one. These relationships are
local models; confirm their validity in the actual device family.

Reuse in-study controls. Bracket with the smallest authorized ladder, fit in
an appropriate coordinate, then confirm once. Existing in-band points can
serve as the confirmation. A miss joins the fit; do not restart the ladder.
Use Q_c-based extrapolation from the current predictor for long-length changes;
the older broad ln(T)-versus-N extrapolation recipe was superseded.

Check finite defect resonance and sample spacing: report Q only when at least
10 points resolve the linewidth. Keep mesher, engine, box, monitors and cluster
consistent for comparisons. Retain current auto-shutoff settings unless the
study explicitly tests convergence. Do not fill license capacity automatically
or add a redundant control just because a historical recipe suggested it.
Report target, measured value, residual, parameter and provenance; distinguish
measured/derived/expected. Original knob history and its dated validity limits:
[Claude source](references/claude-original.md).

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
