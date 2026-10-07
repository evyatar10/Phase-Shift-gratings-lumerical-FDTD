---
name: lumopt2-design
description: "Run, debug, resume, validate, and extend the current TE/TM lumopt2 inverse-design program, including seed geometry, objective normalization, adjoint gradients, monitors, campaign persistence and hardware gates."
---

# lumopt2-design

Read [project context](../project-context.md), the current dated continuation,
`runners/lumopt2_design/THEORY.md` section13, and the installed
[pi-shift-inverse-design skill](C:/Users/evyat/.codex/skills/pi-shift-inverse-design/SKILL.md).

Current TE v3 contract: N98/side, free60/side, bare; layout(60) has296 slots
and181 active variables. S1 uniform and S2 apodized have separate measured
width anchors/bands. Peak3 consumes abs(T_out); modal abs(S)^2 is logged
separately. Width is the production y-integrated envelope FWHM, steered by a
bounded QP with wavelength trust. Width-row reuse is off. Do not transplant
old TM sigma/kappa penalties,191-slot slices,softmax calibration or task IDs.

Read `campaign_te_s1.py`, `campaign_te_s2.py`, `validate_te.py` and the actual
engine path involved. Check live constants; S2's missing calibration must never
be bypassed with S1 coefficients. Recovery checkpoints carry exact source and
spec identities. Freeze their dependencies until queued recovery tasks finish.
A port calibration requires signs, per-class and held-out errors at the
specified gates; field calibration limitations remain visible independently.

Validation: offline gates; hidden CAD for scene/source/monitor changes;
authorized bounded hardware smoke; then device physics validation. Passing a
smoke does not authorize a full campaign or certify an optimum. For monitors,
check extents at parameter bounds, parity, port directions, frequency sampling
and the explicit single-frequency twin. Geometry containment is not convergence.

Check persistent evaluation identity, profile paths/MAC, interrupted JSONL,
accepted/rejected eligibility and atomic optstate before resume. Verify the
candidate's integration status in the dated continuation; tests in an isolated
copy do not mean production was updated. Keep reported best FOM paired with
its selected vector. Preserve expensive FSP/HDF5 and original logs.

For calibration failures, use the current inverse-design skill and dated
failure report: exact source/point identity, full spectra, center-repeat scale,
odd/even finite-step response, and all held-out errors. Reweighting cannot fix
an inconsistent two-coefficient model merely because one probe improves.
Keep peak3(Tout), modal peak and spatial/spectral FWHM distinct. Save raw
checkpoint bytes for SHA pins; a pretty-printed copy is not the same artifact.

The full historical method, vendor API pitfalls, old TM experiments and
platform lessons remain in [the original Claude runbook](references/claude-original.md).
Consult its relevant section as historical evidence, checking current API/code
and newer TE contracts before applying a recipe. It is not a live job queue.

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
