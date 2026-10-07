---
name: predict-q3db
description: Predict long pi-shift-grating observables (T, lambda, Q_L, spectral+spatial FWHM) and design Q3dB-style devices (any dB point, any mode width) from stored calibration — one confirmation run instead of a tuning ladder. Use when the user gives a new device result and asks what a longer/shorter device gives, where the -3 dB (or other) crossing is, what corrugation hits a width target, or asks to refit/validate the q3db predictive engine.
---

# predict-q3db

Read [project context](../project-context.md) and
`python_tools/Q3DB_PREDICTOR_HANDOFF.md` before using the predictor. CMT is
allowed for this separate predictive program; do not reintroduce it as the
lumopt2 optimizer's width wall.

Tools: `bragg_cmt.py` is the shape/envelope engine; `calibrate_q3db.py` fits
stored families and runs hold-out checks; `predict_q3db.py` exposes observe,
design, extend and compare modes. Inspect current top-level knobs and outputs
before editing them. Preserve existing calibration output before regenerating.
After model edits run its existing gates and hold-out matrix.

For a new row establish polarization, corrugation, pitch,N, transmission,
Q_L,resonance,width and numerics. Use a matching polarization/corrugation
family, or explicitly label transferred shape assumptions. Extending adds
uniform outer periods; it does not model arbitrary changes inside the core.
Tooth phase shifts are not modeled as a kappa change.

Extrapolate Q_c, not ln(T); fit Q_i saturation separately. Calibrate kappa from
width or Q_c growth, not a single ill-conditioned Q level. A corrugation
transform needs both rate and intercept and must reproduce the stored ladder.
Decompose a miss into Q_c/Q_i before changing the model. Keep conformal/PVA
families separate. Quote empirical hold-out bands separately from parameter
sensitivity. Old accuracy figures apply only to their measured device classes.

Pre-register prediction and pass bands before an authorized confirmation;
then compare measured results and update the relevant calibration. Scope and
worked examples: [Claude source](references/claude-original.md).

The user prioritizes reuse of existing data over long simulation ladders.
Before a new confirmation, inspect saved anchors/confirmation rows and test
the extrapolation offline. Separate parameter identification from agreement
at a fitted anchor: a single short-device row cannot independently establish
coupling growth, intrinsic-loss saturation and asymptotic spatial width.
Report sensitivity to borrowed family shape and the extrapolation distance.
Check TCMT coupling symmetry/background assumptions and whether T is measured
modal power, total flux or a fitted quantity before decomposing Q. Treat model
inaccuracy, ill-conditioned inference and solver nonconvergence separately.
An offline fit is not physical validation. Preserve original calibration CSV
and raw data; regenerate into a new destination or isolated copy during review.

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
