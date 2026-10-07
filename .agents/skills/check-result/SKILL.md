---
name: check-result
description: Load a result_*.mat FDTD result and report transmission, resonance wavelength, Q, and spatial mode width correctly — with the in-window / dead-device sanity check. Use when asked to inspect, summarize, or sanity-check a simulation result .mat file.
---

# check-result

Read [project context](../project-context.md). Resolve the requested MAT file
or study from the user's context. Read without changing the source; use scipy
loadmat, or an HDF5 reader for v7.3 after checking the actual file format.

Report stored defect resonance, transmission at that resonance,
Q_L=lambda_res/abs(spectral_fwhm), and spatial envelope FWHM when relevant.
Never replace the defect resonance with global argmax(T) in the passband.
Check finite in-window resonance, a resolved linewidth, meaningful transmission
and successful field extraction before reporting derived observables.

Spatial width uses the production y-integrated envelope convention. Sigma,
raw-line FWHM and spectral FWHM are different quantities. Pre-August18,2026
inverse-design spatial widths are invalid unless re-extracted correctly.
Q_i=Q_L/(1-sqrt(T)) is model-derived; 1-T is not radiation loss.
For v3 inverse design, label modal abs(S)^2 separately from objective abs(T_out)
peak3. Do not apply a fixed conversion factor between them.

For multiple files provide a compact comparison with geometry/numerics and
sanity failures. Include clickable absolute paths. Preserve partial or suspect
downloads before fetching verified replacements under fresh names.
Original field conventions: [Claude source](references/claude-original.md).

The bundled runbook is historical evidence. Current user instructions, AGENTS.md, this Codex entrypoint and live code take precedence. Interpret repository paths from the repository root; historical memory references remain external and are not bundled.
