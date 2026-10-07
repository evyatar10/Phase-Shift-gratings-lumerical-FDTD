# S2 calibration failure and bounded diagnosis

All twelve finite-difference legs and both peak-objective gradient
reconstructions completed. Fit job170754_2 failed intentionally after saving
its result because the scientific accuracy gate failed. The queue was empty
at the subsequent preflight. The read-only recovery watcher exited as designed;
this is not a VPN failure. Preserve the complete recovery chain and its sources.

## Independently verified verdict

The current peak3(abs(T_out)) fit has rank2, matching gradient signs, and
condition number8539.56. A sign match alone is insufficient.

| Probe | Fitted relative error | Leave-one-out error |
|---|---:|---:|
| corr1 | +16.70% | +37.74% |
| corr30 | +162.11% | +170.99% |
| avg1 | -8.31% | -10.34% |
| shift1 | +13.34% | +13.37% |
| shift30 | +0.64% | +315.81% |
| cavity width | -31.95% | -39.64% |

The unchanged gate requires every class and every held-out probe within10%.
All six held-out probes fail. The old softmax calibration also fails. Install
neither fit; no optimizer smoke or campaign may use these coefficients.

Independent NumPy reconstruction matches the saved coefficient and held-out
vectors. A separate linear program minimizes max_i|a*Re_i/FD_i+b*Im_i/FD_i-1|.
Its optimal worst relative residual is58.1015%, verified by direct substitution.
Changing least-squares weights cannot make one global complex factor satisfy
10% for these stored vectors. This identifies an inconsistency in the current
calibration evidence/model, not its physical cause. Do not introduce arbitrary
per-parameter correction factors to hide it.

Evidence: `readiness_evidence_20261007/s2_peak_fit_independent_review.json` and
the immutable `results_from_athena/recover_te_s2_20261006/watch/` checkpoints.

## Stored-spectrum checks, no additional FDTD solve

One read-only transfer preserved baseline plus all twelve small port spectra
in `C:/Users/evyat/AppData/Local/Temp/s2_peak_failure_spectra_ppyzxc75/`.
The archive SHA256 is
`b996142caf4a66d73336ed2845a564626e6fbd9e0e9a695ee89bf498e7522204`.
Its `analyze.py` reproduced every leg's recorded peak3 score within1e-12 and
verified identical wavelength grids. This rules out a local score-copy error.

The baseline T_out peak is at sample252 and the modal peak at250. This is a
real observable distinction, not proof of geometry mismatch. Grid spacing is
about0.004nm and recorded modal linewidth about0.2023nm (roughly50.6 samples).
The T_out parabola is concave, vertex offset-0.1442 samples, with valid weights
[0.0825151,0.9791982,-0.0617133]. A grossly unresolved few-sample peak is not
supported by this evidence. The original geometry/field provenance still needs
an actual check; the mean of shifted FD spectra is not a nominal-center solve.

One-sided peak differences relative to the stored center are highly asymmetric
for several probes: corr30 forward+1.09067e-4 versus backward-7.15698e-5 pernm.
That could reflect curvature, numerical effects or a baseline mismatch;
asymmetry alone does not prove the central derivative is inaccurate. The
shift30 endpoints move the peak by about45 samples each (~0.18nm). At this
finite step, applying the baseline Jacobian to the endpoint spectral difference
is not an infinitesimal derivative. Neither fixed-center nor baseline-weighted
endpoint diagnostics rescue the global fit. A smaller-step/provenance check is
needed before selecting a physical correction.

Evidence: `readiness_evidence_20261007/s2_peak_spectral_review.json`.

## Independent consultation, with corrections

The restricted ask-claude helper succeeded with response metadata confirming
exactly `claude-opus-5-5`. Its bounded review is saved as
`readiness_evidence_20261007/claude_s2_failed_fit_review.json`.
Adopted: inspect stored spectra/field provenance and finite-step behavior before
new optimization; do not install failed coefficients. Rejected or corrected:

- It said four held-out probes fail; the measured vector shows all six fail.
- It claimed a fitted corr1 sign flip; all fitted signs actually match. Raw
  Re/Im signs are not the sign of the corrected gradient.
- Raw quadrature magnitudes and FD/Im ratios do not prove failure of a complex
  linear combination. The independent minimax computation supplies that test.
- It called shift30 motion0.09nm; that number is a slope in nm/nm. Each2nm leg
  moves about0.18nm, comparable to this linewidth, not proven many linewidths.
- A one-sample peak offset or leg-average discrepancy does not prove fields
  belong to the wrong geometry. No such provenance verdict is established yet.
- The port-source/Jacobian separation is supported by inspected vendor code;
  changing `fom.fct` is read dynamically by `get_jacobian`, not a cached closure.

## Next bounded diagnostic and independent software progress

Prepare a distinct forward-only diagnostic at the exact recovered te_point:
first repeat the nominal center with the FD recorder/mesh settings, then review
baseline agreement. Only after that review consider corr30 ±0.5nm, preserving
the existing ±2nm data. No optimizer or adjoint is part of this diagnostic.
The center repeat answers a reproducibility question that the old center with
different recorder settings and perturbed endpoints cannot settle. Do not call
the diagnostic a new calibration pass; the full gate remains unchanged.

At05:47 preflight: Athena user queue empty, quota225G/300G soft (330G hard),
12h_4g available in QOS listing, shared license server/daemon up with0/50 solve
seats used. A100 diagnostic remains one job at a time,256G host memory,8CPUs,
the established compatible QOS. Recheck if dispatch is delayed materially.

The separately tested restart/modal candidate was integrated locally after
the chain ended. All thirteen integrated offline suites pass after repairing
one standalone test import path. No failed coefficients or opt-in seed flags
were installed; nothing was uploaded by that integration. Read
`TE_INTEGRATION_2026-10-07.md` for exact integrated hashes and retained logs.
