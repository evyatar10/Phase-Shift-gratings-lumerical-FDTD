# S2 calibration failure and bounded diagnosis

## Measured smaller-step and provenance results, 07:54 Israel

Athena170906_1/2 completed exit0. Exact source/center identities and wavelength
grids matched; exported spectra reproduced saved peak scores. corr30 h0.5nm
peak3(Tout) derivative=3.667260923146e-5/nm versus h2nm1.874864880122e-5/nm
(95.6% increase). Even response is-1.20475e-6 versus+1.80637e-4. Using the
old center changes that small even response to-7.29632e-7, without changing
the central derivative. The h0.5 peak3 spectral-Jacobian contraction gives
3.667761171813e-5/nm, differing by0.01364%; all peak indices are252.
Modal derivatives also depend on step:4.408675869771e-5/nm versus
2.144186733535e-5/nm. Two sizes do not establish convergence or its order.

CPU-only no-solve probe170921_0 completed in28seconds with all critical
evidence available. It compared723 mapped properties in each original
forward/adjoint layout: maximum discrepancy1.6941e-21m from te_point.
Saved Re/Im forward/adjoint coordinates match exactly. Generated seed mesh
and nominal index axes differ from saved coordinates by at most6.7763e-21m,
which is floating-point scale, not a meaningful displaced grid. Original FSP
hashes and companion HDF5 stats are unchanged; full HDF5 hashes were not taken.
Field shapes are inferred from coordinates, not full E/H extraction.

The observed adjoint Port2 shift-49.106nm is intentional: inspected vendor
port_fom.py adds sign*local resolution and a numerical-propagation phase.
Backward sign=-1 gives a1.25-step phase formula. Both saved phases are
23.2815894deg; the numerical wavevector needed to independently reconstruct
that phase was not exported, so formula agreement is not a fresh phase test.
These results do not support gross saved geometry/grid mismatch as the cause.
Next bounded question is step convergence and numerical dEps step dependence;
full calibration remains failed. See `s2_provenance_review.json` and
`s2_fdstep_two_sizes_comparison.json` under readiness evidence.

The coordinate API uses symmetry-unfolded getdata option2, checked against
[official Ansys documentation](https://optics.ansys.com/hc/en-us/articles/360034409834-getdata-Script-command).
No new field solve or old-project save was used for this provenance check.

## Interpretation plan for the pending four-step and dEps checks

Compare h = 2, 1, 0.5 and 0.25 nm without assuming smaller is automatically
better. Report successive derivative differences, signs, both nominal centers,
and peak3 spectral-Jacobian projections. A smooth halving trend on corr30 can
justify a chosen step for remaining class probes, but cannot establish the
full calibration. Nonmonotonic changes require further diagnosis; they do not
alone identify solver noise, a mesh defect or non-differentiable geometry.

The new stored-field probe must first reproduce its original h = 1 nm raw Re
contraction, then Im, before its smaller dEps steps are interpreted. It uses
original-forward Jacobians, whereas new forward differences use the repeated
center for even terms. Preserve that distinction. If the new extraction does
not reproduce the original contraction, diagnose the extraction/subset path
before treating a discrepancy as physics. No coefficient is fitted by this
probe. The six-class held-out gate and width/optimizer hardware gates remain.

## Additional numerical control to record, 08:08 Israel

Official [Ansys mesh-refinement guidance](https://optics.ansys.com/hc/en-us/articles/360034382614-Selecting-the-best-mesh-refinement-option-in-the-FDTD-simulation-object)
recommends PVA for inverse design and distinguishes its `meshing refinement`
accuracy setting from the `mesh refinement` method. It documents default5
and maximum12. Our source sets the PVA method but does not explicitly set
that accuracy integer; the completed provenance probe did not extract it.
Therefore its actual value is currently unmeasured, not established as5.
The prepared no-solve dEps probe will record it without changing it. This is
a candidate control if finite-step sensitivity persists, not a diagnosed
cause or permission to substitute a different numerical family silently.

## New center evidence and next diagnostic, 06:43 Israel

The nominal repeat170901_0 completed exit0. Peak3(Tout) differs by+4.75118e-7
from the original; full Tout spectrum max difference1.22018e-6 and
RMS2.00109e-7. Wavelength/frequency grids match exactly and the parameter
vector/source identities were checked. This observed center difference is
only0.634% of the original corr30 +/-2nm central numerator; it is too small
to explain the large one-sided asymmetry by this measured center offset alone.
One repeat cannot bound run-to-run noise, eliminate all numerical effects or
validate adjoint-field provenance. The failed calibration is unchanged.

After explicit review, the exact new checkpoint SHA was recorded in the
remote center marker; serial array170906 now evaluates corr30+/-0.5nm.
The purpose is to measure step dependence and odd/even response, not to tune
one coefficient until a selected probe passes. Source files are frozen.
See TOP continuation and saved scalar/full-spectrum comparison for hashes,
resources, evidence paths and next actions. All original results are retained.

### Smaller-step interpretation prepared while the jobs run

For scalar f and step h, use odd O=(f+ - f-)/2, even
E=(f+ + f- - 2f0)/2, D=O/h and C=2E/h^2. Report even terms using both
old and repeated centers; the interval is center sensitivity, not a confidence
interval. At h2nm, corr30 O=3.74973e-5 and E=1.80637e-4; even response is
about4.82 times odd response. For a locally quadratic response, O shrinks
fourfold and E sixteenfold at h0.5nm. Deviations can also arise from smooth
higher derivatives; they do not alone diagnose a mesh kink or source error.

The conditional Richardson expression is [16D(0.5)-D(2)]/15. Two sizes do
not establish its assumed h^2 error law. Compare moving peak3 FD with both
the fixed center sample FD and the exact center peak3 Jacobian applied to
the spectral FD. Finite-step disagreement does not itself establish different
infinitesimal observables or a defective adjoint. Inspect the actual peak
indices, concavity and spectra before interpreting any discrepancy.

New exact Opus5.5 text-only review succeeded and is saved in
`readiness_evidence_20261007/claude_s2_center_repeat_review.json`. Its odd/even
and Richardson formulas were checked. Corrections: it wrongly imported the
shift30 wavelength slope into corr30; both stored corr30 endpoints have the
same sampled modal lambda1560.8782189638007nm as center. Its claimed0.045nm
corr30 peak displacement and derived0.18 fixed-wavelength change are unsupported.
Its noise-sigma expressions assume independent equal-variance noise, which
one repeat cannot estimate; no sigma/confidence bound is adopted. Shape-fit
residuals above the repeat difference or non-h^2 scaling do not prove mesh
error, since shape deformation and smooth higher-order response also contribute.
Do not prescribe another solve solely from those heuristic thresholds.

Prepared offline calculator `readiness_evidence_20261007/analyze_s2_fdstep.py`;
analytic quartic/Richardson, off-grid parabola and invalid-spectrum checks
passed exit0. It refuses absent endpoints and exclusively creates output.
After fetching exact task01/02 JSON and small NPZ spectra under the same local
evidence directory, run it with `--evidence-dir docs/readiness_evidence_20261007`,
`--old-spectra C:/Users/evyat/AppData/Local/Temp/s2_peak_failure_spectra_ppyzxc75/stored_port_spectra.zip`,
and a NEW `--output` path. This calculator is not a physics gate.

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
