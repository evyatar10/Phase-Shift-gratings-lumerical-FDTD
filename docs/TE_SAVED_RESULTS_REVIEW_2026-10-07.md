# Saved TE results and Claude-claim audit — 2026-10-07

Independent offline review for the retained TE, N98/free60, bare inverse-design
lane. No server access, upload, cancellation, CAD session or FDTD solve occurred.
Existing source/results/history were read only. The newly created evidence is
[te_saved_results_audit.json](readiness_evidence_20261007/te_saved_results_audit.json).
It includes full selected vectors, source/result SHA256 hashes, bounds checks,
spectral reconstructions and the preserved scratch-script location.

**Verdict:** the latest stored S1 vector is a feasible, unconverged PVA restart
candidate. S2's unchanged seed has higher recorded modal transmission than this
S1 vector, under a different retained width band. Neither comparison establishes
an equal-width optimum, nor a validated new restart or current peak3 gradient.
Do not initialize the TE lane from old 191-slot TM/comb campaign vectors.

## Reproducible eligible inputs

MEASURED below means read directly from the named saved JSONL this session;
these are logged FDTD measurements, not independently re-extracted spatial
profiles. DERIVED means algebra or reconstruction using those saved arrays.

| Input | Modal sampled peak T | Spatial envelope FWHM (µm) | Peak λ (nm) | Recorded FOM | FOM definition |
|---|---:|---:|---:|---:|---|
| S1 original production-window anchor | 0.905321818084 | 19.120888652048 | 1560.899984430 | 0.674197562543 | old softmax |
| S1 toy seed, copied campaign line 1 | 0.905444180777 | 19.119801343548 | 1560.883985445 | 0.859961895300 | T_out peak3 |
| S1 toy final, copied campaign line 4 | 0.934602544703 | 19.410602967118 | 1560.963984624 | 0.889772414031 | T_out peak3 |
| S1 latest campaign, line 8 | 0.952760309296 | 19.385006185682 | 1561.043989595 | 0.908596630955 | T_out peak3 |
| S2 original production-window anchor | 0.973087733005 | 19.635976884298 | 1560.407361198 | 0.728316205034 | old softmax |

Sources:
- `results_from_athena/campaign_te_s1/lumopt2_te_s1_v3_evals.jsonl`.
- `results_from_athena/validate_te/lumopt2_te_s1_anchor_evals.jsonl`.
- `results_from_athena/validate_te/lumopt2_te_s2_anchor_evals.jsonl`.
- `results_from_athena/validate_te/lumopt2_te_s1_v3_toy_evals.jsonl`.

DERIVED using current `param_bounds`, seed construction and width eligibility:
all eight S1 campaign rows satisfy their own parameter bounds and physical S1
band, 18.73858–19.50342 µm. The last row maximizes the recorded objective among
those rows; the original optstate has an empty `ineligible` list. Calling the
live `_best_from_log` independently returns exactly the same 296-vector and λ.
Its byte identity is SHA256 of little-endian float64 vector bytes:
`66e7c00468886c448467ee8432d878eb36ac2451bf2d508608b1dea707be1dd8`.
Use the vector in the evidence JSON; `eval=3` alone is ambiguous because copied
toy and restarted campaign evaluations share numbers.

The stored S1 and S2 anchor vectors exactly equal their respective current seed
constructors, including inert slots. Both are TE/N98/free60/bare. S1 uses pitch
500 nm, corrugation bounds 100–450 nm, mean width 775–825 nm, cavity width
750–1150 nm. S2 uses pitch 491.06 nm, corrugation bounds 0–750 nm, mean width
900–1100 nm, cavity width 850–1250 nm. Both current specs use PVA; S1 scans
10 nm/501 points, S2 2 nm/501 points. The physical S2 width band is
19.24328–20.02872 µm. Bands are retained as given; no common target was imposed.

The latest S1 vector is a deformation of the original uniform seed: first
three full corrugations 153.071748, 180.088996, 221.949125 nm; cavity width
826.171618 nm. Its largest shift is 0.263014 nm. DERIVED shift counts are 16
within 1e-8 nm of zero, and 57 below 0.01 nm. Claude's shorthand “57 at the
zero bound” is not an exact active-set count.

## Width and improvement claims

DERIVED from the copied toy seed and last S1 row: modal transmission increases
by 0.04731613 absolute, T_out peak3 increases by 0.04863474, and spatial FWHM
increases by 0.26520484 µm. Relative to the 19.121 µm configured seed anchor,
the final width is +1.380713%. Claude's “fixed-width gain” is therefore valid
only in the sense of remaining inside the retained ±2% band. The optimization
does use some permitted widening. No exact-width production retrim is present.

The last S1 row's loaded Q is 1549.829797, only +0.73986% versus the copied toy
seed. Its `q_i=64831.288323` is the two-port-model estimate
`Q_L/(1-sqrt(T_modal))`, not an independent radiation-Q measurement. Likewise
`loss=0.044801435309` is `1-T_modal-R_modal`; it cannot on its own identify
radiation, absorption, coupling asymmetry or finite-domain leakage.

MEASURED width authority in the present engine is the y-integrated sum of all
three electric-field intensities, cropped to the grating, then cubic envelope
through standing-wave peaks and FWHM relative to the profile floor. Relevant
functions are `_line_from_res` and `fwhm_env_of_line` in the engine, which call
the production `sim_helpers` envelope/FWHM routines. Sigma, raw-line FWHM,
`softw_um`, `softw_adj_um` and `fwhm_hat_um` are not interchangeable with this
observable. Pre-August-18 inverse-design width claims are not valid inputs.

UNAVAILABLE locally: the S1 final raw spectrum and spatial profile, and anchor
field arrays required to independently repeat these spatial measurements.
The campaign directory contains JSONL, optstate, four gradient NPZs and an
editable design figure/PNG. A geometry figure cannot reconstruct field width.
The directory has no saved per-evaluation field-profile NPZs, so its historical
MAC values can be read but not independently recomputed from this local bundle.

## Objective normalization reproduced from saved spectra

The installed `lumopt2/fom/simulation_results.py::PortResults` explicitly uses
`abs(T_out)`. The callback's `t_pk` is modal `abs(S21)**2`. The same stored port
arrays yield the following DERIVED values using the current engine's resonance
selector and peak3 implementation:

| Stored spectrum | Modal peak3 | T_out peak3 | Modal sampled peak λ (nm) | T_out sampled peak λ (nm) |
|---|---:|---:|---:|---:|
| S1 toy final | 0.934610016846 | 0.889772414031 | 1560.963984624 | 1560.983985700 |
| S2 recentered calibration point | 0.971810214976 | 0.925748506956 | 1560.878218964 | 1560.870219008 |

Sources: `results_from_athena/recover_te_s2_20261006/s1_v3_toy_ports.npz` and
`s2_gate_ports.npz`. The reconstructed S1 T_out objective differs from its
JSONL FOM by about 1e-16. Reconstructing modal power as
`abs(a2/a1)**2 * abs(N2/N1)` matches exported S power to at most 6.7e-16 across
all three local spectra, including `s2_p0_neg_ports.npz`.

The spectrum shapes and selected peak wavelengths differ. Even over samples
where modal T exceeds half its peak, T_out/modal ranges approximately
0.8501–0.9574 for S1 and 0.9158–1.0548 for S2. A fixed +0.045 correction, a
fixed multiplicative factor, or `T_out/T_in` is not an exact conversion.
Claude's historical “near-constant offset” observation over early S1 steps was
an empirical local observation; it cannot define the objective or gradients.

The S2 recentered gate vector includes calibration perturbations and is not
the unchanged S2 seed. Its peak3 T_out is not a measured peak3 objective for
the original S2 anchor. The original S2 anchor's 0.728316 FOM is softmax and
must not be ranked against S1's 0.908597 peak3 value. S2's modal anchor T is
0.02032742 above S1's last modal T, but the widths, geometries and sampling
differ. An equal-width optimization claim is unsupported.

## Diagnostics and calibration limits

The latest logged S1 width-surrogate residual is +0.052445 µm, beyond the
engine's 0.05 µm warning threshold. Its width twin is at 1561.003986 nm,
lagging the recorded modal peak by 0.0400036 nm. The measured physical width
still meets its band. This warning requires investigation, not automatic
rejection of the stored field width.

DERIVED from the three campaign transitions and projection rows:

| Transition | measured/predicted FOM gain | moving-width prediction error (µm) | λ prediction error (nm) |
|---|---:|---:|---:|
| 0→1 | 0.982931 | -0.004479 | +0.020610 |
| 1→2 | 0.951232 | -0.011703 | -0.015884 |
| 2→3 | 0.842508 | -0.018414 | +0.006810 |

Width prediction uses `v3_pred_rows[0]`, including the moving-resonance term;
`dw_pred` alone is not the correct comparison. Transition 1 narrowly exceeds
a 0.020 nm wavelength-error criterion, on a roughly 0.020 nm sampled-peak
grid. It must not be rounded into an unconditional pass. Stored MACs exceed
0.99998, but their missing local profiles limit independent verification.
All five arrays in each of the four gradient NPZs are finite 296-vectors and
their parameter vectors exactly match corresponding logged points. Finiteness
does not verify gradient accuracy.

The original S1 toy `_best.json` reports FOM 0.877210070719 and its earlier
vector, while the copied log includes the later physically compliant point
0.889772414031. The toy projection log explicitly records a width-margin
rejection for its final trial. Later acceptance-policy changes and the warm
start are documented in the historical continuation. Do not choose the
largest FOM in `_best.json` filenames or infer decisions solely from eval IDs;
classification/persistence must be handled by the reviewed restart candidate.

MEASURED historical qualifications read from current source comments and
Claude's `project_te_inverse_design_lane.md`, not a new calibration refit:
- S1 port factor `(0.945335,+0.117012)` had an in-sample exact-LSQ fit with
  component residuals at most about 5%, but held-out `shift_30` was -16.5%.
  It was calibrated on softmax; it is not a strict held-out ≤10% certificate
  for all components of peak3.
- S1 field factor `(0.966720,+0.036560)` was reported to fit its three sampled
  components within 0.2%. Three points for two coefficients give limited
  generalization evidence.
- S2 field factor `(0.983623,-0.088013)` has reported vector residual 1.25%,
  but `shift_1` +23.8% relative component residual at a weak sensitivity. A
  vector norm pass is not a universal componentwise ≤10% pass.
- S2 port factor remains `None` in the retained source. Its current recovery
  needs its actual peak3 per-class, signs, rank and held-out criteria. This
  audit made no live-status claim about the calibration jobs.

The earlier S1 toy final and resumed campaign initial row share an identical
vector but have FOMs 0.889772414031 and 0.889646848723. Their twin wavelengths
also differ. The difference is smaller than current S1 slack 5e-4, but this
is not a controlled identical-numerics noise experiment; do not substitute it
for the dedicated noise floor.

## Older uniform, nonuniform and overshoot evidence

The current TE S1 starts uniform but its saved optimized inner teeth become
nonuniform. Current S2 is Itai's native-20-µm reoptimized overshoot seed,
imported from `runners/sweeps/itai_hh_nt60w20.py`, with its first 60 tooth
entries free and its 61st already equal to bulk. It is not the older
15.1-µm HH table multiplied by 0.562.

MEASURED historical CSV `results_from_igum/itai_hh_nt60w20_summary.csv` reports
native S2 N98: T=0.97498, envelope FWHM=19.633 µm, QL=7680, λ=1559.8597 nm;
N130: T=0.92377, width=19.713 µm, QL=45058. These are the conformal historical
family. The current PVA anchor agrees closely in width and T but differs in λ.
The CSV lacks raw arrays and full numerical provenance, so it is not new
PVA/mesh/domain confirmation for a proposed optimized vector. N130 is outside
the retained N98 design and its Q_i is model-derived.

Old `campaign_v2_uniform.SPEC` and `campaign_v2_projection.SPEC` are TM,
free25/N100, `bare=False`, 191-slot vectors, with anchor 18.346 µm. Their
local logs under `results_from_athena/v2_gpu_gradient_pause/jsonl/` show why
ranking by T alone is unsafe: uniform s5's highest modal row has T=0.959116
at W=20.373796 µm, outside its own band; nonuniform v2proj_s2's highest modal
row has T=0.967289 at W=19.065778 µm, also outside that old band's upper
18.71292 µm. A legacy uniform trial reaches T=0.955767 at W=32.267893 µm.
Other old rows may satisfy their old bands, but none is an eligible TE vector.
Results predating the corrected profile/mesh convention need separate
re-extraction; the mere presence of a scalar `fwhm_env_um` is insufficient.

## Criteria still unestablished by this review

1. Physical optimizer smoke after the mixed-normalization wavelength fix and
   persistence integration, with current calibration, immutable profiles and
   committed decisions. Offline regression evidence does not establish this.
2. Final S2 peak3 port calibration satisfying its existing per-class and
   held-out gates; transfer from S1 cannot fill it.
3. Independent final S1 field-width/MAC extraction from saved profile arrays,
   or exact-width production retrim. The local bundle has scalar evidence only.
4. Convergence to an optimum, a common-width S1/S2 basin comparison, and
   calibrated modal-objective derivatives if that objective is chosen later.
5. Mesh/conformal/domain/PML convergence of a newly optimized vector and an
   independent radiation-loss/Q decomposition. Prior seed/domain studies
   are useful context, not validation of a new final shape.

No existing data or source was altered. The scratch audit exited 0; all
computed comparisons are captured with hashes and vectors in the evidence
JSON. Current authorization and continuation documents control future work.
