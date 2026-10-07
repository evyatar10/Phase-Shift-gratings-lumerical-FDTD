# S2 temporal100 offline linewidth and normalization comparison

Preparation/analysis only. No repository, server, CAD, coefficients or simulation state changed. The earlier repeated nominal center and shift30 +/-0.0625 nm geometries match the corresponding100 ps checkpoints exactly; all501 frequencies and wavelengths match bitwise across both ports and all six files. New runner SHA555f3d13b42c494171e7658cc4263c588ce418e5d937da1497d690824b41d44a and artifact manifest SHA94b062900bd3765dd4a3ef78543dd6998378720ed429264e5aa7833339d5e919 were verified. All100 ps runs completed the full requested duration with early shutoff disabled. Earlier auto-stop durations are geometry dependent; they are not an equal-duration baseline.

| Nominal spectral FWHM | Earlier auto-stop |100 ps fixed duration |
|---|---:|---:|
| abs(T_out) |0.209867042 nm /25.8244415 GHz |0.209420626 nm /25.7695090 GHz |
| abs(S21)^2 |0.202344374 nm /24.8985589 GHz |0.201609961 nm /24.8081798 GHz |

Half-height means half the sampled peak value, with linear interpolation between the measured bracketing samples in frequency. Both nominal spectra have a unique local maximum above half-height and two in-window crossings. These are spectral linewidths, not spatial FWHM. No tail extrapolation or analytic line-shape fit was used.

At fixed bin252 (1560.870219008 nm), nominal Tout changes0.925724841 ->0.933257926 (+0.007533086), modal0.966142886 ->0.974254464 (+0.008111578), incoming Tin0.953428986 ->0.953187111 (-0.000241875), and R=Tout/modal0.958165561 ->0.957920092. Across the whole nominal grid, maximum absolute changes are0.00753309 Tout,0.00819568 modal and0.00700612 Tin; these are deterministic differences, not estimated noise.

Using available complex a and N, A1=abs(a1)^2*Re(N1), A2=abs(a2)^2*Re(N2). The observed identity modal=A2/A1 reproduces all six saved arrays within7.78e-16 maximum absolute error. Exact symmetric finite ratio decomposition at bin252 attributes modal change +0.007862507 to the outgoing numerator term and +0.000249071 to the inverse incoming normalization term. Likewise Tout=R*modal gives +0.007771239 modal term and -0.000238154 ratio term. Thus changes occur in both observables and the numerator, not only normalization. This is endpoint algebra; it neither establishes a temporal mechanism nor certifies a local derivative. At flank244, numerator and incoming-normalization terms partly cancel; at260 both modal terms are negative. The report JSON retains each term for all three matched geometries and bins244/252/260.

The shift30 h0.0625 central fixed-frequency derivatives at bins244/252/260 (per parameter nm) are Tout [0.372791,-0.00924693,-0.384266] earlier versus [0.403100,-0.00869644,-0.413837] at100 ps; modal [0.341089,-0.124823,-0.484512] versus [0.376139,-0.127862,-0.513870]. These measured changes remain relevant to the failed gradient gate; no convergence/noise floor/infinite-time limit is inferred.150 ps results remain future evidence.

Sourcepower is absent from these NPZs; raw P is not relabeled sourcepower. Mode-expansion Tout, modal S21 and Tin remain separate; no universal conversion or grid-dispersion correction is invented. The older nominal checkpoint lacks ports_sha256, so its raw checkpoint and NPZ are independently bound by the pinned review; newer five checkpoints additionally bind the NPZ SHA internally. Saved peak3 values are reproduced from each spectrum within1e-12 before comparison.

Reproduce from repository cwd:

```powershell
python C:/Users/evyat/AppData/Local/Temp/s2_temporal100_width_03666eafc1df/test_analysis.py
python C:/Users/evyat/AppData/Local/Temp/s2_temporal100_width_03666eafc1df/analyze.py --evidence docs/readiness_evidence_20261007 --output NEW_EXCLUSIVE_DIRECTORY
```

Five focused arithmetic tests passed, exit0; actual pinned analysis passed, exit0. Tests cover analytic Lorentzian crossing width, grid order, truncated/multipeak/flat rejection, finite/grid rejection and exact normalization decomposition. No broad suite or fits. The initial attempt exposed the older nominal checkpoint's absent internal NPZ hash and exited1; it was corrected with the explicit independent immutable pins, not an identity relaxation. Existing empty first output directory retained. Source code, raw input and output hashes are in manifest.json. Figure validated_output/spectra_comparison.png was visually inspected for readable units and labels.
