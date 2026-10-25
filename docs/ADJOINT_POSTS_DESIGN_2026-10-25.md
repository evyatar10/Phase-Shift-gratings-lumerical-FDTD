<!-- Design note drafted by GPT-6-Astra (session in docs/ASK_GPT_BRIEF.md), reviewed by Claude 2026-10-25. NOT authorised: the 2-D-before-adjoint gate (HANDOFF sec. 1) still holds until the user lifts it. S2 refers to the scratch model-v2 RESULTS.md (not in the repo). -->
# Post-only FDTD adjoint: decision note
2026-10-25 · Design only. **Recommendation: differentiate peak transmission; accept on intrinsic Q and spatial width.** Full Maxwell fields include collective coupling and grating response (INFERENCE). The cloud-before-adjoint restriction remains [S1:51]; this note authorizes no execution.

## 1. Objective and parameterization

**INFERENCE — start with the actual resonant modal transmission, T\*, with resonance tracked locally, not globally pinned.** Under symmetric single-mode coupling, Qi = Qc sqrt(T\*)/[1−sqrt(T\*)]; hence d ln Qi = d ln Qc + dT\*/[2T\*(1−sqrt(T\*))]. The supplied ~1% coupling change supports this surrogate locally, but does not bound its derivative in every new direction. Measure Qc and left/right coupling on accepted trials; reject T gains without Qi gains. Freeze the ring-down extraction version: model-v2 already revised Qi after changing the truncation-correction origin [S2:61–65].

**FROM-CODE/SOURCE:** existing peak3 differentiates `abs(T_out)`, whereas reported T uses `abs(S)^2`; these are not interchangeable [S3:780–784; E:1764]. Reuse its machinery only after an actual-objective FD gate, or implement the consistently normalized modal-power functional. Use an interpolated frequency-grid peak, not the old broadband softmax. At an exact smooth peak, the wavelength-chain term in peak height vanishes; the moving-resonance width still needs gW = gW_fixed + (dW/dλ)gλ [S3:796–805].

**INFERENCE — alternatives:** do not optimize current tube power: model-v2 reports detuning-dependent increment bias [S2:16–26]. Ultimate direct objective is 1/Qi = Pi/(ωU), using isolated resonant fields, integrated stored energy, and all nonguided loss channels. Its quotient derivative must include U and guided-channel subtraction, not just a surface source. This is substantial new machinery; neither a tapered open tube nor one cladding-plane ∫|E|² certifies it. The existing `rad_fom` implements the latter [S3:856–871]. Differentiating a finite-time ring-down fit requires a time-domain adjoint or a validated differentiated correction; it is not supplied by the ordinary frequency-domain product.

**FROM-CODE → proposal:** keep full-height cylinders and y/z symmetry; independent x positions on the two longitudinal arms. The builder mirrors each listed site across y and exposes radius/x/y [B:1252–1329; R:158–165]. Thus 300 sites on +y imply ~900 parameters for mirrored pairs; freeing −y doubles parameters and loses y symmetry. Freeze tooth geometry and post count; start radius-only, then release positions.

Use `Parametrization` mapping parameters to cylinder properties. **FROM-CODE:** dε/dp uses ± geometry perturbations and cropped index extraction, not FDTD solves [V/d_eps_calculator.py:1007–1068]; stock `ClosedCurve` also uses mesher FD [V/closed_curve.py:2489–2501]. Cost remains O(parameters) CAD/meshing work, potentially doubled by mirror properties; benchmark before replacing it. Cache the sparse Jacobian across objective/width contractions [E:1747–1763], with geometry/grid/material hashes. Optimization fields must contain every post excursion, including ±42 µm extensions; current region sizing is tooth/old-comb based [E:1687–1717].

**FROM-SOURCE → INFERENCE:** an eventual cylinder boundary integral uses Δε E∥f·E∥a − Δ(ε⁻¹)D⊥fD⊥a, with normal velocities 1, cosθ, sinθ for r,x,y (adjoint phase/prefactor fixed by convention). This avoids meshing each parameter, but reconstructing continuous boundary fields on coarse curved cells is risky; it is not automatically the derivative of discrete PVA [Johnson 2002, Eq. 12](https://math.mit.edu/~stevenj/papers/JohnsonIb02.pdf). Prefer mesher FD first.

**INFERENCE — hard geometry constraints:** r∈[50,100] nm; every physical/mirrored pair satisfies ||pi−pj||−ri−rj≥100 nm. Add distance-to-local-tooth-polygon clearance, domain/PML margins, and an infinity-norm step bound. Solve a sequential constrained QP with linearized distances plus width bands; check exact geometry before simulation. Do not repair an optical step afterward by clipping/deleting posts. Keep the measured spatial-envelope width as acceptance authority.

<div style="page-break-after: always;"></div>

## 2. Cost, ordered gates, and failure criteria

**INFERENCE — budget:** one forward + port adjoint + width adjoint per gradient; tiling adds sources, not solves [S3:788–794; S4:1182–1204]. At the supplied, unverified 3.5 h/solve: **10.5 GPU-h + meshing/I/O**, plus 3.5 h/trial. Reusing accepted trials gives ~10.5 h/subsequent iteration. Recentring/retries add solves; a larger refined field region changes runtime.

**INFERENCE — duration:** QL≈10⁵ at 1.62 µm gives field lifetime τA=QLλ/(πc)≈172 ps. At 340 ps a pure pole retains ~14% amplitude; <1% requires ~0.8 ns after excitation, approximately **8 h/solve** by simple runtime scaling. This is neither measured cost nor a gradient guarantee. Require converged Fourier fields, not necessarily continuous-wave excitation. Short solves need a gradient-duration gate; scalar gamma correction alone does not correct its derivative. Longer runs may conflict with the >6 h restriction [S1:50].

All gates below are **PROPOSED (INFERENCE)** in the policy's dependency order [C:216–280]; hardware gates remain conditional on authorization.

| Gate | Smallest sufficient scene and pass condition |
|---|---|
| 1. Math | Synthetic resonance/profile and 3 disks: peak/width Jacobians, units, mirror derivatives, constraints/QP versus independent FD. Known-bad signs/indices must fail. |
| 2. Real-wrapper plumbing | Actual FOM/autograd/parameterization with synthetic arrays; hidden CAD of 3 posts/short strip. Verify property readback, perturbation/restoration, locked grid, sparse support, containment and nominal rebuild; no solve. |
| 3. Engagement | Force restoration, active gap/radius bounds, recentering, rejection, cache invalidation and restart. Assert branch markers **and outputs**; count live field arrays. |
| 4. Smallest hardware | Empty box for GPU/source/tile/API issues. Three posts beside a strip (near-wall and close-pair cases): all 9 r,x,y derivatives and held-out directions versus FD at h,h/2. Proposed vector error ≤10%, correct resolved signs; finer-grid/longer-duration check. Cannot certify cavity Qi/width. |
| 5. Pipeline smoke | Low-Q miniature Bragg cavity plus posts: real forward, both adjoints, mesher FD, constrained step, acceptance, interruption/restart and tiling. Validate port/field phase conventions on held-out directions; no inherited tooth calibration. |
| 6. Device gate | N=96 incumbent, fresh feasible direction, half/full trials. Require measured/predicted objective gain 0.5–1.5 when resolved, Qi gain, width in its frozen band, stable mode/coupling, exact gaps. Validate duration and matched finer-mesh improvement; small-scene calibration alone is insufficient. |

**INFERENCE — resume:** atomically checkpoint every forward, adjoint and dε stage: FSP/companion fields, parameter/grid/source/objective hashes, resonance, accepted/rejected IDs, radius and solver state. Verify and reuse completed stages on restart; lose ≤one solve [C:488–493]. Existing journal/state utilities help [P:74–85,132–146]; stage reuse still needs an interruption gate.

**Ranked risks (INFERENCE):** wrong temporal/normalization gradient; coarse-cell boundary error; coupling/width gains masquerading as Qi gains; field-memory exhaustion; fixed-count local stagnation. Stream contractions. Anchor PVA separately from conformal dy29. Old C_field had a conjugation error despite correct signs [S4:1043–1053]; do not fit away class-dependent errors. Gains/runtime remain unmeasured.

**Source key.** Repo=`C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes`. E=`runners/lumopt2_design/lumopt2_design.py`; B=`bragg_device.py`; R=`runners/sweeps/farfield_sph_15um_postarray_overshoot.py`; P=`runners/lumopt2_design/persistence.py`; C=`CLAUDE.md`; S1=`docs/HANDOFF_postarray_2026-10-24.md`; S3=`runners/lumopt2_design/THEORY.md`; S4=`runners/lumopt2_design/HANDOFF.md`; S2=`../modelv2/RESULTS.md` relative to this note. V=`C:/Program Files/Lumerical/v261/api/python/lumopt2/parametrization`. Result documents are FROM-SOURCE, not independently rerun evidence.
