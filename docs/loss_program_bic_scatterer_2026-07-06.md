# TM Loss-Reduction Program — BIC / Scatterer / Cross-Field Routes (theory-first)

> **Purpose.** Standalone brief so the NEXT chat can execute this without re-deriving
> anything. Written 2026-07-06 after the shape program closed (rectangles win; every
> fixed-width shape is neutral-to-worse). The user asked to pursue a fresh set of
> physics-first ideas — BIC, upgraded scatterers, and cross-field transplants — all
> simulated on Athena, theory BEFORE GPU, honest about ceilings.

---

## 0. How to resume (read this first)

- **Device (anchored TM):** pitch 516.83 nm, corrugation 400 nm (wide 1000 / narrow 600),
  height 350 nm, n_core 1.97 / n_clad 1.444, N=80/side, λ_res ≈ 1558.6 nm (opt mesh) /
  ≈1556 nm (accurate). Converged box **y = 6.8 µm, z = 8.8 µm** (span_mult 5.42).
- **Current best in-bound device = "the stack"** (W1050 cavity + gap-shift pair [+20,+20]
  + see-saw wide teeth 1040/980): loss **0.0545**, T 0.9449, fwhm +0.9%, Q 1404.
  Plain **rect-1050**: loss 0.0773 (opt) / 0.0823 (accurate), T 0.922. W800 baseline loss
  0.110. **Report devices by geometry, never "champion."**
- **Dispatch:** `bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.<module>`;
  fetch via `--results-no-fsp`. `add-study`, `athena-preflight`, `fetch-results` skills.
- All studies use the established runner pattern: `from runners.sweeps.tm_scatterer_scan
  import build_base`, then `BASE.scatterer.enabled=False`, converged box overrides,
  window 1558.5/40/3001. Row 0 = in-study no-change control; jitter partner in every study.

---

## 1. HARD CONSTRAINTS (every idea is gated on all of these)

1. **Mode width preserved:** |Δfwhm_m| ≲ 1% (the spatial energy width vs baseline). This is
   the constraint that killed apodization and every delocalizing route.
2. **SINGLE resonance (user, 2026-07-06 — CRITICAL).** The device must keep ONE usable
   defect resonance in the operating window. When two π-shift defects were brought close
   enough to interact, the mode split into a **doublet (two resonances)** — observed, and
   it did NOT help. **⇒ The two-spatially-separated-defect / supermode route is
   DE-PRIORITIZED.** Any BIC we pursue must be a *single-resonance* BIC (see §4.1).
3. **Uniform height (planar 2D fab).** In-plane shapes only, 350 nm everywhere. This is
   what makes TM's ~40% *vertical* radiation share untouchable by most tricks (§3).
4. **Start from a known-good baseline** (rect grating / rect-1050 / the stack), not LHS.
5. **Honesty gate:** register the predicted effect size in the runner docstring BEFORE
   dispatch; report nulls; all Δ vs the in-study control at identical numerics; a candidate
   near the jitter floor is confirmed at accurate mesh + half-mesh-offset partner before
   it counts.

---

## 2. THE PHYSICAL WALL (why fixed-width shapes all failed — don't re-litigate)

Radiation loss = the fraction of the mode's **spatial Fourier spectrum inside the light
cone**. Localize the envelope to width Δx ⇒ its k-spectrum has width ≥ 1/Δx (Fourier
uncertainty) ⇒ a narrower mode necessarily has MORE light-cone content ⇒ more loss.
**Loss and localization are conjugate.** Apodization works only by widening Δx. Rectangles
beat fancier shapes because a rectangle is already near the optimal same-width envelope.

There are exactly three escapes, and they sort the whole program:
- **(A) Spectral-null shaping** — make |envelope(k)|² have a *zero* at the radiating k, not
  just be narrow. = the "derived golden profile" route. **Done; sat at a local optimum.**
  Largely spent (kept as a falsification result).
- **(B) Interference / symmetry decoupling (BIC)** — cancel two radiating amplitudes so the
  mode does not couple to the continuum. *Evades the wall* (it is not a property of one
  envelope). **This is the only direction with real headroom. §4.1.**
- **(C) Relax a constraint** — height (→ vertical channel) or width. Off-limits, EXCEPT the
  TE observation in §5 changes the accounting.

---

## 3. ESTABLISHED MEASUREMENTS (already done — inputs, do not repeat)

From `tm_radiation_polarimetry` (job 117907) + scatterer program (jobs 115787→116940):
- **Radiation split: ~62% in-plane, ~38% vertical** (energy audit closes 99–100% on
  peak-centered rows; validated at accurate mesh and vs bigger box). ⇒ **HARD CAP: any
  purely in-plane trick (scatterers, in-plane reflectors) tops out at ~55–62% of the loss.**
- **Zero polarization conversion (f_TE ≈ 0.00 every row)** — in-plane radiation keeps guided
  TM polarization (E‖z). The literature "TM→TE lateral leakage / magic-width" mechanism is
  measured DEAD here (fully-etched ridge, no slab). Upside: radiation hits a vertical wall as
  s-pol (higher grazing R).
- **Radiation is NEAR-AXIAL** (|ux| ≈ 0.98, ≤ ~11° from the guide axis) — the hardest
  direction to reflect/redirect, because it is close to the guided mode itself.
- **Spatially distributed:** ~29% of radiating weight within ±3 pitches of the defect; ~70%
  spread along the arms. Local budget is ~harvested (that is what rect-1050 + the stack did).
- **Scatterer route VALIDATED after the box fix.** The earlier z-PML error (1.8λ z-box ate
  TM's vertical reactive tail, made loss read 0.19 vs true 0.11) was fixed; the pillar was
  re-tested at the converged box + accurate mesh (job 116896): **ΔT = +0.0026** (10–60× the
  jitter floor) — REAL but ~3% of the 11.7% radiated budget. Arrays do not multiply it;
  close-packed combs self-destruct (shadowing + multiple scattering); only a "lobe-ray
  diagonal" builds coherently (×2.4) but off a weak anchor.

---

## 4. THE IDEA CATALOG (theory-first → honest ceiling → Athena experiment)

Each idea has a **Phase-0 theory step (zero GPU, local)** that must pass its gate before any
dispatch, then a sim design. Deploy strategy in §6.

### 4.1 Single-resonance BIC — THE headline route (from atomic physics)

**Theory.** Friedrich–Wintgen BIC (1985, two autoionizing atomic resonances) and Dicke
subradiance (1954) are the same mechanism: two resonances coupled to the *same* radiation
channel, tuned so their radiation tails **destructively interfere** → one supermode's
radiative loss collapses (Q → ∞). A true BIC decouples from the **entire** continuum —
in-plane AND vertical — so it is the ONLY planar idea that can also touch TM's untouchable
40% vertical loss.

**The single-resonance requirement rules out the obvious version.** Two spatially separated
π-shifts → symmetric+antisymmetric supermodes → a **doublet** (what the user saw; bad). So we
pursue BICs that keep ONE resonance in the window:

- **(a) Symmetry-protected BIC.** A single defect mode whose symmetry is orthogonal to the
  near-axial radiation continuum's symmetry → radiation forbidden by symmetry, no second
  resonance introduced. Phase-0 task: classify the defect mode and the near-axial continuum
  under the device's mirror symmetries (y→−y lateral, x→−x about the defect); find a defect
  variant whose operating mode is ODD under the mirror the continuum is EVEN under. Candidate
  knobs (single resonance preserved): an antisymmetrized defect (sign-flipped/odd π-shift
  arrangement), or a lateral-symmetry-breaking-then-restoring perturbation. **Gate:** the
  group-theory table must predict a forbidden coupling for a realizable, single-resonance
  defect; else drop (a).
- **(b) Friedrich–Wintgen at ONE location (two mode *families*, not two defects).** Co-locate
  a second resonance of DIFFERENT character with the defect (e.g. a superimposed second-order
  grating harmonic, or a localized Mie-like index perturbation at the defect) and tune so the
  DEFECT mode becomes the BIC while its partner is pushed OUT of the operating window → only
  one resonance remains usable. Phase-0 task: 2-mode coupled-mode-theory (CMT) model — write
  the 2×2 non-Hermitian Hamiltonian, find the detuning/coupling where Im(eigenvalue)→0 for the
  operating mode AND the partner's real part leaves the ±20 nm window. **Gate:** CMT predicts a
  Q-boost ≥ ×3 with the partner ≥ 15 nm away and the operating mode's Δfwhm_m ≲ 1%.

**Honest ceiling.** In a finite device this is a *quasi*-BIC (finite Q, not infinite), and the
dark state tends to **spread spatially** → the width enemy again. The bet: a regime where the
interference is already deep before the width penalty exceeds 1%. Empirical, untried. Highest
value *and* highest realizability risk. Potentially the only route to the +0.05 T headline for
TM (because it can also cancel vertical radiation).

**Sim design (accurate mesh; ~2 arrays).** For (b): a 2-D scan of {partner-coupling strength}
× {partner detuning} around the CMT-predicted BIC point, ~25–35 rows, tracking loss, Q,
spectral-FWHM, AND fwhm_m at every point (a Q-spike with fwhm_m still in bound = win; a spike
that arrives only with widening = null). Include a control, a jitter partner, and a
"single-resonance check" (confirm only one peak in the window). For (a): ~10–15 rows over the
symmetry-breaking knob with the same tracking. Register the CMT-predicted separation/coupling
as the pre-registered prediction.

### 4.2 Scatterers, done properly — Kerker/Huygens + everything proposed (from nanophotonics)

The scatterer IS a **Green's function secondary source** (user's intuition — correct): total
radiated field = leaked field + Σ (scatterer field ⊗ free-space Green's function). Loss
suppression = arrange secondary sources so their propagated fields destructively interfere with
the leak in the far zone (a photonic dark state). Every pillar tried so far was a plain cylinder
= an omnidirectional monopole. Unexplored knobs:

- **(a) Kerker/Huygens directional tuning.** A dielectric scatterer at the radius where its
  electric and magnetic dipole resonances OVERLAP (a₁≈b₁) scatters **directionally** (a
  Huygens' source) — it can throw its field backward to cancel the near-axial leak with far
  less parasitic forward scatter. Phase-0: Mie analysis for a SiN pillar (350 nm tall, in
  oxide) at λ≈1556 nm — find the radius/shape (may need an elliptical/rod cross-section since
  a finite-height pillar isn't a perfect Mie sphere) giving the first Kerker (zero-backward →
  invert for max-backward) condition. **Gate:** Mie model predicts a directionality contrast
  ≥ 3:1 achievable in-band.
- **(b) Sparse phased array on the phase-matched arcs** (not close-packed combs — those
  self-destruct). Place Kerker scatterers at the constructive-interference radii λ_res/(2·n_clad)
  along the lobe-ray diagonal (the one geometry that built coherently, ×2.4). Phase-0: the
  phase-return model from the scatterer study, now with the Kerker directional phase.
- **(c) Re-run the best prior scatterer configs at the Kerker radius** to isolate directionality
  from size (matched control: same positions, plain-cylinder vs Kerker-tuned).

**Honest ceiling.** Still IN-PLANE ONLY (~55% of the loss), and distributed radiation needs the
sparse array. Realistic +0.01–0.02 T if directionality helps; but it is the route with an
existing validated +0.0026 anchor and a genuinely unused physical knob.

**Sim design.** Needs the rect-scatterer builder extension already spec'd in the old plan
(`simulation_config.py` ScattererConfig x_span/y_span; `bragg_device.py` addrect branch;
`generate_file_tag` rect tag). ~2 arrays: (1) Kerker-radius sweep single pair at the validated
x=810 site (size × cross-section to hit a₁=b₁), (2) sparse lobe-ray array of Kerker scatterers,
N∈{2,3,4}, vs plain-cylinder matched control. Accurate mesh; jitter partners; drain-vs-recycle
control (broadband T drop = drain, resonance-only = recycle).

### 4.3 Counterdiabatic / shortcut-to-adiabaticity apodization (from quantum control)

**Theory.** Apodization reduces loss by making the π-shift transition adiabatic — but adiabatic
= slow = spread = wide mode. Counterdiabatic driving (transitionless quantum algorithm, Berry;
"shortcut to adiabaticity") adds an engineered counter-term that cancels the non-adiabatic
(radiating) transitions WITHOUT slowing the transition → adiabatic-quality loss suppression at
the FAST (localized) length. Translated: a specific **non-monotonic duty-cycle / corrugation
modulation** near the defect that cancels the radiative coupling the abrupt π-shift generates,
at fixed localization length.

**Phase-0.** Write the coupled-mode envelope equation for the grating with a position-dependent
κ(x) and the π-shift; identify the non-adiabatic coupling term (∝ dκ/dx or the shift
discontinuity); derive the counterdiabatic counter-term H_cd and map it to a realizable
duty-cycle/width profile. **Gate:** the counter-term maps to a binary-etch-realizable profile
that is NOT just the already-found derived profile (else it collapses onto §2A, spent).

**Honest ceiling.** May be unrealizable in binary etch at fixed height, or may reduce to the
derived-profile local optimum. Speculative — but it is a NEW optimization target (a designed
non-monotonic modulation), not another fixed shape. Modest priority; cheap to test if Phase-0
yields a concrete profile.

**Sim design.** ~10–12 rows: the counterdiabatic profile × amplitude scale (incl. sign flip as
falsifier, per the derived-profile template), vs rect-1050 and vs the plain apodization ladder
at MATCHED fwhm_m (the real claim is "same loss as apodization at smaller width"). Accurate mesh.

### 4.4 Green's-function anti-radiation framing (unifying theory, not a separate sim)

The organizing principle behind 4.1 and 4.2: radiated power = overlap of the effective current
distribution with the radiation-continuum Green's function. Loss = 0 ⇔ that overlap = 0 (a dark
state / BIC). Use this in Phase-0 to (i) compute the current distribution from an existing field
export, (ii) identify which secondary-source placement or symmetry makes the overlap integral
vanish. This tells the scatterer array and the BIC symmetry analysis where to aim. Zero GPU.

---

## 5. TE vs TM (user asked — real asymmetry)

`te_span_z_check` (job 116891) measured TE **insensitive to the vertical box** (ΔT ≤ 0.003)
vs TM's +0.019 ⇒ **TE radiates far less vertically.** So the "untouchable ~40% vertical" wall
that caps TM barely exists for TE. ⇒ For **TE**, the in-plane toolkit (scatterers, Kerker,
in-plane reflectors, in-plane BIC) has a MUCH higher ceiling; for **TM**, only the true BIC
(§4.1) can reach past the in-plane 55%. The BIC/subradiant mechanism is polarization-agnostic.
**Plan:** primary target TM; but run the KEY BIC and Kerker studies for BOTH polarizations
(cheap add — polarization is a sweepable field) so we learn whether a TE-only win exists.

---

## 6. ATHENA DISPATCH STRATEGY (user: "deploy a lot together"; we log out)

- **Jobs are server-side (sbatch) — logging out is FINE once submitted.** The problem is only
  the number of concurrent arrays.
- **Reconcile "deploy a lot" with the serialize rule (CLAUDE.md §6):** two `--option3` arrays
  CANNOT run at once (shared `data/sweep_list.txt` — the clobber kills pending tasks). So the
  way to launch many things "together" is **combine multiple studies' rows into ONE large
  sweep array** (one runner, one sweep_list), which runs unattended as a single array. Build a
  few BIG combined runners (e.g. all-BIC-scan, all-scatterer-Kerker), not many small ones.
- **QOS `24h_1g` caps: 100 submitted / 4 running per user.** Arrays > 100 rows go in chunks
  (`--array-tasks=1-100`, then the rest as the queue drains). Count with `squeue -r` (plain
  `squeue` collapses a pending array to one line).
- **Per-run:** `athena-preflight` first (ports 1055/2325 OPEN by IP = license OK; `lmstat -96`
  is a FALSE NEGATIVE — do not block on it); unique `generate_file_tag()` per row; converged
  box; accurate mesh for anything claimed; `SBATCH_MEM=256G` only for field/far-field rows.
- **Every dispatch turn ends with the job/array ID + task count** (or a prominent "NOT
  dispatched because Y").

---

## 7. EXECUTION ORDER (proposed; theory gates first)

1. **Phase 0 (zero GPU, local):** the four theory steps — BIC symmetry table + FW-CMT model
   (4.1), Kerker Mie analysis (4.2), counterdiabatic derivation (4.3), Green's-function overlap
   from an existing field export (4.4). Each writes its gate verdict. **Cheapest, do first.**
   Reuse/extend `python_tools/lateral_radiation_theory.py`, `derive_boundary_profile*.py`.
2. **Checkpoint with user:** which gates passed; pick which sims to build.
3. **Batch 1 (biggest headroom):** single-resonance BIC scan (4.1b CMT-guided), TM + TE, one
   large combined array, accurate mesh, width-tracked.
4. **Batch 2:** Kerker scatterer (4.2) — needs the rect-scatterer builder extension first
   (smoke-test locally per §5 of CLAUDE.md), then one combined array.
5. **Batch 3 (if Phase-0 yields a profile):** counterdiabatic apodization (4.3).
6. Each winner → accurate-mesh confirm + half-mesh-offset jitter partner; FINDINGS.md +
   MATLAB .fig/PNG; full absolute local paths stated.

---

## 8. What is DE-PRIORITIZED / already closed (do not redo)

- Two spatially separated π-shifts / supermode doublet — splits the resonance (§1.2); the
  user confirmed it did not help. Only the SINGLE-resonance BIC forms survive.
- Fixed-width passive shapes (barrel/hann/gauss/tri/ellipse/wedge/tilt): neutral-to-worse;
  helpful ones only added area and a rectangle uses that area at least as well
  (`cavity_hann_sweep` job 118529 closed the Hann-widening variant too).
- Derived "golden" boundary profile: local optimum, both signs worsen (spectral-null route ~spent).
- Broadside near-cavity strip reflectors: their channel is already cancelled by rect-1050.
- Plain-cylinder scatterer arrays / close-packed combs: self-destruct; ceiling +0.003.
- Distributed π-shift, in-core holes (parasitic), odd anti-moment: closed.

---

## Sources (BIC / Kerker / STA)
- FW-BIC in a 1D grating on a DBR — https://pmc.ncbi.nlm.nih.gov/articles/PMC11501903/
- Existence of Friedrich–Wintgen BICs — https://arxiv.org/html/2505.12297
- BIC nonlinear/laser device review — https://www.nature.com/articles/s42005-022-00884-5
- Kerker/Huygens directional scattering — https://arxiv.org/pdf/1410.2315
- All-dielectric Huygens meta-waveguides — https://onlinelibrary.wiley.com/doi/full/10.1002/lpor.202200860
- Shortcuts to adiabaticity in optical waveguides — https://pubmed.ncbi.nlm.nih.gov/28085803/
- Counterdiabatic term cancels radiation — https://journals.aps.org/pra/abstract/10.1103/PhysRevA.108.022217
