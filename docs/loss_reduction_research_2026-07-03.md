# Reducing the TM Radiation Loss of the Pi-Shift Bragg Grating — Research Memo

**Date:** 2026-07-03
**Scope:** literature survey (two independent web sweeps) + full re-read of paper_8, cross-checked
against this project's measured FDTD results. Constraints assumed throughout: keep the cavity
mode spatially **short**, and only **single-layer 2D in-plane patterning** of the SiN layer
(no multilayer / bottom mirrors).

**Device context.** Pi-shift Bragg grating, SiN strip (n_core = 1.97, SiO2 cladding n = 1.444),
height 350 nm, TM anchored geometry pitch 516.83 nm / corrugation 400 nm, resonance ≈ 1558.6 nm.
After the z-box convergence study, the **true resonant radiation loss is ≈ 10 % of T** (the old
19 % was roughly half numerical: the 1.8λ z-boundary sat inside the reactive near field).
Already tried in this project: apodization (works, but widens the mode too much), tooth shift
(no effect on TM), single SiN pillar pairs as recyclers (+0.002 T, confirmed at accurate mesh),
SiO2 holes in the core (parasitic-to-neutral; ~−0.45 nm λ trimming knob).

---

## 1. The hard ceiling and the governing bound

The only recoverable energy is the propagating radiated power — **≈ 10 % of T**. Every scheme
below is bounded by it (paper_8 Eq. 13 says the same).

**Coupled-mode / time-reversal bound.** A cavity with radiative rate γ_rad and an external
reflector that returns amplitude-overlap fraction *f* of the radiation pattern with round-trip
phase φ has

> γ_eff = γ_rad (1 − f·cos φ)

to first order. Q_rad improves by at most 1/(1−f); wrong phase *increases* loss by the same
amount. Derivations/demonstrations:

- Single ion in front of a distant mirror (the exact analog of our geometry): Eschner et al.,
  Nature 413, 495 (2001) — https://www.nature.com/articles/35097017 ; theory Dorner & Zoller,
  PRA 66, 023816 (2002) — https://arxiv.org/pdf/quant-ph/0206080
- f → 1 limit (all radiation returned, emission fully suppressed/enhanced): Hoi et al.,
  Nature Physics 11, 1045 (2015) — https://www.nature.com/articles/nphys3484
- CMT formalism: Fan, Suh, Joannopoulos, JOSA A 20, 569 (2003) —
  https://opg.optica.org/josaa/abstract.cfm?URI=josaa-20-3-569
- On-chip proof that decay-channel interference can change Q ×4: Tanaka et al.,
  Nature Materials 6, 862 (2007) — https://www.nature.com/articles/nmat1994

**Implications for this device (λ = 1558 nm, n_clad = 1.444):**

- Our measured +0.002 T pillar pair ⇒ effective f·cosφ ≈ 2 % of the radiation channel —
  the physics behaves exactly as CMT predicts; it is weak because the scatterer is weak.
- **Gains scale linearly in N** phase-correct scatterers (the N² in paper_8 Eq. 16 is returned
  *power*; the loss-rate correction — what T measures — is linear in returned *amplitude*),
  and linearly in polarizability α per element (self-loss ∝ α² sets a finite optimum, but a
  non-resonant SiN post in SiO2, Δn = 0.53, is far below it; unitary dipole limit:
  Ruan & Fan, PRL 105, 013901 (2010) — https://link.aps.org/doi/10.1103/PhysRevLett.105.013901).
- **Wavelength bandwidth is a non-issue**: the cos φ > 0.5 window is ±14 nm for a reflector at
  ρ = 10 µm (±35 nm at 4 µm) vs a ~1 nm cavity linewidth.
- **Radial placement tolerance IS the issue**: ±50–90 nm for ≤ π/3–π/6 phase error — fab-relevant,
  and the reason the effect only survived at dx ≈ 35 nm mesh (dx = 50 nm quantizes position at
  the scale of the whole useful phase window).

---

## 2. What paper_8 actually delivers (honest verdict: more than it seemed)

Re-read in full. It earns its keep on four counts:

1. **It predicted this project's negatives.** Tooth shift "recycles vertical radiation, helps TE
   but does nothing for TM, because TM has almost no vertical radiation to recycle" (§5) —
   matches the measured null. "Envelope apodization ... is largely spent for TM, since the
   baseline loss is already small" (§10) — matches.
2. **The diagonal placement intuition is its Fig. 9.** Inclusions belong on concentric arcs
   spaced λ₀/2n_clad ≈ 0.544 µm **along the forward/backward lobes**; a fixed-y lateral row
   "catches only the weak broadside tail." (The queued tm_scatterer_array study tests same-arc
   and lobe-ray rows against the fixed-y comb head-to-head.) Tempering caveats from the paper
   itself: the TM lobe hugs the axis within ~10°, so a few µm out "diagonal" is nearly in-line
   with the guide; and a discrete scatterer "re-radiates a dipole pattern ... it cannot cancel
   that radiation in every direction at once" — the structural reason scatterers rank below the
   two-cavity route.
3. **Its single concrete recommendation is still untried here:** two in-line π-shifts tuned to
   the **subradiant supermode** (Eq. 21): Q_sub/Q_single = 1/(1 − η|cos(k_rad·d)|), maxima at
   d = m·λ₀/2n_clad, ceiling 1/(1−η). Sweet spot ≈ 43 periods (≈ 22 µm), which simultaneously
   gives the subradiant phase and a balanced partial mirror R ≈ 0.46. In-line is preferred
   because each defect's near-axial radiation heads toward the other (high channel overlap η).
4. **It flags the reactive-vs-propagating distinction** ("the wide TM near field is reactive and
   stores energy without losing it") — precisely the effect the z-box convergence study measured
   numerically.

---

## 3. Ranked options (most promising first)

### 3.1 Short interface taper at the π-shift — NOT the apodization already tried ★ top pick

Quadratic ramp of the corrugation (mirror strength) over only **~4–15 periods adjacent to the
defect**, mirror body left at full strength. The abrupt envelope kink at the π-shift is what
generates spatial-Fourier components inside the cladding light cone — i.e. the radiation source
itself. Published scaling (the key asymmetry): **Q_rad grows exponentially with taper periods
while mode length grows only linearly** — a fundamentally better trade than full-envelope
apodization.

- Quan & Lončar, "Deterministic design of wavelength scale, ultra-high Q photonic crystal
  nanobeam cavities," Opt. Express 19, 18529 (2011) — https://arxiv.org/abs/1108.2675
  (Q_rad = 5×10⁹ designs; waveguide-coupled Q = 1.3×10⁷ at T = 97 %; method stated universal
  incl. TM; residual far field hugs the waveguide axis exactly like our TM lobe)
- Mode-profile matching is the dominant mechanism, achievable in 2–4 periods:
  Sauvan, Lalanne et al., PRB 71, 165118 (2005) — https://arxiv.org/pdf/cond-mat/0502664
- Experimental, wavelength-scale cavities with few-period tapers, measured Q ≈ 1.5×10⁵:
  Velha et al., Opt. Express 15, 16090 (2007) —
  https://opg.optica.org/oe/fulltext.cfm?uri=oe-15-24-16090&id=145446 ;
  Md Zain et al., Opt. Express 16, 12084 (2008) —
  https://opg.optica.org/oe/fulltext.cfm?uri=oe-16-16-12084&id=170159

Caveats: headline demos are TE hole-gratings in Si membranes; the mechanism (killing light-cone
components from the interface kink) is polarization- and direction-agnostic, but the magnitude
for a TM sidewall grating needs FDTD. Fully single-layer. Keeps the mode short.

### 3.2 Widen the waveguide — pull n_eff off the cladding light line

The TM n_eff sits barely above n_clad = 1.444; that is the **root cause** of the axis-hugging
radiation cone (paper_8 Eq. 9b). Quan & Lončar design step (iv), verbatim physics: larger width
raises the mode's effective index, pulls it away from the light line, and reduces **in-plane**
radiation loss (roughly exponentially in the margin β − n_clad·k₀). One-parameter sweep,
single-layer, preserves mode length. Caveats: pitch must be re-trimmed (pitch ↔ width coupled,
as always in this project), higher-order-mode onset, and corrugation 400 nm was chosen for
mode-width matching — κ changes with width, so the match must be re-checked.

### 3.3 k-space diagnostic before any further geometry work (free)

FFT the simulated resonant field envelope along x and weigh the components inside
|k| < n_clad·k₀: this attributes the 10 % between the defect-interface kink and the mirror body,
and directly ranks 3.1 vs everything else. Formalism:

- Srinivasan & Painter, Opt. Express 10, 670 (2002) —
  https://opg.optica.org/oe/abstract.cfm?uri=OE-10-15-670
- Englund, Fushman & Vučković, Opt. Express 13, 5961 (2005) —
  https://web.stanford.edu/group/nqp/jv_files/papers/cavity-theory-2005.pdf

Applies unchanged to in-plane radiation (the near-axis lobe = components just inside the
light-cone edge). Can run on field data already recorded (demo study slices).

### 3.4 Two in-line π-shifts (subradiant supermode) — paper_8's own #1

Mechanism independently supported at large factors in the BIC / Friedrich–Wintgen literature:

- BICs in fiber Bragg gratings (closest published geometry — grating waveguide radiating into a
  cladding continuum; quasi-BIC Q enhancement ~2 orders): Gao, Zhen, Soljačić, Chen & Hsu,
  ACS Photonics 6, 2996 (2019) — https://pubs.acs.org/doi/10.1021/acsphotonics.9b01202
  (arXiv: https://arxiv.org/pdf/1707.01247). Caveat: those BICs live at one isolated k_z of a
  *propagating* band, not at a band-edge defect mode — no published BIC-protected phase-shift
  cavity exists.
- Supercavity / avoided-crossing quasi-BICs (10–100×, collapses ∝ 1/δ² off the crossing):
  Rybin et al., PRL 119, 243901 (2017); coupled-resonator chains —
  https://arxiv.org/html/2307.05937 ; FW-BICs in coupled stubs: PRB 109, 235431 (2024) —
  https://journals.aps.org/prb/abstract/10.1103/PhysRevB.109.235431
- Review: Hsu et al., Nature Reviews Materials 1, 16048 (2016) —
  https://scholar.harvard.edu/chsu/publications/journals/NRM

Honest costs: the composite subradiant mode spans both cavities (~22 µm — check against the
short-mode requirement before investing), and the boost is detuning-fragile. Serves TE too
(more radiated power to suppress there).

### 3.5 Coherent Bragg arcs — the correct scaling of the scatterer/diagonal idea

Replace isolated pillars with **chirped concentric trench arcs** (continuous, not pillars)
covering each ±15° lobe, several rows deep:

- Cylindrical-wave Bragg design theory (rings must be chirped near the source to match the
  Hankel phase; periodic λ/2n spacing is only asymptotically right): Scheuer & Yariv,
  JOSA B 20, 2285 (2003) — https://opg.optica.org/josab/abstract.cfm?uri=josab-20-11-2285 ;
  Erdogan & Hall circular DFB: https://www.osti.gov/biblio/6803501 and
  https://ieeexplore.ieee.org/document/124985/
- Ideal-planar ceiling for n = 1.97/1.444 quarter-wave rows: 3 pairs → R ≈ 0.54, 5 → 0.84.
  Discrete pillars do far worse (diluted fill); and the radial-Bragg literature's documented
  killer is **vertical scattering** of the unguided cladding wave — US patent 8718112 —
  https://patents.google.com/patent/US8718112B2/en (deliberately weakening the mirror raised Q
  by orders of magnitude). This matches our own z-convergence finding that z is where the
  power goes.
- Partial arcs still work (reduced effect): bowtie circular-Bragg cavities, Nanophotonics 14
  (2025) — https://www.degruyterbrill.com/document/doi/10.1515/nanoph-2024-0485/html ;
  bullseye reviews: Sci. Rep. 13 (2023) — https://www.nature.com/articles/s41598-023-32359-0

**Novelty note:** no published demonstration was found of in-plane radiation recycling of a
waveguide cavity by external scatterers or arcs — the current experiment appears to be
unpublished territory (a clean positive OR a bounded null is publishable). But the expected
magnitude stays bounded by the 10 % budget and grows only linearly with coverage.

### 3.6 Tooth shape at fixed κ (cheap, untested)

Radiation per period at fixed mirror strength is shape-dependent — e.g. sinusoidal or
equivalent-κ hole-based modulation vs deep rectangular sidewall teeth (H-shaped holes:
https://www.sciencedirect.com/science/article/abs/pii/S1386947714001635 ; shape+rotation-tapered
SiN nanobeams report ~8× Q over width-only tapers). No published comparison exists for TM
sidewall gratings — an easy in-house FDTD test.

---

## 4. Dead ends (confirmed, with reasons)

| Route | Why not |
|---|---|
| Far-field engineering by mirror perturbations (Portalupi et al., Opt. Express 18, 16064 (2010) — https://opg.optica.org/oe/fulltext.cfm?uri=oe-18-15-16064&id=203838 ; Tran et al., PRB 79, 041101(R)) | Only *redirects* radiation (collection optics figure of merit), generically **costs** Q. Power must return to the guided mode here — wrong tool. |
| Weak / mode-gap modulation (Kuramochi et al., APL 88, 041112 (2006); on SiO2: Opt. Express 18, 15859 (2010)) | Proven to Q ~ 10⁵–10⁶ on oxide, but the mode penetration 1/κ grows in exact proportion — quantitatively the same mode-widening trade as apodization. Excluded by the short-mode constraint. |
| Bottom mirrors / DBR under-cladding (SiN grating coupler +3.2 dB: Sci. Rep. 9 (2019) — https://www.nature.com/articles/s41598-019-49324-5) | Multilayer — excluded by fabrication constraint. Cited only as proof that recycling a known loss lobe works when mode-matching is easy. |
| Vertical tooth shift for TM | paper_8 §5: TM has almost no vertical radiation to recycle — confirmed by this project's null result. (Remains the natural first lever for TE.) |
| More isolated pillar pairs | Linear-in-N, tiny per-element α; strictly dominated by coherent arcs at the same fab cost. |

---

## 5. Bottom line

1. The recoverable budget is **≈ 10 points of T**, period. Any claim above that is a numerics
   artifact (compare only within identical, z-converged numerics).
2. The scatterer/recycling mechanism is real, behaves exactly per CMT, and appears **novel as
   published work** — but it recovers slices of the budget linearly and is placement-critical to
   ±50–90 nm. The queued array study (ρ-combs vs same-arc vs lobe-ray) settles the geometry
   question empirically.
3. The two genuinely new levers this research surfaced, both compatible with the short-mode
   constraint and single-layer fab, are the **short interface taper at the π-shift** (published
   exponential-gain-vs-linear-cost scaling — top pick) and the **width / light-line margin**
   knob (attacks the root cause of the axis-hugging lobe). The free **k-space diagnostic**
   should run first to attribute the loss and pick between them.
4. paper_8's two-in-line-π-shifts subradiant route remains the highest-ceiling interference
   scheme, at the cost of a ~22 µm composite mode and detuning fragility.
