# THE DESIGN, AND THE METHOD — pi-shift Bragg grating, inverse design

> ### ℹ️ You probably want `HANDOFF_SELF_CONTAINED.md` instead
> This file is the **method chapters only**, and it is *entirely contained in*
> `HANDOFF_SELF_CONTAINED.md`, which adds the 191-parameter design vector, the
> code, the raw data and the run record. This file exists as the **editable
> source** — change the prose here, then regenerate the self-contained version.
> Hand THAT one to a chat session, not this one.


**What this file is.** The handoff for the inverse-design programme. `HANDOFF.md` is *state* — jobs, numbers, what to run
next. This is the *explanation*: the device we have built, why we are doing
inverse design at all, why a single cost function provably cannot do this job,
and what our algorithm actually does instead.

Diagrams: §7 specifies what to draw and how. Every number is **MEASURED** (from a named file),
**DERIVED**, or flagged as **THEORY**.

---

## 1. Two tracks, and where each one stands

The programme runs on two legs, and they answer different questions.

### Track A — the design we actually have  ✅ *this is the asset*
A **parametric device**: the geometry is described by tables of per-tooth
values (corrugation, mean width, longitudinal shift) plus a comb of flanking
posts and a cavity width. It was brought to its current performance by
**successive hand-guided adjustment** — reading the mode width `σ` and the
transmission after each change, adjusting the tables, re-measuring. Not a
black-box optimizer run; a physicist steering a parametric model.

**It works.** MEASURED, and currently being validated further (see §7).

### Track B — the inverse design  🔧 *this is the method under construction*
Adjoint-based optimization over all 191 parameters at once. The goal is to do
in a machine loop, and better, what Track A did by hand — and, crucially, to
do it **while holding the mode width on spec**, which is the part that has
made this hard.

★**Track A is the deliverable. Track B is the multiplier.** Nothing in Track B
is required for the device to exist; it is required for the device to get
substantially better without spending months of hand-tuning per iteration.

---

## 2. The device we have

**Physics.** A pi-shift Bragg grating in SiN (`n_core = 1.97`,
`n_clad = 1.444`), core height 350 nm, pitch 516.83 nm, TM polarization. The
corrugation opens a photonic bandgap; a half-period defect at the centre puts
one resonant mode inside that gap. Resonant light tunnels through; the rest of
the stopband reflects.

**What we optimize for.**
- **Transmission `T` at resonance.** `1 − T` is cavity loss — resonant energy
  radiated out of the guide instead of transmitted.
- **Spatial mode width `W`** — FWHM of the resonant field envelope along `x`,
  in µm. This is the **sensing aperture** for the acousto-optic application.

★**The width is a HARD, TWO-SIDED SPEC.** The detector must overlap an
acoustic field of a given extent, so a *narrower* mode is off-spec, not a
bonus. This one fact is what makes the problem non-trivial — remove it and you
simply lengthen the cavity until radiation vanishes.

### The best design we hold — `BEST_T9636`
Full 191-parameter vector stored in `best_designs.py` (never re-pasted;
import it). Origin: v2 campaign, Athena job 136465, eval 12, **converged**
(evals 10–12 identical to 5 decimals; the optimizer took a zero step).

| quantity | PVA mesh (design numerics) | conformal mesh (spec numerics) |
|---|---|---|
| **Transmission T** | **0.96361** | **0.97805** |
| resonance λ | 1566.444 nm | 1560.907 nm |
| mode width (FWHM) | 18.353 µm | 19.008 µm |
| loaded Q | 2021.6 | 1714.2 |
| intrinsic Q | — | **155 358** |

Geometry at that point: mean corrugation 357.95 nm, cavity width 961.1 nm,
cavity elongation `2·Σshift` = 132.6 nm, winner comb.

**Cavity loss `1−T` fell from 0.0717 at the uniform origin to 0.0220** — a
**−69%** reduction — while the mode width was *kept* (−0.88% vs origin, i.e.
slightly narrower, comfortably in band).

★**The two mesher columns are not interchangeable.** PVA and conformal are
different discretizations; the same device reads λ +5.3 nm and FWHM −8% apart
between them. Never compare a number across meshers. What *does* transfer is
the **ranking** — origin < see-saw < best holds under both — and that was
verified before any conformal number was quoted.

---

## 3. Why we are doing inverse design

Hand-tuning worked, but it explores a 191-dimensional space one or two
coordinates at a time, guided by intuition about which knob does what. The
adjoint method gives the derivative with respect to **all 191 parameters for
the cost of two simulations**, not 191. That is the whole promise: full-space
descent at fixed simulation budget.

The obstacle is not getting a gradient of `T`. That part has worked for a
while. The obstacle is the **constraint**.

---

## 4. ★ The cost functions we tried, and why each one failed

This section is the heart of the document. Three figures of merit were tried in
sequence; each failed for a *different structural reason*, and understanding
those reasons is what produced the current method.

### 4a. Attempt 1 — σ, the second-moment width  ❌ *the constraint could not see the violation*

The first width measure was **σ**, the RMS width of the intensity profile:

```
σ = sqrt( ∫ (x−µ)² I(x) dx  /  ∫ I(x) dx )        (sigma_of_line, :1720)
```

It is the obvious choice: one line of code, smooth, differentiable, no
peak-finding. **It does not work, and the way it fails is instructive.**

σ is a *second moment*, so it is dominated by the profile's **tails and bulk**.
The spec quantity — FWHM — is set by where the envelope crosses half its peak.
Apodization, which is exactly what the optimizer does to suppress radiation,
reshapes the envelope *near the half-max* while leaving the far tails much as
they were. So the optimizer could reshape the mode substantially and σ would
barely register it.

**MEASURED:** against the true `fwhm_env`, σ is **24 percentage points** off,
and a peak-ratio measure is 21 pp off, where the later soft-level-set measure
tracks it to **≤2 pp**. That is not a calibration error — it is a different
quantity.

**Consequence, and it is the worst kind:** the optimizer bought transmission
*with width*, and **σ hid it.** Both design lineages went width-buying while
the constraint reported healthy. The recorded best had grown **+14.9%** in
true width — a flat spec violation — while the FOM was satisfied throughout.

**A second, compounding failure.** To make σ cheap inside the loop it was
replaced by a *fitted linear surrogate*:

```
σ̂  =  17.49  +  0.0051·(2Σshift)  +  0.109·(w_cav − 800)      [µm; nm inputs]
```

Fitted at one operating point, used everywhere. It **overstated corrugation's
width authority by ~30%**, so it labelled trade rows "width-neutral" that were
actually spending a third of the remaining band — and it **falsely rejected a
genuinely in-band design at T 0.9591.**

★**The rule that came out of this, now enforced in code**
(`check_sigma_surrogate`, `:887`): *if a surrogate can be checked against a
real measurement on the very same evaluation, check it there, every time, and
make the disagreement loud.* Both numbers are already in hand, so it costs
nothing — and it converts a silent modelling error into a visible one. This is
why the current method logs `wg_resid_um` on every single evaluation.

### 4b. Attempt 2 — a single combined cost function  ❌ *the mode kept expanding*

Next: keep the honest width measure, and fold it into one scalar objective with
a penalty on violating the band:

```
J  =  J_T  −  μ · penalty(W)
```

then tune `μ`. **This is the attempt whose failure is most worth explaining,
because the symptom was unmistakable: run after run, transmission rose and the
mode widened — monotonically, not erratically — until the device left the band
and the campaign was worthless.** Two full campaigns ended that way.

It is tempting to read that as "μ was mistuned". It is not. There are three
structural reasons, and no value of μ fixes any of them.

**Reason 1 — a scalar fixes the exchange rate before you know the landscape.**
Collapsing two goals into one number means committing, in advance, to how much
width a unit of transmission is worth. Every subsequent step trades at that
rate. Too small and the width runs away; too large and the optimizer stalls
against the penalty and stops finding transmission. There is no correct value,
because the true marginal trade *varies across the space* — and the shadow
price we now log confirms it varies by more than an order of magnitude.

**Reason 2 — a deadband penalty prices nothing inside the band. ★This is the
direct cause of the observed expansion.**
The spec is two-sided with a ±2% deadband. Inside that band the penalty is
*identically zero*, so the width is **completely unpriced**. Meanwhile widening
almost always buys transmission. So the gradient of `J` inside the band is
simply the gradient of `T` — and it points, reliably, toward a wider mode. The
optimizer drifts to the edge because nothing opposes it, crosses, gets shoved
back by the now-active penalty, and thrashes at the boundary.

**Monotone widening is not a bug in the tuning; it is the exact behaviour this
formulation specifies.**

**Reason 3 — a scalar penalty can be blind to entire directions.**
A penalty acts through whatever quantity it is written on. The tooth-level wall
was found (audit, 2026-08-24) to be **rank-deficient**: it priced only the
*mean* corrugation, leaving the **see-saw** direction — alternating corrugation
up and down at fixed mean — completely unpriced. The optimizer walked freely
along the one direction the constraint could not see. A scalar sees a scalar;
the constraint is a 191-dimensional object.

### 4c. What the two failures have in common

Attempt 1 failed because the constraint **could not measure** the violation.
Attempt 2 failed because the constraint **could not price** it in the region
where it mattered. Both are failures of *compressing the constraint into one
number* — first into a bad scalar, then into a good scalar that is still a
scalar.

★**Conclusion: transmission and width must remain SEPARATE objectives with
SEPARATE gradients.** That is what the current method does, and §5 is why
that buys something a scalar never can.

## 5. ★ What separate objectives buy — and why it costs a second adjoint

This is the finding that reorganized the whole programme, and it came from
watching the earlier inverse-design runs fail in a *consistent* way.

**The observation.** Run after run, the optimizer raised transmission and
**the mode kept widening.** Not erratically — monotonically, every campaign,
until the device left the width band and the run was worthless. Two full
campaigns ended out of band that way.

The natural first response is to add a penalty: optimize

```
J  =  J_T  −  μ · penalty(W)
```

and tune `μ`. We did that. It does not fix the problem, and it is worth being
precise about *why*, because the reasons are structural rather than a matter of
tuning harder.

**Reason 1 — a scalar objective fixes the exchange rate in advance.**
Collapsing two goals into one number means committing, before you know the
landscape, to how much width a unit of transmission is worth. Every step then
trades at that rate. Too small a `μ` and the width runs away; too large and the
optimizer stalls against the penalty and stops finding transmission. There is
no correct value, because the true marginal trade varies across the space.

**Reason 2 — a deadband penalty prices nothing inside the band.**
Our constraint is two-sided with a ±2% deadband. Inside the band, the penalty
is identically zero — so the width is **completely unpriced**, and the
optimizer is free to drift toward the edge because widening usually *does* buy
transmission. Then it crosses the edge, gets shoved back, and thrashes. The
observed monotone widening is exactly what an unpriced-until-violated
constraint produces.

**Reason 3 — a scalar penalty can be blind to whole directions.**
A penalty acts through whatever surrogate quantity it is written on. An earlier
width wall was found to price only the **mean** corrugation, which left the
*see-saw* direction — alternating corrugation up and down at fixed mean —
completely unpriced. The optimizer walked freely along the direction the
constraint could not see. A scalar sees a scalar; the constraint is a
191-dimensional object.

★**The conclusion: transmission and width must stay SEPARATE objectives with
SEPARATE gradients.** Not blended into one number.

---


Keep `∇T` and `∇W` as two distinct 191-vectors. Now you can do something a
scalar objective can never do: **construct a step that provably does not
change the width to first order.**

```
step  =  α · ( D∇T  −  coef · D∇W ),      coef chosen so that   ∇W · step = 0
```

That is an orthogonal projection of the transmission gradient into the
**null space of the width gradient**. Follow it and, to first order, the width
does not move *at all* — while transmission climbs. No exchange rate is chosen,
because nothing is being traded: we move only in directions the constraint is
indifferent to.

**This is what "we need something that holds the width constant while finding
the gradient" means concretely.** It is not a heuristic — `∇W · step = 0` is
exact, and is verified numerically to 8.3×10⁻¹⁷ in
`gates/gate_projection_local.py`.

**And this is why we need more than one adjoint.** The adjoint source is
`dJ/dfield` — it is *built from the objective*. Transmission and width are
different functionals of the same fields, so:

```
dT/dfield   ≠   dW/dfield        ⇒   different adjoint source   ⇒   a second solve
```

You cannot extract `∇W` from the transmission adjoint by any post-processing;
the information is not in there. Hence **two adjoint solves per iterate** — one
driven by the port mode, one driven by a weighted field-region source.

★A pleasing physical detail: the width adjoint's source profile is literally
`dsoftW/dI`, which is sharply peaked **at the two half-max crossings of the
envelope**. The width gradient is asking *"how do I move the half-max
points?"*, and its source sits exactly there.

★**The trade this makes explicit:** the ratio `λ = (∇T·D∇W)/(|D∇W||∇W|)` — the
*shadow price* — is logged every iterate. It is the marginal transmission
available per unit width spent. With a scalar penalty this quantity is buried;
here it is a readout, and it tells you when the constraint has genuinely
stopped being affordable.

---

## 6. What the algorithm actually does

### 6a. The parameters (191)
All in nm. `N_FREE = 25` free periods per side, `N_COMB = 57` posts.

| slice | n | meaning |
|---|---|---|
| `SL_CORR` 0:25 | 25 | corrugation depth per tooth — sets local coupling κ; apodization lives here |
| `SL_AVG` 25:50 | 25 | mean tooth width — sets local effective index / detuning |
| `SL_SHIFT` 50:75 | 25 | per-tooth longitudinal shift; the cavity absorbs `2·Σshift` |
| `SL_R` 75:132 | 57 | comb post radii |
| `SL_X` 132:189 | 57 | comb post positions |
| `I_DCOMB` 189 | 1 | comb transverse offset |
| `I_CAV` 190 | 1 | cavity width — the most width-efficient lever measured |

Only the innermost 25 periods are free; the outer ones are pure mirror and the
surrogate `N` is chosen so the mirror is already effectively infinite
(`2κL ≳ 3.5`).

### 6b. The transmission objective
A **windowed power-mean** over the recorded spectrum:

```
J_T = ( mean_{i ∈ window} |T_i|^12 )^(1/12),   window = |λ_i − λ_pk| ≤ 2.5·FWHM
```

*Soft-max, not `max`*: a hard maximum has zero gradient at every non-maximal
sample and its argmax jumps between grid points as the resonance drifts — the
optimizer would see a staircase. *Windowed*: the global maximum of `T(λ)` sits
in the passband, not at the defect resonance, so an unwindowed objective
optimizes the wrong feature entirely.

### 6c. The width: one observable, one differentiable carrier
- **`fwhm_env`** — the spec quantity, built by fitting a cubic envelope through
  the standing-wave peaks. Identical by construction to the programme's stored
  `fwhm_m`. **Not differentiable** (peak-picking, interpolation).
- **`softW`** — a smooth surrogate: smooth the profile, take a soft-max peak
  and an edge-window floor, form a sigmoid indicator of "above half-max", and
  integrate it. Differentiable end to end, so it can drive an adjoint.

They are tied by a **delta anchor** measured at one reference point, and the
residual between prediction and measurement is logged **every evaluation** with
a loud warning if it drifts. Carry the surrogate, but keep the real observable
beside it.

★All branch decisions are made on the **MEASURED** `fwhm_env`, never on the
surrogate. The surrogate only ever supplies a *direction*.

### 6d. One iterate, start to finish
```
1. forward solve at p                    → T(λ), field profile I(x)
2. measure λ_pk, FWHM, fwhm_env          → the observables
3. adjoint solve × 2                     → port-driven and width-driven fields
4. assembly pass 1  → ∇T
   assembly pass 2  → ∇W                 (same fields, zero extra solves)
5. choose the branch on MEASURED width:
      W below target − margin/2  → CLIMB   step = α·D·∇T, clipped so it lands
                                            exactly on the ceiling, never over
      W within the margin        → RIDE    the null-space step: ∇W·step = 0
      W above target + margin/2  → RESTORE step straight back along ∇W
6. clip to the bounds box; cap the step
7. accept or reject on a filter over (transmission, distance-to-target);
   on reject, re-step from the last accepted point at half the step length
   using its STORED gradients — no re-solve
8. log everything; persist for resume
```

**The strategy is deliberately ceiling-riding**: sit just under the maximum
allowed width and spend the whole allowance on transmission, rather than
hugging the seed width and leaving performance unclaimed.

★**Seed-dependent exception (MEASURED, b1 lane 2026-08-29, lifted from
campaign_v2_proj_best.py before its archival):** a NEAR-CONVERGED seed must
NOT inherit the ceiling-ride target. At BEST_T9636 the width-blind climb to
the ceiling bought +0.00097 T for +0.272 µm — 0.0036 T/µm, 30× below the
uniform lane's rate — because at a converged point ∇T is aligned with the
width direction and climbing just spends band for nothing. A best-seeded
lane sets `wgp_target_um` = the seed's own fwhm_env, so the constrained
(null-space) law engages from iterate 0.

### 6e. Two engineering results that made this affordable
- **The width adjoint runs on GPU via source tiling.** The full-width source
  was rejected by a per-source CUDA launch bound; splitting it into 4 narrow
  sources enabled in **one** solve is *exact* (sources superpose linearly, the
  gradient is linear in the adjoint field). MEASURED: **~1.8 h/gradient on GPU
  vs 8.7–12.1 h on CPU.**
- **`∇T` and `∇W` come from the same solved fields at zero extra cost**,
  because the gradient assembly is linear in the objective's Jacobian — re-run
  the assembly with a different objective selector and you get a different
  component out of the same physics.

---

## 7. Diagrams — what to draw, and how

Two pictures carry this whole method. They are the analogue of the standard
neural-network training diagram, and they answer two different questions.

### 7a. The optimization loop  *(the "training loop" picture)*

Same role as a forward/backward-pass diagram in a network: it shows what is
computed, in what order, and where the gradient comes from. The point to make
visually is that **one iterate = three solves and two gradients**, and that the
resonance-chain term rides along for free.

```mermaid
flowchart TB
    P["parameters p<br/>(191 values: corrugation, width,<br/>shift, comb, cavity)"]
    F["FORWARD solve<br/>full-wave FDTD"]
    M["measure<br/>T(λ) · λ_pk · mode profile I(x)"]
    A1["ADJOINT 1<br/>source = port mode"]
    A2["ADJOINT 2<br/>source = dsoftW/dI<br/>(peaked at the half-max crossings)"]
    G1["∇T<br/>191-vector"]
    G2["∇W<br/>191-vector"]
    C["+ resonance chain term<br/>gλ = dλ_pk/dp<br/>(0 extra solves)"]
    S["choose the step<br/>CLIMB · RIDE · RESTORE"]
    U["p ← p + step<br/>clip to bounds"]

    P --> F --> M
    M --> A1 --> G1
    M --> A2 --> G2
    M -.->|"two selector passes<br/>over the SAME fields"| C
    C --> G2
    G1 --> S
    G2 --> S
    S --> U
    U -->|next iterate| P
```

**What a reader should take from it:** the two adjoints are *parallel and
independent* — that is the visual argument for why one cost function cannot
work. If T and W were combined into a scalar, there would be only one adjoint
box, and no way to construct a width-preserving direction downstream.

### 7b. The projection geometry  *(the picture that actually explains the method)*

This is the money diagram. Draw it in the 2-D plane spanned by `∇T` and `∇W` —
a slice through the 191-dimensional space:

```
            ↑ ∇W  (direction that widens the mode fastest)
            │
  W = W_hi  ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  ← the CEILING (hard spec)
            │
            │        ∇T ↗            ← raw transmission gradient:
            │       ↗                   climbing it walks INTO the ceiling
            │      ↗
  W = W_tgt ├╌╌╌╌╌●━━━━━━━━━▶ d      ← the PROJECTED step: the component of
            │     ┊         ↖           ∇T with all of its ∇W content removed.
            │     ┊          ╲          ∇W · d = 0  EXACTLY.
            │     ┊           ╲      ← what was subtracted: (∇T·û)û
            │     ┊
            │   contours of constant T ──────
            └──────────────────────────────────────────→
                                          (all other directions)
```

**How to draw it, concretely:**
1. Horizontal axis = "everything else"; vertical axis = `∇W`, the width
   direction. Any 2-D slice is a lie in 191-D, but *this* slice is the honest
   one, because the projection only ever acts in the plane of `∇T` and `∇W`.
2. Draw two horizontal dashed lines: the **target** width and the **ceiling**
   (target + margin). Shade above the ceiling as forbidden.
3. Draw `∇T` as an arrow with a clear upward component — that is the whole
   problem in one stroke: *the direction that most improves transmission also
   widens the mode.*
4. Draw the projected step `d` as strictly horizontal. Draw the removed
   component as a faint vertical arrow, labelled `(∇T·û)û`.
5. Optionally add faint contours of constant `T` so the reader sees `d` still
   climbing them, just more slowly than `∇T` would.

**The one sentence the diagram must land:** *we give up some transmission per
step in exchange for spending exactly zero width.*

### 7c. A third panel worth having — the failure being fixed

To show the resonance-chain defect visually, draw the **same** geometry twice
side by side:

- **left, "what the optimizer believed":** `∇W` drawn at fixed wavelength, and
  `d` correctly perpendicular to it.
- **right, "what was true":** the *real* width gradient rotated away from the
  drawn one by the unpriced `(dW/dλ)(dλ_pk/dp)` term — so the same `d`, which
  looks perpendicular on the left, has a visible upward component on the right.

That single rotation is the entire defect, and it explains why the width crept
up by a small amount every iterate while the projection reported `∇W·d = 0`.

### 7d. Data figures worth plotting (from stored logs, no new simulation)
- **W against λ_pk**, both baselines, with fit lines — the coupling finding.
  Data and slopes come straight from `gates/derive_dwdlam.py`.
- **Mode envelope `I(x)`**, uniform origin vs `BEST_T9636`, overlaid, with the
  FWHM marked on each — shows the width was *kept* while loss fell 69%.
- **T(λ)** for the same pair — shows the resonance sharpening.
- **Width trajectory per iterate**, uncorrected control vs corrected run — the
  before/after of the fix, once a corrected run exists.

---

## 8. Where each track stands right now

**Track A — validating, and running as of this writing.** The design is being
confirmed under the spec mesher and pushed along an `N`-ladder toward the
production device. Two jobs were RUNNING on IGUM at the time of writing
(`63540_3`, `63595_4`). The conformal re-measure already gave **T 0.97805 at
N=100** with the mode width kept, and the mesher ranking-transfer question that
gated quoting conformal numbers has been settled.

**Track B — the constrained optimizer is not yet delivering.** Honest status:
the projected method runs, the GPU width gradient works, but the loop has not
yet produced a width-controlled improvement we would stand behind. The most
recent understanding — that a large part of the observed widening was the
**resonance drifting**, and that the width gradient was being evaluated at a
frozen wavelength and so could not see it — has a correction implemented and
gated offline, but it **has not completed an iterate on hardware**. Treat it as
unproven.

★**This does not weaken Track A.** The device stands on its own measurements.

---

## 9. What is genuinely open

- **Can a machine-driven run reach a `BEST_T9636`-class design on its own?**
  If yes, the earlier stalls were mispriced constraints, not a rugged
  landscape.
- **Is there a *family* of equally good designs, or is this a needle?** Its
  corrugation profile drops abruptly at the edge of the free region — physical,
  or an artifact of where we froze the parameters?
- **Can transmission rise at genuinely fixed resonance?** ★Not answerable from
  the runs we have: transmission and resonance wavelength are 0.996-correlated
  in them, so the two cannot be separated. The corrected optimizer is precisely
  the experiment that decides it — **and a negative answer is a real result**,
  telling us the two are physically locked for this device.
- **Will `BEST_T9636` survive the production device** at the full period count
  and the fine mesh, outside the optimizer's own builder?

---

## 10. ★ The route we have NOT taken yet — a proper augmented Lagrangian

§4b rejected a *fixed-μ penalty*. An **augmented Lagrangian** is not that, and
it is the strongest alternative to the projection method. It deserves a fair
statement, because parts of it are already built.

### 10a. What it is, and why it escapes §4b's Reason 1
Instead of guessing an exchange rate, AL **learns the correct one**. It carries
explicit multipliers `λ_hi`, `λ_lo` alongside a quadratic term:

```
J = J_T − [ λ_hi·max(0, g_hi) + ½μ·max(0, g_hi)²
          + λ_lo·max(0, g_lo) + ½μ·max(0, g_lo)² ]

  g_hi = fhat − 1.02·f0        (over the band)
  g_lo = 0.98·f0 − fhat        (under the band)
```

After each inner solve the multipliers are updated on the **measured**
violation:

```
λ_hi ← max(0, λ_hi + μ·g_hi)          (and likewise λ_lo)
```

That update is the whole point. At convergence `λ` equals the true shadow price
of the constraint — the exchange rate is *discovered*, not assumed. Reason 1 of
§4b dissolves. And unlike a plain penalty, AL does not need `μ → ∞` to enforce
the constraint exactly, so it stays well-conditioned.

### 10b. What is already implemented
`width_band_penalty` (`:1811`) and the multiplier update (`:2472`) exist, and
the knobs are on `CampaignSpec`: `wg_mu = 8.0` (per µm²: 0.05 µm over-band ⇒
0.01 FOM), `wg_lam_hi = wg_lam_lo = 0.0` initially. So the *ingredients* are
there; what is missing is the outer loop that makes it an AL method rather than
a penalty with an unused multiplier.

### 10c. ★ What would need to be done — concretely
1. **Fix defect #19 first.** ★AL uses `∇W` exactly as the projection does, so
   it inherits the *same* frozen-wavelength error. An AL run on the uncorrected
   gradient would chase a constraint it is mis-measuring, and would fail in the
   same direction. **This is a prerequisite, not a detail.**
2. **Build the outer loop.** Inner solve to loose tolerance → update `λ` on the
   measured violation → tighten. Currently the update fires per restart, which
   is incidental rather than a schedule.
3. **Escalate μ only on stall.** Standard rule: if the violation did not fall
   by ~25% over an outer iteration, `μ ← 2μ`; otherwise leave it. Escalating
   every round destroys conditioning.
4. **Decide inner tolerance.** AL is only cheap if the inner problem is solved
   loosely early on. With ~2.4 h per iterate, the natural budget is 3–5 inner
   iterates per outer round.
5. **Keep the honest readout.** Multipliers must be updated on **measured**
   `fwhm_env`, never the surrogate — that is precisely how the σ̂ wall went
   wrong (§4a).
6. **Reason 3 still applies.** AL fixes the *exchange rate* problem, not the
   *rank-deficiency* problem. If the penalty is written on a quantity blind to
   a direction (the see-saw case), AL will be blind to it too. Write the
   constraint on the honest width, not a reduced surrogate.

### 10d. How to choose between AL and the projection
They are not really rivals; they answer different questions.

| | projection (current) | augmented Lagrangian |
|---|---|---|
| width held | exactly, to first order, every step | approximately, converging |
| exchange rate | never needed | discovered via `λ` |
| cost | 2 adjoints/iterate | **1 adjoint/iterate** — a combined scalar |
| best when | the spec is hard and you want to ride the ceiling | you want the true trade-off curve |

★**The one-adjoint saving is real and is AL's strongest argument** (~−33% per
iterate). But note it is a *consequence* of recombining into a scalar — and
therefore it is **incompatible with the projection**, which needs `∇T` and `∇W`
separately to build the null space. Choose the formulation; you cannot have
both the null-space guarantee and the single-adjoint cost.

**Recommendation for whoever picks this up:** validate the corrected gradient
on the projection first (it is instrumented, gated, and one short run from an
answer). If the projected `‖∇T‖` collapses — i.e. transmission and width really
are locked — then the trade-off *curve* is the interesting object, and AL is
the right tool to map it.

---

## 11. ★ What happens next

Read this together with §8 (where each track stands) and §9 (the open
questions). This section is the *plan*, ordered, with **who can actually do
each step** — that matters, because not every reader of this document has the
same powers.

### 11a. Who can do what

| capability | Claude in a chat window | Claude Code session | must be a human |
|---|---|---|---|
| reason over the data in the appendix | ✅ | ✅ | |
| design the next experiment | ✅ | ✅ | |
| read repo files / run the gates | ❌ | ✅ | |
| ssh to Athena or IGUM, dispatch, fetch | ❌ | ✅ | |
| commit, deploy | ❌ | ✅ | |
| approve a §2 numerics change | ❌ | ❌ | ✅ |
| decide the formulation (projection vs AL) | ❌ | ❌ | ✅ |

★**If you are reading this in a chat window, you cannot run anything.** That is
fine — most of the valuable work left is *analysis and design*, and the
appendix was built precisely so you can do it without tools. See §11d.

### 11b. The ordered sequence — for whoever has cluster access

**1. Fetch the IGUM results first.** ⚠️ *Before anything else.*
The conformal / q3db ladder was still running at the pause (jobs 63423, 63438,
63540, 63595). Those results exist **nowhere else** — a cluster holding the only
copy of anything is the one situation this programme treats as an emergency.
They belong to Track A, the deliverable.

**2. Resolve the `bragg_device.py` mesh question.** ⚠️ *Blocking for Track B.*
A parallel session changed the fine-mesh y-span to size from
`max(width_wide_per_tooth_m)` rather than the scalar width. It is a genuine bug
fix — the old behaviour ate 448 nm of PML standoff and inflated T above 1 — but
it is a **§2 named-numerics change** on a shared file, and the stored control
(job 137075) ran *before* it. For the current seed both widths agree (0.9625 µm)
so the domain is unchanged there; divergence appears only once a tooth is drawn
wider than the scalar, which is exactly what a per-tooth optimizer does.
**Needs: a scene-snapshot diff against the committed references, and a decision
on whether the control must be re-measured.**

**3. Run the offline gates.** Six of them, all local, all seconds, zero GPU.
They must all pass before any dispatch. Expected outputs are stated in
`HANDOFF.md`.

**4. Dispatch the 3-iterate validation toy.** ~9 h, one task.
★**This is a prerequisite, not the first item in a queue.** The production
campaign is configured and ready at 30 iterates — roughly 81 GPU-hours — on a
gradient that has **never completed a single iterate**. Do not skip to it.

**What the toy decides**, in order of what to look at:
- Does the resonance-chain term *execute*? Look for `gLam_n` present in the
  proj log with **no** `λ-CHAIN SKIPPED` line. If it skipped, nothing else in
  the run means anything.
- Does predicted `gλ·dp` match the measured `Δλ_pk`? The control drifted about
  **+0.04 nm per iterate**; a correct chain term should predict that.
- Does `ΔW` per iterate fall below the control's **+0.0110 / +0.0122 µm**?
- ★**The falsification test:** does the projected `‖∇T‖` collapse toward zero?
  If it does, transmission and width are **genuinely locked** for this device.
  That is a real physical result, not a failure — and it arrives in ~5 GPU-hours
  instead of a wasted multi-day campaign.

**5. Only then, the production campaign** — and only if step 4's verdict
supports it.

### 11c. Decisions that need a human

- **Projection or augmented Lagrangian?** (§10) They are not interchangeable:
  the projection guarantees zero first-order width change and costs two
  adjoints; AL discovers the true exchange rate and costs one. You cannot have
  both. Recommendation in §10d: validate the projection first, because it is
  one short run from an answer — and if that answer is "locked", the trade-off
  *curve* becomes the interesting object and AL is the right tool to map it.
- **Is Track A's device final?** If the answer is yes, Track B's remaining
  value is scientific rather than practical, and the priority order changes.
- **How much more GPU time is this worth?** Track B has consumed a great deal
  and has not yet produced a width-controlled improvement.

### 11d. What a chat session can do right now, with no tools at all

The appendix contains the real data, so these are all genuinely available:

1. **Re-read the earlier results through the λ-detrend lens.** Every past
   conclusion about "this change widened the mode" was drawn before we knew
   that width tracks resonance at ~0.37 µm/nm. Some of those conclusions are
   probably wrong. The tables in A4 are enough to re-examine them.
2. **Interrogate the design vector in A1.** The corrugation profile, the shift
   distribution, the comb spacing — is the freeze-boundary discontinuity at
   tooth 26 costing anything? Is the shift profile doing what a taper should?
3. **Design the next experiment on paper.** What is the smallest run that
   distinguishes "T and W are locked" from "the optimizer has not found the
   right direction"? Specify it precisely enough that a Claude Code session can
   dispatch it without re-deriving anything.
4. **Sanity-check the method itself.** The derivations in §5, §6 and §10 are
   all written out; a careful reader may well find something wrong. This
   programme has repeatedly been saved by someone checking the algebra rather
   than the code.
5. **Write.** The physics story here — a constraint that turned out to be
   mostly a proxy for something else — is a genuinely interesting result and is
   not yet written up anywhere except these documents.

★**What a chat session should NOT do:** invent numbers, assume a run happened,
or claim the λ-chain fix works. It has never completed an iterate on hardware.
Everything about it in this document is *implemented and gated offline*, which
is not the same as *validated*.

---

## 12. Pointers

| for | read |
|---|---|
| jobs, numbers, resume commands | `HANDOFF.md` (top box) |
| the 191-vector of every named design | `best_designs.py` |
| offline gates, all zero-GPU | `gates/` |
| project invariants and the trap list | `../../CLAUDE.md` |
| the defect history in full detail | `HANDOFF.md`, and the memory file `project_v2_width_gradient_plan.md` |
