# THE DESIGN, AND THE METHOD — pi-shift Bragg grating, inverse design

**What this file is.** `HANDOFF.md` is *state* — jobs, numbers, what to run
next. This is the *explanation*: the device we have built, why we are doing
inverse design at all, why a single cost function provably cannot do this job,
and what our algorithm actually does instead.

Figures to be added later. Every number is **MEASURED** (from a named file),
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

## 4. ★ Why a single cost function cannot do this — the central lesson

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

## 5. ★ What separate objectives buy — and why it costs a second adjoint

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

## 7. Where each track stands right now

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

## 8. What is genuinely open

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

## 9. Pointers

| for | read |
|---|---|
| jobs, numbers, resume commands | `HANDOFF.md` (top box) |
| the 191-vector of every named design | `best_designs.py` |
| offline gates, all zero-GPU | `gates/` |
| project invariants and the trap list | `../../CLAUDE.md` |
| the defect history in full detail | `HANDOFF.md`, and the memory file `project_v2_width_gradient_plan.md` |
