# Literature check — width-constrained cavity inverse design (2026-08-29)

Researched by background agent, reviewed by Fable. Question: is our approach
(maximize resonance T at held envelope FWHM via projected adjoint gradients
with the resonance-shift chain term) sound, and what do others do?

## Headline findings

1. **Our defect-#19 chain-term fix is independently confirmed as THE key
   ingredient** — Shaker, Martinez de Aguirre Jokisch, Chao, Johnson
   (Nov 2025, arXiv:2511.16643): re-solve the resonance each iteration and
   include dF/dp = ∂F/∂p|_ω + (∂F/∂ω)(dω*/dp). Omitting the chain term
   leaves O(Q²) ill-conditioning in the Hessian; including it dropped their
   dominant eigenvalue ~10⁵→~1 and converged ~1000× faster. At our Q≈14-16k
   we are squarely in that regime. Also explains why the T-side stop-gradient
   is safe (∂T/∂λ=0 at the peak — envelope theorem) while the W-side chain
   was fatal (W is not extremal in λ; measured dW/dλ=+0.37 µm/nm).
2. **~30 iterates is EARLY, not slow** — published adjoint cavity
   optimizations run 100-1000+ iterations (L-BFGS-B stalls ~100, MMA ~200;
   Asano & Noda 101 its to Q=1.1e7; Shaker et al. up to 5e5 CCSAQ its).
3. **Constraining envelope FWHM specifically has NO published precedent** —
   the field constrains mode VOLUME (Işıklar et al. 2022: min V s.t. Q-bound,
   Pareto family). Our FWHM spec is novel; no exchange-rate numbers exist to
   compare against.
4. **"Fast then rescale" has direct precedent** — Sauerzopf et al. 2025
   (arXiv:2510.27476): grating couplers retuned by uniform geometric scaling,
   objective SURVIVES. Caveat for us: pitch-only rescale with fixed 350 nm
   height + n(λ) dispersion is NOT the exact scale-invariance trick — a
   one-parameter retune to be measured, not assumed (task 49 measures it).

## Ranked implementable changes

1. **CCSAQ/SLSQP with the width as an explicit nonlinear constraint**
   (replaces the hand-rolled projection; same two adjoints, zero extra
   solves; what Shaker et al. and MEEP use for this problem class).
2. **If keeping projection: add Feppon's range-space restoration term**
   (Feppon, Allaire, Dapogny 2020, ESAIM:COCV 26:90 + open-source
   `nullspace_optimizer`): our ride step is only the null-space half (ξ_J);
   a range-space ξ_C term makes constraint violation decay exponentially
   instead of drifting. ~20 lines; our "restore" phase is a crude version.
3. **Global scale as a 192nd parameter** — the cheapest λ-restoration
   direction, near-orthogonal to shape; turns fast-then-rescale into a
   continuous lever. Verify against dispersion first.

NOT recommended: switching the width observable to participation-ratio /
softmax-V — no precedent, and it abandons the spec's FWHM convention.

## Sources
- arXiv:2511.16643 (resonance-tracked LDOS optimization, chain term + CCSAQ)
- arXiv:2510.27476 (rescale-retuned grating couplers)
- Feppon et al., ESAIM:COCV 26:90 (2020), hal-01972915 (null/range-space)
- Işıklar et al., Opt. Express 2022 (Q-V trade-off, DTU)
- Wang et al., APL 113:241101 (2018), arXiv:1810.02417 (Q/V bowtie)
- MEEP adjoint tutorial (CCSAQ/epigraph practice)
- arXiv:2310.15751 (eigenvalue shape derivatives)
- arXiv:2303.15070 (quasi-Newton TO pace)
- Asano & Noda, Nanophotonics 2018 (101-iteration L3 cavity)
