**INFERENCE — Stop adding antenna geometries for now. You do not yet have a feasible cavity optimum or a reliable linear response model. The point-current reach map is being treated as a fabrication prescription; that is the main conceptual mistake.**

1. **What is wrong or weak.**  
   **INFERENCE:** The 950 nm cavity violates the previously specified \(|\Delta\lambda|\le0.25\) nm and \(|\Delta W|\le0.1\,\mu m\). The 270 nm cavity violates both more strongly. They are useful measurements, but not admissible improvements unless retuning or different bands are explicitly included.  
   A **0.64 depth-linearity residual invalidates the shoulder Jacobian** at those depths. Excellent left/right additivity does not repair it: two independently nonlinear responses can still add. Establish whether this is physical nonlinearity, geometry-dependent meshing, or normalisation error. Lorentzian correction alone does not establish constant stored-energy normalisation.  
   The source fit demonstrates representability, not six independently controllable emitters. The 76–86% selectivity belongs to prescribed point currents; fabricated features acquire their amplitudes and phases from the actual cavity.  
   **FROM-SOURCE / INFERENCE:** Mirror selection requires symmetry of the electromagnetic system and its excitation. Because this cavity lacks x symmetry, mirrored *geometry* does not enforce the claimed x-even/x-odd current combinations. [Vector mirror conventions](https://meep.readthedocs.io/en/latest/Exploiting_Symmetry/). Also, \(l\sim kx_0\) is a scale, not a minimum reachable order.  
   The shoulder’s component outside the width/length span is encouraging, but does not prove useful new control: it could be numerical error, unwanted radiation, or a response reproducible by other existing parameters.

2. **Establish the baseline before measuring the antenna basis.**  
   **INFERENCE — first two new rows:** matched finer-mesh plain control and right-shoulder 10 nm. Require the improvement to survive beyond measured uncertainty; inspect complex response and actual stored energy. Use the existing 5/10 nm pair to diagnose, rather than conceal, the failed linear model.  
   Then finish the pending width/length rows and retain only correctly normalised, feasible points. Fix the wavelength/width bands and whether pitch retuning is allowed; give baseline and antenna designs the same retuning freedom.  
   **Width alone is insufficient.** Start with \(q=(w_{\rm cav},L_{\rm cav},c_{\rm inner})\), where \(c_{\rm inner}\) is a precisely defined common contrast change on the innermost period pair. This is a bounded local baseline, not a global optimum of all existing inverse-design parameters.  
   A local quadratic in width×length needs **six geometries plus one independent optimum check**; adding inner contrast needs **ten plus one**. Reusing three suitable width points leaves approximately **four or eight new rows**, respectively. These counts establish a local model, not guaranteed convergence.  
   For the three-variable case: measure ±length, ±contrast, then width–length, width–contrast and length–contrast mixed probes; simulate the predicted feasible optimum. Reuse valid existing axial probes. Choose length increments from the measured detuning slope—±20 nm is too large for a local model under the original bands.  
   If the validation disagrees materially, narrow the region and refit. Call the result **“best validated cavity in this parameter box”**, not “optimal TE cavity.”

3. **Separate new shape control from cavity retuning correctly.**  
   **INFERENCE:** Compare
   \[
   J_0^*=\min_{q\ {\rm feasible}}P_{\rm rad}(q,0)/U,\qquad
   J_{\rm feat}^*=\min_{q,\alpha\ {\rm feasible}}P_{\rm rad}(q,\alpha)/U .
   \]
   “Gain survives reoptimising the cavity” is not an independent test: if the feature point is feasible, allowing further optimisation cannot worsen its best attainable value. The meaningful comparison is the two constrained optima, with comparable search effort and verification.  
   Demonstrate that the feature introduces geometry absent from the existing parameterisation, then show selective **complex-coefficient** changes and a verified incremental gain. That establishes new control; it does not establish physics unavailable to inverse design in principle.  
   The TM see-saw is a useful experimental template—baseline first, incremental feature second. Its stored 1% gain is not a TE pass threshold, and its tooth modifications belong to the excluded baseline family here.

4. **Cheapest designed single-harmonic test after the baseline.**  
   **INFERENCE:** Prefer **E(3,±3)** first: the symmetry-linked pair contains about 20% of the supplied plain-control power, versus 13% for M(1,0). Recompute these fractions at the optimised cavity; do not carry over the old target ranking or response columns.  
   Measure **four independent local shoulder/notch directions** at that parent, initially around \(\pm a,\pm3a\), selected for independent predicted responses rather than imposed x parity. One complex target plus λ/W gives four real constraints generically; rank, bounds and unwanted radiation can still make four knobs insufficient.  
   Solve for target reduction while minimising total radiation and limiting untargeted power. Use actual measured addition/removal directions; the failed depth gate forbids treating a finite notch as automatically minus a tab.  
   Validate the solved geometry at full and half amplitude: **six new rows minimum**, plus any missing linearity/control rows. Proposed passes: response selectivity \(\|P_t b\|^2/\|b\|^2\ge0.70\); target **absolute power falls ≥50%**; untargeted absolute power does not increase beyond measured uncertainty; λ/W hold; full/half complex residual ≤20–25%. Confirm the survivor on a finer mesh.  
   Report both target suppression and total reduction. A falling harmonic *fraction* or a selective response without destructive phase is not cancellation.

5. **A stopping rule that prevents another open-ended search.**  
   **INFERENCE:** Pre-register one expansion to at most **6–12 validated local knobs** at the optimised parent. Stop this TE harmonic-antenna programme if its uncertainty-aware optimistic bound cannot deliver a useful incremental gain—suggest **10% total radiation reduction**—or if predicted selective combinations repeatedly fail direct full/half and finer-mesh tests.  
   If the apparent gain disappears against an adequately searched baseline, classify it as cavity improvement and stop calling it an added-antenna result. This would close the tested route decisively; it would not prove that every possible TE perturbation is incapable of harmonic cancellation.
