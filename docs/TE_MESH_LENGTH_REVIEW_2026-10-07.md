# TE mesh, length and spatial-width review

Updated October 7, 15:20 Israel. Production remains TE, N98 per side, 60 free
inner periods per side, no surrounding posts, with the existing seed width
bands. **Current mesh/length convergence is not yet established.**

## What the evidence says

| Question | Verified evidence | Conclusion |
|---|---|---|
| Is the working mesh about 50 nm? | Current S2 saved axes give 49.106 x 50 x 49.3626 nm; 2676 x 69 x 70 coordinates in the symmetry-reduced domain | Yes, approximately. Nominal settings alone are insufficient. |
| Is 50 nm mesh converged? | An older uniform N80 workbook changed T from 0.8226 to 0.7929 and resonance from 1560.259 to 1555.864 nm when x spacing changed 50 to 25 nm | The old blanket convergence claim is unsupported. These are different materials/meshing from current S2, so this does not measure today's error. |
| Was the transverse box tested? | Raw uniform N98 box sweep: 6.8 x 6.8 to 12 x 12.8 um changes width 19.128691 to 19.151675 um, T 0.9116845 to 0.9118256, resonance by +0.315069 nm | Small width/transmission sensitivity in that conformal family; resonance is more sensitive. Not a PVA/current-shape certificate. |
| Does width saturate with length? | Uniform corrugation250 ladder N166 to215: width20.455916 to20.537896 um while T0.491903 to0.088366 | Width can stabilize while mirror coupling and transmission change strongly. Length is a physical design variable. |
| Is N98 the shortest suitable device? | Native S2 CSV reports N98 to130 width19.633 to19.713 um, but N130 raw profile was not found locally. No matched current S2 N80 comparison exists | No minimum length has been established. Longer-length evidence does not prove shorter-length suitability. |

N98 means 98 periods **on each side**, not 98 total. S2's physical grating is
about96.49 um long, with a131.36 um full simulation domain. N80 retains the
inner60 and removes18 outer bulk periods on each side; the apodization must
remain indexed from the cavity.

The independent audit selectively read18 MAT files (including one TM reference)
and the original mesh workbook. All18 stored envelope FWHMs reproduce exactly,
and saved T arrays equal abs(S21)^2. The parent repeated the extraction and
obtained identical inventory data. This is validation of saved processed
profiles, not reintegration of every large raw E array. Larger files were
identified by metadata and selected contents; full-file SHA was deliberately
not claimed for those files. Exact paths and qualifications are in the archive.

## Efficient next checks

1. Recover the broadband field_profile data from existing solved100/150 ps
   center files. The diagnostic disabled optimization_dft; source inspection
   shows field_profile is a separate recorder. Actual saved data availability,
   wavelength and coordinates must still be checked. A guarded server reader
   is being prepared; no new baseline solve is needed if this succeeds.
2. Finish the already-running150 ps step pair171276 and compare both h values
   using the same peak3 definition, exact grids, full spectra and fixed-center
   Jacobian projections. Existing differences are not a convergence/noise bound.
3. After reviewing duration adequacy, compare one consistent x refinement on
   the same current point. Global and locked-region x settings must agree;
   dy/dz remain fixed for this x-sensitivity test. Record realized mesh, dt,
   source/monitor/PML geometry and tracked resonance. Two levels establish
   sensitivity, not an asymptotic convergence order.
4. Prepare an N80 comparison with the same inner vector and initially the same
   N98 domain, source/ports and monitor coverage. This isolates physical
   truncation. Only if its width/mode remains suitable should a separate
   compact-domain comparison measure the possible speed benefit.

The production FWHM uses the y-integrated intensity-like profile, a cubic
envelope through standing-wave peaks, and a half-height relative to the floor
within the grating crop. Shortening N changes that crop. Record the normalized
floor, endpoint levels and shape; supplemental common-window/floor diagnostics
must not replace the production observable. An endpoint intensity is not a
modal electric-field amplitude or a radiation-loss measurement.

At fixed duration, measured-cell/CFL estimates suggest x refinement costs
about1.61 times the current work (roughly81-105 minutes using a50-65 minute
baseline). Simultaneous y/z refinement to25 nm could cost about6.78 times and
substantially more memory. These are estimates, not benchmarks. Compact N80
at unchanged edge gaps saves only about13.5% of domain cell work; free60
field-recording costs do not shrink proportionally. The initial fixed-domain
N80 test has essentially no domain-cell saving.

## Acceptance and limitations

Report absolute/relative changes in both Tout and modal transmission, resonance,
spectral linewidth, production spatial FWHM, profile shape, and realized
numerics. Report membership in the existing width band separately from
strict no-widening and separately from numerical convergence. No new accuracy
tolerance or relaxed width band has been adopted.

Reliable gradients on the exact working discretization are necessary before
an optimizer smoke. Fully mesh-converged continuum gradients are a separate
claim. Coarse-grid gains need refined/device confirmation before being called
physically robust. Mesh/length checks do not bypass the failed calibration gate.

Ansys likewise separates mesh, PML, source and monitor errors and recommends
choosing an application-relevant accuracy target. Its general guidance does
not prescribe this device's pass threshold. [Ansys convergence guide](https://optics.ansys.com/hc/en-us/articles/360034915833-Convergence-testing-process-for-FDTD-simulations).

Verified text-only Claude Opus5.5 reviewed the plan through the installed
[ask-claude skill](C:/Users/evyat/.codex/skills/ask-claude/SKILL.md). Useful points
were crop/floor sensitivity, cavity-centered indexing and fixed-grid versus
continuum accuracy. Corrections to its response: CFL dt does not itself move
the configured monitor frequency bins; own-peak3-J is the differential of the
same functional, and a finite moving central difference need not equal it;
a shorter crop does not prove its floor rises; x-only refinement does not
require changing dy/dz. Its recommendation to leave all decisions to the user
does not override already-authorized routine work. No new solve was released
by that consultation.

Evidence: `readiness_evidence_20261007/te_mesh_length_reviewed_1520.zip`, including
the full audit, measured inventory, parent equality check and corrected review.
