# TE objective and adjoint source facts — 2026-10-07

Read-only audit for the objective reassessment. No source, CAD, server, job or
calibration changes. Only this note was added. Frozen production engine SHA256:
`8f09350583a53146091faf590c2c93e8d93ac01a46b48006890db62c72a295e5`.
The isolated restart candidate remains separate; existing readiness gates apply.

## Active objective and wavelength policy

- `SPEC_V3` selects `make_fct_peak`: a three-point frequency-grid parabola of
  **abs(T_out)** at the selected defect resonance. The callback's `t_pk` is
  sampled **abs(S21)^2**; it additionally logs each observable's peak3 value.
  These are distinct, geometry-dependent observables, not related by an exact
  fixed factor. Sources: `lumopt2_design.py:859–875,1759–1781,2169–2187`;
  `TE_RECOVERY_2026-10-06.md` reports the exact stored-spectrum reconstruction.
- Width acceptance uses the y-integrated production spatial-envelope FWHM;
  `softW` is a differentiable steering surrogate. Existing bands are S1
  18.73858–19.50342 µm and S2 19.24328–20.02872 µm. A literal no-growth ceiling
  differs from these +2% upper bounds.
- Current v3 already allows cumulative wavelength drift: its wavelength row is
  a **per-step local trust bound**, S1 ±0.25 nm and S2 ±0.05 nm, with scan
  recentering. The old ns2 global latch is not the v3 equation. Removing this
  local guard would weaken model validity, even without a physical wavelength
  target. Sources: `THEORY.md:757–815`, `campaign_te_s1.py:135–142`,
  `campaign_te_s2.py:110–117`, `lumopt2_design.py:2942–2970`.

## Actual configuration and solve count

The live code supplies no `ProjectConfig` to either result entry:
`PortResults("Port_2",...,501 wavelengths)` (`lumopt2_design.py:1640`) and
`WidthResults` (`:2360`). Both therefore have `config=None` (installed
`lumopt2/fom/simulation_results.py:45–53,182,283`). Installed
`utils/config_map.py:109–145` groups them into **one** configuration:

| Files per fresh geometry | Role |
|---|---|
| `fwd_default.fsp` | Broadband Port_1 forward; also records production profile and single-frequency width twin. |
| `adj_default_Port_2.fsp` | Broadband reverse port-mode adjoint, with port offset/phase handling. |
| `adj_default_field_profile_adj.fsp` | Weighted `conj(E_forward) * dsoftW/dI * y-quadrature` source, four x tiles in one GPU solve. |

`Project.set_fwd` loops configurations (`project.py:536–570`); `set_adj` saves
one file per result entry (`:595–627`). There is **no live `wg_single` config**.
No current second forward is required merely because the twin is
single-frequency. Source-derived cost is `tf_default + ta_port + ta_width +
assembly/CAD`, not three equal-duration units. Reuse removes `ta_width` and its
assembly; wall-clock savings depend on measured durations and rejected trials.

Port/production grids have 501 frequency-uniform samples: S1 10 nm scan,
S2 2 nm. The width twin has one point, explicit center, zero span and source
limits off. Its center follows the **previous evaluation's modal resonance**,
including rejected evaluations, and can lag. Tiled source wavelength follows
the twin's actual result. Sources: `lumopt2_design.py:770–778,1347–1415,
1772–1777,2355–2371,2414–2499`; inherited GPU setting
`campaign_v2_proj.py:100–126`.

## Which gradients cost extra solves

`MixedFom.calculate_gradient_fields` (`:2505–2558`) assembles gT, two port
selector rows and gW separately from those two solved adjoints. The selectors
at approximately half a spectral linewidth around the modal peak cost **zero
additional FDTD solves**. Their T_out-based denominator yields
`gLambda = -(gThi-gTlo)/(Tprime_hi-Tprime_lo)` (`:1786–1840,3345`). This is a
local translating-lineshape estimate; mixed modal/T_out peak definitions remain
a qualification. v3 forms `gW_res = gW_fixed + cW*gLambda`, with cW measured
from the same forward's spectral profiles (`:1889–1904,2951–2965`). Allowing
wavelength drift does not remove this moving-resonance width chain term.

The QP needs separate gT, gW_res and gLambda rows. One scalar combined gradient
cannot determine those independent rows without additional basis information.

## Reuse and historical combined-adjoint proposal

Live reuse requires ns2, k≥2, accepted stored gW, no dirty/reject state, age<k−1,
measured width drift ≤0.025 µm since the last fresh row, accumulated geometry
travel plus next cap ≤40 nm, and current width in the v3 inner band
(`:3228–3257`). Optional Broyden updates use twin softW and subtract known twin
wavelength motion (`:3258–3288`). Both TE v3 specs set k=0: toy169105 changed W
0.08–0.12 µm per step, preventing the reuse gate from opening.

Exact historical Claude sources (under the memory root recorded in
`.agents/skills/project-context.md`):

- `project_v2_width_gradient_plan.md:125–139`: TM two-point row-angle probe
  job139256, 0.685° per 10 nm; historical reuse/parallelization decision.
  `:69–79` corrects unsafe per-iterate extrapolation to a travel budget;
  `:11–16` records smoke139345 with 2/4 reused rows. This is TM evidence,
  not a TE reuse certificate.
- `project_te_inverse_design_lane.md:439–446,537–547`: TE v3 hardware smoke
  and toy169105; failed "never REUSED" marker and decision to disable reuse.
- `project_v2_width_gradient_plan.md:318–328` links the banked
  `AL_COMBINED_DESIGN.md` (2026-08-28, **design only**).
- `AL_COMBINED_DESIGN.md:18–55` proposes source superposition, source-specific
  complex calibration before summing, and retention of the width chain term.
  `THEORY.md:552–632` discusses a proper outer AL loop; §13 explicitly says
  historical fixed-penalty failures do not disprove scalar constrained methods.

Vendor source resolves the proposal's "verify first": `BaseFom` applies
`jac_of_fom` in **post-solve accumulation** (`base_fom.py:511`). Merely changing
MixedFom's scalar function still runs both adjoint files. A real merged source
must handle broadband port weights, the single-frequency lagged twin,
per-component phase/scaling and moving-resonance width sensitivity. Retaining
separate selector passes after merging is not automatically sufficient to
recover uncontaminated gLambda. These are engineering/validation requirements,
not a proof superposition is impossible.

The existing `width_band_penalty` (`:2144–2157`) is hinged linear plus quadratic
penalty, not standard shifted PHR AL. It is inactive under projected v3;
historical AL ingredients do not establish a tested combined-source AL optimizer.

## Cheapest defensible next step

Complete existing S2 peak3 calibration review, integrate the isolated restart
candidate after frozen dependencies finish, then inspect the already prepared
bounded **fresh-width** hardware smoke. This uses measured instrumentation and
does not need a new formulation. Saved fields can support assembly/diagnostics
without rerunning equivalent forwards; trial results must still pass measured
FWHM/MAC/peak validity gates. Conditional width reuse is the least invasive
future solve-saving experiment, but needs TE prediction-error evidence and a
fresh endpoint check. Port-only proposals with measured width rejection could
also be studied, but are not currently validated constrained optimization.

A future one-adjoint AL comparison requires a real merged-source implementation,
consistent chosen transmission objective, moving-resonance derivative, measured
width multiplier updates/outer schedule, source-superposition and FD gates, and
equal-budget feasible-endpoint comparison. It does not justify skipping current
scientific gates or starting a full campaign.
