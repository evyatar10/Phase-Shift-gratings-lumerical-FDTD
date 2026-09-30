# Calibration dossier — TE overshoot seed, plain TE pi-shift device, TM calibration points

Date: 2026-09-26. Everything below is labelled **MEASURED** (read from a named file this
session), **DERIVED** (computed from measured values), **EXPECTED** (theory/estimate), or
**ABSENT / NOT STORED**. Code facts carry `file:line`.

---

# A. Existing material

## 1. Geometry from the scene builder

Two devices. Neither uses tooth shifts; neither has PDMS anywhere in the repo.

### 1a. Plain uniform TE pi-shift device (baseline)

Pure `SimulationConfig()` defaults, no per-tooth arrays.

| quantity | value | source |
|---|---|---|
| avg corrugation width | 800 nm | `simulation_config.py:27` |
| corrugation depth | 300 nm | `simulation_config.py:28` |
| → width_narrow / width_wide | 650 / 950 nm | derived, `simulation_config.py:54-61` |
| access/feed waveguide width | 1000 nm | `simulation_config.py:30` |
| core height | 350 nm, z-centred at 0 | `simulation_config.py:29`, `bragg_device.py:847,863-864` |
| pitch | 500 nm | `simulation_config.py:92` |
| n_periods_each_side | 80 | `simulation_config.py:93` |
| duty cycle | 50% (half_pitch narrow + half_pitch wide) | `bragg_device.py:1114-1122` |
| apodisation | **OFF** — uniform depth every tooth | `simulation_config.py:194`, `bragg_device.py:1060` |
| phase-shift / cavity segment | pitch/2 = 250 nm, width = avg = 800 nm | `bragg_device.py:186,1128`; `cavity_neg_detuning_nm=0` at `simulation_config.py:94` |
| tooth shifts | **ABSENT** (all zero) | `simulation_config.py:97,110`, `bragg_device.py:1089-1092` |
| n(SiN) / n(SiO2) | **1.97 / 1.444**, constant, dispersionless | `simulation_config.py:253-254`, drawn as object-defined dielectric `:263` |
| PDMS | **ABSENT** — no PDMS anywhere in the repo | — |

The TE Q3dB variant of the same uniform device overrides corr → 250 nm, N → 166
(`runners/sweeps/te_q3db_20um.py:53-54`).

### 1b. TE overshoot / apodised seed (Itai HH)

`runners/sweeps/itai_hh_apod.py`, `runners/sweeps/itai_hh_asdrawn.py`.

**This is not an envelope-generated taper.** It is an explicit 61-value drawn
(narrow, wide) table, innermost tooth first: `APOD_NARROW_NM` at
`itai_hh_apod.py:89-98`, `APOD_WIDE_NM` at `:99-108`. Beyond tooth d=61 the arm is
uniform bulk (746.9, 1257.1) nm, i.e. Δw 510.2 nm (`:65`, `:144-145`).

**The overshoot:** Δw rises 0 → **1200 nm at d ≈ 24–26** (wide 1915.1 nm, narrow
577.0 nm), dips to 603 nm, a second lobe of 1188 nm at d ≈ 50–54, then bulk
(`itai_hh_apod.py:12-16`; peak asserted at `itai_hh_asdrawn.py:58`). Peak/bulk =
1200/510.2 = **2.35× overshoot**.

| quantity | value | source |
|---|---|---|
| avg width | 1000 nm | `itai_hh_apod.py:63` |
| cavity segment width | 950.3 nm | `itai_hh_apod.py:64` |
| cavity length | pitch/2 | `bragg_device.py:186` |
| N_SIDE | 98 periods/side (apodised core = 60 periods, `N_APOD`) | `itai_hh_apod.py:66-67` |
| indices | 1.97 / 1.444, unchanged | spec overrides no material field |
| tooth shifts | **ABSENT** | — |

Scaled rows: `teeth(n_side, scale)` (`:136-154`) preserves the shape, multiplies depth
by `scale`, then re-solves each period's mid width via `_solve_mid` (`:117-133`) so the
period-averaged n_eff stays at the scaled bulk value. Dispatched scales 0.52, 0.58 (TE)
and 0.72 (TM) (`ROWS`, `:199-202`). DERIVED this session for `teeth(98, 0.52)`: bulk
corr 265.3 nm, narrow 737.0–984.9 nm, wide 984.9–1432.8 nm, max Δw 695.8 nm at index 23.

Pitch is per-row, solved by `pitch_for(pol, scale)` (`:157-174`) from a 20 µm-FWHM
Gaussian-weighted ⟨n_eff⟩: TE scale 0.52 → **489.42 nm**. The as-drawn row instead uses
**Itai's own pitch 514.0 nm, unretuned** (`itai_hh_asdrawn.py:50`) with the full-depth
table (`:55-56`).

## 2. Vertical stack and PML distances

**The stack is simpler than the question assumes.**

- **Bottom oxide thickness: ABSENT as a drawn object.** The whole domain is a uniform
  oxide background — `set("background material", clad_material)` + `set("background
  index", n_clad_const)` (`bragg_device.py:764-766`). Only the core polygons are drawn
  (`:857-866`).
- `geometry.substrate_thickness_m = 10 µm` (`simulation_config.py:31`) is passed through
  (`:596`) and stored (`bragg_device.py:130`) but **used nowhere else in the builder —
  it is a dead parameter.**
- **Si substrate: ABSENT.** No Si material, no substrate object.
- **Top material: the same oxide background**, semi-infinite to the PML. No air, no
  separate top cladding.

Span computation:

- x: `x_grating_end = N*pitch + cavity/2` (`bragg_device.py:285`);
  `dist_grating_to_port = 20*pitch` (`simulation_config.py:228`, snapped to dx at
  `bragg_device.py:1306`); `dist_port_to_pml = 5.0*λ_B` with
  `λ_B = 2*n_eff_guess*pitch`, `n_eff_guess = 1.55`
  (`simulation_config.py:229,256`; `bragg_device.py:178,288`);
  `fdtd_x_span = 2*(x_port + dist_port_to_pml)` (`:289,333`).
- y: `y_span = max_drawn_wide + span_mult*λ_c` (`simulation_config.py:546`), where
  `_max_drawn_wide_m` (`:498-511`) takes the max of the per-tooth array (the 2026-08-26 fix).
- z: `z_span = core_height + span_mult*λ_c` (`:551`).
- `span_mult` default **1.8**, or 5.0 when far-field recording is on (`:495`).

DERIVED this session, plain TE: y_span **3.758 µm**, z_span **3.158 µm**,
x_grating_end 40.125 µm, x_port 50.125 µm, PML boundary at ±57.875 µm,
`fdtd_x_span` **115.75 µm**. Lateral PML clearance from the widest tooth edge
**1.404 µm**; vertical clearance from the core face **1.404 µm**.

The Itai rows override the box: `BOX_Y_UM=6.8`, `BOX_Z_MULT=4.14`
(`itai_hh_apod.py:192`) → y 6.8 µm, z ≈ 6.81 µm. The as-drawn row uses
`Y_SPAN_UM=9.0`, `SPAN_MULT=5.4` (`itai_hh_asdrawn.py:51`) → z ≈ 9.10 µm. DERIVED for
the TE scale-0.52 row: x_grating_end 48.09 µm, fdtd_x_span ≈ 130.9 µm.

## 3. Boundary conditions, mesh, mesh overrides

All six faces default to PML (`bragg_device.py:733-734`). **No PML layer count or
profile is ever set** — Lumerical's default (8 layers, "standard") applies.

Symmetry is polarization-driven (`bragg_device.py:740-750`), and the comment there states
the rule: the BC must match the parity of the injected mode's E field or the mode is
filtered out.

| polarization | y min bc | z min bc |
|---|---|---|
| TE (E along y) | **Anti-Symmetric** | **Symmetric** |
| TM (E along z) | Symmetric | Anti-Symmetric |

Both enabled by default (`simulation_config.py:293-294`); both devices here are TE and
use both planes. `force symmetric y mesh = 1` (`:742`) and `force symmetric z mesh = 1`
**always, BC or not** (`:750`) — the latter is the fix for the 178-vs-179-cell knife
edge that shifted n_eff by +0.0022 (+1.6 nm band shift, job 128925).

Global mesh (`bragg_device.py:772-789`): `mesh type = "custom non-uniform"`;
`dx = pitch/(2*cells_per_half_period)`; `dy = dz = 50 nm` hard-pinned; grading off in x,
on in y/z with factor 1.41421; `mesh refinement = "conformal variant 0"`;
`dt stability factor = 0.7`.

`cells_per_half_period`: `"optimization" = 5` → dx = pitch/10; `"accurate" = 7` →
dx = pitch/14 (`simulation_config.py:219-222,243`). Default `"optimization"` (`:230`) —
**both devices run there**. dx: plain TE **50.0 nm**; Itai TE s=0.52 (pitch 489.42)
**48.94 nm**; as-drawn (pitch 514) **51.40 nm**.

**One** mesh override region, `"mesh_override"` (`bragg_device.py:830-843`):

| axis | centre | span | step |
|---|---|---|---|
| x | 0 | `M*dx`, `M = ceil((fdtd_x_span + 2 µm)/dx)` parity-matched (`:807-811`) — the whole domain + 2 µm | global dx |
| y | 0 | `1.2 × max(width_port, width_wide, width_narrow, max per-tooth wide)` (`:823-825`): plain TE 1.200 µm; Itai s=0.52 **1.719 µm** | `dy = width_narrow/13` (`:826`): plain TE 50.0 nm; Itai s=0.52 66.7 nm |
| z | 0 | `core_height` = 350 nm (`:827`) | `dz = core_height/7 = 50.0 nm` (`:828`) |

Note the override dy uses the **scalar** narrow width, not the per-tooth minimum, so on
per-tooth devices the transverse mesh is coarser than the scalar naming suggests.

## 4. Excitation and extraction

**Source: mode-expansion Ports, not a plain mode source.** `fdtd.addport()`,
`injection axis = "x"`, `mode selection = "fundamental TE mode"` (string built from
`polarization`), `frequency dependent profile = 1` (`bragg_device.py:1321-1331`).
Port_1 at `-x_port` forward, Port_2 at `+x_port` backward (`:1334-1335`); Port_1 is the
source. Port y span = `1.2*y_span`, z span = `1.2*z_span` (`:1316,1328`).

`source.polarization = "TE"` (`simulation_config.py:279`) sets both the port mode and
the symmetry parity (`:274-277`) — one knob, two consequences.

Band: `setglobalsource("wavelength start"/"stop")` (`bragg_device.py:1506-1507`) with
`lam_min/max = λ_c ∓ scan_width/2` (`:1503-1505`). Defaults λ_c = 1560.1 nm,
span 20 nm, 3001 points (`simulation_config.py:208-211`). Runner overrides:
TE Q3dB centre 1560.0 / span 30.0 / 4001 pts (`te_q3db_20um.py:57-59`);
Itai TE centre 1559.79 / span 30.0 / 4001 pts (`itai_hh_apod.py:75,77,79`);
as-drawn TE 1620.0 ± 35 nm (`itai_hh_asdrawn.py:61`).

**DFT / time apodisation window: ABSENT.** There is no `set("apodization", ...)`
anywhere in `bragg_device.py` or `sim_helpers.py` — Lumerical's default (None) applies
to every monitor. If the report needs a stated window, it must be added.

Simulation time **2000 ps** (`bragg_device.py:769`; env `TM_SIM_TIME_PS` is opt-in only).
Auto-shutoff **1e-7** (`:770`; `simulation_config.py:231`).

Transmission: `get_s_and_t_matrix()` (`bragg_device.py:1519`) via
`post_processing.extract_s_parameters` (`post_processing.py:88-105`), with feed
de-embedding and Bragg-slope/phase correction both on by default
(`simulation_config.py:412-413`).

**How Q is computed — this is a linewidth measurement, not a Q object and not a
ring-down fit.** There is no Q analysis object and no ring-down fit anywhere in the
codebase. Two implementations, both `Q = λ_res / |spectral_fwhm_nm|`:

- Python: `python_tools/analyze_batch.py:53-55`, with `spectral_fwhm_nm` from
  `find_resonance` → `scipy.signal.peak_widths(T, [idx], rel_height=0.5)` scaled by the
  λ grid step (`post_processing.py:120-130`, stored `:353`).
- MATLAB: `matlab_plotting/plot_transmission.m:390-424`, FWHM from linear interpolation
  of the half-max crossings of `T_f`.

The resonance index comes from `find_bragg_resonance`, which scores local maxima by
`(prominence/(width+1)) * (1 - base_level)` (`sim_helpers.py:171-198`) — never
`argmax(T)`, because the global T max sits in the passband.

Note `spectral_fwhm_nm` is often stored **negative**; always take the absolute value.

## 5. Mode-width definition, exactly as coded

`post_processing.extract_field_profile` (`post_processing.py:137-154`) is a thin wrapper
around `sim_helpers.extract_and_process_field_profile` (`sim_helpers.py:271-310`). That
is the **single** mode-width path, and `fwhm_m` is its return value
(`post_processing.py:152`).

| aspect | what the code actually does | line |
|---|---|---|
| source monitor | `"field_profile"`, a 2D Z-normal plane at **z = 0** through the core | `bragg_device.py:1347-1357` |
| field read | `getresult(..., "E")` | `sim_helpers.py:279` |
| wavelength | the recorded point **nearest the resonance**, not band centre | `:285` |
| quantity | total electric intensity \|Ex\|² + \|Ey\|² + \|Ez\|² — **all three components** | `:297` |
| transverse reduction | **integrated over y**, `trapezoid(I_xy, f_y, axis=1)` | `:298` |
| crop | \|x\| ≤ `sim.x_grating_end` | `:301-304` |
| smoothing | an **envelope**, not a filter: `extract_envelope_peaks` finds local maxima of the standing-wave pattern and cubic-interpolates through them, nearest-neighbour at the edges | `:248-268` |
| criterion | **half maximum relative to the floor**, `target = y_min + 0.5*(y_max - y_min)`; width = distance between the outermost linearly-interpolated crossings; returns 0.0 on fewer than two crossings | `calculate_fwhm_relative`, `:218-243` |

Three points that matter for the report:

1. **The criterion is half-max-above-floor, not −3 dB of the peak.** On a profile with a
   non-zero pedestal these differ.
2. **The historic y-integration bug is gone** — the y-integrated form at `:298` is the
   only one present. But every `sigma`/`FWHM` logged **before 2026-08-18 is VOID**
   (T / λ / Q / R / loss are unaffected).
3. The monitor's transverse extent is only `1.5 × width_wide` (`bragg_device.py:1353`)
   and uses the **scalar** `width_wide`, so on per-tooth devices the y-integration window
   can be narrower than the real teeth (Itai: window 1.5×1132.7 = 1.70 µm against a
   1432.8 nm tooth).

**Values** — see item 6 for the plain device and items 6b/14 for the seed.

## 6. Per-device measured results

### 6a. Plain uniform TE pi-shift devices

All MEASURED from `results_from_athena\te_q3db_20um\results\` (study
`runners\sweeps\te_q3db_20um.py`; pitch 500 nm, height 350 nm, avg 800 nm). Q computed
as λ/|spectral_fwhm_nm|; port powers read at the resonance index of the stored
`wl_nm`/`T`/`R`/`loss` arrays.

| device | file | λ_res (nm) | spec. FWHM (nm) | Q | T_pk | fwhm_m (µm) | R@res | loss@res |
|---|---|---|---|---|---|---|---|---|
| **N=80 baseline, corr 300** | `result_N80_avg.mat` | 1558.9264 | −1.10570 | **1409.9** | **0.87049** | **15.561** | 0.00514 | 0.12437 |
| **N=166, corr 250 (Q3dB 20 µm deliverable)** | `result_N166_avg_C250.mat` | 1559.7883 | −0.12089 | **12902.9** | **0.49190** | **20.456** | 0.09186 | 0.41624 |
| N=168, corr 250 | `result_N168_avg_C250.mat` | 1559.7883 | −0.11506 | 13556.9 | 0.47172 | 20.463 | 0.10081 | 0.42747 |
| N=176, corr 250 | `result_N176_avg_C250.mat` | 1559.7883 | −0.09599 | 16250.1 | 0.38980 | 20.487 | 0.14399 | 0.46622 |
| N=190, corr 250 | `result_N190_avg_C250.mat` | 1559.7883 | −0.07329 | 21282.5 | 0.25502 | 20.512 | 0.24757 | 0.49742 |
| N=215, corr 250 | `result_N215_avg_C250.mat` | 1559.7883 | −0.05321 | 29315.1 | 0.08837 | 20.538 | 0.49363 | 0.41800 |
| N=110, corr 233 | `result_N110_avg_C233.mat` | 1560.1708 | −0.85808 | 1818.2 | 0.92120 | 21.595 | 0.00211 | 0.07669 |
| N=170, corr 233 | `result_N170_avg_C233.mat` | 1560.1708 | −0.14852 | 10504.8 | 0.65174 | 22.542 | 0.03928 | 0.30897 |

Independent N=80 repeat at a different z-box
(`results_from_athena\te_span_z_check\results\result_N80_avg_Ybox3p8_Zbox8p8.mat`):
λ 1559.2260, Q 1415.8, T 0.87323, fwhm_m 15.565 µm, R 0.00497, loss 0.12180 —
confirms the baseline to 0.3% in T across a 3.2 → 8.8 µm z-box.

### 6b. TE apodised overshoot seed — INCOMPLETE ON LOCAL DISK

**The raw `result_*.mat` files for this study are NOT on local disk.** Only
server-side-reduced summaries were pulled (the ~700 MB volumes stayed on IGUM).
Jobs (IGUM, 2026-08-25/26): 63237/63424 **VOID** (undersized box, T+R>1), 63438 box
ladder, 63441 as-drawn, 63451, 63454, 63491, 63722, 63752.

**HH profile as drawn, full depth (the true overshoot seed), TE, pitch 514 nm, N=98/side:**

| quantity | value | source |
|---|---|---|
| fwhm_m (spatial) | **15.265 µm** | `results_from_igum\hh_asdrawn_spatial_TE.mat` (`fwhm_um`; box 9.0 × 9.1 µm, job 63441) |
| λ_res, T, Q, spectral FWHM, port powers | **NOT STORED locally** | — |

TM as-drawn twin for contrast: fwhm_m 17.222 µm
(`results_from_igum\hh_asdrawn_spatial_TM.mat`).

**HH amplitude-scaled to hit the 20 µm spec** — `results_from_igum\itai_hh_summary.csv`
(columns `who,pol,N,Tres,Qi,fwhm`; box 6.8 × 6.81 µm = inverse-design numerics).
**λ and spectral FWHM are not in this file, and the Q column is Q_i, not Q_L:**

| pol | N | T_res | Q_i | fwhm_m (µm) |
|---|---|---|---|---|
| TE (scale 0.52) | 98 | 0.98821 | 378044 | 20.72 |
| TE (scale 0.58) | 98 | 0.98393 | 427874 | 19.688 |
| TE | 140 | 0.96348 | 621270 | 19.815 |
| TE | 155 | 0.94093 | 570920 | 19.834 |
| TE | 175 | 0.90947 | 631194 | 19.851 |
| TE | 195 | 0.85955 | 674598 | 19.860 |

**Itai's re-optimised Nt60 profile, natively 20 µm, TE, untouched** — the
best-documented rows, `results_from_igum\itai_hh_nt60w20_summary.csv`:

| N | λ_res (nm) | T_res | R | Q_L | Q_c | Q_i | fwhm_m (µm) | job |
|---|---|---|---|---|---|---|---|---|
| 98 | 1559.8597 | 0.97498 | 0.00014 | 7680 | 7778 | 610032 | 19.633 | 63722 |
| 130 | 1559.8638 | 0.92377 | 0.00162 | 45058 | 46880 | 1159234 | 19.713 | 63752 |

**Do not quote** `results_from_igum\itai_hh_apod_task0_slice.mat` (λ_res 1559.6258,
T_pk 1.0255 — unphysical, T+R>1, fwhm_m 17.608 µm): it is from the VOIDED round-1
undersized box.

## 7 & 17. Far field — what exists, and the honest status of the TE claim

**Projection chain (code):** `sim_helpers.extract_farfield` (`sim_helpers.py:15`) calls
Lumerical `farfield3d(monitor, idx, res, res)` plus `farfieldux`/`farfielduy` on a
**201×201 direction-cosine grid**, and optionally `farfieldvector3d` → complex
`Ex_c/Ey_c/Ez_c` (`cfg.farfield.save_complex`). So **both intensity and complex are
available**, complex only when that flag is set.

**Monitors** (`bragg_device.py:1464-1498`): `side_monitor` = 2D **Y-normal** at
y = 1.5·width_wide (in-plane/lateral radiation), `top_monitor` = 2D **Z-normal** at
z = 1.5·core_height (vertical radiation). Both at x = cavity_length/2, both
**1 frequency point**. Caution: they are set with `"use source limits"=1` + 1 frequency
point, which is exactly the band-centre-not-resonance trap; in the files checked the
offset is small (ff λ 1558.545 vs λ_res 1558.6117, i.e. 0.07 nm) but it is not zero and
must be stated in the report.

**Axis caveat that will bite the report:** on the **side** monitor (Y-normal) the two
in-plane cosines are x and z, so the struct field named `uy` is physically **u_z**. On
the **top** monitor (Z-normal) `uy` really is u_y.

**TE far-field data that exists locally:**

- `results_from_athena\comb_physics_rethink\data\scat_z_teffmap__result_N80_avg_Ybox16p0_Zbox8p8_ff.npz`
  — TE uniform N=80 control, **side AND top**, E² **and** complex Ex_c/Ey_c/Ez_c,
  201×201, at resonance λ 1559.389 nm (in-file: T 0.87563, R 0.00495, loss 0.11942,
  Q 1411.07, fwhm 15.606 µm). **This is the single best TE far-field file.**
- Three TE comb variants, same format:
  `...\comb_physics_rethink\data\scat_te_comb__result_N80_avg_Ybox16p0_Zbox8p8_scR110_arr41_*_ff.npz`.
- Originals: `results_from_igum\scat_z_teffmap\results\result_N80_avg_Ybox16p0_Zbox8p8_ff.mat`
  and four `results_from_igum\scat_te_comb\results\...arr41_X-11{357,505,652,800}to..._C300_pair_ff.mat`.
- `results_from_athena\run_te\results\result_N80_avg_ff_te_fields_smp.mat` — 1.7 GB MAT
  v5, TE N=80 with far field plus full field volumes. Not opened (size).

### The "TE radiates mainly vertically" statement — QUALIFIED, and the record is inconsistent

This needs fixing before it goes into a report.

1. **MEASURED this session** from the npz above: integrating E²/u_z over the hemisphere
   gives top 8.64e-15 vs side 5.84e-15 (arb. units), **ratio ≈ 1.48**, i.e. top > side.
   **Caveat: the two monitors have different apertures and normalisations, so this is
   not a calibrated power split** — only a same-file comparison.
2. `docs\comb_physics_rethink_2026-09-11.md` §8 states: far field is a broad double hump
   at |u_x| ≈ 0.75–0.8, **top monitor > side**, 1–2% in the grazing bin.
3. **Contradicting prose:** `docs\loss_program_bic_scatterer_2026-07-06.md` §5 concludes
   "**TE radiates far less vertically**", arguing from TE's insensitivity to the vertical
   box (confirmed above: T 0.87107 → 0.87323 over z 3.2 → 8.8 µm, vs TM's +0.019).
4. `docs\research_overview_briefing_2026-09-13.md` L173: "TE radiates upward in a broad
   cone with no needle" (job 97112).

The two are reconcilable — TE's **total** loss is small and its radiation is near-axial,
so "most of a little" and "far less than TM" are both true — but the phrasing in the
record is not internally consistent.

**NOT STORED: any quantitative TE in-plane/vertical power split.** The only calibrated
split (Poynting-flux polarimetry with a 99–100% energy audit, via
`extract_monitor_polarimetry`) was run for **TM only** —
`results_from_athena\tm_radiation_polarimetry\FINDINGS.md`, job 117907: **62% in-plane /
38% vertical, f_TE ≈ 0.00**. There is no TE row in that study and no TE equivalent
anywhere. A numeric "TE is X% vertical" claim would be **UNSUPPORTED by stored data**.

## 8. Monitors in the seed scene

Always present (`bragg_device._add_source_and_monitors`):

| name | type | position / span | line |
|---|---|---|---|
| `Port_1` | port, x-normal | x = −x_port, y span 1.2·y_span, z span 1.2·z_span | `:1334` |
| `Port_2` | port, x-normal | x = +x_port | `:1335` |
| `field_profile` | 2D Z-normal | x=0, x span `2·x_grating_end + 2 µm`, y span `1.5·width_wide`, z=0, **501 freq pts**, source limits | `:1347-1357` |
| `time_input` | time, point | x = −x_grating_end − 0.5 µm, y=0, z=0 | `:1368` |
| `time_cavity` | time, point | x = 0 | `:1369` |
| `time_output` | time, point | x = +x_grating_end + 0.5 µm | `:1370` |

Present because `monitors.record_2d_fields = True` by default
(`simulation_config.py:360`) and neither runner disables it:

| name | type | position | line |
|---|---|---|---|
| `field_profile_2D_XY` | 2D Z-normal ("Top view") | x=0, span `2·x_grating_end + 1 µm`, z=0 | `:1384-1399` |
| `field_profile_2D_YZ_cross` | 2D X-normal | x = `cavity_length/2` (defect centre) | `:1402-1417` |
| `field_profile_2D_XZ_side` | 2D Y-normal ("Side view") | y=0, x=0 | `:1420-1434` |

Frequency points are 5 at build, overridden to `n_2d_monitor_points = 51` by
`apply_monitor_overrides` (`sim_helpers.py:590-595`, `simulation_config.py:212`).

- `field_profile_3D`: **ABSENT** — `record_3d_fields = False` (`simulation_config.py:365`).
- `side_monitor` / `top_monitor`: **ABSENT in the seed** — `farfield.enabled = False`
  (`:381`) and neither runner enables it.
- **CLOSED BOX of monitors: ABSENT.** There is no enclosing 6-face monitor set anywhere
  in the builder — only the three orthogonal cut planes above, plus the two optional
  far-field planes when enabled.
- **H is never read.** Every 2D readout requests `"E"` (plus optionally Poynting `"P"`)
  — `post_processing.py:177,190,197,210,216,229`; the 1-D `field_profile` path reads
  `"E"` alone (`sim_helpers.py:279`). Lumerical profile monitors store H internally by
  default, but nothing in this codebase retrieves it. **Exception:**
  `sim_helpers.extract_monitor_polarimetry:119-120` does read both E and H — on one
  plane, reducing to scalars and 1-D profiles server-side.

---

# TM calibration points

## 9. Internal oxide circles (SiO2 holes in the SiN core)

**Geometry** (`runners/hole_lattice/tm_hole_lattice.py`): cylinders of n = 1.444 inside
the SiN core, **r = 100 nm, y = 0 (on-axis), FULL core height 350 nm (through)**, one per
narrow-segment centre, **160 sites** spanning ±41.3 µm (all 160 periods). Device: TM
W800, pitch 516.83, corr 400, h 350, N = 80/side, optimization mesh, window
1545 ± 75 nm / 6001 pts. Job 123303. All rows from
`results_from_athena\tm_hole_lattice\*.mat`:

| row | λ_res (nm) | T | Q | fwhm_m (µm) |
|---|---|---|---|---|
| **control, no holes** (`result_N80_TM_avg.mat`) | 1558.559 | 0.8250 | 1209 | 15.54 |
| matched lattice Λ 516.83, corr 400 | 1571.466 † | 0.9113 † | 338 † | 63.07 † |
| jitter twin (+25 nm) | 1571.311 † | 0.9115 † | 332 † | 63.0 † |
| matched lattice, corr 300 trim | 1549.029 | 0.1598 | 2401 | 10.22 |
| period-detuned Λ 545 nm, 151 sites | 1548.777 | 0.3403 | 918 | 16.00 |
| lattice shifted +pitch/4 | 1547.871 | 0.4094 | 1767 | 12.45 |

**† TRAP — do not quote these.** For the two matched-lattice rows the stored
`resonance_wavelength_nm` is a **finder mis-pick of the passband**. The real defect peak
is at **1547.8 nm, T = 0.032** (jitter twin 0.040) — the peak is annihilated. This is
exactly the failure mode the resonance-sanity rule exists to catch.

**Verdict: route CLOSED, every variant strongly harmful.** Predecessor single-hole scan:
`runners/archive/sweeps/scatterers/tm_hole_scan.py`, jobs 116152/116272 —
parasitic-to-neutral, never beneficial.

**Separate in-core oxide *comb* family** (r 80/50, y = ±250 nm, h 350 through-core,
Λ ≈ 524–536, 31 holes; `runners/scatterers/scat_x*_incore*.py`). Control =
`results_from_athena\scat_h_retrocomb\results\result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat`:

| variant | λ (nm) | T | Q | width (µm) | file |
|---|---|---|---|---|---|
| **ctrl (no holes)** | 1558.612 | 0.8851 | 1327 | 15.53 | scat_h_retrocomb `..._ff.mat` |
| 31 holes r80 y±250 Λ531 φ0 | 1554.579 | 0.8460 | 1074 | 25.64 | `results_from_igum\scat_x_incore\results\...arr31_X-7567to8363_Y250...` |
| 9 holes | 1557.943 | 0.8654 | 1298 | 17.85 | same dir, `...arr9_X-1726to2522...` |
| 9 holes (other phase) | 1555.415 | 0.5438 | 1022 | 16.63 | same dir, `...arr9_X-1992to2256...` |
| r50 + corr 477 (**equal-width test**) | 1556.239 | 0.7922 | 2102 | 15.39 | `results_from_athena\scat_x8_incore_r50_c477\results\...C477...` |
| on-axis r60 + corr 465 (equal-width) | 1556.571 | 0.8198 | 1920 | 15.52 | `results_from_athena\scat_x17_incore_axis_eqwidth\results\...C465...` |
| on-axis r80 + corr 503 (equal-width) | 1555.083 | 0.7490 | 2335 | 15.79 | same dir, `...C503...` |

At matched mode width the in-core holes **radiate more** than the plain device
(Q_i −9% to −23%) — CLOSED NEGATIVE. Narrative: `runners/scatterers/COMB_HANDOFF.md` §6.

## 10. External SiN post comb

**Geometry** (`runners/scatterers/COMB_HANDOFF.md` §2–3): SiN cylinders in the oxide
cladding, two rows mirrored at ±d, **h = 350 nm = core height (single-litho)**, comb
period **Λ = 531 nm** (deliberately NOT the grating pitch 516.83 — it is matched to the
radiation cutoff, not to the teeth), rigid shift δx = 401 nm, hence
**phase φ = 360·δx/Λ = 270°, referenced to the centre of the cavity segment**
(x = 0; the cavity spans ±129.21 nm).

**q3db operating point, corr-325** (jobs 130458/130548,
`results_from_athena\comb_q3db\results\`): r = 80 nm, d = 1.9 µm, 57 posts/row, 2 rows.

| row | N | λ (nm) | T | Q | width (µm) |
|---|---|---|---|---|---|
| **BEFORE — control, no comb** | 165 | 1559.001 | **0.4906** | **13930** | **19.97** |
| AFTER — comb 270° | 165 | 1559.011 | 0.5361 | 14584 | 19.90 |
| comb 90° (sign check, loses) | 165 | 1559.016 | 0.4371 | 13143 | 20.05 |
| comb Λ = 536 (aim hedge) | 165 | 1559.016 | 0.5283 | 14476 | 19.96 |
| comb 270° | 167 | 1559.011 | 0.5160 | 15352 | 19.90 |
| comb 270° | 168 | 1559.011 | 0.5059 | 15761 | 19.91 |
| **comb 270° — the −3 dB lock** | 169 | 1559.011 | **0.4961** | **16203** | **19.91** |

⇒ **+16.3% Q at the same −3 dB spec and the same 20 µm mode, λ unpulled (+10 pm).**

**Newer r-110 comb device** (`results_from_athena\comb_q3db_lock\results\`, jobs
146639/146681), same control 0.4906 / 13930:

| device | λ (nm) | T | Q | width (µm) |
|---|---|---|---|---|
| single 61-post comb, N171 | 1559.036 | 0.5060 | 17557 | 19.79 |
| **pair (31 @ +12 µm δx 88.5 nm + 31 @ −12 µm δx 177 nm), d 1.8 µm, Λ 531, N172** | 1559.031 | **0.5003 (−3.01 dB)** | **18093** | **19.76** |

corr-400 family (r 110/96/89, d 1.8 µm, 31–53 posts,
`results_from_athena\scat_p_antineedle\`, job 129989): T 0.8851 control → 0.8966–0.9001;
phase circle 0/90/180/270° = 0.8694 / 0.8586 / 0.8689 / 0.8797. Those tables carry
**T only** — no per-row λ/Q/width without re-reading each .mat.

## 11. The Λ = 551 nm period-matched scatterer comb — it EXISTS

Λ = 551 nm = λ / (2·n_clad·u_x) at the measured "needle" peak u_x = 0.980, θ = 11.6°.
Posts **r = 110 nm**, h = 350 nm (planar) or 12000 nm (full-z), 151 per row, ±41.3 µm,
mirrored ±y. Control (same numerics, box y = 16 µm, 1501 pts):
**λ 1558.612 / T 0.8851 / Q 1327 / 15.53 µm**.

**Stage H — standoffs d ≥ 3 µm** (`results_from_athena\scat_h_retrocomb\results\`,
jobs 123561/123563, runner `runners/scatterers/scat_h_retrocomb.py`):

| d (µm) | λ (nm) | T | Q | width (µm) |
|---|---|---|---|---|
| control | 1558.612 | 0.8851 | 1327 | 15.53 |
| 3.0 | 1558.612 | 0.8846 | 1326 | 15.54 |
| 3.7 | 1558.612 | 0.8843 | 1327 | 15.53 |
| 4.4 | 1558.612 | 0.8846 | 1327 | 15.53 |
| 5.1 | 1558.612 | 0.8854 | 1328 | 15.53 |
| 2 rows, 3.0 + 5.68 | 1558.612 | 0.8849 | 1326 | 15.54 |

**Verdict NULL** (`results_from_athena\scat_h_retrocomb\FINDINGS.md`): ΔT flat within
±0.0008, below half the 0.0018 numerical floor; the needle bin never reduced; rows do
not stack.

**Stage O — the same 551 nm comb moved into the near zone, d = 1.8 µm**
(`runners/scatterers/scat_o_comb1800.py`, `results_from_athena\scat_o_comb1800\results\`):

| variant | λ (nm) | T | Q | width (µm) |
|---|---|---|---|---|
| planar h 350 nm | 1558.652 | **0.8732** | 1313 | 15.51 |
| full-z h 12000 nm (through both z-PMLs) | 1558.972 | **0.6868** | 1151 | 15.46 |

The full-z collapse (0.873 → 0.687, with a new +10 dB lobe at u_x ≈ −0.925) is the
accident that started the anti-needle comb programme: it showed a post row out-couples
the guided carrier, which led to retuning Λ from 551 to **531 nm** (COMB_HANDOFF §4).

Other periods on record: **531 (the lock)**, 536 (aim hedge), 545 (stage P circle),
524/527/530 (in-core), 590 (TE comb). The values 88.5 nm and 177 nm are comb **δx
shifts** of the Λ-531 pair, not periods.

## 12. Scatterer linearity / superposition gate

Stage B: `runners/scatterers/scat_b_gates.py` + analyser
`runners/scatterers/solve_response_matrix.py` (gate constant `PAIR_ERR_MAX = 0.05` at
line 51). Jobs 120817 + 120961. Report:
`results_from_athena\scat_b_gates\results\gates_report.json`.

**How superposition was tested:** mirrored ±y SiN pillar rows at y = 700–1000 nm on the
TM corr-400 N=80 device, λ-locked, complex far field saved. Two singles are run alone at
x = −135 nm and x = +135 nm (a cavity-straddling arrangement, the worst case = strongest
drive), then a run with both present. The error metric
(`solve_response_matrix.py` ~line 306) is over the **complex far-field vector**:

```
e = ||(r_both − r_ctrl) − (r_A − r_ctrl) − (r_B − r_ctrl)|| / max(||r_A−r_ctrl||, ||r_B−r_ctrl||)
```

| r (nm) | y = 700 | 800 | 900 | 1000 |
|---|---|---|---|---|
| 60 | 2.28% | 1.92% | 1.64% | 1.84% |
| **80** | **4.39% PASS** | 3.64% | 3.15% | 3.50% |
| 100 | 7.73% FAIL | — | — | 5.81% FAIL |
| 125 | — | — | — | 9.45% FAIL |
| 150 | — | 18.25% FAIL | — | 14.92% FAIL |

**The ~5% number is the r = 80 nm / y = 700 nm row: pair_err = 4.391%**, just under the
5% gate. That is the `"recommended"` entry in the JSON (snr 7.15e4, resonance pull
80 pm) and the (r, y) the whole response-matrix programme then adopted
(`_common.py` RADIUS_NM = 80.0, Y0_NM = 700.0).

Also in the same report: a separation series (Born test) at r 80, s =
135/270/405/540/1080 nm → only **270 and 1080 nm pass**, giving
`min_spacing_nm = 1080`; noise floor `floor_cross_norm = 9.13e-14` against a reference
response of 4.51e-08 (5e5 headroom).

Downstream (from memory, not re-read from .mat this session — treat as EXPECTED):
stage-A baseline T 0.8862 / λ 1558.6156 / spectral FWHM 1.181 nm (Q ≈ 1319, DERIVED);
best predicted binary combo `[0, 270]` gave T 0.886 → 0.909 (+0.0227), loss
0.110 → 0.0885, with predicted 30.0% vs measured 30.0% leak cancellation.

## 13. Adjoint work — width gradient, resonance gradient, 0.365 µm/nm, far-field FOM

Files: `runners/lumopt2_design/lumopt2_design.py` (= `L` below), `THEORY.md`,
`HANDOFF.md`, `HANDOFF_2026-09-01.md`.

### 13a. Width gradient — `fwhm_env` (observable) vs `softW` (carrier)

- **`fwhm_env` is the observable and is NOT differentiable.** `fwhm_env_of_line(x, I)`
  (`L:1818-1833`) is a cubic envelope through the standing-wave peaks plus half-max
  relative to the profile floor, calling `sim_helpers`' own `extract_envelope_peaks` /
  `calculate_fwhm_relative` so it is byte-identical to `post_processing`'s `fwhm_m`.
  Peak-picking plus cubic interpolation admits no gradient (THEORY.md:343-345).
- **`softW` is the autograd carrier.** `soft_width_of_line(x, I)` (`L:1874-1889`):
  boxcar over one 258 nm fringe convolved with a 0.25 µm Gaussian
  (`WG_FRINGE_UM`/`WG_GAUSS_UM`, `L:1845-1849`) → softmax peak `P` (β = 60 on
  `Is/scale`, the scale deliberately kept **inside** the graph, `L:1879-1884`) → fixed
  edge-window floor `F` (5% of samples each end) → sigmoid super-level-set indicator at
  half max `h = F + 0.5(P − F)`, temperature `WG_EPS = 0.05` → trapezoid integral.
  Differentiable end-to-end in `I`.
- **What is differentiated:** `dsoftW/dI` via `autograd.grad` (`L:1892-1898`). **That
  vector is the adjoint source weight**: `MixedFom.setup_adjoint_simulation` imports
  `conj(E_fwd)·W(x,y)` on the resonance λ plane only, other planes zeroed, with
  `W = dsoftW/dI ⊗ wy` (`L:2153-2175`). Stock `FieldFom` cannot do this — its source is
  hard-coded plain `conj(E)`, correct only for Σ|E|² FOMs (`L:2085-2088`;
  `lumopt2/fom/field_fom.py:32-60`). Physically the weight peaks at the two half-max
  crossings (THEORY.md:293-300).
- **Position in the FOM vector:** `softW` is a second FOM entry, always **last**, so the
  flat autograd input is `x = [T(λ_0) … T(λ_{n−1}), softW]` and the width selector is
  `x[-1]` (`L:2256-2258`, `:2277-2279`, `make_fct_v2` `:1918-1944`). This flat layout is
  the plumbing trap that killed job 137267. Under `wg_project=True` the fct is pure
  windowed softmax-T and the width jacobian entry is exactly 0 (`L:1927-1941`); the width
  adjoint still runs and ∇W is recovered by re-running the **same** assembly with fct
  `x[-1]` (`L:2245-2250`, `:2291-2296`) — **zero extra solves**, because assembly is
  linear in the fct jacobian.
- **How the two relate: a delta anchor, not a fit.**
  `fhat = anchor["fwhm"] + (softW − anchor["softw"])` (`L:1905` penalty path, `L:1697`
  per-eval logging). The anchor `{softw, fwhm}` is measured on the **same** evaluation and
  re-anchored per restart (`L:244-248`, set at `:2989-3004`). The residual
  `wg_resid_um = fhat − fwhm_env` is logged every eval with a loud warning above
  `WG_RESID_WARN = 0.05 µm` (`L:1694-1704`, `:1850`). **All branch decisions use the
  MEASURED `fwhm_env`** (`WidthTrip`, `L:1745-1747`); softW supplies direction only
  (THEORY.md:355; `L:241-243`).
- Validation as documented: softW tracks measured `fwhm_env` growth to ≤ 2.2 pp across
  +4.9%…+26.6% designs where sigma errs by 24 pp; autograd ≡ FD to 1e-8
  (`L:1837-1844`, THEORY.md:131). The field-adjoint amplitude/phase is corrected by a
  separately fitted `C_field` (`L:373-374`, applied `:2236-2238`; provenance
  `runners/lumopt2_design/fit_c_field.py:1-16`).

### 13b. Resonance gradient — the λ chain term (defect #19)

- **The defect:** the adjoint `gW` is `∂W/∂p` **at fixed λ**, but W is specced at the
  device's own moving resonance, so
  `dW/dp = ∂W/∂p|_λ + (dW/dλ)·(dλ_pk/dp)` was missing its second term
  (HANDOFF.md:313-316; `L:415-428`).
- **What is differentiated w.r.t. what:** `gλ = dλ_pk/dp`, from the **implicit function
  theorem on the peak condition ∂T/∂λ = 0**:
  `gλ = −(∂²T/∂λ∂p) / (∂²T/∂λ²)` (`L:2729-2732`).
- **How it is obtained — matched stencil, zero extra solves.** In the callback, pick
  `k = round(0.5·fwhm/dl)` grid points either side of `i_pk`, giving `i_lo, i_hi`;
  compute `T'` at both by central difference off the already-solved spectrum; store
  `_wg_lam_idx` and `_wg_dTp = T'_hi − T'_lo` (`L:1610-1657`). In
  `calculate_gradient_fields`, two extra **selector passes** with fct `abs(x[i_lo])` /
  `abs(x[i_hi])` over the SAME solved fields give `gfields_Tlo/Thi` (`L:2277-2287`).
  The driver assembles `gLam = −(g_hi − g_lo)/dTp`, then `chain = wg_dwdlam·gLam`
  (`L:2761-2763`).
- **Why "matched":** for any translating lineshape `T = A(p)·S(λ − λ_0(p))` with S even,
  the amplitude part is even and cancels in **both** antisymmetric differences, so the
  stencil truncation cancels exactly for any h — hence a **wide** stencil is better
  (`L:1624-1637`). The naive pair (central difference of ∂T/∂p over a second difference of
  T) has error exactly `1/(1+x²)` with x = h/g — **49.4% low at a half-linewidth**
  (`L:1632-1636`; gate `gates/gate_lam_chain.py` passes at 0.0034%).
  **∇T needs no chain term** because ∂T/∂λ = 0 at the peak (HANDOFF.md:334).
- **Guards:** `dTp < 0` ⟺ the stencil straddles a maximum, else LOUD skip; a relative
  curvature floor `|dTp| < 5%·dTp0` skips the step (the IFT gain 1/dTp is unbounded as
  the peak flattens); thinning warning at 25% (`L:2738-2755`); λ-descending spectra
  swapped so `wl[i_lo] < wl[i_hi]` (`L:1638-1647`); index-wrap guard
  `1 < i_pk < len−2` (`L:1610-1617`). `compute_gradient_from_fields` is deliberately
  deferred to the driver — doing it inside the FOM crashed on hardware in analysis mode
  (`L:2756-2760`, `:2265-2272`).
- **Two uses:** legacy fold-in `gW += chain` (`L:2790`), or — under `ns2` — `gLam` kept as
  its **own** constraint row so `gLam·d = 0` and the fitted coefficient cancels from the
  feasible directions entirely (`L:2783-2789`, `_ns2_step` `:2363-2400`). A
  rotation-sensitivity audit at coefficient ×0.8/×1.2 is logged as `proj_rot_deg`
  (`L:2764-2777`).
- **Honest status:** `wg_lam_chain` defaults **False** (`L:429`); HANDOFF.md:353-357 states
  the fix had "never completed a single iterate on hardware" at that writing.
  HANDOFF_2026-09-01.md:27 later reports a measured `ΔW ≈ 0.3655·Δλ_pk` every iterate
  on d1.

### 13c. The 0.365 µm/nm figure — FOUND, but the denominator is NOT a knob

- **Value and location:** `CampaignSpec.wg_dwdlam = 0.3655` µm/nm, `L:430`. Also quoted at
  `.claude/skills/lumopt2-design/SKILL.md:1058`, `HANDOFF.md:281-298`,
  `HANDOFF_SELF_CONTAINED.md:829,1026`.
- **The denominator is the device's own RESONANCE WAVELENGTH `lam_pk_nm`, not a geometric
  parameter.** It is `dW/dλ`: spatial mode width (`fwhm_env_um`) per nm of resonance
  shift — the "width is slaved to λ" coupling (HANDOFF.md:281).
- **How it was measured: neither finite difference nor adjoint — a least-squares line fit
  to stored eval logs.** `runners/lumopt2_design/gates/derive_dwdlam.py:75-84`:
  `np.polyfit(lam, W, 1)` over in-band rows of `lumopt2_v2_uniform_s5_evals.jsonl` and
  `lumopt2_v2_seesaw_evals.jsonl` in `results_from_athena\v2_gpu_gradient_pause\jsonl\`.
  The filter rule (`derive_dwdlam.py:34-66`) requires finite `lam_pk_nm`/`fwhm_env_um`,
  `fom > 0.5·max(fom)`, and dedup of identical (λ, W) pairs — dropping either clause
  moves the slope to 0.59 or pollutes it.
- **Provenance:** uniform_s5 **+0.3654**, r = 0.984, n = 9, explaining 93% of that run's
  raw width growth; seesaw **+0.300** (~20% spread); pooling the two is explicitly wrong
  (0.288) — HANDOFF.md:288-299, `derive_dwdlam.py:88-108`. **The stored constant comes
  from the uniform baseline alone.**
- **Caveat stated in-repo:** it is a **path** derivative, not a device constant
  (`CHANGES_2026-08-28.md:30`) — which is why `wg_dwdlam_fit=True` re-fits it online
  (`L:392-398`) and why the ns2 design makes it cancel.

### 13d. Can the adjoint take a far-field amplitude or plane-wave overlap as FOM? — NOT TODAY

**What exists (read, not inferred):**

- Exactly two FOM families ship in lumopt2: `PortFom` (mode-expansion S-parameters at a
  port) and `FieldFom` (|E|² over a field region) —
  `lumopt2/fom/{port_fom.py:13, field_fom.py:12}`. The results containers likewise:
  `PortResults` accepts metric `"transmission"` only and raises `NotImplementedError`
  otherwise (`simulation_results.py:196-210`); `FieldResults` accepts `"intensity"` only
  (`:286-300`). **There is no far-field FOM and no near-to-far transform anywhere in the
  adjoint path.**
- This program's `MixedFom` (`L:2114`) already generalises the field branch: it overrides
  `setup_adjoint_simulation` to import an **arbitrary complex weight**
  `W(x,y)·conj(E_fwd)` on one λ plane (`L:2153-2175`), with GPU tiling across 4 narrow
  FieldRegion sources (`import_tiled_source`, `L:2013+`, `:2196-2201`) and a fitted
  `C_field` (`:2236-2238`).
- The far-field monitors exist in the device builder but are **plotting-only and are not
  in the lumopt2 scene**: `bragg_device.py:1463-1497` adds `side_monitor`/`top_monitor`
  only when `record_farfield=True`, while `build_base_fsp` adds only `field_profile` plus
  the single-λ `field_profile_adj` FieldRegion (`L:1202-1231`).

**Verdict: feasible on this machinery, and it is new code — moderate, not a rewrite**,
because the hard part (an arbitrary weighted adjoint source) is already built. Both
candidate FOMs are linear-in-E functionals over a plane,
`J = |∫ W*(x,y)·E(x,y) dA|²`, with `W` the target plane wave or a far-field kernel
`exp(−i k·r)` for one direction or a set of directions — the same *shape* as the width
FOM. What would have to be written:

1. A `FarFieldResults(FieldResults)` that reads a near-field plane and forms the overlap.
   The existing `WidthResults` (`L:2097-2112`) is the template. The metric strings must be
   bypassed, since neither stock class supports it.
2. The adjoint source `dJ/dE = 2·Re{⟨W,E⟩}·W` (**INFERRED** — standard, not read from this
   repo), plugged into the same branch at `L:2153-2175` in place of
   `W = dsoftW/dI ⊗ wy`. **Important structural difference:** the width source is
   `conj(E)·W` (a real, intensity-derived weight), whereas a plane-wave-overlap source is
   the conjugate of a **prescribed** kernel scaled by the overlap scalar — the `conj(E)`
   factor drops out. That is a genuine code change, not a parameter.
3. A fresh `C_field` fit for the new source (`fit_c_field.py:1-16` states explicitly that
   the port C does not transfer to the field path and must be redone per version bump),
   plus a finite-difference `check_gradient` gate on a tiny problem.
4. Two structural traps already documented and re-applicable: (i) the twin plane must not
   sit on the z=0 anti-symmetric BC or an import source injects nothing
   (`check_import_src_injects`, `L:1946-1970`); (ii) a broadband FieldRegion is rejected at
   `generate()`, so the FOM plane is single-λ and must be pinned to the resonance
   (`L:1203-1214`).
5. If the true angle-resolved far field is wanted rather than a single plane-wave
   projection, the NTFF itself would have to be made differentiable or expressed as a
   weighted near-field integral — the latter is the only tractable route here (**INFERRED**).

---

# Profiles — what exists as arrays

Two facts govern this whole section:

- **Every** `result_*.mat` written by `post_processing.py` (~line 362) carries `field_x`,
  `field_energy_density_1D`, `field_envelope_1D`, `fwhm_m` — the y-integrated |E|² along
  x from the `field_profile` monitor at z=0, extracted at the recorded λ nearest
  `resonance_wavelength_nm`. So item 14 is ubiquitous.
- **Stored arrays are FULL domain (unfolded).** MEASURED: `xy_y` runs −7.610 → +7.610 µm
  for a Ybox = 16 µm run, `xz_z` −4.009 → +4.009 for Zbox 8.8, far-field `ux`,`uy` in
  [−1, +1]. Lumerical unfolds the Anti-Symmetric (TE) / Symmetric (TM) y-BC and the z-BC
  in monitor output. **No half-domain array was found anywhere.**

## 14. Envelope along x — EXISTS-ARRAY for both

**Plain uniform TE:**
- `results_from_athena\te_q3db_20um\results\result_N80_avg.mat` — `field_x` (1,1606)
  −40.125…+40.125 µm, `field_energy_density_1D`, `field_envelope_1D`,
  `fwhm_m` = 15.5605 µm; λ_res 1558.926, T 0.8705, corr 300, N=80.
  Jobs 128580/128581/128593/128730/128733 (header of
  `matlab_plotting\studies\plot_te_q3db_20um.m`).
- Same dir: N166/168/170/176/190/215 corr-233/250 — e.g. `result_N166_avg_C250.mat`,
  `field_x` 3326 pts ±83.125 µm, fwhm 20.456 µm, T 0.4919.
- TE control that also carries far field:
  `results_from_igum\scat_z_teffmap\results\result_N80_avg_Ybox16p0_Zbox8p8_ff.mat`.

**TE apodised / overshoot seed:**
- `results_from_igum\hh_asdrawn_spatial_TE.mat` — `x_um` (1,1966) −50.50…+50.50,
  `envelope`, `energy_density`, `fwhm_um` = 15.2655, `polarization` = 'TE', plus a `note`
  field with the full provenance string. TM twin: `hh_asdrawn_spatial_TM.mat`
  (fwhm 17.222 µm).
- `results_from_igum\itai_hh_apod_task0_slice.mat` — same three arrays (1,1966), but
  **T 1.0256 > 1**, the voided undersized-box round.
- The re-optimised Nt60/W20 TE envelope (IGUM 63722) is **embedded as literal arrays
  inside** `matlab_plotting\plot_itai_hh_nt60w20_envelope.m` (x and envelope, ~±48 µm) —
  usable, but there is no .mat; the figures are
  `results_from_igum\itai_hh_nt60w20_envelope.{fig,png}`.

## 15. Transverse field at z=0 near the centre, complex E

**TM N=80, corr-400, W800 (avg), Ybox 16 / Zbox 8.8 — EXISTS-ARRAY, complex, all three
components:**
- `results_from_athena\scat_i_fieldmaps\results\result_N80_TM_avg_Ybox16p0_Zbox8p8_SLICE.npz`
  — `xy_x` (1626) ±41.976 µm, `xy_y` (311) ±7.610 µm,
  `xy_Ex`/`xy_Ey`/`xy_Ez` (1626, 311) complex64 at **z = 0**; plus `xz_*` (1626, 163) at
  y = 0; `xy_lam_used_nm` = 1558.545, `resonance_wavelength_nm` = 1558.6117. Job 123991
  (header of `matlab_plotting\plot_scat_i_fieldmaps.m`). Two sibling devices in the same
  dir.
- Richer copies including the YZ plane:
  `results_from_athena\comb_physics_rethink\data\result_N80_TM_avg_Ybox16p0_Zbox8p8_PLANES_RES.npz`
  (+2 siblings) — `xy_ax1/ax2`, `xy_Ex/Ey/Ez`, `xz_*`, `yz_ax1(311)/ax2(163)`,
  `yz_Ex/Ey/Ez`, `lam_res_nm`, `lam_slice_nm`, `T`, `fwhm_um`.
- Intensity-only siblings (no phase): `..._planes.npz/.mat` in the same dirs.
- `results_from_athena\tm_field_export\results\result_N80_TM_avg_Ybox6p8_Zbox8p8_EZSLICE.mat`
  — `x`(652), `y`(127), `Ez_re`/`Ez_im` (652,127) at z=0, `lam_used_nm`,
  `resonance_wavelength_nm`.

**Note on the request's "complex E_y preferred": for TM the dominant component is E_z, not
E_y.** All three components are stored in the npz files above, so either can be taken.

- **TM N=150 corr-400 (trench study):**
  `results_from_athena\trench_n150_full\results\result_N150_TM_avg_Ybox8p0_Zbox8p8_PLANES.npz`
  — **intensity only** (`xy_E2` (3026,151), `xy_x`, `xy_second_axis`, `xz_E2`), no complex.
- **TM corr-325, N 165–172 (the comb/hole baseline): NOT FOUND as a 2-D field.**
  `results_from_athena\comb_q3db\results\result_N165_TM_avg_C325_Ybox8p0_Zbox8p8.mat` and
  `comb_q3db_lock\results\result_N17{1,2}_...mat` contain only the 1-D `field_x`(3305) /
  `field_envelope_1D` / `field_energy_density_1D` (±85.4 µm), λ_res 1559.001,
  T 0.4906, fwhm 19.970 µm (N165 ctrl) and λ 1559.031, T 0.5003, fwhm 19.763 µm
  (N172 comb pair). **No xy/yz plane and no complex E exists at corr-325.**
- **TE seed transverse plane: NOT FOUND.** No SLICE/PLANES file in any TE directory —
  only 1-D envelopes. The nearest TE 2-D artefacts are PNG/FIG only.

## 16. TM far field, the ±11.5° peaks — EXISTS-ARRAY (and complex)

- **The plot source for the claim:**
  `results_from_athena\scat_e_validate\figures\scat_farfield_data.mat` — `w800_ux` (1,201),
  `w800_P` (1,201), `w800_needle_neg = −0.980`, `w800_needle_pos = +0.980`, and the same
  four for `w1050`. P = side-monitor E² integrated over the second cosine.
  **The stored axis is the direction cosine u_x, not degrees**; the "11.5°" label is
  arccos(0.980) = 11.48° measured **from the waveguide x-axis**.
- **Underlying full 2-D maps, intensity + complex:** any `..._ff.mat`, e.g.
  `results_from_athena\air_trench_dscan\results\result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat`
  — struct `farfield_side` = {`E2` (201,201), `ux` (201) ∈ [−1,1], `uy` (201) ∈ [−1,1],
  `lam`, `Ex_c`, `Ey_c`, `Ez_c` (201,201) complex128}; the same for `farfield_top`; plus
  `polarimetry_side/top` (`x` (1163), `prof_total`/`prof_tm`/`prof_te`).
  Far-field λ = 1558.545 nm vs `resonance_wavelength_nm` 1558.6117 — the 1-freq-point
  + source-limits offset, 0.07 nm here.
- **Axis caveat (repeat):** the side monitor is Y-normal, so its struct `uy` is physically
  **u_z** (the MATLAB script labels the axis `u_z`). On the top monitor `uy` is genuinely
  u_y.
- Reduced/pre-sliced copies:
  `results_from_athena\comb_physics_rethink\data\*_ff.npz` with
  `farfield_side_ux/uy/E2/Ex_c/Ey_c/Ez_c` and `farfield_top_*`.
- Trench-figure map set: `results_from_athena\air_trench_dscan\figures\trench_ff_maps.mat`
  — `w800_ctrl_E2/_ux/_uy`, `w800_tr_*`, `w1050_ctrl_*`, `w1050_tr_*`, each
  (201,201)+(1,201). Jobs 124379 (W800 d-scan) + 124400 (W1050), 2026-07-22.
- Scale: ~600 TM `*_ff.mat` files across ~50 study dirs (largest: scat_c3_ygrid 156,
  scat_c_response 98, scat_b_gates 56).
- Related k-space diagnostic:
  `results_from_athena\radiation_kspace_diag\kspace_diag_N80_TM.mat` — `x_um`(3359),
  `y_um`(87), `kx_um`, `spec_kx`, `kc_um`, `beta_um`, `E_leak_abs2` (3359,87),
  `Ez_axis_re/im`.

## 17. TE far field — EXISTS-ARRAY, but sparse (6 files)

See the far-field subsection under item 7 above for the file list and the status of the
"radiates vertically" claim. Structurally identical to TM: `farfield_side`/`farfield_top`
with `E2` (201,201), `ux`, `uy` (201), `lam`, complex `Ex_c/Ey_c/Ez_c`. TE control:
λ_res 1559.389, T 0.8756, ff λ 1558.936, N=80, corr 300 nm.
**There is no TE equivalent of the ±11.5° needle** — the TE map is a broad double hump
at |u_x| ≈ 0.75–0.8 (registered in `runners\scatterers\scat_z_teffmap.py`'s docstring).

## 18. Provenance per profile

| item | run / job | λ of extraction | monitor + position | domain |
|---|---|---|---|---|
| TE N80 envelope (`te_q3db_20um`) | Athena 128580/128581/128593/128730/128733 | λ_res 1558.926 (nearest of 501 recorded) | `field_profile`, 2D Z-normal, z=0, y=0, x span ~±41 µm | full (unfolded) |
| TE HH as-drawn envelope | IGUM, `runners\sweeps\itai_hh_apod.py` (docstring says job TBD) | λ_res 1559.626 | same `field_profile` | full |
| TE Nt60/W20 envelope | IGUM 63722 | λ_res per script | `field_profile` | full |
| TM N80 corr-400 xy/xz/yz complex planes | Athena 123991 | slice λ 1558.545 vs λ_res 1558.612 (nearest recorded of 5 pts) | `field_profile_2D_XY` z=0 / `..._XZ_side` y=0 / `..._YZ_cross` x=cavity_length/2 | full (±7.61 µm y, ±4.01 µm z) |
| TM N150 planes | Athena 124551+ | `xy_plane_lambda_nm` stored with `xy_plane_offset_pm` | same three monitors | full |
| TM corr-325 N165/171/172 | `runners\metal_mirror\comb_q3db*.py`; 130458 for the Q 13930 ctrl | λ_res 1559.001 / 1559.031 | `field_profile` only | full, **1-D only** |
| TM far field (all `*_ff.mat`) | per study dir; air_trench = 124379/124400 | ff monitor records **1 freq pt** (`lam` in struct; 1558.545 vs λ_res 1558.612) | `side_monitor` Y-normal at y = 1.5·W_wide; `top_monitor` Z-normal at z = 1.5·h; both x = cavity_length/2 | far field on the full ux–uy grid [−1,1]², 201×201 |
| TE far field | IGUM `scat_z_teffmap` / `scat_te_comb` (docstrings say TBD) | ff λ 1558.936 vs λ_res 1559.39–1559.42 | same two monitors | full |

**Gaps worth flagging:** no complex transverse field at corr-325 N165–172, and none for
any TE device. Both would need new runs (`record_2d_fields_top_and_cross` plus the
server-side SLICE extraction that produced the corr-400 npz files). Several runner
docstrings still say `Job(s): TBD`; the recoverable job IDs live in the MATLAB plot
headers and the `layouts/*.log` files.

---

# B. Future checks — feasibility and what is new code

Monitor architecture today is item 8 above. Against that baseline:

## 19. Closed 6-face box of complex E and H in the cladding — ABSENT, new code

Only **two** of the six faces exist (`side_monitor` at +y, `top_monitor` at +z), and only
when `record_farfield=True`, which the seed does not set. Missing: −y, −z, +x, −x.

**The symmetry BCs make this subtler than it looks.** With `use_symmetry` and
`use_z_symmetry` on (`bragg_device.py:739-749`), the −y and −z half-spaces are folded
away in the *simulation*; monitor output is unfolded by Lumerical (MEASURED — stored
`xy_y` spans ±7.610 µm), so recorded planes are full-domain, but a monitor **placed** at
−y or −z is inside the folded region. Cleanest route: run the box with symmetry OFF, or
record the +y/+z faces and reconstruct the mirror faces by parity.

E and H are both retrievable — `sim_helpers.extract_monitor_polarimetry:119-120` already
does `getresult(mon,"E")` and `getresult(mon,"H")` on one face. But that helper
deliberately reduces to scalars and 1-D profiles server-side because full complex maps on
an arm-length monitor are hundreds of MB. **A raw 6-face complex export must be cropped in
x or downsampled**, or it will not survive the ~0.5–1 MB/s link.

**Mode-expansion amplitudes on the two end faces already exist:**
`bragg_device.get_s_and_t_matrix` reads `"expansion for port monitor"` on Port_1/Port_2.

**The apodisation window does not currently exist** (item 4) — it would have to be set
explicitly and then recorded alongside the faces.

## 20. Complex E in the core on z=0 and on the two sidewall planes over the central 40 µm

- **z = 0 plane: EXISTS** as `field_profile_2D_XY` (2D Z-normal at z=0), and
  `field_2d_x_span_m` is the crop knob — **40 µm is a one-line config change**. The
  extraction that produces complex npz output is the one behind the corr-400 SLICE files.
- **The two sidewall planes (y = ± w/2): ABSENT.** `field_profile_2D_XZ_side` sits at
  **y = 0**, not at the sidewall. New code: two Y-normal profile monitors at
  y = ±core_width/2. With y-symmetry ON only one of them is physical.

## 21. Stored energy in a fixed cavity region and net Poynting flux through the box

- **Poynting: HALF EXISTS.** `extract_monitor_polarimetry` already surface-integrates S_y
  and S_z per face and returns `P_total`/`P_tm`/`P_te` plus 1-D profiles. Summing over a
  closed box is trivial **once item 19 provides six faces**.
- **Stored energy: ABSENT.** There is no energy monitor and no
  `U = (1/4)(ε|E|² + µ|H|²)` integration anywhere. It needs a 3-D monitor over the
  cavity region (`record_3d_fields` + `field_3d_span_m` crop, currently off) **plus
  ε(r)** — and there is no index monitor in the scene today, so ε must either come
  from the builder's known geometry or from a new `addindex()`.

## 22. Parity of the TE mode in x at the pi shift, from the near field — NO NEW MONITOR NEEDED

Directly obtainable from the **existing** `field_profile` (2D Z-normal, z=0, spanning the
whole device, 501 frequency points): take complex E_y(x) at the resonance index and test
E_y(−x) against E_y(+x). The monitor currently feeds
`extract_and_process_field_profile`, which reduces to an intensity envelope — but the
complex array comes from the same `getresult(..., "E")` call, so **this is a
post-processing addition only**.

Note x-parity is **not** a symmetry BC here (x min/max are PML), so nothing is folded in x
— the stored array is the full range.

## 23. Plane wave from a chosen direction inside the oxide as an adjoint source

Injectable in principle, but **not wired into the lumopt2 adjoint today** — see item 13d
for exactly what would have to be written.

**The blocking issue is symmetry.** An off-axis plane wave with a general (kx, ky, kz)
breaks **both** the y=0 and z=0 mirror planes, so `use_symmetry` and `use_z_symmetry` must
go OFF (`bragg_device.py:739-749`), costing roughly a 4× volume/time increase.

- A k lying in the **y = 0 plane** keeps the y symmetry if its polarization parity matches
  (TE: y=0 Anti-Symmetric).
- A k tilted only in x–z keeps y-symmetry and breaks z.

**Recommendation: restrict the adjoint plane-wave directions to the y=0 plane and keep one
symmetry.** Also re-check `check_import_src_injects` (`L:1946-1970`) — a source plane
sitting on an anti-symmetric BC injects nothing, silently.

## 24. Finite-difference pair: 20 nm centre oxide circle at ±h in area

Supported by the existing scatterer machinery (`scatterer_radius_nm`, material,
`scatterer_y_m`), but two documented traps apply directly:

1. **This sits exactly where the noise-floor rule bites.** A candidate effect near the
   numerical floor is not a result. The prior incident (2026-07-02) measured a dx=50 nm
   jitter of 0.0018 in T, collapsing to 0.0001 at dx≈35 nm. So run it as the documented
   two-step: **measure the floor by repeating points offset by half a mesh cell at
   `simulation_mode="optimization"` (dx=50 nm), then confirm survivors at `"accurate"`
   (dx≈35 nm)** — which is precisely the "Q, width and resonance versus mesh" the question
   asks for.
2. **The no-scatterer control REQUIRES `scatterer_radius_nm=[0.0]` explicitly**, or the
   scatterer defaults ON and the "control" is not a control. Guard by checking the
   `generate_file_tag()` match.

A 20 nm circle is small relative to a 50 nm cell — EXPECTED: at dx=50 nm it will be at or
under the meshing threshold and the ±h area perturbation may not be resolved at all. The
accurate-mesh leg is likely to be load-bearing here, not optional.

## 25. Runtime per seed run on Athena

**MEASURED** — `sacct`, last 30 days, COMPLETED `lum_pipeline_array` tasks, n = 105
single-simulation rows (optimizer/campaign driver tasks excluded: they carry billing 88–89
against 45 for a sim, and run 3:43–7:48).

| lane | n | min | **median** | max |
|---|---|---|---|---|
| a100-public | 82 | 20.3 min | **37.9 min = 0.63 GPU-h** | 47.6 min |
| h200-shared | 21 | 9.9 min | **18.5 min = 0.31 GPU-h** | 64.0 min |
| l40s-public | 2 | 19.7 min | — | 19.8 min |
| all GPU types | 105 | 9.9 min | 36.9 min | 64.0 min |

**h200 is ≈0.50× a100 wall-clock**, proven within a single split array (job 148812: a100
tasks 32.9–35.0 min, h200 tasks 16.8–18.5 min — same study, same numerics). The h200 tail
at 54–64 min is jobs 146639/146681, the long N172 corr-325 comb device — longer device,
not a different lane.

**Calibration budget (DERIVED): ≈0.65 GPU-h per sim mixed-lane, ≈0.8 GPU-h for a long
(N > 170) device.** A 10-point calibration ladder is ≈6 GPU-h on a100 or ≈3 GPU-h on h200.

**No wall-clock runtime is stored locally anywhere.** `results_from_*/**/layouts/*.log`
contain only Lumerical *physical* simulation time (e.g. `1.370941e-11s of Simulation
Time`) and "Max time remaining" progress lines; there are no `slurm-*.out` copies and no
`elapsed` field in any json/jsonl. **sacct is the only runtime record** — worth fixing if
the budget number needs to be auditable later.
