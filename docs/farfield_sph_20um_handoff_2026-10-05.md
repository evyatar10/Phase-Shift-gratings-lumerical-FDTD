# Far-field spherical-harmonic study of three ~20 µm-mode pi-shift gratings — HANDOFF (2026-10-05)

Session 2026-09-29 → 2026-10-05. Cluster: Athena. All numbers below are MEASURED from the files
named unless marked DERIVED. Status: round A complete, analysed, figures delivered. Nothing running.

## 1. Question asked
Same mode width (~20 µm) and same length (N = 98 periods/side) for three devices; record the complex
far field AT RESONANCE; decompose it into 3D vector spherical harmonics (electric/magnetic multipoles)
and report the power fraction per (type, l, m). First determine the TE far-field box (TM's was settled).

## 2. Devices (all N = 98/side, h 350 nm, n 1.97/1.444, mesh "optimization" dx 50 nm)
| | geometry | box y/z (µm) | λ_res (nm) | T | R | radiated (1−T−R) | Q_L | spatial FWHM |
|---|---|---|---|---|---|---|---|---|
| A TE plain | pitch 500, W800, corr 250 | 6.8 / 6.81 | 1559.990 | 0.912 | 0.003 | 8.6 % | 1694 | 19.13 µm |
| B TE overshoot | Itai Nt60 re-optimized profile, job-63722 geometry untouched, pitch 491.06, avg W 1000 | 6.8 / 6.81 | 1559.867 | 0.973 | 0.000 | 2.7 % | 7696 | 19.63 µm |
| C TM plain | pitch 516.83, W800, corr 325 | 8.0 / 8.8 | 1559.065 | 0.915 | 0.002 | 8.3 % | 1652 | 19.18 µm |

Widths are 19.1–19.6 µm, not 20: at N = 98 the mode is end-truncated (stored 19.99 µm for A is at N = 166).
Row B reproduces its stored control (63722: 1559.8597 / 0.97498 / Q 7680 / 19.633) exactly.

## 3. Runs (Athena)
- 164883 smoke FAIL — `apply_monitor_overrides` (sim_helpers) silently reset the far-field monitors to
  1 frequency point; also the 2D field planes made a 197 MB .mat at N = 10. Both fixed.
- 164891 smoke PASS (11-point monitors, projection at point 6/11, 0.8 MB).
- 164893 round A, 6 tasks: TE box ladder on A at 6.8/6.81, 8.0/8.8, 10.0/10.8, 12.0/12.8 + B + C.
  Solve times 15–45 min; all far fields within 0.05 nm (≤ 5 % of a linewidth) of the found resonance.
- Files: `results_from_athena/farfield_sph_20um/results/result_N98_*_ff.mat` (+ `_multipoles.csv` each).

## 4. Engine change (default-inert; snapshot gate 6/6 byte-identical, twice) — UNCOMMITTED
`FarFieldConfig.farfield_freq_points` (default 1 = legacy band-centre). > 1: far-field monitors record the
band; `sim_helpers.extract_farfield(..., lam_target_m)` projects at the recorded point nearest
`resonance_wavelength_nm` (passed from `post_processing`). Reason: every pre-existing *_ff.mat in the
repo was projected at the scan band CENTRE (stored TE example 41 % of a linewidth off). Files touched:
simulation_config.py, bragg_device.py, sim_helpers.py (also the override block), post_processing.py.
KNOWN GAP: `extract_monitor_polarimetry` returns nothing for multi-point monitors → `polarimetry_*`
absent from these .mat files. Fix before the next far-field run (select the same λ index).

## 5. Tool — `python_tools/farfield_multipole.py <mat> [--lmax 200] [--csv]`
Full sphere from top (+z) and side (+y) planar monitors (nearest-normal patchwork; −y/−z halves by the
device's mirror symmetry with the parity sign READ from the data); Jackson X_lm / n×X_lm projection
(Legendre recurrence, φ by FFT). Validated: Legendre + derivative identity vs scipy; five synthetic
dipoles → 100 % in l = 1 with the right E/M type, Parseval 1.000, parities correct. On real data:
transversality ~1e-7, Parseval 0.999, parities s_y/s_z = −1/+1 (TE), +1/−1 (TM). Reported caveat: the
band within ~6° of the waveguide axis is in neither monitor's half-space (0.1–0.6 % of power TE, 4.8 % TM).

## 6. Results
**TE box verdict:** converged at 6.8/6.81 µm — vs 12/12.8 every harmonic within 0.8 points, T identical.
(Measured on device A only; B was run at 6.8 only — one run at 10/10.8 would close that.)

**Multipole content (% of radiated power; ±m summed; E/M = electric/magnetic):**
| | A TE plain | B TE overshoot | C TM plain |
|---|---|---|---|
| E / M split | 52 / 48 | 53 / 47 | 50 / 50 |
| l ≤ 1 (dipoles) | 13.7 (all M(1,0)) | 5.2 | 10.1 (E(1,0) 7.1) |
| l ≤ 5 | 92.9 | 29.3 | 80.4 |
| l ≤ 12 | 98 | 51 | 91 |
| power-weighted ⟨l⟩ | 3.7 | 16.3 | 6.3 |
| largest terms | E(3,±3) 19.4, M(1,0) 13.1, E(4,±4) 12.3, M(2,±1) 12.0, E(2,±2) 11.3 | M(2,±1) 5.9, E(8,±8) 3.5, E(4,±2) 3.3, M(1,0) 3.2, E(18,±18) 2.8 | M(2,±2) 18.1, M(3,±3) 11.1, E(3,0) 7.6, E(1,0) 7.1, E(3,±2) 6.4 |

B's spectrum is a slowly decaying comb (l-period ≈ 4–5), 32 % in sectoral terms (|m| = l) reaching l ≈ 25;
no single high harmonic dominates (biggest above l = 12: E(18,±18) 2.8 %).

**Where the radiation comes from (DERIVED: inverse Fourier transform of the far field = flux along x on
the monitor plane, resolution 0.8 µm):** A and C radiate from the π-shift cusp itself — 98 % (A) / 72–88 %
(C) within ±5 µm, the arms do not radiate measurably; B radiates from the apodization lobes at ≈ ±6, ±13,
±25 µm with ~50 % at the cavity. Flux at the monitor ends (±40 µm) is zero in all six monitors → no
contamination from grating entrances or ports; the 80 µm x-span was adequate (run this check on every
far-field result; it is not a general guarantee).

## 7. Interpretation (stated to the user)
- A is a compact radiator (kR ≈ 5) → multipoles are the right basis; its only dipole is the z-MAGNETIC
  dipole, which a single dielectric pillar (electric dipole) cannot cancel → consistent with the pillar
  programs saturating at ~30 % of the leak. Cancelling M(1,0) perfectly would be worth +0.011 in T (DERIVED).
- B: the apodization already removed the low-order leak (2.7 % vs 8.6 %); the residue is high-rank
  (~60 terms for 90 %) and comes from the overshoot lobes — not cancellable by a few scatterers.
- Multi-harmonic cancellation = linear superposition; the response-matrix program
  (`runners/scatterers/`, 97 pillar responses, LS solve) already does it in the (ux,uy) basis. Projecting
  those stored responses into the multipole basis would show which (l,m) pillars can reach. No model yet
  maps "cancel (l,m)" → required Δε; first-order it is the overlap of Δε·E_mode with the regular (l,m)
  multipole wave (buildable from stored 3D fields).

## 8. Figures (results_from_athena/farfield_sph_20um/, .fig + .png; script matlab_plotting/studies/plot_farfield_sph_20um.m)
farfield_sph_20um_triangles (3 devices × E/M, l ≤ 12) · farfield_sph_20um_overshoot_l24 ·
farfield_sph_20um_on_resonance (T(λ) with the far-field λ marked) ·
farfield_sph_20um_triangle_{te_plain,te_overshoot,tm_plain} (apex-up, m = −l..l, log colour) ·
farfield_sph_20um_where_radiation_crosses.png (diagnostic, matplotlib).

## 9. Open items
1. Commit the engine change + runner + tool + plot script (user permission needed).
2. Fix `extract_monitor_polarimetry` for multi-point monitors.
3. Optional: overshoot device at box 10/10.8 (convergence on B itself); project the 97 stored pillar
   responses into the multipole basis; corrugation retune if exactly 20 µm at N = 98 is wanted.
