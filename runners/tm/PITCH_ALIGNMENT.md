# Aligning a polarization's resonance to a target wavelength (pitch correction)

How to move a Bragg/cavity resonance onto a target wavelength by adjusting the
grating **pitch** — e.g. shifting the **TM** resonance onto the **TE** wavelength.
This is the repeatable version of the FDE-calibrate → FDTD-confirm → secant loop.

## Physics

The cavity/Bragg resonance sits at

    λ_res = 2 · n̄_eff · Λ          (n̄_eff = grating-average effective index, Λ = pitch)

To move λ_res up, increase Λ. The catch: **n̄_eff is wavelength-dependent even
though the material index is held constant** — that's *waveguide* (geometric)
dispersion. So the naive "scale the pitch by the wavelength ratio" (the one-step
formula `Λ' = λ_target / (2·n̄_eff(λ_old))`) **undershoots**, because it uses
n̄_eff at the *old* wavelength. FDE gives n̄_eff(λ) so you can evaluate it at the
*target* — that's the whole point of the calibration.

## Procedure

1. **Baseline comparison** — get the resonances at the baseline pitch (500 nm):
   `bash athena/deploy_athena.sh --option2 --run=run_tm_vs_te --pol-array`
   then `python -m runners.tm.run_tm_vs_te --stitch <results_dir>`.

2. **FDE calibration → predicted pitch** (local, needs a MODE license):
   `python -m runners.tm.calibrate_neff --out <results_dir>`
   Reads the TE/TM anchors from the result `.mat`, runs FDE on the wide+narrow
   tooth cross-sections (constant indices), **anchors the FDE curve to the FDTD
   n_eff**, and prints the recommended pitch. It saves `neff_calibration.mat`.
   - **Reuse (no MODE rerun):** for a *new* target on the *same* geometry,
     `python -m runners.tm.calibrate_neff --out <dir> --predict-target <λ_nm>`
     reads the saved calibration and prints the pitch.

3. **Confirm with one FDTD run** at the predicted pitch:
   `TM_PITCH_NM=<pitch> bash athena/deploy_athena.sh --option2 --run=run_tm`
   The pitch flows through `TM_PITCH_NM` and is encoded in the filename
   (`result_..._P<pitch>.mat`, 0.1 nm resolution) so iterations never collide.

4. **Secant / bracket refine** — FDE gets within ~1 nm of pitch (~3 nm of λ); the
   FDTD run cleans up the residual. With one point **below** and one **above** the
   target, false-position on the bracket gives the next pitch:

       Λ_next = Λ_lo + (λ_target − λ_lo)·(Λ_hi − Λ_lo) / (λ_hi − λ_lo)

   Repeat step 3 (usually 1 iteration). λ_res is read off the scan grid (25 pm at
   the default 6001-pt / 150-nm window), so practical precision is ~0.05–0.1 nm;
   narrow the window (`COMPARE_*` in `_tm_vs_te_common.py`) to go finer.

## Worked example (this device: 500 nm pitch, 80 periods, n_core 1.9963 / n_clad 1.444)

Goal: TM resonance onto λ_TE = 1570.74 nm.

| pitch (nm) | method                | TM λ_res (nm) | off (nm) |
|-----------:|-----------------------|--------------:|---------:|
| 500.00     | baseline              | 1523.57       | −47.2    |
| 515.3      | one-step λ-ratio      | 1563.74       | −7.3     |  ← undershoots (ignores dispersion)
| 519.18     | FDE (anchored)        | 1573.58       | +2.8     |
| 518.09     | FDTD secant           | 1570.00       | −0.75    |
| 518.32     | bracketed             | ≈ target      | ~0       |

n̄_eff,TM fell 1.5236 (@1523.6) → 1.5127 (@1570.7) over the 47 nm — that drop is
the waveguide dispersion the one-step formula misses.

## Files

- `calibrate_neff.py` — FDE n_eff(λ) for TE & TM + pitch recommendation; `--predict-target` reuse.
- `neff_calibration.mat` — the saved calibration (reusable for new targets, same geometry).
- `run_tm.py` / `run_te.py` — single-polarization scans (`TM_PITCH_NM` overrides pitch).
- `_tm_vs_te_common.py` — `run_one_scan`, `run_stitch`, `stitch_dir`, `COMPARE_*` window.
