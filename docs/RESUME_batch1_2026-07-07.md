# RESUME CHECKLIST — BIC/Kerker/CD program (2026-07-07, autonomous run)

**Connection to Athena dropped mid-run (Technion VPN down ~00:40 local).** Jobs
are server-side (sbatch) and keep running. This is the exact-command resume so
nothing is missed. A background watcher (`scratchpad/wait_vpn.sh`) is polling for
VPN return; when it fires, execute the steps below in order.

## State at drop
- **Batch-1 = job 118618** (Athena array 0-34, 35 tasks) RUNNING when the link
  dropped. Runner `runners/sweeps/tm_bic_kerker_batch1.py`.
- **Batch-1b (FW-BIC) = `runners/sweeps/tm_fw_bic_scan.py`** — code-complete,
  smoke-passed, NOT dispatched. Waits for 118618 queue EMPTY (serialize rule).
- Local edits made this session (all smoke-verified, backward-compatible):
  `simulation_config.py` (`avg_corrugation_width_2_m` + two-device abs y-box),
  `experiment_card.py` (`avg_width_2_nm` map+field), `sweep_spec.py` (field),
  `sim_helpers.py` (file tag `_avg2W`). NOT committed (no request).

## STEP 1 — confirm batch-1 finished, download, analyze
```bash
ssh evyatarrubin@athena.technion.ac.il "squeue -u evyatarrubin -r -h | wc -l"   # want 0
bash athena/deploy_athena.sh --results-no-fsp        # downloads to results_from_athena/tm_bic_kerker_batch1/
python python_tools/analyze_batch.py tm_bic_kerker_batch1 --tsv scratchpad/b1.tsv
```
Row map (index → meaning), Δ vs the named control:
- **Controls:** row 0 = rect-1050 opt (control for alternation); row 1 = the
  stack, accurate (control for Huygens + CD).
- **Huygens (2-10, accurate, vs row 1 stack loss 0.0545):** 2=r200@A 3=r250@A
  4=r300@A 5=r250@B 6=r150@L 7=r250@L 8=r250@A+jitter 9=r250@L+jitter 10=2-pair.
  Sites A=(380,1020) B=(620,1400) L=(810,1000) nm. Jitter floor = |row3−row8| and
  |row7−row9|. **WIN** if a Huygens row gives ΔT ≥ +0.003 AND > jitter floor;
  Kerker-helps if r250@A (row3) beats r150@L (row6, the old +0.0026 anchor).
  **KILL Kerker** if row3 ≤ floor and ≤ row6 → directionality adds nothing,
  cap stays +0.0026; do NOT build a larger Huygens array.
- **Vertical 2Λ alternation (11-26, opt, vs row 0 rect-1050 loss ~0.077):**
  11/12=±2 13/14=±4 15/16=±8 17/18=±16 (wide, 16 teeth); 19-24=narrow ±4/8/16;
  25/26=±8 (8 teeth). **WIN** if a row cuts loss below row 0 by > opt jitter with
  a +/- SIGN ASYMMETRY (coherent vertical leak). **KILL 4.1c** if no row beats
  row 0 beyond jitter and +dw≈−dw (incoherent leak) → report null.
- **CD falsifier (31-34, accurate, vs row 1 stack):** 31=+1 32=−1 33=+2 34=−2
  scale. Expected NULL (|Δloss| ≲ 0.003). Confirm null → 4.3 closed as predicted.
- **TE Huygens (27-30, accurate):** 27=control 28=r200 29=r260 30=r320.
  Kerker TE peak at r260 → row 29 should be best.
- **Sanity every row:** analyze_batch flags OFF-WINDOW / DEAD / npk. Any Huygens/
  alternation row with npk≠1 or DEAD → distrust, re-check the .mat.

## STEP 2 — dispatch FW-BIC (only after 118618 queue == 0)
```bash
bash athena/deploy_athena.sh --status                # re-confirm empty
bash athena/deploy_athena.sh --option3 --spec=runners.sweeps.tm_fw_bic_scan
```
End the turn with the job/array ID + task count (20).

## STEP 3 — FW-BIC row 0 is a BOX-SANITY GATE (check before trusting the grid)
When the first FW results land:
```bash
bash athena/deploy_athena.sh --results-no-fsp
python python_tools/analyze_batch.py tm_fw_bic_scan --tsv scratchpad/fw.tsv
```
- **Row 0 (gap 1.5, avg_2 1000) MUST be physical:** T ≤ 1, loss ~0.077 (isolated
  rect-1050 at this pair box), resonance in window. If T>1 / negative loss /
  off-window → the two-device box is wrong (y-pad↔z coupling). **HALT the FW
  interpretation** and run a pair domain-convergence check (vary y_span_override
  10→12→14 µm at one cell) before believing any FW number.
- **Degenerate row 14 (avg_2 800) is the positive control:** expect a DOUBLET
  (npk=2) — confirms coupling is real.
- **FW WIN:** any cell (rows 2-13) with device-1 loss BELOW the reference (rows
  0/1) AND one usable peak in window (partner ≥15 nm away) AND fwhm_m within ~1%.
  Map device-2 λ vs avg_2 from the reference rows first.
- **KILL FW-BIC** if NO cell across the whole (gap 0.4-0.7 × avg_2 900-1300)
  plane drops device-1 loss below the reference → radiation-pattern overlap ρ is
  too small (< ~0.8), FW-BIC dead for this side-coupling geometry. Report the
  null honestly (the CMT flagged ρ as the unknowable risk).

## STEP 4 — accurate confirm of survivors (batch-1c, build reactively)
For any batch-1 or FW survivor: copy its winning row into a small accurate-mesh
array WITH a half-mesh (17 nm at accurate dx≈35) jitter partner + in-study
control, per CLAUDE.md §2. Only then is an effect "claimed."

## STEP 5 — write FINDINGS + MATLAB figs, report full local paths.
Plots: scatterer rows → `matlab_plotting/plot_scatterer_scan.m`; others need a
simple loss-vs-parameter plot (build when data is in hand).
```
