FOURTH FOLLOW-UP (2026-10-06): the v3 toy finished, a long campaign started from it, and the S2 calibration
puzzle has a candidate cause. Please VERIFY what was done — find mistakes, not reassurance.
Context: briefing log at the bottom of docs/ASK_GPT_BRIEF.md, your earlier answers in docs/ask_gpt_*_answer.md.
Code changed since your last answer: `git log 2a9cef3..a4b006b` (commits 860ff86, 2fd827a, 930120c, 59a5cf3,
a4b006b). Main files: runners/lumopt2_design/lumopt2_design.py (run_projected), v3_step.py (restore_lam_step),
validate_te.py, campaign_te_s1.py, gates/gate_v3_local.py. Rules as before: verdict first, evidence,
FROM-CODE / FROM-DATA / INFERENCE labels, <= 150 lines, adversarial.

WHAT CHANGED (after your checkpoint answer)
C1 c_W: centred slopes at 1/4, 1/8, 1/16 linewidth half-spans from the same forward; keep the narrowest slope
   that agrees with the next wider one to 20 %, else flag curved (the running toy used the OLD estimator).
C2 growth probe only when both the current and the 1.5x-radius QP are full "ascent".
C3 Broyden twin-lambda correction only when c_W was not flagged at the accepted point.
C4 restore_lam_step (your H5.3): projected QP first; LP computes the attainable width change y*; if the QP gets
   less than y_t = sgn*min(|need|, 0.5*|y*|), return the minimum D-norm step with gW.d = y_t under the same
   lambda band, box and cap; fallback = LP vertex scaled to y_t.
C5 gradient vectors of every accepted iterate are now saved (they were not, so the anomaly in DATA A is
   undiagnosable from stored data).
C6 acceptance tolerance: the QP band is the INNER band (margin 0.10 um inside the +-2 % spec, here
   [18.8386, 19.4034] um); a trial is now width-rejected only if its violation of the inner band exceeds
   marg/2 = 0.05 um (before: any violation).
C7 width-row reuse switched OFF under v3 (its gate needs |dW| <= 0.025 um since the last fresh solve; every v3
   step moved W by 0.08-0.12 um, so it never opened, and the toy's marker check failed on "never reused").
C8 the S2 C-recipe gates are now centred at the gate point's own measured resonance (DATA C).
C9 the campaign (job 170253) warm-starts from the toy's best width-compliant eval (selection filter = +-2 %
   spec), with the toy's optstate (cap 11.25 nm after one reject).

DATA A — S1 v3 toy (job 169105, real device, PVA, 10 nm / 501-point window, band as above):
  eval  fom        t_pk      fwhm_env   lam_pk       Q_i     c_W (all flagged curved, old estimator)
  0     0.859962   0.905444  19.119801  1560.883985  31752   0.29574
  1     0.866987   0.912487  19.197581  1560.903984  34410   0.245475
  2     0.877210   0.922442  19.314666  1560.943984  38986   0.143214
  3     0.889772   0.934603  19.410603  1560.963985  46450   0.184612
  step 0: cap 10, no row active, dF pred +0.006940 / meas +0.007025; dW pred +0.0873 / meas +0.0778
  step 1: cap 15 (grow), no row active, dF pred +0.011514 / meas +0.010223; dW pred +0.1217 / meas +0.1171
  step 2: cap 22.5 (grow), WIDTH ROW ACTIVE (mu_W 0.003793, mu_lam 0), dF pred +0.014003 / meas +0.012562;
          dW pred +0.0888 (to exactly the inner edge) / meas +0.0959 -> 0.007 um past the inner edge ->
          rejected under the old rule (would pass C6); retry at cap 11.25 proposed, toy then hit max_iter.
  width-row norm |gW|: 0.0461 (it 0), 0.3827 (it 1), 0.4408 (it 2); cosine between successive fresh rows:
  -0.560 (it 0 -> 1), +0.99984 (it 1 -> 2). Step shares it 1 (nm): corr 15.0, cavity 10.48, avg 0.21,
  shift 0.04. The seed is a uniform grating (all free teeth 250 nm corrugation, avg widths 800 nm).
DATA B — campaign iterate-0 forward at toy eval 3 (window centre moved 1560.936 -> 1560.96 = rounded best lambda):
  fom 0.8896468 (toy 0.8897724), t_pk 0.934394 (0.934603), fwhm_env 19.410527 (19.410603), Q_i 46303 (46450).
DATA C — S2 (apodized seed, linewidth ~0.20 nm) field-adjoint gate. The gate point te_point() = seed with
  shift_1 = shift_30 = 5 nm, cavity +10 nm, corr_1 lifted to 5 nm. One forward there (job 169655):
  lam_pk 1560.8789 vs the gates' centre 1560.464 (the seed's resonance), linewidth 0.2045 nm, T 0.9681,
  W 19.6408. The field adjoint samples softW at ONE wavelength = the scan centre, i.e. ~2 linewidths off.
  FD at that centre: +-4 nm [5.9165e-4, 9.9079e-4, 1.028914e-2]; +-1 nm [-1.994673e-4, 1.602142e-4,
  9.963433e-3]; adjoint Re [8.416e-4, 1.25511e-3, 1.115668e-2] (corr_1, shift_1, cavity width).
  Rerun centred at 1560.8789 (+-1 nm FD, 2 nm / 151-point window) is running. The S2 PORT gate (softmax FOM
  over a 2 nm window, +-2 nm FD) also sat at the old centre; its job timed out before finishing.

QUESTIONS
J1. Verify C1-C9 in the code. Specifically: (a) C6 + C9 together — the campaign starts at a point that only
    C6 makes acceptable; what does the first QP do with W 0.007 um above the inner edge (band upper bound
    negative) — ascent with an active row, or restore_lam, and is either a problem? (b) Is marg/2 the right
    tolerance given the measured width-prediction errors (ratios 0.89 / 0.96 / 1.08 at 0.08-0.12 um moves)?
J2. C4: is "half of the attainable change, minimum D-norm" sound? D = ((hi-lo)/2)^2 per parameter — does the
    D metric make the min-norm step favour wide-bound classes in a way that hurts on hardware?
J3. DATA C: does the off-resonance sampling explain BOTH the step dependence and the bad complex fit? What
    should the recentred rerun show if this is the whole story, and what if it is not? Must the port gate
    be recentred too (its FOM is a softmax over a window that still contains the peak)?
J4. DATA A |gW| 0.046 -> 0.38 with cosine -0.56, then stable. Plausible causes (uniform-seed symmetry, the
    twin wavelength 56 pm ABOVE the peak at it 0 vs 20 pm BELOW at it 1, softW threshold) and how to test
    each cheaply with the vectors now saved.
J5. Campaign: anything here that should make me stop job 170253 now? What should I check on its first three
    iterates, with numeric pass bands?
J6. Anything else you read as a warning sign.
