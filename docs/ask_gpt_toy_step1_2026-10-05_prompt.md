THIRD FOLLOW-UP (2026-10-05 evening): first hardware results of v3 + one calibration puzzle.
Context: briefing log at the bottom of docs/ASK_GPT_BRIEF.md and your three earlier answers in
docs/ (ask_gpt_*_answer.md). Since your v3 code review Claude fixed: width-violating trials
rejected regardless of FOM; missing/curved c_W = loud DEGRADED row with the lambda bound halved;
1.5x-radius growth probe; feasibility-aware stop; driver-side recenter of accepted points;
Broyden twin-lambda correction; and — after the v3 smoke's first hardware iterate dropped the
lambda row and moved the peak +3.75 nm in one step — the lambda bound is never dropped
(restoration mode "restore_lam" with the resonance-neutral part of the width row as objective)
and the log callback no longer restarts the campaign before the driver classifies a trial
(commits 721dfe9, 2a9cef3; gates gate_v3_local V4-V16). Rules as before: verdict first,
evidence, FROM-CODE / FROM-DATA / INFERENCE labels, <= 150 lines, adversarial.

DATA A — v3 smoke, N=70 surrogate (job 169002, OLD build; plumbing only): accepted step with
predicted dFOM +1.47e-3, measured +1.54e-3 (ratio 1.05); width 19.664 -> 19.395 um (band
upper edge 19.403).

DATA B — v3 TOY on the real TE S1 device (job 169105, fixed build; PVA mesher, 10 nm window,
501 points = 20 pm grid, trust radius 10 nm, lambda bound 0.25 nm halved to 0.125 because
cw_curved=True, width band [18.8386, 19.4034] um):
  eval 0: fom 0.859962, lam_pk 1560.883985, t_pk 0.905444, Q_L 1538.45, Q_i 31752, loss 0.093685,
          spectral FWHM 1.014584 nm, fwhm_env 19.119801 um, softW 18.752813, softW(twin) 18.737004,
          twin_lam 1560.94, c_W 0.29574 um/nm (curved flag True)
  step 0: mode ascent, multipliers [0, 0] (no row active), predicted dFOM +0.00694,
          predicted rows [dW +0.0873 um, dlam +0.0158 nm]
  eval 1: fom 0.866987, lam_pk 1560.903984, t_pk 0.912487, Q_L 1540.12, Q_i 34410, loss 0.086726,
          spectral FWHM 1.013493 nm, fwhm_env 19.197581 um, softW 18.836432, softW(twin) 18.818075,
          twin_lam 1560.883985, c_W 0.245475 um/nm (curved flag True)
  => measured dFOM +0.007025 (ratio 1.012); d fwhm_env +0.0778 um (ratio 0.89), d softW +0.0836
     (0.96), d softW(twin) +0.0811; dlam +0.020 nm (one 20 pm grid step; predicted 0.0158).
  The trial was accepted; iterate 1's gradients are being computed. Note: the optimizer's FOM
  is ~0.95 x the logged t_pk at the peak in every run of this programme (TM too) — a
  normalisation difference between lumopt2's T array and the logged |S21|^2.

DATA C — calibration puzzle on the SECOND seed (S2 = apodized "overshoot" device, Q_L 7694,
spectral linewidth 0.203 nm; S1 has 1.014 nm). Width-adjoint gate (J = -softW at the single-
lambda twin monitor pinned at the scan centre), same three parameters as S1 (corr_1, shift_1,
cavity width), central finite differences with +-4 nm steps:
  S1: FD [8.848e-4, 1.60834e-2, 8.00252e-3]  Re [9.1042e-4, 1.634131e-2, 8.15273e-3]
      Im [8.0486e-5, 7.80897e-3, 3.34312e-3]  -> one complex constant (0.96672, 0.03656)
      fits to <= 0.2 % per parameter.
  S2: FD [5.9165e-4, 9.9079e-4, 1.028914e-2]  Re [8.416e-4, 1.2551e-3, 1.115664e-2]
      Im [1.1752e-4, 1.3423e-4, 1.16977e-3]   -> Re/FD = 1.42 / 1.27 / 1.08, Im/Re =
      0.14 / 0.107 / 0.105 (quadratures nearly parallel, exact-LSQ fit ill-conditioned:
      a 1.62, b -6.71, held-out errors up to -282 %). S1's constant applied to S2 leaves
      +38 / +23 / +5 %.
  On S2 the operating point has corr_1 = 5 nm (the seed's first tooth has zero corrugation,
  lifted to 5 nm so the FD leg stays >= 0), shift_1 = 5 nm, cavity width +10 nm.
  Claude's hypothesis: the adjoint is fine and the FD is nonlinear — softW is sampled at a
  FIXED lambda, and a +-4 nm move of a cavity-adjacent tooth shifts the resonance by a large
  fraction of S2's 0.2 nm linewidth, while the cavity-width parameter barely moves it (it
  agrees best). A rerun of the FD at +-1 nm is in progress (job 169360).

QUESTIONS
H1. Read DATA B against the criteria you set in G8. What does one step with ratios 1.01 /
    0.89 / ~1 establish, and what does it NOT establish? Which of your earlier concerns does
    it address, and which remain untested by this step (name the specific next observation
    that would test each)?
H2. The c_W "curved" flag fired on both evaluations (c_W 0.296 then 0.245 um/nm from softW at
    lambda_pk and +-1/4 linewidth). Is halving the lambda bound the right response, or should
    the estimator change (narrower stencil, 5 points, fit in frequency, use fwhm_env instead of
    softW)? Is a ~0.25-0.30 um/nm slope physically plausible for a 19 um mode with a 1 nm line?
H3. The step was not limited by either row (multipliers 0) and used the full 10 nm radius; the
    1.5x probe will decide growth. With predicted gain +0.0069 per 10 nm step and noise ~1e-5,
    what radius policy would you run for the remaining 3 iterates of this toy, and what would
    you change before a long campaign (the user wants results; each iterate costs ~1.5-2 h)?
H4. DATA C: assess Claude's FD-nonlinearity hypothesis against the alternatives (a real
    class-dependent adjoint error on the apodized device; the near-zero-corrugation tooth; the
    twin wavelength sitting off the resonance of the detuned operating point; softW's
    smoothing interacting with a narrower line). What does each predict for the +-1 nm rerun,
    quantitatively? If the +-1 nm ratios come back ~1.0, what FD-step rule should the gate
    use (in units of linewidth), and does the S2 port-adjoint gate (softmax FOM, +-2 nm steps,
    still running) need the same treatment?
H5. Anything in DATA A-C that you read as a warning sign I have not flagged.
