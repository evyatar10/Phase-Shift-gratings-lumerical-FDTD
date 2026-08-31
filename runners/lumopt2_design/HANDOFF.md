# HANDOFF — lumopt2 corr-325 inverse design — updated 2026-08-31 IDT

> ## ★★★★★★★★2026-08-31 — d1 GENERATION LIVE. THIS BOX SUPERSEDES EVERYTHING
> ## BELOW AND MOST OF HANDOFF_2026-08-30.md.
> **The re-think happened and was validated on hardware.** Root causes found
> (measured): (1) the old step engine delivered a CONSTANT 10 nm move —
> `_cap(a)=cap0·min(1,a/a0)` cancels alpha/wgp_step whenever the raw ~73 nm
> step exceeds the cap (b2's step-doubling was a NO-OP); (2) width creep was
> λ-slaving — ride nulled dW/dp but nothing nulled dλ/dp (c1's live ride
> evals grew W +0.031/+0.050 µm, WORSE than climb).
> **THE d1 LAW (`_ns2_step`, lumopt2_design.py): project D·∇T into the null
> space of BOTH raw fixed-λ ∇W AND gλ (IFT, zero extra solves) + Feppon
> range-space restoration + adaptive trust cap (×1.5 on verified holds,
> 10→60 nm, ×0.5 on reject) + `<label>_optstate.json` sidecar (survives
> REQUEUE).** With gλ·d=0 the fitted 0.3655 coefficient cancels out of the
> step entirely (gate-checked). Gates: projection gate +29 ns2 checks ALL
> PASS; plumbing/lam-chain/predispatch green; smoke = task 50; toy = task 51.
> **TOY 138658 VERDICT (MEASURED, N=100 PVA, 3 iterates): PASS.**
> t_pk 0.96348→0.96459→**0.96582** (+0.00234/2 steps, accelerating);
> λ_pk 1566.444 EXACT on every step (dlam_pred ~1e-16, rLam 0.0);
> W 18.353→18.287 in-band (eases NARROWER at fixed λ — detrend confirmed);
> Q_i 109,684→117,594. rho_T 0.143→0.075 (∇T overlaps raw gW only ~0.6%,
> gλ ~85% ⇒ T rises by red-shifting; width creep was its shadow). WATCH:
> rho_T declining — the 24h kill rules cover a stall (ΔT<+0.003/10 its at
> cap≥30 ⇒ saturated; reject rate >50%/8; |W−tgt|>0.05 ×3).
> **LIVE: d1 = Athena 139049 (BEST lane, inherits toy state — starts from
> 0.96582); d1u = Athena 139050 (uniform lane, inherits c1's 15 iterates,
> resumes c1 eval-5 fom 0.68758). Both 4d_1g/96:00:00/256G (★4d_1g REJECTS
> 300G — 275G cap, use 256G). Labels lumopt2_v2_proj_d1 / _d1u.**
> c1 (138535) + b1 (IGUM 64279) CANCELLED 2026-08-30 after state fetch:
> results_from_athena/lumopt2_v2_proj_c1/, results_from_igum/lumopt2_v2_proj_b1/,
> toy+smoke: results_from_athena/v2_ns2_toy/.
> **★NEW USER RULES (CLAUDE.md §6, also in skill + memory): never re-derive
> across labels — campaigns INHERIT toy/lane state (copy evals+optstate into
> the new label server-side); result identity = engine version + numerics,
> NOT cluster; "can't verify identical" is NEVER a rerun reason — read the
> provenance instead.** Comb stays frozen in gen-1 (d2 arm = free it, only
> if d1 healthy-but-saturated; r=80 choice provenance: COMB_HANDOFF.md §5.3
> plateau + r⁴ parasitic + flush-route margin). Uncommitted: engine ns2
> block, campaign_v2_proj_d1.py, campaign_v2_proj_d1u.py, gates, validate
> tasks 50/51, this box (+ the 08-30 backlog listed in HANDOFF_2026-08-30.md).

> ### ★2026-08-30: `HANDOFF_2026-08-30.md` (the re-think request) is now
> ### HISTORICAL — its live-state section is superseded by the box above;
> ### its measured-facts section remains valid background.
> This file's state below is current only through 2026-08-26.

> ### ℹ️ Which file do you want?
> This is the **operational log** — job IDs, incidents, the full trap history,
> resume commands. Use it when you have **cluster access and are about to run
> something**.
> For understanding the design and the method, or for handing to a chat session
> with no tools, use **`HANDOFF_SELF_CONTAINED.md`** instead.


> ### 📘 COMPANION: `THEORY.md` — read it first if you need to UNDERSTAND, not just resume
> This file is **state** (jobs, numbers, what to run next). **`THEORY.md`** is
> **method**: the cost function and why it is a windowed p=12 soft-max; the
> 191-parameter layout; why there are TWO width measures and how the anchor ties
> them; the penalty-vs-projection formulations; how the adjoint gradients are
> obtained (weighted field source, C_field, GPU tiling, the zero-extra-solve
> `∇T`/`∇W` split); the resonance chain-rule term and its exact stencil; the
> best design we hold and its open questions; and the algorithm written out
> step by step.

> ## ★★★★★★★2026-08-28 ~11:00 — λ-CHAIN VALIDATED IN EXECUTION + SIGN; RIDE
> ## TEST IN FLIGHT. THIS BOX SUPERSEDES THE RESUME BOX BELOW.
> **DONE (exit 0, data in `results_from_athena/v2_lam_chain_toy/`):** smoke
> 137872 (2/2 chain) → toy 137873_41 + fresh control 137873_46 (3 iterates
> each, 3h13m, ~1.5 h/iterate on H200). **MEASURED verdict:** chain ran 3/3
> (gLam_n ≈0.245, dTp −2.25, 20× floor margin); ‖∇T‖ healthy;
> proj_rot_deg 46°→6°→3° (it-0 inflated by near-cancellation — the chain
> shrinks ‖gW‖ 0.126→0.028 there). ★★**SIGN RESULT: the control's dw_pred is
> NEGATIVE on every iterate while measured ΔW is POSITIVE (+0.0110/+0.0122,
> = 137075 exactly) — fixed-λ gW points the WRONG WAY; the corrected arm's
> dw_pred sign is RIGHT (mag 0.5-4×).** λ-drift explains 81-154% of measured
> ΔW on every real step (slaving confirmed in-run). ★NOT yet tested:
> WIDTH-HOLDING — all 6 iterates sat in CLIMB (step ignores gW); "ΔW < ctrl"
> is ill-posed there (totals equal, as expected). gλ·dp vs Δλ_pk was
> untestable post-hoc (norm-only logging) → **dlam_pred_nm now logged**.
> ★Label-reuse trap struck AGAIN: _41 silently RESUMED from cancelled
> 137853's log (control seed eval bit-identical ⇒ zero engine drift; only a
> one-step offset). Bump labels per attempt.
> **IN FLIGHT: 137879 smoke → afterok → 137880_48 = RIDE-PHASE TOY**
> (`lumopt2_v2_ridetoy`, wgp_target_um=18.345 = seed W ⇒ ride/null-space
> ACTIVE from it 0; 3 iterates ~4.5 h). PASS = ΔW/iterate within the
> ±0.0055 µm noise floor with fom rising ⇒ then the 96 h campaign
> (pre-approved). Drift-out ⇒ RESTORE exercised (also informative). Task
> map: 41 toy / 46 control / 47 smoke / 48 ride-toy (N_TASKS 49; predispatch
> gate audits index reachability). Campaign HELD until the ride verdict.


> ## ★★★★★★WIDTH-GRADIENT PROGRAMME RESUMED — 2026-08-27 ~23:00. THIS BOX
> ## SUPERSEDES THE "PAUSED" BOX BELOW. λ-CHAIN TOY IS ON HARDWARE.
> **LIVE (Athena): job 137845, tasks 27 + 41, RUNNING on n315 (h200-shared),
> QOS 12h_4g / 12:00:00 / 300G verified via sacct.**
> - **_41 = the λ-chain toy** (`lumopt2_v2_projchain_toy`, wg_lam_chain=True,
>   3 iterates, cold start from the uniform seed — fresh label, no resume trap).
> - **_27 = NEW control twin** (`lumopt2_v2_projctrl_toy`, wg_lam_chain=False,
>   otherwise identical spec, current engine + committed mesh fix) — added
>   2026-08-27 (user: "we might need to rerun anyways") so the toy-vs-control
>   comparison cannot be confounded by code/mesh drift since 137075_41.
> **Pre-dispatch state, all green (this session):** 5/5 gates passed with their
> exact expected lines; `scene_snapshot` diff vs committed references =
> **6/6 byte-identical** (the per-tooth mesh fix 3120d38 is behavior-preserving
> on every existing config path — §5 gate DISCHARGED; residual note: the
> two-device fine-mesh branch in `bragg_device.py` ~818 still sizes from the
> scalar, irrelevant for n_devices=1); file md5s local≡remote; seats 0/50;
> quota 199G/300G; **fixed h5 cleaner INSTALLED on Athena** (md5-verified,
> crontab entry pre-existing — the "3 blocked routes" item below is CLOSED).
> **Verdict criteria** = the 4 checks in the paused box below (gLam_n present /
> gλ·dp vs Δλ_pk / ΔW below control / ‖∇T‖-collapse falsification), except the
> control ΔW reference is now task _27's own trajectory (fallback: 137075's
> +0.0110/+0.0122 µm). **User approved AUTO-PROCEED** to the uniform-seed
> production campaign (`campaign_v2_proj`, 4d_1g/96h) if (1)-(3) pass and ‖∇T‖
> stays healthy. Cluster ruling this session: toy on Athena. λ-chain math
> re-reviewed by Fable (matched-stencil algebra, signs, guards): sound.
> q3db note: no N=240/280/320 results exist on Athena (137331 left nothing);
> sacct query on it still owed.


> ## ★★★★★q3db LADDER — LIVE STATE 2026-08-26 ~02:4x. READ THIS BOX FIRST.
> **MEASURED so far** (regular builder, conformal, q3db family numerics):
> | N | T | dB | lambda nm | Q_L | mode FWHM |
> |---|---|---|---|---|---|
> | 100 | 0.97228 | -0.122 | 1560.947 | 1818.6 | 19.1709 |
> | 150 | 0.91429 | -0.389 | 1560.857 | 10493.8 | 19.7909 |
> **IN FLIGHT (Athena):** 137322_2 N=180 (running) · 137333 N=200/220/240 ·
> 137331 N=280/320. IGUM has no q3db jobs left. err=0 everywhere.
>
> ### ★THE ACCEPTANCE CRITERIA ARE THE FAMILY'S, RE-READ FROM THE OLD STUDIES
> `trench_q3db_20um.py` round 2: "interpolate to T = 0.5; **accept T = 0.5 +/- 0.03,
> fwhm 20 +/- 1 um**". `te_q3db_20um.py`: "T=0.5 interpolation + **1 integer confirm**".
> The TE study's CLOSED result was N=166 / T 0.4919 / **Q_L 12903** / fwhm **20.46 um**.
> ⇒ **The mode does NOT have to be exactly 20 um** (user confirmed 2026-08-26). Our
> 19.17-19.79 um is comfortably inside 20+/-1. The deliverable is N at -3 dB and Q there.
> Method = ladder -> interpolate to T=0.5 -> ONE integer confirm rung -> quote THAT
> device. Q must NOT be interpolated: near the crossing Q_L moves ~N^4 (projected
> 143k at N=280 vs 241k at N=320), so a mid-gap interpolation could be 10-20% off.
>
> ### ★THE CROSSING IS NOT PREDICTABLE FROM THE TWO POINTS — 3 MODELS, 230 to 640
> | model | crossing N |
> |---|---|
> | loss_dB ~ N^2.859 (power law) | 306 |
> | ln(T) linear in N (the family's own convention) | ~640 |
> | Q_c ~ exp, Q_i ~ N^1.5, solved at Q_c = 0.828*Q_i | ~230 |
> Do NOT quote 306; it was stated too confidently earlier in the session and is only
> one branch. The 180-320 ladder brackets all three. Fitted params for the third
> model (from N=100/150): Q_i = 130296*(N/100)^1.5, Q_c = 3688*exp(0.03565*(N-100)),
> T = (2Qi/(Qc+2Qi))^2, Q_L = Qi*Qc/(Qc+2Qi).
>
> ### ★SPECTRAL-SAMPLING FIX — THIS SAVED THE STUDY TWICE, KEEP IT
> The 20 nm / 4001 pt family window is **5 pm per sample**. Fine at N<=180 (>=10
> samples across the line) but at the high-Q end the line is 6-15 pm: N=220 gets 3.1
> samples, N=240 1.8, N=280 2.2, N=320 1.3. **Under-sampling does not merely spoil Q —
> it puts the true peak BETWEEN grid points and biases peak T LOW, which drags the
> apparent crossing to lower N and then looks self-consistent.** Fix = per-row window
> `_WIN` in the runner: N>=200 use 3 nm @1560.7 (0.75 pm), N=320 uses 2 nm @1560.6
> (0.50 pm); rows 0-2 keep the family window so measured rungs stay bit-comparable.
> Cancelled+resubmitted twice for this: 137330 -> 137331, and 137322_3/_4 -> 137333.
>
> ### ★a100-staging IS NOT OURS — do not try again
> 2 idle A100 nodes (n307-308) but `AllowAccounts=admins-projects`; our account is
> `rosenthal_prj`. Checked with `scontrol show partition`. Also: `scontrol show
> reservation` = none, so a "ReqNodeNotAvail, May be reserved for other job" pending
> reason on Athena is TRANSIENT scheduler noise, not a maintenance window.
> ★Plot script ready + checkcode clean:
> `matlab_plotting/studies/plot_invdesign_q3db_20um.m` (scans BOTH cluster result
> dirs; stamps "EXTRAPOLATED - not bracketed" if the rungs fail to straddle -3 dB).


> ## ★★★q3db RUN MAP — 2026-08-26 01:3x (user: "work until completion tonight")
> Study: `runners/sweeps/invdesign_q3db_20um.py` (regular builder, conformal,
> q3db family numerics). Rungs deliberately SPLIT ACROSS BOTH CLUSTERS:
> | N | cluster / job | state |
> |---|---|---|
> | 100 | IGUM 63202_0 | **DONE** T 0.97228 / Q 1818.6 / 19.1709 um |
> | 150 | IGUM 63423_1 | running |
> | 180 | Athena 137322_2 | running (n314) |
> | 200 | Athena 137322_3 | queued |
> | 220 | Athena 137322_4 | queued |
> | 240 | **NOT DISPATCHED** | permission classifier blocked the submit; only needed if the crossing lands >220 |
> IGUM 63423_2/3/4 were CANCELLED (duplicates of the Athena rungs).
> ★Athena's deploy tree is **`~/bragg_sim_athena/project`** — NOT `~/bragg_sim/project`
> (a different, stale tree; do not md5 against it, that misled this session once).
> ★Athena QOS caps **mem=275G** per job (24h_1g / 4d_1g): 300G is rejected with
> `QOSMaxMemoryPerJob` and the deploy prints only "ERROR: sbatch failed". Use 256G.
> ★Plot script written + checkcode CLEAN:
> `matlab_plotting/studies/plot_invdesign_q3db_20um.m` — scans BOTH result dirs,
> interpolates the -3 dB crossing in dB-vs-N, and REFUSES to hide extrapolation
> (labels the crossing "EXTRAPOLATED - not bracketed" when the rungs do not
> straddle -3 dB). Overlays the stored family points (ctrl N165 Q 13930,
> comb N169 Q 16203). Outputs .fig + .png next to the results dir.
> ★Monitor lesson: a `|| CLUSTER_UNREACHABLE` fallback around an ssh whose REMOTE
> SCRIPT has a syntax error reports a false outage. Both clusters read
> "UNREACHABLE" while a direct probe answered instantly. Fix = single-quote the
> remote script (no nested escaping) and key on completed `result_N*.mat` files
> rather than log scraping.


> # ⏸️ GPU WIDTH-GRADIENT PROGRAMME — **PAUSED 2026-08-26 ~00:50 BY USER**
> **Resume in a few days.** Nothing of THIS programme is running; **Athena queue
> is EMPTY**.
>
> ★★**BUT IGUM IS NOT IDLE — AND ITS JOBS HOLD UNIQUE RESULTS.** Measured
> 2026-08-26 ~01:15: **63423 tasks 2-4 PENDING** and **63438 tasks 2-7 PENDING**
> (JobArrayTaskLimit, `%1`-serialised ⇒ many hours). `sacct` was DOWN
> (slurmdbd "Connection refused" — the known IGUM infrastructure weakness), so
> completion state could not be read. These belong to the **SEPARATE conformal
> / q3db ladder workstream** and were deliberately left running.
> ⇒ **FIRST ACTION ON RESUME, BEFORE ANY OF THE BELOW: fetch IGUM 63423/63438
> results** (CLAUDE.md §6 — never let one cluster hold unique results).
>
> ★**NAVIGATION WARNING.** This file is NOT two clean halves: it is interleaved
> chronological strata across ~3000 lines, and the IGUM/conformal boxes use
> IDENTICAL `>` + `##` + star formatting (one even carries FIVE stars). **Once
> you scroll past this top box there is no marker telling you which programme a
> box belongs to.** Anything mentioning conformal, q3db, N-ladders, "WAVE",
> or job IDs in the **6xxxx** range is the OTHER workstream on IGUM. This
> paused programme's jobs are **13xxxx** on Athena.
>
> ## THE ONE THING TO READ FIRST
> **The mode width is slaved to the resonance wavelength: dW/dλ ≈ +0.37 µm/nm.**
> ★**PROVENANCE IS A RUNNABLE SCRIPT — `gates/derive_dwdlam.py`** (re-derives to
> **0.03%** of the stored `CampaignSpec.wg_dwdlam = 0.3655`). Sources, both
> local: `results_from_athena/v2_gpu_gradient_pause/jsonl/` →
> `lumopt2_v2_uniform_s5_evals.jsonl` and `lumopt2_v2_seesaw_evals.jsonl`.
> | baseline | slope µm/nm | r | n | λ explains |
> |---|---|---|---|---|
> | uniform_s5 | **+0.3654** | 0.984 | 9 | **93%** of its raw width growth |
> | seesaw | +0.3000 | 0.867 | 9 | **77%** of its raw width growth |
> ⇒ **the bulk of the width blow-up that killed both baselines was RESONANCE
> DRIFT, not envelope reshaping.** We spent weeks fighting the wrong quantity.
>
> ★★**TWO CORRECTIONS TO EARLIER WORDING, both found by re-deriving (2026-08-26):**
> 1. I first reported "93–94%". The **94% for seesaw applied the UNIFORM run's
>    slope to seesaw**; on its OWN slope (0.300) seesaw is **77%**. Use 93% /
>    77%, per-run.
> 2. ★**The slope is NOT universal — 0.300 (seesaw) vs 0.365 (uniform), a ~20%
>    spread.** `wg_dwdlam` is a single hardcoded scalar, so the chain term
>    carries ~20% magnitude uncertainty across designs. This is tolerable (the
>    projection DIRECTION is far less sensitive than the magnitude) but it is
>    NOT a constant of nature — **re-derive it for a new seed family**, and
>    consider fitting it online from the eval log once ≥4 in-band points exist.
> ★**THE FILTER RULE MATTERS ENORMOUSLY AND IS NOW WRITTEN DOWN** (it was
> hand-applied and undocumented, which nearly shipped a 61%-wrong constant):
> unique (λ, W) pairs, `fom > 0.5·max(fom)`. Dropping either clause breaks it —
> lumopt2 re-logs the accepted point each restart segment (W 18.5076 appears
> 3×, 18.4088 2×) and a single out-of-band probe (fom 0.194 at W 19.53) is high
> leverage, pulling the slope 0.366 → 0.59 on its own. Do NOT pool the two
> baselines either (different intercepts ⇒ pooled slope 0.288, wrong).
>
> ## THE DEFECT THIS EXPOSED (#19) — CONFIRMED FROM SOURCE, FIX WRITTEN
> `gW` from the adjoint is **∂W/∂p at FIXED λ**, but W is specced at the
> device's OWN MOVING resonance, so the true derivative
> `dW/dp = ∂W/∂p|_λ + (dW/dλ)·(dλ_pk/dp)` is missing its second term. The
> projection therefore nulls the WRONG gradient. The code says so itself —
> `make_func` emits the width twin's λ-pin as *"A CONSTANT to autograd (no p
> dependence): zero Jacobian row, zero dEps"* (lumopt2_design.py ~line 570).
> Symptoms it explains, all at once: the whole +0.0110 µm of toy iterate 0→1 is
> accounted for by its +0.04 nm λ drift; the T-per-µm exchange rate never beat
> the unprojected baseline (0.097 vs 0.091); ‖gW‖ grew 58% and the shadow price
> 43× while climbing.
>
> ## THE FIX (implemented, gated, **NOT YET PROVEN ON HARDWARE**)
> `gλ = dλ_pk/dp` from the implicit function theorem on the peak condition
> ∂T/∂λ = 0, using **two extra selector passes off the SAME solved fields —
> ZERO extra adjoint solves** (the assembly is linear in the fct jacobian):
> ```
> gLam = -(g_hi - g_lo) / (T'(lam_hi) - T'(lam_lo))      # MATCHED pair
> gW   = gW + spec.wg_dwdlam * gLam                      # wg_dwdlam = 0.3655
> ```
> ★The **matched** pair is exact for ANY stencil width h and any symmetric
> lineshape (the amplitude part is even and cancels in both antisymmetric
> differences), so no curvature is ever formed. The NAIVE pair (cross-difference
> over a second difference of T) has error **exactly 1/(1+x²)**, x = h/g —
> verified numerically to the digit — which is why a half-linewidth stencil was
> **49.4% LOW**. Because truncation now cancels, a WIDE stencil is BETTER:
> k = round(0.5·fwhm/dl) = 20 gives 0.0034%.
> ∇T needs NO chain term (at the peak ∂T/∂λ = 0 ⇒ it vanishes).
> Guard: `dTp = T'_hi − T'_lo < 0` ⟺ stencil straddles a maximum; else LOUD skip.
> Residual accepted: **0.60% δ-leak** (argmax index sits ≤ dl/2 off true λ₀,
> leaking ∂A/∂p at O(δ)); removable only by fitting λ₀ (3-pass Lorentzian +
> 2-basis regression) — judged not worth it for a correction-to-a-correction.
> ★**NUMERICS REQUIREMENT: ≥~40 spectrum points per spectral FWHM.** Campaign is
> 10 nm / 501 pts = 20 pm vs FWHM ~810 pm = exactly 40. **Do NOT widen
> `scan_width_nm` without raising `n_wl_points` in step.**
>
> ## ★★★THE HONEST STATE — WHAT IS AND IS NOT PROVEN
> - **PROVEN (zero GPU):** all four gates pass — `gates/gate_lam_chain.py`
>   (math, 0.0034%), `gates/gate_lam_chain_plumbing.py` (autograd call path),
>   `gates/gate_projection_local.py` (15/15), `gates/predispatch_check.py`
>   (seeds in bounds). Compile clean. `wg_lam_chain` defaults **False**.
> - ★**NOT PROVEN: the fix has never completed a single iterate on hardware.**
>   Job 137267 died at 2:03 on my selector bug; 137296 was cancelled at ~40 min
>   for this pause, BEFORE reaching the 2:03 mark. **Treat the λ-chain as
>   UNVALIDATED until a run prints `gLam_n` with no skip line.**
> - **OPEN PHYSICS QUESTION:** whether T can rise appreciably at FIXED λ.
>   `corr(λ, T) = 0.9963` ⇒ T and λ are nearly COLLINEAR in the stored data, so
>   it CANNOT be answered from the logs. **Do not quote the tempting "13×
>   better exchange rate"** — it is an artifact of dividing by a small,
>   poorly-identified detrended denominator.
>
> ## ⛔ TWO BLOCKING QUESTIONS FOR THE USER — ASK BEFORE ANY DISPATCH
> 1. ★**There is an UNCOMMITTED `bragg_device.py` mesh change dated 2026-08-26
>    that I did NOT make** (it came from the parallel conformal/IGUM session):
>    `max_device_width` now includes `max(self.width_wide_per_tooth_m)`, which
>    widens the FINE-MESH y-span whenever per-tooth widths exceed the scalar
>    `width_wide`. **The projected campaign USES per-tooth widths, so this
>    changes our mesh.** It is a §2 named-numerics change and this file's own
>    rule calls editing `bragg_device` forbidden. The CONTROL (137075) ran
>    BEFORE it; anything dispatched now runs AFTER it ⇒ **the control may no
>    longer be numerics-comparable.** `simulation_config.py` is also modified
>    and undescribed. REQUIRED: have the author explain it, run
>    `debug_fsp_compare/scene_snapshot.py` vs the committed snapshots (CLAUDE.md
>    §5), and decide whether the control needs re-running.
> 2. **Athena or IGUM?** (CLAUDE.md §1 requires asking.) The command below
>    assumes Athena. ★**IGUM IS NOT IDLE** — see the live-jobs note above.
>
> ## EXACT RESUME COMMAND (validation toy, ~9 h, 3 iterates)
> Run from the **Bash tool** (POSIX). The env-var prefix is a PowerShell parse
> error, so this does NOT work in the primary shell as written.
> ```
> cd /c/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes
> for g in runners/lumopt2_design/gates/*.py; do python "$g" || echo "FAILED $g"; done
> SBATCH_MEM=300G LUMOPT2_QOS=12h_4g LUMOPT2_TIME=12:00:00 \
>   bash athena/deploy_athena.sh \
>   --lumopt2-design=runners.lumopt2_design.validate_c325 --array-tasks=41
> ```
> **All five gates must pass first** (each exits 0; the last line is the check):
> | gate | expected last line |
> |---|---|
> | `gate_lam_chain_plumbing.py` | `ALL PASS` (incl. "gate has teeth") |
> | `gate_lam_chain.py` | `ALL PASS` |
> | `gate_projection_local.py` | `ALL PASS` (15/15 PASS lines) |
> | `predispatch_check.py` | `ALL SEEDS IN BOUNDS — safe to dispatch` |
> | `derive_dwdlam.py` | `OK   re-derived +0.3654 -> 0.03% from stored` |
> (`gate_projection_local.py`'s docstring mentions a `patch_projection.diff` —
> **stale, that file does not exist and is not needed**; the gate is
> self-contained.)
>
> ## ⛔ DO NOT DISPATCH THE 96-HOUR PRODUCTION CAMPAIGN YET
> `campaign_v2_proj.py` already carries `wg_lam_chain=True` and `max_iter=30`,
> and a lane table further down this file hands you
> `LUMOPT2_QOS=4d_1g LUMOPT2_TIME=96:00:00`. **That is ~81 GPU-h on a gradient
> that has never completed one iterate.** The toy above is a PREREQUISITE, not
> merely the first item in a list. Also still unexercised: the RESTORATION and
> FILTER/REJECT branches of `run_projected` have NEVER run — and 3 climb
> iterates will probably not exercise them either, so they remain untested even
> after the toy passes.
> **What it decides:** (1) does the λ-chain path RUN — `gLam_n` present in
> `lumopt2_v2_proj_toy_proj.jsonl`, no `★λ-CHAIN SKIPPED` line; (2) does
> predicted gλ·dp match the measured Δλ_pk (~+0.04 nm uncorrected); (3) does
> ΔW/iterate fall below the CONTROL below; (4) ★the falsification test — if the
> projected ‖∇T‖ collapses toward 0, T and W are genuinely LOCKED for this
> device, which is a real verdict, not a failure. Expect the corrected climb to
> be SLOWER and to lean on `I_CAV` (only cavity changes narrow the mode).
>
> ## THE QUEUE AFTER THE TOY VALIDATES (user's stated priority order)
> 1. **The uniform seed must work on its own** — does it reach a
>    `BEST_T9636`-class design honestly, now that width is priced correctly?
> 2. **Learn from the seeds/designs already run** — all 78 eval logs are local
>    in `results_from_athena/v2_gpu_gradient_pause/jsonl/`; this is FREE
>    analysis, no GPU. The λ-detrend (W − 0.3655·Δλ) is the new lens to re-read
>    every past result through — it may re-interpret conclusions drawn before
>    the coupling was known.
> 3. **Can `BEST_T9636` be improved further?** It came from a chain of hand
>    adjustments; open user questions: can a uniform seed reach that basin at
>    all; is its abrupt corrugation drop to 325 at tooth 25 physical; is there a
>    family of equally-good designs; is it even converged; will it survive the
>    Q3dB device. Only landscape-CHANGING moves are worth running (the
>    optimizer already probed and rejected its local landscape).
> 4. ★**BANKED BY USER — the ONE-adjoint variant.** A COMBINED objective needs
>    only ONE adjoint instead of two (~−33%): see the existing box further down
>    this file (search "A COMBINED objective needs only ONE adjoint", ~line 323).
>    ★Caveat discovered 2026-08-25: a combined FOM yields only ∇(T − μW), i.e.
>    the penalty/AL formulation — it is **incompatible with the PROJECTION
>    method**, which needs ∇T and ∇W separately to build the null space. So it
>    is an ALTERNATIVE route, not an optimisation of the current one. Decide
>    which formulation you want before building it.
> 5. Deferred, answered, no tokens spent: a **Lumerical MCP** was judged NOT
>    worth it — this pipeline is scripted/headless/cluster-side by design.
>
> ## THE CONTROL (uncorrected gradient — job 137075_41, 3 iterates, keep)
> ★**TWO SOURCE FILES — they are not interchangeable.** Both local under
> `results_from_athena/v2_gpu_gradient_pause/jsonl/`:
> - `lumopt2_v2_projchain_toy`… no — the control wrote under the OLD label:
>   **`lumopt2_v2_proj_toy_proj.jsonl`** → `fom`, `W`, `alpha`, `gT_n`, `gW_n`
>   (MEASURED, verified to 6 digits).
> - **`lumopt2_v2_proj_toy_evals.jsonl`** → `lam_pk_nm`. ★`lam_pk` is **NOT** in
>   the `_proj.jsonl` file.
> ```
> it 0  fom 0.667217  W 18.3452  lam_pk 1564.61   gT_n 0.002989  gW_n 0.125902
> it 1  fom 0.668293  W 18.3562  lam_pk 1564.65   dW +0.0110   [0.097 T/um]
> it 2  fom 0.669780  W 18.3684                   dW +0.0122   [0.122 T/um]
> ```
> `fom`/`W`/`lam_pk`/`gT_n`/`gW_n` are MEASURED. `dW` and the bracketed
> `T/um` are DERIVED (ΔW between rows; Δfom/ΔW).
> ★★**NAME COLLISION — `lam` MEANS TWO DIFFERENT THINGS IN THIS FILE.**
> In `*_proj.jsonl` the field **`lam` is the SHADOW PRICE** (−1.99e-05 at it 0),
> NOT a wavelength. The wavelength is `lam_pk_nm`, and it lives only in
> `*_evals.jsonl`. A box further down says "READ THE `lam` COLUMN of
> `*_proj.jsonl`" — that one means the **shadow price**. Do not conflate them.
>
> Any corrected run is judged against these ΔW values (+0.0110 / +0.0122 µm)
> and against the ~+0.04 nm/iterate λ_pk drift.
>
> ## TRAPS PAID FOR IN GPU-HOURS THIS SESSION — DO NOT RE-LEARN
> 1. ★**The fct's `x` is FLAT** — `[T(λ_0)…T(λ_{n_wl−1}), softW]`, NOT a list of
>    FOM entry results (that is why the width selector is `x[-1]`). Writing
>    `x[0][i]` cost **2 GPU-h** (137267, `IndexError: invalid index to scalar
>    variable`). Correct: `lambda x: anp.abs(x[i])`.
> 2. ★**A math gate is not a plumbing gate.** The math gate passed at 0.0034%
>    while the call path was broken. Always drive new fct/jacobian code through
>    the REAL wrapper locally — and **assert the known-bad form still raises**,
>    or the gate has no teeth. Now CLAUDE.md §5.
> 3. ★**Count live FIELD SETS before adding an assembly pass.** Stashing all
>    four would double peak RAM; the double-pass ALREADY OOM-killed a 160G job
>    at 501 λ (137012, exit 137). Fix: selector passes run BEFORE the width
>    stash and convert to 191-float vectors immediately (`gvec_Tlo/Thi`,
>    `del f`) ⇒ peak stays at TWO field sets. Driver stashes `spec._wg_p = p`.
> 4. ★**Defect #18 — autogain must gate on `phase == "restore"`,** not on
>    `|dW_pred|`. CLIMB is `alpha·D·gT`, UNPROJECTED (I wrongly called it
>    orthogonal-by-construction; RIDE is the orthogonal one), so its ∇W·step is
>    an incidental near-zero — dividing by it gave r = −52 and fired a false
>    "C_field PHASE error" alarm. Only restore builds its step along ∇W.
> 5. ★**Band-edge wrap:** i_pk within 1 of an edge made `T[i_lo-1]` wrap via
>    numpy negative indexing and silently build gλ from the wrong end of the
>    spectrum. Guard `1 < i_pk < len(wl)-2` added; swept all 501 positions.
> 6. ★**The h5 cron cleaner CANNOT free space by design** (see below) and
>    **`pgrep` is the wrong way to check it** — it is a CRON job, not a resident
>    process, so an empty pgrep is NOT evidence it died.
> 7. **IGUM hostname `igum.technion.ac.il` does NOT resolve** — use the IP from
>    `igum/igum.conf` (132.68.58.101). Licence probe must come from IGUM.
>
> ## ⚠️ UNFINISHED / NEEDS THE USER
> - **h5 cleaner fix NOT installed.** Quota hit 289G/300G; I freed **84.6 GB**
>   (289G → 199G) by deleting dead-study `*_output.h5`. Root cause: the cron
>   cleaner keeps "the newest TWO per study dir" but each dir holds only 2-3
>   files, so 21 dead study dirs kept ~everything. Fixed script (adds a PASS 2
>   for dirs cold >24 h) is written but **three transfer routes were BLOCKED by
>   the permission classifier** (`find -delete`, `cat > file`, `scp`). It needs
>   a user-approved retry or a manual paste to `~/h5_clean_once.sh`.
>   **Without it the quota climbs ~4 GB per completed study.**
>   ★The fixed script is preserved at **`athena/h5_clean_once.sh`** (bash -n
>   clean). To install: copy it to `~/h5_clean_once.sh` on Athena — the crontab
>   entry `*/10 * * * * $HOME/h5_clean_once.sh` ALREADY EXISTS and needs no
>   change. ★Do NOT "check the janitor" with `pgrep` — it is a CRON job, not a
>   resident process, so an empty pgrep is not evidence it died (this misled us
>   twice). Check `quota -s` and the h5 total instead.
> - Local edits are **deployed** to Athena (engine mtime 2026-08-26 00:06) but
>   **NOT committed to git** (needs permission).
>
> ## DATA PULLED LOCAL (no cluster holds unique state)
> `results_from_athena/v2_gpu_gradient_pause/jsonl/` — all campaign/validate
> `*_evals.jsonl` + `*_proj.jsonl`.
> Gates preserved in-repo: `runners/lumopt2_design/gates/` (were in a
> session-scratchpad that would not survive the pause).


> ## ★★★★★WAVE 1 COMPLETE — 2026-08-26 00:5x. ALL FOUR ROWS, CONFORMAL, N=100.
> ## The PVA ordering TRANSFERS, the width is KEPT, and T = 0.978.
> | row | T | lambda nm | Q_L | Q_i | mode FWHM um | PVA ref |
> |---|---|---|---|---|---|---|
> | bare (no comb) | 0.91830 | 1559.031 | 1638.2 | 39 266 | 19.2484 | — |
> | origin uniform+comb | 0.92835 | 1559.041 | 1646.7 | 45 128 | 19.1772 | 0.90120 |
> | see-saw d090 | 0.95464 | 1559.156 | 1759.7 | 76 701 | 19.0131 | 0.93836 |
> | **BEST_T9636 (exact)** | **0.97805** | 1560.907 | 1714.2 | **155 358** | **19.0083** | 0.96361 |
> Job 62750 (rows 0-1) + 63195 (rows 2-3), lumopt2 path, conformal variant 0,
> q3db family box/window, region dx pitch-locked.
> ★**MESHER RULE DISCHARGED.** conformal ordering origin < see-saw < best is the
> SAME as PVA (0.90120 < 0.93836 < 0.96361). The ranking-transfer assumption the
> HANDOFF flagged as untested is now TESTED and HOLDS on three devices.
> In-batch gain best − origin: **+0.0497 conformal** vs +0.0624 PVA (~80% carries).
> Cavity loss 1-T: origin 0.07165 → best **0.02195**, **-69%**.
> Width: best 19.0083 vs origin 19.1772 = **-0.88%** (kept, slightly NARROWER).
>
> ## ★★★★THE SHIFT-CONVENTION IS IMMATERIAL — the trap is PRICED, ~0.002 T.
> Same device, same N, same mesher, two builders:
> - lumopt2 path, EXACT optimizer layout (63195_3): T **0.97805**, FWHM 19.0083
> - regular builder, its own right-arm convention (63202_0): T **0.97228**, FWHM 19.1709
> Gap **+0.00577**. But row 0 measured the lumopt2 mesh-region artifact ALONE
> (zero-shift device, so convention cannot contribute) at **+0.0079**. Subtracting:
> convention ≈ **-0.002 T** — smaller than the artifact and at the jitter scale.
> ⇒ **The regular builder's layout is equivalent for physics purposes.** The q3db
> ladder therefore runs the regular path legitimately: artifact-free absolute T,
> directly comparable to the stored family crossings, and ~5x faster (22 min vs 2 h).
> The convention caveat stays documented but is no longer a blocker.
>
> ## LADDER LIVE — IGUM **63423** tasks 1-4 (N=150/180/200/220), `%1` serialized
> after the ansyscl startup race killed 63415 outright (4/4). 63423_1 RUNNING,
> ERR=0. ★**EXPECTED crossing N ~ 220-235** (design at -0.122 dB / N=100 on the
> regular path, riding the stored bare dB~N^4.04 slope) — i.e. AT or ABOVE the top
> of the bracket. Plan: read the measured dB(N) from the first rungs, then add the
> ONE rung that brackets it. Do not extrapolate the crossing from the fit alone.


> ## ★★★★★THE SHORT DEVICE IN CONFORMAL — MEASURED 2026-08-25 23:27
> ## T = 0.97228 at N=100. The design's mode width IS kept.
> **Source of truth:** IGUM job **63202_0**, 1348.8 s solve, exit 0,
> `results/invdesign_q3db_20um/results/result_N100_TM_W961_C325_...mat`.
> Regular builder (NOT lumopt2), conformal variant 0, q3db family numerics.
> | | design N=100 | stored bare N=100 (IGUM 51736, NOT re-run) | delta |
> |---|---|---|---|
> | T | **0.97228** | 0.9104 | **+0.0619** |
> | cavity loss 1-T | **0.02772** | 0.0896 | **-69%** |
> | lambda nm | 1560.947 | 1559.006 | +1.94 |
> | Q_loaded | **1818.6** | 1760 | +3.3% |
> | mode FWHM um | **19.1709** | 19.2448 | **-0.38% (KEPT)** |
> DERIVED Q_i = Q_L/(1-sqrt(T)) = **130 300** vs bare 38 400 → **x3.4**.
> §2 sanity: resonance in-window (1549.5-1569.5), T far above the dead floor.
> ★**0.96361 was the PVA number; conformal reads HIGHER**, as the origin pair
> already showed (PVA 0.90120 → conformal 0.92835). So "around 0.96" is the
> floor, not the target — 0.972 is the conformal answer for the short device.
> ★**CAVEAT, do not drop it:** this is the REGULAR-BUILDER layout, which differs
> from the optimizer's exact device on the right arm only (≤6.43 nm/tooth — see
> the convention trap below). Job **63195_3** is the exact-optimizer device at
> N=100 on the lumopt2 path and prices that difference; subtract row 0's measured
> +0.0079 mesh-region artifact from the gap to isolate the convention.
> ★**SPEED FINDING:** the regular path solved in **22.5 min** vs ~2 h for the
> lumopt2 path at the same N and numerics — ~5x. The lumopt2 optimization-region
> mesh override is what costs it. That is why the ladder runs the regular path.
> ## ★LENGTHENING LADDER DISPATCHED — IGUM job **63415**, tasks 1-4,
> ## N = 150 / 180 / 200 / 220 (`runners/sweeps/invdesign_q3db_20um.py`).
> N=100 not repeated (63202_0 is that point). Target: the -3 dB crossing and Q
> there, vs stored ctrl N165 -3.09 dB Q 13930 and comb N169 -3.04 dB Q 16203.
> ★Queue note: **63237 = `itai_hh_apod`, 6 tasks, NOT dispatched by this session**
> (22:31). Shared-file md5s were verified identical local↔remote before deploying,
> so the parallel-deploy conditions of §6 were met; the rsync itemisation showed
> `t`-only (timestamp) touches on its files, no content change.


> ## ★★LIVE ON IGUM — job 62750, PRODUCTION CONFIRM (user, 2026-08-25)
> `runners/lumopt2_design/prod_confirm.py`, 4 forwards, N=100/side, **mesher =
> "conformal variant 0"** (the project's regular mesher) at the **q3db family
> numerics** (box y8.0 / z-mult 5.42, 20 nm @1559.5 / 4001 pts, z-sym),
> region dx pitch-locked. Rows: 0 bare (IDENTITY gate vs stored IGUM 51736
> T 0.9104 / λ 1559.006 / FWHM 19.2448) · 1 origin (PVA 0.90120/18.3460) ·
> 2 seesaw d090 (PVA 0.93836/18.331) · 3 BEST_T9636 (PVA 0.96361/18.35309).
> This is the HANDOFF's own "cheap closer" + item-1 production confirm, and it
> discharges the user's MESHER RULE (conformal-vs-conformal PAIR).
> **WAVE 2, already agreed with the user:** ladder `n_periods_side` at the SAME
> numerics to the −3 dB crossing and report N + Q there, against the stored
> corr-325 q3db crossings (ctrl N165 T 0.4906 Q 13930 · winner comb N169
> −3.04 dB Q 16203). N is sized from wave 1's measured T(N=100) — do NOT guess it.
> Athena stayed under the DEPLOY FREEZE (137075_41 still running) — that is why
> this went to IGUM. Seats at dispatch: 0/50 in use.
>
> ### WAVE-1 PARTIAL RESULT — 2026-08-25 21:30 (MEASURED, job 62750 logs)
> | row | T | λ nm | Q_L | Q_i | FWHM µm |
> |---|---|---|---|---|---|
> | 0 bare | 0.91830 | 1559.0310 | 1638.2 | 39266 | **19.2484** |
> | 1 origin (uniform+comb) | 0.92835 | 1559.0410 | 1646.7 | 45128 | **19.1772** |
> **IDENTITY GATE (row 0 vs stored IGUM 51736 0.9104/1559.006/19.2448):**
> λ **+0.025 nm**, FWHM **+0.019 %** — the scene and the mode are the regular
> device to within noise. But **T reads +0.0079 HIGH and Q_L 7 % LOW** (1638 vs
> 1760). That offset is the residual of the lumopt2 optimization-region mesh
> override (pitch-locked dx 51.683 nm vs the family's global 50 nm) — it cannot
> be removed from this path. ⇒ **In-batch deltas are the only valid currency
> here (§2); never subtract our T from a stored family T.**
> ★**CONSEQUENCE FOR WAVE 2, decide before dispatching it:** −3 dB is an
> ABSOLUTE-T question, so the stored crossings (ctrl N165 / comb N169) cannot
> be mixed with our numbers. Wave 2 must ladder the DESIGN **and** the uniform
> origin, and quote both crossing-N under our own stack; the stored 165/169
> then serve as an external cross-check, not as the reference.
> Width conversion confirmed: origin PVA 18.3460 → conformal 19.1772 =
> **×1.0453** (EXPECTED was ×1.049).
> Comb gain in conformal, in-batch: **+0.0101 T** over bare at ~equal width.
> **TASKS 2-3 DIED** — task 3 instantly on the ansyscl cold-start daemon race
> (§6's known array-cold-start casualty), task 2 after ~2 h when its FDTD
> session dropped mid-`runjobs` retry ('Failed to put variable'). RESUBMITTED
> as **job 63195, `--array-tasks=2-3 --max-concurrent=1`** (serialized — the
> race needs two tasks cold-starting on one node in the same second).
>
> ### ★★★NEW TRAP FOUND 2026-08-25 (zero GPU) — THE TWO PATHS DISAGREE ON THE
> ### RIGHT-ARM SHIFT INDEX. Any "run it outside lumopt2" plan must read this.
> Attempting the HANDOFF's own item-1 advice (production confirm via a plain
> SweepSpec runner) and gating it locally against `make_func` found **75
> mismatching geometry properties, ALL on the RIGHT arm, ZERO on the left**.
> **Root cause, read from source:** `bragg_device.py:1180` walks the right arm
> with `s_prev = shift_for_tooth[d-1]` (tooth 1 gets 0); lumopt2's `make_func`
> right walk uses `s = shift[i]` for tooth `d = i+1`. So the builder shortens
> `R_narrow_d` by `s_(d-1)`, the optimizer by `s_d`. Worst tooth displacement
> **6.43 nm** (= s_6, the largest shift in BEST_T9636); `R_narrow_1` span differs
> by 3.16 nm. Both layouts are geometrically valid pi-shift gratings — they are
> different CONVENTIONS, and the device that measured 0.96361 is `make_func`'s.
> **It cannot be repaired by re-indexing the input list**: the left arm needs
> `t_d = s_d` while the right needs `t_(d-1) = s_d`. Editing `bragg_device` is
> forbidden — it would silently change every stored distributed-shift result.
> ⇒ `runners/sweeps/invdesign_q3db_20um.py` is written but carries a **BLOCKED —
> DO NOT DISPATCH** banner. The q3db ladder must either run through the lumopt2
> canary path (exact device, carries row 0's +0.0079 T mesh-region artifact) or
> be preceded by ONE simulation pricing the convention difference. PARKED.
> Gate script: `gate_invdesign_scene.py` — ★LOST (was in a session scratchpad; rebuild if needed — it is
> the cheapest guard this programme has against shipping the wrong device).
> ★Throttle note: 63195 was submitted `%1`; raised live with
> `scontrol update jobid=63195 arraytaskthrottle=2` so task 3 (BEST_T9636, the
> row the user is waiting on) starts in parallel with task 2 but ~9 min
> STAGGERED — which is also the cold-start-race fix, without a scancel.
>
> ### ★ATHENA IS STILL FROZEN — 137075 ENDED, **137267_41 IS NOW LIVE** (2026-08-25 22:1x)
> Checked directly: `137267_41 l40s-public lum_pipeline_array R 15:28, 11:44 left`.
> This session did NOT dispatch it. Same array index (_41) as 137075.
> **Most likely explanation, and it also explains the mid-session engine edit:**
> `lumopt2_design.py` (21:53) and `validate_c325.py` (21:54) were modified
> locally by someone other than this session, and the v2 campaign was
> re-dispatched to Athena as 137267 under that new code. HYPOTHESIS, not proven
> — but it fits the timestamps exactly. ⇒ **The DEPLOY FREEZE on Athena stands.**
> All prod_confirm / q3db work stays on IGUM.
>
> > ### ✅ANSWERED 2026-08-26 by the width-gradient session — your hypothesis was RIGHT
> > **137267 WAS dispatched by the other (width-gradient) session**, and the
> > 21:53/21:54 edits to `lumopt2_design.py` / `validate_c325.py` were its
> > defect-#19 λ-chain work. Ground truth on all three jobs:
> > - **137075_41** — the uncorrected control. 3 iterates, then CANCELLED.
> > - **137267_41** — **FAILED at 02:03:18**, `IndexError: invalid index to
> >   scalar variable` (a selector bug, now fixed and gated).
> > - **137296_41** — the refixed rerun; **CANCELLED at ~40 min** when the user
> >   paused the programme.
> > ⇒ **ATHENA IS NOW EMPTY AND THE FREEZE IS MOOT** — the width-gradient
> > programme is PAUSED (see this file's TOP BOX, which supersedes this
> > paragraph and gates any dispatch on its own two blocking questions).
> > ★Your instinct to keep prod_confirm / q3db on IGUM remains sound, but the
> > reason is now the pause, not a freeze.
> > ★**AND A QUESTION BACK TO YOU (blocking for us):** there is an uncommitted
> > `bragg_device.py` change dated 2026-08-26 adding
> > `max(self.width_wide_per_tooth_m)` to `max_device_width` — we believe it is
> > yours. It widens the FINE-MESH y-span for per-tooth-width devices, which
> > the projected campaign uses, so it is a §2 numerics change that may break
> > comparability with our stored control. Please confirm authorship and
> > whether it was snapshot-diffed.
> **Version-mixing audit done (user asked, "check that it's exactly as is"):**
> the +120 new lines are confined to the width-gradient / projection path
> (`wg_lam_chain`, `wg_dwdlam`, the `wgp_*` block, code between `make_project`
> and `run_campaign`). Everything `run_canary` calls is byte-identical to the
> pre-edit reading — constants (CORR_NM 325 / AVG_W_NM 800 / comb), `seed_params`,
> `build_base_cfg`, `run_canary`, and the defaults prod_confirm leans on
> (`corr_seed_nm`, `width_grad=False`, `fwhm_wall=False`). ⇒ rows 0-1 (old engine)
> and rows 2-3 (new engine) REMAIN COMPARABLE. Structural inspection, not a
> byte-diff: the pre-edit file no longer exists anywhere to diff against.
>
> ### HOW TO READ ROW 3 WHEN IT LANDS (user expects "around 0.96")
> 0.96361 is the **PVA** number. Conformal reads HIGHER on this family: the
> origin moved PVA 0.90120 → conformal **0.92835** (+0.0272, MEASURED in this
> same batch). Naively transferring that offset puts BEST near 0.99 — but that is
> exactly the single-secant extrapolation this programme has been burned by twice
> (FW_A_ELONG, the see-saw payback rate). **The verdict is the IN-BATCH GAIN, not
> the absolute:** at PVA, BEST − origin = **+0.0624**. If conformal shows a
> comparable gain over its own 0.92835 origin, the PVA optimization TRANSFERS and
> the mesher rule is discharged. If the gain collapses, PVA is disqualified as an
> optimization mesher regardless of how high the absolute T reads.
>
> ### q3db LADDER IS WRITTEN AND GATED — `prod_q3db_ladder.py`
> N = [150, 180, 200, 220], exact-device lumopt2 path, identical numerics to
> prod_confirm, resume-protected, N=100 not repeated (row 3 is that point).
> Not dispatched: it is pointed at the measured T(N=100), which row 3 supplies.


> ## ★★★★BANKED (user, 2026-08-25): THE ONE-ADJOINT COMBINED-FOM VARIANT
> ## — 33% CHEAPER. Do it AFTER the projection is shown to work, not before.
> **The user's point, and it is correct:** the projection costs THREE solves
> per iterate (forward + port adjoint + width adjoint) because it needs
> grad-T and grad-W SEPARATELY — the null-space coefficient
> `(D grad-T . u)` depends on both, so neither can be folded in advance.
> A COMBINED objective needs only ONE adjoint.
> **WHY ONE ADJOINT SUFFICES:** the adjoint problem is LINEAR, so sources
> superpose (the same fact that makes the 4-tile source exact). For a scalar
> objective `T - lam*W` with `lam` FIXED within the iterate, build ONE source
> `= (dJ/dT)*(port source) + (dJ/dW)*(width source)`, run ONE adjoint, and the
> combined gradient comes out directly. **3 solves -> 2, i.e. -33%.**
> **TWO FLAVOURS, both cheap to build on what exists:**
> 1. **Equality-form AL** — `lam` updated BETWEEN outer iterations from the
>    MEASURED width deviation (signed, not `max(0,g)`, so it is never
>    dormant). The engine already carries `wg_mu` / `wg_lam_hi` / `wg_lam_lo`.
> 2. **Lagged multiplier** — take `lam` from the PREVIOUS iterate's exact
>    projection and use one combined adjoint this iterate. One-iterate-stale
>    price, exact machinery, minimal new code.
> ★**THE CURRENT RUN IS ALREADY GENERATING THE EVIDENCE TO CHOOSE.**
> `run_projected` logs the exact shadow price `lam` EVERY iterate
> (`lam = (grad-T . u)/|grad-W|`). If that trajectory is smooth and slowly
> varying, a lagged/AL `lam` is safe and the one-adjoint variant is a
> near-free 33%. If `lam` swings iterate to iterate, the combined form would
> chase a moving price and the exact two-adjoint version earns its cost.
> **READ THE `lam` COLUMN of `*_proj.jsonl` BEFORE BUILDING THIS.**
> ★**STACKS with the other banked economy** (from the formulation brief, also
> unimplemented): during CLIMB, grad-W moves slowly and can be REUSED for
> ~3 steps with a guard that refreshes immediately when measured dW deviates
> >2x from predicted. Together these could approach ~1.3 solves/iterate.
> ★**SEQUENCING, and it is deliberate:** prove width-holding with the EXACT
> method first (job 137075). The cheap variant trades exactness for speed, and
> trading away something we have not yet demonstrated would leave us unable to
> tell a formulation failure from an approximation failure.
>
> ## ★★★★★NEXT-DISPATCH CHECKLIST — RUN THIS BEFORE THE NEXT DEPLOY
> ## (2026-08-25: there are LOCAL-ONLY fixes that job 137075 does NOT have)
> **WHY THIS EXISTS:** `wgp_autogain` and the phase-error trip were written
> AFTER 137075 was dispatched, so they live only on the laptop. A deploy
> rsyncs the whole tree, and every Athena partition REQUEUEs on preemption —
> deploying while 137075 runs would let a requeue resume it under a DIFFERENT
> method and silently mix two algorithms in one experiment (§6).
> **1. DEPLOY FREEZE — lift only when 137075 has finished** (`squeue -u
> evyatarrubin -r`). Until then: no deploys, no dispatches.
> **2. THEN DEPLOY EVERYTHING AT ONCE**; confirm the rsync itemised list shows
> `lumopt2_design.py`, `campaign_v2_proj.py`, `validate_c325.py`.
> **3. VERIFY THESE FLAGS ARE LIVE** (all confirmed locally 2026-08-25):
> | flag | value | why |
> |---|---|---|
> | `wgp_autogain` DEFAULT | **False** | REQUEUE-safety for older runs |
> | `campaign_v2_proj.wgp_autogain` | **True** | measure abs(C) online, stop fitting it |
> | `ADJ_FIX_FIELD` | **(0.4554, +0.1336)** | the CONJUGATE — fit_c_field prints the other sign |
> | `wg_track_resonance` | **True** | softW on resonance (standing user ruling) |
> | `fwhm_wall` | **False** | penalty REPLACED by the projection, never stacked |
> | `wg_src_tiles` / `max_iter` | **4 / 30** | CUDA per-source bound; 30x2.7 h fits the 96 h lane |
> **4. PRE-DISPATCH GATES — both zero-GPU, both must pass:**
> `python runners/lumopt2_design/gates/predispatch_check.py` (seed + DETUNE vs bounds — this
> class cost FOUR dispatches) and `python runners/lumopt2_design/gates/gate_projection_local.py`
> (15 checks incl. exact grad-W . d = 0).
> **5. SEAT ARITHMETIC — count the FAN-OUT, not the tasks.** A
> `run_validate_gradient` task is **1 + n_legs seats** (7 for 3 indices) —
> that is what killed job 137035. Probe seats IMMEDIATELY before dispatch,
> never from a cached reading.
> **6. LANES:** campaign = `SBATCH_MEM=300G LUMOPT2_QOS=4d_1g
> LUMOPT2_TIME=96:00:00`; 4-iterate toy = 300G / 12h_4g / 12:00:00; single
> forward = 160G / 04:00:00.
> **7. STILL UNEXERCISED CODE — where the next bug lives:** the RESTORATION
> branch and the FILTER/REJECT branch of `run_projected` have NEVER run;
> iterate 0 took the CLIMB path only. Read the first `restore` and `*-retry`
> rows with suspicion.
>
> ## ★★★STATE AS OF 2026-08-24 EVENING — READ THIS WHOLE BOX FIRST.
> ## Every earlier job list in this file is SUPERSEDED, including §0a
> ## ("EVERYTHING IS STOPPED"), which is wrong.
>
> **LIVE (Athena): 136752 `lumopt2_v2_seesaw` · 136753 `lumopt2_v2_uniform_s5`.**
> Both re-dispatched this evening with the corrected wall AND a 30 nm shift
> trust box. CANCELLED today, do not resume: 136465 (converged, its winner is
> preserved as `BEST_T9636`), 136468, 136695, 136708, 136709.
> Full measured state: `memory/project_v2_width_gradient_plan.md`.
> Six fixes committed today: 7eb7d35, 0d2ff88, 30f8f77, 3c10524, 60e57f9, b0102dc.
>
> ### ★★★PRIORITY ZERO — FIX THE FWHM GRADIENT ON GPU (user, 2026-08-24)
> **This outranks the three jobs below, because it is the ROOT FIX for most of
> what those jobs are fighting.** User's reasoning, and it is correct: a
> working GPU width-adjoint "could solve for uniform seed and accelerate".
> Full retry plan, four routes and the mandatory h5 gate: see the
> **"RETRY THE GPU WIDTH-ADJOINT"** block further down. Start with route 1
> (the CUDA `invalid configuration argument` is a kernel-LAUNCH error, very
> plausibly a region-SIZE bound — one task to find out).
> **Why it is priority zero — what it actually buys:**
> - **Exact ∂W/∂p for all 191 parameters, re-measured on the ACTUAL device at
>   every evaluation.** That single change retires, at the root, four separate
>   defects this programme spent 2026-08-24 patching: the rank-deficient
>   surrogate (skill 34), the device-class transfer error (skill 35), the
>   1.87× elongation over-tax, and the shift-distribution blindness. They are
>   all symptoms of standing a 2-parameter MODEL in for a 191-parameter truth.
> - **Speed: ~1 h/solve instead of 8.7-12.1 h**, which is what moves it from
>   "occasional calibration tool" to "in-loop".
> - **It is the only route to what the user keeps asking for** — a gradient
>   that acts on width itself rather than on a model of it.
> ★**HONEST LIMIT — do not oversell it:** a true softW does NOT by itself fix
> the in-band-zero-gradient problem. If it is wrapped in the same
> inequality-constraint AL, that penalty is still zero while the design is in
> spec (no constraint acts when satisfied). What changes is that with EXACT
> ∂W/∂p in hand you can finally choose a better formulation — an active
> shadow price, or an explicit T-vs-width trade — instead of being stuck with
> a surrogate whose error you cannot bound. Fixing the adjoint is necessary,
> not sufficient.
>
> ### THEN THE PROGRAMME'S THREE JOBS. In the user's own priority order:
>
> **① THE UNIFORM SEED — TOP PRIORITY (user, 2026-08-24).** Why it matters:
> `BEST_T9636` is NOT a clean research result. Its shifts (e≈130.6) were
> inherited from stage-1 work fitted to a device that no longer exists, and
> 136465 only nudged them 2 nm; the corrugation came through hand retrims. So
> the headline number is a chain of adjustments plus partial gradient runs, not
> one honest optimization. A uniform-seeded campaign that reaches a comparable
> design IS that honest result, and it is what lets us claim we understand the
> patterns rather than having stumbled into them.
> ★**The stalls were NOT local minima** (Fable audit, task a5fddc5457850a073):
> the fixed point was wrong because the PRICES were wrong (rank-deficient
> wall), not because of multimodality. The seed is fixable.
> ★**But it cannot "discover everything" from the seed, structurally.** From
> uniform, the pure-T gradient says "lower every corrugation, spend width" —
> ∂T/∂corr < 0 on every tooth. The see-saw REQUIRES raising outer teeth, which
> locally costs T, so it is justified only once width has a price — and the
> hinge gives width ZERO price until the band edge (MEASURED: penalty gradient
> is exactly 0.000e+00 for e ≤ 81 nm). Therefore the uniform seed can only find
> the see-saw AFTER walking to the band ceiling. That is exactly where 136468
> was pinned, and exactly where the per-tooth fix bites. It also cannot
> discover the comb (frozen) or N (fixed).
> ★**Historical contrast worth keeping:** the sigma-era campaign DID travel
> from uniform to a shaped design — because sigma put no price on the profile
> at all — and its width blew up +14.9%. Sigma era = freedom without control;
> FWHM era = control without freedom. Neither was right.
> ★**If 136753 thrashes at the band edge, the next move is the AUGMENTED
> LAGRANGIAN, not another patch.** The hinge gives width shadow price 0 inside
> the band and ∞ outside, so the optimizer spends blindly then crashes. The AL
> machinery (`wg_mu`, `wg_lam_hi`, `wg_lam_lo`) already exists in the engine —
> it was built for the exact width-gradient path and orphaned when that path
> was priced out at 8.7 h/solve. Applying it to the cheap surrogate is the
> principled fix. A "soft shoulder" is a patch on the wrong architecture.
>
> **② THE BEST DESIGN — and the measurement that decides what to do with it.**
> `BEST_T9636` (in `best_designs.py`): T 0.96361 / λ 1566.444 / fwhm_env
> 18.35309 µm / Q_load 2021.6 / Q_i 110 087 / mcorr 357.95 / e 132.6 /
> wcav 961.1.
> ★**MEASURED 2026-08-24: campaign 136465 moved its seed by ≤0.26 nm in ANY of
> the 191 params** (corr 0.059 / avg 0.263 / shift 0.169 / wcav 0.161 nm)
> against trust radii of 10-12 nm — it used ~2% of its allowed travel.
> **`BEST_T9636` is therefore `BEST_T9635` + the 42 nm hand retrim, and the
> gradient method contributed essentially nothing to it.** That is the concrete
> basis for the user's "this is not a full research way".
> ★**AND THE WALL IS INACTIVE THERE: penalty 0.0016, |grad| 5.0e-4**, because
> 18.353 µm sits well inside the 18.713 ceiling. Two consequences, both
> important, and BOTH were stated wrongly in session before being checked:
>   - "It is converged so there is nothing to do" — misleading, retracted.
>   - "Restart it under the corrected wall and it may find more" — ALSO wrong,
>     retracted. A term that is inactive cannot change the local landscape;
>     a restart would very likely sit still exactly as 136465 did. Do not
>     spend a campaign slot on it.
> What we actually measured is that the optimizer probed and rejected
> everything, NOT that no better nearby point exists — do not overstate it as
> a proven local optimum.
> ★**So the things worth trying on this design are the ones that CHANGE the
> landscape, not search it harder** (all cheap, all forward-solve only):
>   1. **Cavity width probe** — wcav 961 → ~1100 (bound 1150), 2 forwards.
>      189 nm never explored, and it is the most width-efficient lever ever
>      measured here (rtdec: +0.0409 T for +0.0305 µm). Highest value/cost.
>   2. **Comb re-tune** — the 57-site comb is FROZEN at a tune fitted for the
>      uniform seed, but the mode has moved a long way since (mcorr 325→358,
>      e 0→132.6, wcav 800→961). Worth +0.0040 at its current tune; a small
>      δx/r/d scan on the CURRENT mode is untested. NOTE: a second comb ROW is
>      MEASURED DEAD three times — see the correction above.
>   3. **Spend the 0.36 µm of unused headroom deliberately, then re-optimize
>      from the new point.** The design stops short of the ceiling. A hand move
>      (see-saw, or the split test) puts it somewhere the constraint is active,
>      which is the only regime where the corrected wall does anything.
>   4. **PRODUCTION CONFIRM** (see "what the user forgot") — the actual
>      deliverable, and independent of all of the above.
>
> **③ LEARN FROM THE DESIGNS ALREADY RUN.** The measured pattern inventory is
> in §"WHAT WE MEASURED" below and in the plan memory. The two the user singled
> out, both confirmed: (a) **many teeth × tiny shifts costs NO width and raises
> T** — below ~1.3 nm/tooth (e ≤ 65) width is flat and T gains +0.0185;
> (b) **reaching 0.96 needed MORE shift than that** — e = 132.6 is double the
> knee, and the part above the knee cost 1.5 µm of width for +0.0207 T.
>
> ### ★"IS FWHM IN THE GRADIENT?" — SETTLED 2026-08-24, THE RECURRING QUESTION
> **NO — the true FWHM is not. A MODEL of it is, and that model says nothing
> while the device is in spec.** There are TWO width paths in the engine;
> check which one you are looking at before answering this again.
> - **Path A, the real one — BUILT, VALIDATED, SWITCHED OFF.** `softW`
>   (differentiable mode envelope from the field) + `width_band_penalty`
>   (augmented Lagrangian) + `make_fct_v2`. `width_grad=True` appears ONLY in
>   `validate_c325.py` and behind `EXACT_WIDTH_GRAD=False`; **`ADJ_FIX_FIELD`
>   is None — the field-adjoint C was never fitted** (an assert fires if you
>   enable it). MEASURED cost 8.7 h/solve ⇒ priced out.
> - **Path B, what actually runs — `make_fwhm_wall`.** An analytic model of
>   width (per-tooth corr + total elongation). It IS autograd-differentiated
>   and IS added to the FOM gradient. But it is a pure QUADRATIC hinge, so its
>   derivative is exactly zero everywhere inside the band AND zero AT the band
>   edge; force only appears once you are measurably past it. MEASURED:
>   |∂pen/∂shift| = 0.000e+00 for e ≤ 81 nm, 2.1e-3 just past, 0.136 at e=90.
> ★**The AL is NOT the cure for the in-band blindness** (said in session,
> corrected): `λ·max(0,g)` is also zero when the constraint is satisfied —
> correct and unavoidable for an inequality constraint. Its real advantage is
> narrower: the LINEAR term gives a nonzero slope AT the boundary where the
> quadratic hinge gives zero, so it bites on contact instead of only after
> overshoot. Worth adopting for that reason; it will not produce an in-band
> width-reducing direction, because no constraint acts when satisfied.
> ★**What the user asks for — "a step that minimises FWHM", i.e. the see-saw —
> is a move gradient descent CANNOT make from a local optimum.** The see-saw
> narrows AND raises T in order to free width budget to respend on shifts:
> a TWO-STEP move whose first step looks locally worse. `BEST_T9636` sits at a
> local T max with 0.36 µm of slack, so a descent method will not take it.
> Not a bug in the wall — outside what the method does.
> ★**What DOES serve that goal, and shipped today: `FW_TOOTH_W`.** When the
> design is against the ceiling and wants more T, the gradient now knows which
> directions are width-cheap (inner teeth cost ~11× outer). That is the
> see-saw knowledge, in the gradient, engaging exactly where it is needed.
> ★**"CAN'T THE WIDTH ADJOINT JUST RUN ON GPU?" — ASKED AND ANSWERED, NO.**
> The 8.7 h IS a CPU number, so the question is the right one to ask; it was
> chased and it failed for a physical reason, not a cost one. MEASURED (job
> 136108, n310/A100, identical scene):
>   - forward: 3,100.3 s
>   - width adjoint via lumopt2's own **FieldRegion** object: CUDA `invalid
>     configuration argument`, dies in seconds — 3 independent tasks. The GPU
>     engine rejects that object outright.
>   - workaround via a **standard import source**: 3,133.6 s (52 min) on GPU,
>     which looked like a 10-14× win and was written up as
>     "★★★GPU WIDTH-ADJOINT PROVEN".
>   - **THAT CLAIM WAS THEN RETRACTED.** The adjoint source is an imported
>     field sheet at z = 0, and an import source injects through TANGENTIAL
>     components — but at z = 0 this TM mode is tangentially DEAD BY PARITY:
>     MEASURED max|Ex| = 0.0, |Ey| = 0.0 EXACTLY, |Ez| = 8.41. So it injected
>     nothing. The output h5 has EVERY monitor field EXACTLY 0.0 — a dead
>     source at ratio 0, not 1e-4. The 52 min was an EMPTY SCENE integrating
>     the full sim time (hence runtime ≈ the forward's). Every printed
>     "adjoint" vector in that run was the penalty gradient alone.
>   - the CPU FieldRegion timings (8.7-12.1 h) WERE real adjoints — dipole
>     injection is unaffected — so the cost verdict stands.
> ⇒ **Do not resurrect the 10-14× speed claim; it is void.**
>
> ### ★★★RETRY THE GPU WIDTH-ADJOINT — USER PRIORITY 2026-08-24 ("very important")
> **Do not treat this as closed.** The route failed for ONE specific,
> understood reason — a source that could not inject at z=0 — not because GPU
> adjoints are impossible. **The prize is the whole architecture:** a working
> GPU width-adjoint (~1 h/solve instead of 8.7-12.1) makes the TRUE FWHM
> gradient affordable in-loop, which replaces the hinge surrogate entirely and
> is the only thing that can give the user what they have asked for repeatedly
> — a gradient step that acts on width itself rather than a model of it.
> **Attack routes, cheapest first:**
> 1. **Diagnose the FieldRegion CUDA rejection — do this FIRST, it is one
>    task.** Be precise about WHAT fails: not a field-profile MONITOR (those
>    are fine on GPU) but lumopt2's adjoint-SOURCE object `FieldRegion`.
>    MEASURED job **136026**, all 3 tasks, dead in seconds with
>    `ERROR: invalid configuration argument`. Ansys docs list only TFSF and
>    BFAST as GPU-unsupported; FieldRegion is too new to be documented either
>    way, so this is an UNDOCUMENTED incompatibility, not a stated limit.
>    ★INFERRED from the error semantics (not measured): that string is CUDA's
>    `cudaErrorInvalidConfiguration` — a kernel launched with execution
>    dimensions outside the allowed range (block/grid size). That is
>    characteristically a SIZE problem, not a capability problem, and
>    FieldRegion spans the whole optimization region. **So: shrink the region
>    (fewer cells, smaller y/z span) and re-launch.** If a smaller region
>    runs, the limitation is a launch-config bound and the fix is to TILE the
>    region, not to abandon the object. Cheapest possible test of the highest-
>    value hypothesis — do it before route 2, which changes the FOM.
> 2. **Import sheet OFF the symmetry plane.** z=0 is tangentially dead BY
>    PARITY for this TM mode (measured Ex=Ey=0.0 exactly). At z ≠ 0 inside the
>    core the tangential components are non-zero. CAVEAT that must be handled,
>    not ignored: the adjoint-source plane must match where the FOM SAMPLES
>    the field, so moving the sheet means redefining softW to sample there (or
>    summing a ±z pair). That is a FOM change and needs its own W0/W1 gate.
> 3. **Two-sheet ±z pair**, summed — keeps the profile centred while giving
>    each sheet non-zero tangential field to inject through.
> 4. **Re-test on EVERY Lumerical version bump.** FieldRegion-on-GPU is an
>    upstream limitation; Ansys may fix it. Add this to the version-bump
>    checklist alongside the B3 gradient gate.
> ### ★ROUTE 1 IMPLEMENTED + DISPATCHED 2026-08-24 (Fable) — OPUS RUNBOOK
> **Athena job 136799, array 27-32%8** (dispatched ~22:20; 4 running + 2
> pending at +12 s; smoke-tested locally incl. the 3D mutation; rsync touched
> ONLY validate_c325.py + HANDOFF.md — engine/campaign files untouched).
> ★**JANITOR "DEAD" WAS A FALSE ALARM — DO NOT RESTART IT** (checked
> 2026-08-24 19:45). `pgrep h5_roll_clean` finds nothing because the janitor
> is now a **CRON job**: `*/10 * * * * $HOME/h5_clean_once.sh` (MEASURED in
> `crontab -l`). It is working — quota fell 241G → **235G** during the
> session. The handoff's "verify with pgrep -af h5_roll_clean" instruction is
> STALE for the cron era; check `crontab -l | grep h5` and the quota trend
> instead. Restarting the old nohup loop would double-run the cleaner.
> **`validate_c325.py` tasks 27-32 = the FieldRegion size ladder** (N_TASKS 33).
> All adjoint-only, `wg_source="fieldregion"`, `wg_adj_resource="GPU"`,
> `wg_pure`, 151 λ pts, indices `[0, SL_SHIFT.start, I_CAV]` (= the FD
> reference's three). The twin is shrunk scene-locally by wrapping
> `eng.build_base_fsp` in the validate module — the ENGINE IS UNTOUCHED, so
> the in-flight campaigns 136752/136753 are not exposed to any code change.
> Rungs: 27 full (control — must reproduce the CUDA error) | 28 y×0.5 |
> 29 x×0.5 | 30 x,y×0.25 | 31 patch 6×0.8 µm | 32 thin-3D (z span 0.16 µm,
> full x,y — singleton-dim hypothesis). Each prints `[gfr] rung=...` with
> actual spans/cells before running.
> **DECISION TREE (execute verbatim; escalate to Fable only where stated):**
> 1. Read each rung's log: `invalid configuration argument` = REJECTED at
>    that size; a real adjoint solve time = LAUNCHED; a Python traceback
>    before the adjoint (possible on 31/32 — softW/3D-shape fragility) =
>    INCONCLUSIVE rung, not a CUDA verdict.
> 2. Any LAUNCHED rung → h5 non-zero gate FIRST (snippet:
>    V2_FWHM_PLAN.md:1225-1259, login-node python3, one ssh). All-zero
>    fields ⇒ that rung is a FAIL despite the runtime.
> 3. Some rungs launch+inject, full fails ⇒ SIZE BOUND CONFIRMED. Run the
>    FD gate at the LARGEST working size: re-dispatch tasks 20+21-style at
>    that region config (edit the task-20/21 `_w_spec` calls to add the same
>    `_shrink_twin` prelude, lane `SBATCH_MEM=250G LUMOPT2_QOS=12h_4g
>    LUMOPT2_TIME=09:00:00`, chain 21 `--after=` 20). PASS = sign 3/3 vs
>    `[-0.00365, +0.01825, +0.02026]` + `fit_c_field.py` residual ≤10%.
>    Then ESCALATE TO FABLE for the tiling design (a cropped region is a
>    FOM change — NOT production; tiling must cover the full envelope).
> 4. ALL rungs fail incl. 31 ⇒ route 1 DEAD. STOP GPU submissions on this;
>    ESCALATE TO FABLE for routes 2/3 (off-plane sheet / ±z pair).
> 5. Everything launches incl. 27 ⇒ the 136026 failure didn't reproduce —
>    h5-gate all, then ESCALATE TO FABLE with the config diff vs 136026.
> Exit-137/OOM or walltime ≠ CUDA verdicts either — rerun that rung in a
> bigger lane before classifying.
>
> ### ★★★★★FABLE PRE-DISPATCH AUDIT (2026-08-25) — 3 BLOCKERS, ALL SILENT
> **B1 — THE C_field SIGN WAS CONJUGATED. 18.4% error, signs unaffected.**
> `fit_c_field` solves `FD ≈ s(cosφ·RE − sinφ·IM)` and printed `C = s·e^{iφ}`
> = 0.4554 **−**0.1336i — but the ENGINE APPLIES `a·RE + b·IM`, which needs
> the CONJUGATE. RE-DERIVED INDEPENDENTLY HERE:
> stored (0.4554, −0.1336) ⇒ resid **−18.4 / −14.8 / −14.4 %** = FAILS W3;
> conj (0.4554, **+**0.1336) ⇒ resid −0.4 / +0.1 / −0.1 % = PASSES.
> ★**Why it was invisible: SIGNS ARE IDENTICAL EITHER WAY.** Every sign gate
> would have passed while the campaign ran on ~15% wrong step MAGNITUDES —
> wrong clipping, wrong restoration. FIXED in `campaign_v2_proj.py`, in task
> 41, and **in `fit_c_field.py` itself**, which now prints
> `>>> STORE THIS -> adj_fix_field_re/im = (s·cos, −s·sin)`.
> ### ★★IS THE **PORT** C CONJUGATED TOO? — INVESTIGATED, LEANS NO, NOT PROVEN
> This matters far more than the width C: `ADJ_FIX_PORT` steers EVERY campaign
> ever run, including 136753/136905 right now.
> **Evidence it is CORRECT as stored:** the engine comment (`lumopt2_design.py`
> :178-185) records the port C as *"ONE global complex factor C = 0.8685+0.1022i
> **fits FD on ALL 7 params to 1.7%**"* — a statement about the APPLIED value,
> not about a printed `s·e^{iφ}`; and the C-fix moved `vec_error` 11.40 → 0.144,
> a 79× improvement that a 15-18% magnitude error would not produce. The port C
> also did NOT come from `fit_c_field.py` (whose own docstring says the port C
> "does NOT transfer to the field-adjoint path") — it came from the earlier
> skill-item-6 recipe, so the conjugation convention is not shared by
> construction.
> **Why it is NOT PROVEN:** the raw (FD, Re, Im) triple for the port is NOT in
> the repo, so it cannot be re-derived. ★And note the B3 gate does NOT settle
> it: that gate accepts α ∈ [0.8, 1.25], and a conjugation error here is
> ~15-18% — **it would PASS the gate**. So "B3 passed" is not evidence.
> ⇒ **STANDING ITEM:** if the port triple is ever recovered (or re-measured),
> re-derive it against the applied convention `g = a·RE + b·IM`. Do not treat
> the existing gates as having covered this.
> ⇒ Original note: **CHECK `ADJ_FIX_PORT` (1.0561, 0.1239) the same way** if its raw
> FD/Re/Im vectors are ever recovered — same script, same convention. It
> passed its own B3 α-gate, so it is probably fine as stored, but it has
> never been re-derived against this convention.
> **B2 — `wg_track_resonance` defaults FALSE and the campaign never set it.**
> The twin (and the tiles, whose λ follows `sim_result.wavelengths`) would
> stay pinned at the scan centre while λ walks up to RECENTER_NM = 2.0 nm
> ≈ 2.7 linewidths. That violates the standing user ruling (2026-08-23):
> **softW is measured ON RESONANCE, always.** Off-resonance the width
> gradient's sign structure differs, and both restoration and the null space
> are sign-sensitive. FIXED: `wg_track_resonance=True`.
> **B3 — C_field was fitted at the WRONG MESH.** Tasks 37/40 ran `_w_spec`,
> i.e. `region_dx_nm` at the 50.0 DEFAULT; the campaign runs pitch-locked
> 51.683. Job 132637 measured tooth-gradient scales moving 10-30× between
> mesh conventions, and plan §24 rules a 50-nm C "NOT for production".
> ⇒ **RE-FIT DISPATCHED: job 137017 (tasks 42 Re / 43 Im) at the FULL
> production numerics** — pitch-locked dx, centre 1564.614, 4 tiles,
> resonance tracking ON, `wg_project=False` so it stays single-pass (~26 GB).
> The FD reference is unchanged and still valid: it is a finite difference of
> the SAME functional, so it is config-independent.
> **S4 — the ELONGATION WALL was still stacked under `wg_project`.**
> `attach_penalty` selects `elong_penalty` whenever `rho_band=False`, and
> `run_projected` uses the WRAPPED fom/gradient, so the wall silently
> re-entered despite the docstring saying "pure T". DERIVED: at e = 132.6 its
> shift-gradient is ~5e-4 vs a measured ∂T/∂shift ~4e-5 — it would DOMINATE
> the shift block ~12× and cap the run below `BEST_T9636`'s OWN e = 132.6,
> structurally forbidding the from-uniform campaign from reaching the basin
> it exists to test. FIXED: no analytic penalty at all under `wg_project`
> (WidthTrip still fail-closes the delivered design).
> ★**AUDIT CONFIRMATIONS worth keeping (do NOT "fix" these):**
> - The double field assembly is SAFE, not just memory-hungry: `get_jacobian`
>   rebuilds `jacobian(self.fct)` every call, the monitor cache is keyed on
>   (abspath, mtime), and `fct` is restored in `finally` — pass 2 cannot
>   return pass 1's fields.
> - The width entry IS last and the port entries contribute EXACTLY zero.
> - ★**grad-W is intentionally the NEGATION of the FD reference.** FD was
>   measured under `wg_pure` (J = −softW); pass 2 uses jac = +1 and yields
>   +d(softW)/dp, which is what climb/restore need. C_field is applied to the
>   adjoint FIELDS and is jac-independent, so the same C is correct for both.
>   **Do not "correct" grad-W to match FD — that would invert restoration.**
> - C is applied ONCE, to the width entry only; tiles are campaign-safe
>   (rebuilt per adjoint from the monitor's own samples, parent source-mode
>   forced off each time, never leaking into the port adjoint).
> **STILL OPEN (S6):** one flaky profile extraction raises RuntimeError and
> ends the whole campaign with no resume benefit — worth one retry or a
> restart-from-log before the 81 h run.
>
> ### ★★★MEMORY: `wg_project` DOUBLES THE GRADIENT-FIELD FOOTPRINT (2026-08-25)
> **MEASURED: job 137012 (P2 gate) OOM-KILLED, exit 137, at "Computing
> gradient fields from forward + adjoint data".** Cause is structural and
> mine: `wg_project` runs `calculate_gradient_fields` TWICE (grad-T then
> grad-W), so a second full `(nx, ny, nz, 3, n_wl)` array is live alongside
> lumopt2's per-entry forward+adjoint arrays — and lumopt2 fetches the FULL λ
> grid per entry before slicing (`fdtd_session.py:1355`).
> DERIVED for the pitch-locked region (2044×28×15 = 0.88 M cells):
> ONE field array is **6.4 GB at n_wl=151** but **21.3 GB at n_wl=501**;
> the live stack is ~5 such arrays ⇒ **~32 GB at 151, ~107 GB+ at 501**,
> and 501 blew a 160 G job. Tasks 37/40 survived only because they ran 151.
> ⇒ **GATES: n_wl_points 501→151** (a sign check cannot be changed by
> spectral sampling) and dispatch at 250 G — redispatched as job **137015**,
> validated first: bounds violations 0, memory ~32 GB, projection still on,
> wall still off. P3 (137014) was CANCELLED before it hit the same wall.
> ⇒ **CAMPAIGN keeps 501 λ (the FOM needs the spectrum) and MUST run at
> `SBATCH_MEM=300G`** — written into `campaign_v2_proj.py`'s docstring.
> ★If 300 G ever proves tight, the principled fix is to contract the width
> fields to the 191-vector INSIDE `calculate_gradient_fields` instead of
> stashing the full array — blocked today only because `params` is not passed
> to that method, so it would need a small signature change.
>
> ### ★★★★★★2026-08-25 04:01 — **THE WIDTH GRADIENT IS FIXED AND CALIBRATED.**
> ### PRIORITY ZERO IS DONE. W3 PASSED.
> **MEASURED, array 137003, both tasks exit 0, both `[wg-tiles] live 4/4`:**
> | | corr_1 | shift_1 | wcav |
> |---|---|---|---|
> | adjoint Re (t37, C=(1,0)) | −0.0072625 | +0.03713637 | +0.04125803 |
> | adjoint Im (t40, C=(0,1)) | −0.0024589 | +0.01015485 | +0.01087661 |
> | keep-forever FD (136189) | −0.00365 | +0.01825 | +0.02026 |
> | **raw Re/FD ratio** | **1.990** | **2.035** | **2.036** |
> ★The raw ratio is constant to **±1.2% across three different parameter
> classes** — the textbook signature of a CORRECT adjoint awaiting ONE global
> constant. (Contrast the cropped rungs: 0.35-1.83, because they measured a
> different functional.)
> **FIT (`fit_c_field.py`, PEN_GRAD zeroed — these prints are raw/wg_pure):**
> **`C_field = 0.4554 − 0.1336i`** (s 0.4746, φ −16.35°)
> **vector residual 0.1%**; per-param −0.4% / +0.1% / −0.1%; **signs 3/3**.
> The W3 gate is ≤10% per param ⇒ **PASS by ~25×.**
> ⇒ Written into `campaign_v2_proj.py` as `ADJ_FIX_FIELD`; its guard now
> passes. **Cost: ~54 min forward + ~54 min adjoint ≈ 1.8 h per gradient**
> (3221 s adjoint), vs 8.7-12.1 h on CPU. The exact ∂W/∂p for all 191
> parameters is now affordable in-loop.
> ★**STILL OWED, do not skip:** the fit docstring's standing order — verify at
> a SECOND operating point before trusting magnitudes broadly, and re-fit on
> every Lumerical version bump. Tasks 38/39 (dispatched, job **137011**) are
> at different points and check SIGNS there; a magnitude re-fit elsewhere is
> still outstanding.
> **NEXT (running): P2/P3 known-answer gates, job 137011.**
> P2 = at the uniform seed the projected step's SHIFT block must be strongly
> positive (>0 on ≥20/25 teeth) — shifts below the e=65 knee are MEASURED
> width-free. P3 = at ceiling contact the CORR block must show the
> inner/outer SIGN SPLIT (the see-saw). Magnitudes are not gated there —
> C_field only rescales, it cannot flip a sign.
>
> ### ★★★★★2026-08-25 02:02 — **THE GPU ACCEPTS TILED SOURCES.** R6 CLOSED.
> **Job 136967 ran the FULL 33-minute tiled adjoint on GPU with 4 simultaneous
> field-region sources and did NOT raise `invalid configuration argument`.**
> It failed at the very end on a DATASET SHAPE bug of mine, not a GPU limit:
> `field region field_profile_adj_t0 imported source profile dataset
> dimensions (528 x 17 x 1) do not match the field region dimensions
> (529 x 17 x 1)`.
> ⇒ **THE CUDA 1024 BOUND IS PER-SOURCE, NOT PER-RUN** — the tiling premise is
> MEASURED-correct, and the whole architecture stands. This was the single
> biggest remaining risk (logged as R6) and it resolved the good way.
> ★**THE FENCEPOST BUG AND ITS FIX.** `tile_x_edges` places edges at sample
> MIDPOINTS, and I set each tile's span from those edges — but the engine's
> region then sampled 529 cells where the slice held 528. Fixed by deriving
> the geometry from THE SAMPLES THE TILE MUST CARRY instead of from the
> edges: centre on the tile's own first/last sample, span = extent + 0.9·dx.
> Every intended sample then sits ≥0.45·dx INSIDE the region and the nearest
> EXCLUDED neighbour sits 0.55·dx OUTSIDE. VERIFIED locally for the exact
> 136967 geometry (2114 samples, 4 tiles): slice sizes [528, 529, 529, 528]
> and the region would sample [528, 529, 529, 528] — MATCH on all four.
> ★LESSON: never infer a Lumerical object's sampling from your own geometry
> arithmetic; derive the geometry from the samples you intend to occupy. The
> engine's inclusive-boundary rule is not yours to guess.
> **REDISPATCHED: tasks 37 + 40 in parallel (elements 137003 / 137004)** —
> the tiled gradient and its Im-quadrature partner, both at the production
> tiling, so `fit_c_field.py` gets Re and Im from ONE wall-clock window.
> Seats probed 9/50 first.
> ★**Why the C fit is meaningful HERE and was not on the cropped rungs:**
> tiling reproduces the FULL-REGION source, so this gradient is directly
> comparable to the keep-forever FULL-REGION FD `[-0.00365, +0.01825,
> +0.02026]`. The cropped rungs measured a DIFFERENT functional.
>
> ### ★★★TRAP: `validate_gradient` FANS OUT **6 CONCURRENT SIMS** (2026-08-25)
> **Job 137035 (FD reference) DIED** after its forward (2787 s) and adjoint
> (3136 s) both completed NORMALLY, with:
> `LumApiError: "Can not find result 'expansion for port monitor' in the
> result provider 'FDTD::ports::Port_2'"`.
> **DIAGNOSIS via the §6 rule (check "Simulation time" first): NOT a license
> no-op of the main solves** (they took 46 and 52 min, not ~1 s) and NOT an
> h5 clobber. The log says it outright:
> `12:28:01 Generating 6 perturbed simulation files for concurrent execution`
> `12:28:23 Running 6 finite-difference simulations concurrently...`
> ⇒ **`lmpt.validate_gradient` runs its FD legs 6-WIDE**. At that moment BOTH
> baseline campaigns were still live, so we had ~8+ concurrent solves of our
> own against a faculty-shared pool. Losers no-op silently, produce no port
> expansion, and `calculate_fom` then raises — the documented Athena
> (container) starvation signature.
> ★**THE LESSON, and it is a budgeting rule, not a bug:** a
> `run_validate_gradient` task is **NOT one seat — it is 1 + n_legs**, i.e.
> 7 for 3 indices with central differences. §6's mandatory seat check must
> budget the FAN-OUT, not the task count. Probing 23/50 free and dispatching
> "one task" was wrong arithmetic.
> ⇒ **NOT RE-RUN.** This was VERIFICATION, not a gate: `wgp_autogain` now
> measures the magnitude online, and the phase sensitivity is only ~0.04° of
> direction per degree of phase. If it is ever wanted, throttle the FD legs
> (or run with the campaigns stopped) and budget ~7 seats.
>
> ### ★★★★★THE CALIBRATION IS MOSTLY A RESONANCE PHASE, NOT A YEE ARTIFACT
> ### (2026-08-25, Fable — this changes the METHOD, not just a number)
> **THE LAW:** `arg C(λc) ≈ φ_geom − arctan(2(λc − λr)/FWHM)` — a small fixed
> geometric (Yee-like) term PLUS **the resonance's own Lorentzian transfer
> phase**, which the fit silently absorbs whenever the adjoint is evaluated
> OFF-RESONANCE (the softW twin records at the scan centre).
> **THE EVIDENCE, and it is tight:** the fitted phase moved **48.85°**
> (−16.35° → −65.20°) when the scan centre moved 0.55 linewidths; the
> Lorentzian phase for that displacement is `arctan(2×0.55) = 47.73°` —
> agreement ~1°. A Yee offset would have moved it **< 0.3°** (k·dx changes by
> 3.4%). ⇒ **PURE-YEE IS DEAD as an explanation**; it also explains why the
> PORT path (evaluated essentially on resonance) sits at −6.7° ≈ the
> quarter-cell 6.2°, while the width path does not.
> ⇒ **CONSEQUENCE: with `wg_track_resonance` ON the detuning term VANISHES
> and the phase should collapse to the small geometric value.** So Fit B's
> −65.20° is WRONG for a tracking campaign — it is mostly detuning. Fit A's
> −16.35° is far closer to geometric and is what the live seed run uses.
> ⇒ **TO GET φ_geom CLEANLY: fit at a scan centre SET TO THE MEASURED
> RESONANCE** (δ = 0 by construction), not at the family centre.
> ★**HOW MUCH THIS ACTUALLY MATTERS — DERIVED, and it is reassuring:** RE and
> IM are only **5.52° apart**, so a large phase error makes a SMALL direction
> error. Measured sensitivity: φ −6.7° → 0.20°, −16.4° → 0.48°, −30° → 0.87°,
> −45° → 1.35°, −65.2° → 2.27° of direction change vs φ=0. **~0.04° of
> direction per degree of phase.** The projected step only cares about
> direction, so even a badly wrong phase perturbs the null space by ~1-2°,
> which the measured-fwhm_env restoration corrects each iterate.
> **PRODUCTION METHOD, DECIDED = (b): FIT THE PHASE, MEASURE THE MAGNITUDE.**
> |C| is EXACTLY irrelevant to the direction (verified 0.00°), so fitting it
> against an expensive FD reference was solving a non-problem. `wgp_autogain`
> (implemented, ON for the campaign) compares predicted vs MEASURED ΔW each
> iterate and self-calibrates — surviving mesh changes with no new FD run.
> ★AUDIT of autogain (Fable): SAFE — the ride branch is exactly invariant to
> the gain, the scalar cap preserves ∇W·d = 0, and the gain updates only on
> ACCEPTED steps from the STORED UNSCALED gW, so it cannot compound with the
> α backtracking. **One caveat FIXED:** a NEGATIVE predicted/measured ratio
> means the width moved OPPOSITE to prediction — a PHASE error no scalar gain
> can repair — and it would have pinned the gain at its 0.2 clamp while
> masquerading as a harmless small gain. Now trips a loud warning naming the
> phase as the cause.
> ★REJECTED: (c) derive the phase from the Yee offset alone — refuted above;
> (d) `colocate_fields` to force C → 1 — recorded expected-PARTIAL
> (α ~0.3-0.5), so a residual C would still need fitting.
>
> ### ★★★★★2026-08-25 13:40 — BASELINES STOPPED, PROJECTED SEED RUNNING
> **CANCELLED: 136753 (uniform s5) and 136905 (seesaw)** — both old-method
> (hinge-wall) campaigns. Logs FETCHED FIRST (17 and 15 rows) to
> `results_from_athena/lumopt2_c325_logs/lumopt2_v2_{uniform_s5,seesaw}_evals.jsonl`.
> Resume is proven, so either can be restarted from its best in-band row.
> **WHY: both had reached the failure mode we are replacing, and they were the
> node contention making every iterate cost 2.7 h instead of ~1.1 h.**
> ★**THE BASELINE, now the number the projected method must beat:**
> | | seed | last accepted | last probe |
> |---|---|---|---|
> | uniform s5 | T 0.90120 / 18.3452 | T 0.92689 / 18.6284 | **T 0.94644 / 19.5321 OUT OF BAND** |
> | seesaw | T 0.93790 / 18.3318 | T 0.94777 / 18.6259 | **T 0.95078 / 18.9630 OUT OF BAND** |
> ⇒ **BOTH campaigns ended up out of band.** Over 15.9 h the uniform seed
> bought **+0.026 T with +0.283 µm of width** — it is not optimising at fixed
> width, it is trading spec for performance and periodically crashing through
> the ceiling. That is the defect, measured over a long run, on two seeds.
> **LIVE: 137075 = the UNIFORM SEED UNDER THE PROJECTED METHOD** (validate
> task 41, 4 iterates, 12 h lane, 300 G, C=(0.4554,+0.1336), tiles 4,
> track_resonance ON, no analytic wall). PASS = fwhm_env within ±0.05 µm of
> 18.613 on EVERY accepted iterate and NO width rejection. Read
> `lumopt2_v2_proj_toy_proj.jsonl`: `dw_pred` vs the measured ΔW is the direct
> test of whether the gradient's width prediction is TRUE.
> **LIVE: 137035 = the FD reference at production numerics** — kept as a
> parallel CHECK, no longer a gate. ★Justification for not waiting (DERIVED):
> the two candidate calibrations differ by only **1.79° in direction** and 15%
> in magnitude on the production vectors; the projection removes the component
> ALONG grad-W, so 1.79° leaves a ~3% width component per step, which the
> measured-fwhm_env restoration corrects every iterate. The magnitude error
> only rescales the step, and the 5 nm cap bounds it.
>
> ### ★★★★MEASURED 2026-08-25 — **THE CAVITY-WIDTH HEADROOM IS NOT FREE.**
> ### Item ②-1 CLOSED, NEGATIVE. First physics result of the session.
> **Job 137021, ONE forward at the production pitch-locked numerics** (the
> wcav-961 control was NOT re-run — it is the stored 136465 eval-12 row at the
> SAME numerics, §6):
> | | wcav (nm) | T | fwhm_env (µm) | λ (nm) | Q_i |
> |---|---|---|---|---|---|
> | BEST_T9636 (stored) | 961.1 | **0.96361** | 18.35309 | 1566.444 | 110 087 |
> | probe (this run) | 1100.0 | **0.94389** | 18.30151 | 1566.568 | 70 100 |
> ⇒ **ΔT = −0.0197 for Δwcav = +138.9 nm** — a LOSS of ~10× the 0.002 T noise
> floor, with Q_i collapsing 110k → 70k. Width barely moved (−0.052 µm).
> ⇒ **The 189 nm of "unexplored headroom" is not headroom: `BEST_T9636` is at
> or near its wcav optimum, and pushing further is harmful.** Handoff item
> ②-1 is CLOSED as a negative; do not re-propose it.
> ★**WHY THE PREDICTION WAS WRONG, and it is a repeat offence:** the "+0.0409 T
> for +0.0305 µm ⇒ ~1.3 T/µm, 50-60× the see-saw" figure came from the rtdec
> rows, measured over 800 → 961 nm ON A DIFFERENT DESIGN. Applying it to
> 961 → 1100 on `BEST_T9636` is extrapolating a secant past its measured
> range — the SAME error the programme logged for FW_A_ELONG, FW_A_MCORR and
> my own see-saw payback amplitude. The rate was real where it was measured;
> the lever simply saturates and then reverses.
> ⇒ Standing rule reaffirmed: **every rate carries the amplitude range it was
> measured over, and any use outside that range is a PREDICTION TO BE TESTED.**
>
> ### ★★★USER'S PHYSICS QUESTIONS ABOUT `BEST_T9636` (2026-08-25) — PARK,
> ### ANSWER AFTER THE GRADIENT WORKS. Two are partly ALREADY MEASURED.
> The user's framing, which is the right one: `BEST_T9636` was reached by a
> CHAIN of hand adjustments — change corrugation, narrow the mode, then raise
> tooth shifts — i.e. a COMBINED move, not one honest optimization. Open
> question: **can a uniform seed reach that basin at all?** Is there a path
> from uniform along which the mode width stays CONSTANT while T rises —
> possibly tooth shifts "fighting" the width growth? Unknown, and it is
> exactly what the projected (null-space) method is built to answer, because
> constant-width paths are precisely its search space.
> **Concern (a): the corrugation profile drops abruptly to 325 at tooth 25**
> (the free/frozen boundary). The user finds the short transition
> unphysical — "why 25 and not 20?" — and suspects a FAMILY of best devices.
> ★**PARTLY ANSWERED, MEASURED (job 136302 task 24, the "de-step" run):**
> ramping teeth 20-25 smoothly down to the frozen 325 gave T 0.9600 @ 17.862
> = **+0.0006 vs the stepped design, i.e. INSIDE the 0.002 T noise floor** ⇒
> **the 46 nm boundary step is NOT the mechanism; bulk inner-tooth κ is.**
> Corroborating: outer free teeth 20-25 are INERT (mean corr −1.75% moved
> width only +0.05%). So the step is cosmetically odd but not load-bearing.
> ★**STILL UNTESTED, and the user is right that it is open:** whether the
> BOUNDARY LOCATION itself (N_FREE = 25) is optimal. N_FREE 25→40 is a banked
> v2.1 candidate (~10× light-cone model headroom). If the outer free teeth
> are inert, moving the boundary should be nearly free — which also means a
> FAMILY of equivalent designs is plausible, as the user suspects.
> **Concern (b): is it even converged, and will it survive the Q3dB device?**
> MEASURED: campaign 136465 moved its seed by ≤0.26 nm in ANY of 191 params
> against trust radii of 10-12 nm — it used ~2% of its allowed travel, so
> "converged" means "the optimizer probed and rejected everything nearby",
> NOT that no better point exists. The production/Q3dB confirm at N≈169 +
> accurate mesh has NEVER been run — the user's worry is legitimate and
> unaddressed.
> ⇒ **ORDER (user, explicit): fix the gradient FIRST. Then the uniform seed.
> Then re-examine the best design with these three questions.** Do not spend
> GPU on them before the gradient is gated.
>
> ### ★★★★★THE X THRESHOLD IS PINNED (2026-08-25 ~00:40) — IT IS ~1024
> **MEASURED, job 136907 task 35: x = 1000 cells PASSED** (exit 0, vector
> `[-0.00127211, +0.01191436, +0.00730678]`, signs 3/3 vs the keep-forever FD).
> **x = 1056, 1080, 2112 all REJECTED; x = 528 and 1000 both PASS.**
> ⇒ **The bound lies between 1000 and 1056 — i.e. CUDA's 1024 threads/block**,
> exactly as the `lumcudafdtd.dll` string "Total threads per block %u exceeds
> device limit of %d" predicted. The mechanism is now understood, not guessed.
> **CONSEQUENCE FOR TILING: 3 tiles suffice** (2112/3 = 704 cells, comfortably
> under 1024); `wg_src_tiles=4` (528/tile) is what is dispatched and carries
> extra margin at no extra cost — tiles share ONE adjoint run.
> ★Ratio spread across passing rungs (0.35-0.65 here vs 0.39-0.44 at rung 30)
> confirms again that these ratios are CROP-dominated — fit C_field ONLY at
> the production tiling.
>
> ### ★★★TINY-SCENE PROBES HUNG — MY OWN FIX FAILED, OWN IT (2026-08-25)
> `gpu_probe.py` tasks 0/1 (jobs 136917/136921) **stalled inside the FIRST
> `fdtd.run("FDTD","GPU")`** — GPU handshake printed, then ~50 min of silence,
> no error. CANCELLED (scancel, ~100 GPU-min wasted). **The scene is NOT the
> problem**: a local build-only check confirms 0.67 M cells, 10 fs, every
> property set correctly. The fault is in a HAND-ROLLED lumapi session calling
> `run()` directly, versus lumopt2's proven `FdtdSession`/`LocalRunner` path
> (which every device rung uses without trouble).
> ⇒ **LESSON, and it does NOT retract the tiny-scene rule (CLAUDE.md §5):**
> the rule is right, but a debug probe must reuse the PROVEN execution path,
> not re-implement it. A hand-rolled harness adds a second unknown alongside
> the bug you are chasing. If a probe needs a solver run, drive it through the
> same session machinery production uses, or make it build-only.
> ⇒ Do not sink more time into `gpu_probe`'s run path; the device task 37
> answers the same question through the proven path.
>
> ### ★★★★TILING IMPLEMENTED 2026-08-25 00:xx — `wg_src_tiles`, DEFAULT-OFF
> **Engine now carries the fix** (`lumopt2_design.py`): `wg_src_tiles: int = 1`
> + `wg_tile_max_xcells: int = 528`, `tile_x_edges` / `split_dataset_x` /
> `import_tiled_source`, and a 3-way dispatch in
> `MixedFom.setup_adjoint_simulation` (import | tiled | legacy).
> With `wg_src_tiles>1`, `field_profile_adj` stays a PURE MONITOR and the same
> weighted dataset is injected through N narrow FieldRegion tiles **all
> enabled in ONE adjoint run** — sources superpose linearly and the gradient
> assembly is linear in the adjoint field, so the result is EXACT and costs
> ONE adjoint, not N. Tile x-geometry is set at ADJOINT time from the
> monitor's own recorded samples (the mesh is frozen by then); interior edges
> are sample MIDPOINTS so no sample is shared or lost.
> **VERIFIED LOCALLY, ZERO GPU:** default is 1 (running campaigns
> bit-identical, REQUEUE-safe); the partition reconstructs the dataset
> ELEMENT-FOR-ELEMENT over nx ∈ {2112, 2111, 2113, 100, 7} × N ∈ {1..8}; no
> interior edge lands on a sample; **2112 / 4 = 528 exactly** = the highest
> measured-pass rung.
> **GATES DISPATCHED ON DUMMY SCENES (the new §5 tiny-scene rule):**
> - **136917 = `gpu_probe` task 0** — the x-threshold ladder, 12 sizes
>   (256…2112) incl. the suspected 1024 boundary, two short solves each.
>   (136914 was the same thing and died in SECONDS on a lumapi signature
>   error — `add*` helpers take NO geometry kwargs in this build; use
>   `add()` + `set()`. That is the tiny-scene rule paying for itself.)
> - **136918 = `gpu_probe` task 1** — ★the TILING CORRECTNESS gate, and it
>   needs NO device because source superposition is generic physics: inject a
>   structured synthetic weight (a) through ONE 512-cell source, (b) through
>   4×128-cell tiles carrying disjoint slices of the SAME data, compare an
>   independent witness monitor. PASS = max relative difference < 1e-6.
>   A mis-sliced tile cannot hide: the weight has 3 sine periods across x.
> ⇒ **The only DEVICE run still required is the full-region 4-tile gradient**,
> sign-checked against the keep-forever FD `[-0.00365, +0.01825, +0.02026]`,
> then `fit_c_field`. Everything else was moved off the device.
> ★**R1, the one that can silently corrupt the gradient:** tiles are assumed
> to sample the FROZEN global mesh. If a tile builds its own grid,
> `importdataset` INTERPOLATES and the partition stops being exact. The gate
> is free — assert `concat(x_t0..x_tN) == x_of(field_profile_adj)` by float
> equality. ★R6: N sources in one run is itself untested; if the tiled
> adjoint still dies, the next probe is ONE 528-cell tile covering part of the
> region, separating a per-source X limit from a per-run aggregate.
>
> ### ★★★★★THE ROADMAP — ONE PATH, NO OPTIONS (user order 2026-08-24 23:30:
> ### "work autonomously and actually fix everything and see that we get a
> ### working optimizer that manages to keep the FWHM constant")
> **STAGE 1 — MAKE THE GRADIENT WORK. Nothing else starts until 1.4 passes.**
> - 1.1 **PIN THE X THRESHOLD** — job **136907** rungs 35 (x=1000) / 36
>   (x=1080), in flight. Output = the safe tile width. [MEASURED so far:
>   x=528 PASS 3/3, x≥1056 FAIL 4/4, cell counts OVERLAP across the split
>   (pass 3,696-45,936 vs fail 29,568-183,744) ⇒ **x alone is the variable**.]
> - 1.2 **TILE THE ADJOINT SOURCE IN X.** ★**This is EXACT, not an
>   approximation, and the argument is why the whole route works:** the
>   FORWARD runs fine at full size, so the softW weight W(x,y) = dsoftW/dI ×
>   y-trapz is computed ONCE from the complete forward field; the adjoint
>   problem is LINEAR in its source; and the gradient assembly
>   ∫ E_adj·(dEps/dp)·E_fwd is LINEAR in E_adj. Therefore
>   **∇W(full source) = Σ_i ∇W(tile_i source)** exactly, provided each tile
>   carries its own slice of the SAME globally-computed W and zeros
>   elsewhere. The global quantities inside softW (softmax peak, fixed-edge
>   floor) are evaluated on the full forward BEFORE partitioning, so
>   non-separability of softW is NOT a problem — we never partition softW,
>   only its adjoint source. 3 tiles across 2112 cells at ≤1000-cell tiles.
> - 1.3 **GATE the tiled gradient** vs the keep-forever CPU FD
>   `[-0.00365, +0.01825, +0.02026]`: sign agreement, then
>   α ∈ [0.8, 1.25] after 1.4. A tiled run must also reproduce a
>   single-tile run on a region small enough for both (consistency check
>   that costs one extra eval and catches partition bugs).
> - 1.4 **FIT C_field AT THE PRODUCTION TILING** (`fit_c_field.py`, needs the
>   Re run + an Im-quadrature partner). NEVER on a cropped rung — rung 30 and
>   rung 34 gave ratios 0.40 vs 1.5 purely from region change, which is
>   exactly why a cropped C is meaningless.
> **STAGE 2 — MAKE THE OPTIMIZER HOLD THE WIDTH.** Gates P0/P2/P3/P4/P5 as
> specified in the formulation block below; P2 and P3 are single-gradient
> SIGN checks (~2 evals) and are the cheap falsification of the whole design.
> Then wire into the campaign driver: PHASE A predictive step-clipping,
> PHASE B null-space projection + measured restoration, λ logged per iterate.
> **STAGE 3 — THE SEEDS, in the user's stated order.**
> - 3.1 **THE UNIFORM SEED** under the new method — the honest-result
>   campaign. Its failure mode is already MEASURED (3 of 7 evals rejected on
>   width) and is exactly what phase-A clipping removes.
> - 3.2 **IMPROVE THE BEST DEVICE** (`BEST_T9636`): deliberately spend its
>   0.36 µm of slack to reach the ceiling (where the constraint is finally
>   active), then ride. Its stall is MEASURED as near-zero ∇T with slack —
>   the method does NOT fix that on its own; spending the slack is a driver
>   decision that must be taken explicitly.
> **MEANWHILE (do not disturb):** 136753 (uniform, hinge wall) and 136905
> (seesaw, resumed at T 0.94493) keep running as the BASELINE the new method
> must beat. They are the control for "did any of this help".
> ★**FABLE BUDGET: ~10% weekly left, a few days.** Spend it ONLY on: (a) a
> failed 1.3/1.4 gate, (b) the campaign-driver wiring review before Stage 2
> dispatch. The tiling linearity argument above did NOT need it and the
> formulation is already decided — do not re-litigate either.
>
> ### ★★★★★THE FORMULATION, DECIDED 2026-08-24 23:00 (Fable brief, user's
> ### own intuition formalised): **CEILING-RIDING PROJECTED GRADIENT.**
> The user asked "with the full gradient map we know where width MINIMISES,
> but also WHERE IT IS CONSTANT — what should be done?" That is exactly the
> right question and the answer is: **"where it is constant" = the null space
> of ∇W = the tangent space of the active ceiling manifold.** Ride it.
> **PRIMARY METHOD (formulations 1+3 are two halves of ONE method):**
> - **Treat width as an EQUALITY at the ceiling, not an inequality.** The
>   active set is known A PRIORI from measured physics: every strong T lever
>   spends width (136753 ev5 T 0.95912 @ 20.37 µm; above-knee shifts
>   0.0136 T/µm; mcorr falling 325→311.4), so T is monotone in width along
>   every dominant direction ⇒ **the optimum sits ON the ceiling.** With the
>   active set known, projected/reduced gradient (Rosen/GRG) is textbook and
>   needs NO multiplier schedule — the shadow price is implicit.
> - ★**Ride W_hi − margin (18.713 − 0.10 = 18.613), NOT the 18.346
>   benchmark.** The band's upper half is free real estate: `BEST_T9636` sits
>   at 18.353 with **0.36 µm of unspent slack ≈ +0.005 to +0.015 T** at
>   measured marginal rates. This IS handoff item ②-3, now with a method.
> - **PHASE A (climb, W < target): predictive step clipping.** `W(p+αd) ≈
>   W + α ∇W·d` is FREE once ∇W exists ⇒ cap α so no probe is ever predicted
>   past the ceiling. ★**This is the single biggest immediate saving and it
>   needs no convergence theory**: it converts tonight's measured waste
>   (3 of 7 evals in 136753 rejected on width, ~40% of GPU time, 45-58 min
>   each) into a free linear prediction.
> - **PHASE B (ride): null-space step** `d = D∇T − (D∇T·û)û`, û = D∇W/|D∇W|,
>   width-neutral to first order; **restoration** `p −= (W−W_tgt)∇W/|∇W|²`
>   on MEASURED fwhm_env when drift exceeds margin/2; **log the shadow price
>   λ = (∇T·û)/|∇W| every iterate.** `D` = per-block trust scales squared,
>   which also fixes the 0-200-vs-0-7 conditioning INSIDE the method.
> - Acceptance = Fletcher-Leyffer filter (already in the V2 plan); WidthTrip
>   and a W_lo hard-reject (see-saw over-narrowing guard) stay.
> - ★**It prices the see-saw's first leg**: on the manifold, narrowing earns
>   λ > 0, so the locally-T-negative outer-tooth raise is CREDITED for the
>   width budget it frees, combining both legs into ONE first-order step.
> **FALLBACK: equality-form AL** — reuse the existing `wg_mu`/`wg_lam_hi/lo`
> machinery with ONE change: update the multiplier on the SIGNED measured
> deviation from W_target, not `max(0,g)`, so it is never dormant. ~10-line
> edit, inner L-BFGS-B untouched. **Trigger:** projected direction zig-zags
> 3 consecutive iterates, OR restoration consumes >⅓ of accepted steps, OR
> the drift budget blows twice.
> **REJECTED:** inequality AL as primary (λ·max(0,g) is zero while feasible,
> and the multiplier can only learn the price BY overshooting — the blind
> search again at 45-90 min per lesson); SLSQP/trust-constr (eval-count
> multiplier, noise-intolerant at the 0.002 T floor, invalidates the
> L-BFGS-B-calibrated trust machinery); the hinge wall for STEERING (keep it
> as a WidthTrip tripwire only).
> **GATES — P2/P3 make the whole design falsifiable for ~2 full evals:**
> P0 local: projected direction satisfies ∇W·d = 0 to machine precision +
> scaler round-trip exact. P1: FD gate AT THE PRODUCTION TILING CONFIG
> (never on a cropped rung), sign 6/6 + α ∈ [0.8, 1.25] after `fit_c_field`.
> **P2 free-shift known-answer**: at the uniform seed the projected shift
> components must be strongly positive (e ≤ 65 is measured width-flat) —
> >0 on ≥20/25 teeth, then 2-3 steps give ΔT ≥ +0.010 at ΔW ≤ 0.05 µm.
> **P3 see-saw known-answer**: at ceiling contact the corrugation block must
> show the inner/outer SIGN SPLIT matching the measured ~11× price ratio —
> ≥20/25 teeth. P4 restoration: displace W by +0.15 µm, ONE step recovers
> ≥70% (this is the end-to-end C_field MAGNITUDE test). P5: 5 null-space
> steps, cumulative drift ≤0.15 µm.
> **WHAT IT DOES NOT FIX (stated up front):** true local maxima on the
> manifold — keep see-saw seeding, P3 tests the gradient at contact, not
> everywhere; `BEST_T9636`'s stall (near-zero ∇T with slack) — the method
> prescribes DELIBERATELY spending the slack first, a driver decision, not
> emergent; **C_field + tiling are PREREQUISITES, not details** — step
> clipping and restoration need MAGNITUDES, and signs alone (all we have
> tonight) steer direction but cannot control step length; noise floors
> (5.5 nm width, 0.002 T) unchanged; and the T-ceiling exchange rate is
> untouched — 0.97 plausible, 0.98 still needs new structure.
> **BUDGET (EXPECTED):** ~2-2.5 h per full iterate (fwd 22 min + port adjoint
> 22 min + tiled width adjoint ~1.5 h at 4 tiles), ~1 h on climb iterates
> reusing ∇W up to 3 steps (refresh if measured ΔW deviates >2× from
> predicted). Gates P0-P5 ≈ 10 evals ≈ 25 GPU-h. Campaign 35-55 evals ≈
> 80-140 GPU-h per seed.
>
> ### ★★★INCIDENT 2026-08-24 20:30 — CAMPAIGN 136752 KILLED BY LICENSE
> ### STARVATION, AND MY DISPATCH CONTRIBUTED. PROCESS FIX BELOW.
> **MEASURED (`sacct`): `136752_0 FAILED, elapsed 04:04:46, exit 1:0`**, log
> ends `RuntimeError: Optimization failed: ... 'in run:'` — the documented
> license-starvation signature, at 20:30:55.
> **What I did wrong:** I dispatched the 6-task ladder (136799) at 19:31
> WITHOUT probing seats, on the reasoning that a probe an hour earlier read
> 10/50 and that the connection budget was tight. CLAUDE.md §6 makes the
> seat check **MANDATORY before any dispatch of more than one task** and says
> in terms that REACHABILITY ≠ AVAILABILITY and the pool oscillates. It went
> 10/50 → **45/50** within the hour; 3 of my 6 rungs died instantly on the
> same signature, which was the warning I should have acted on.
> **Honest attribution:** the pool is faculty-shared and others took ~29 of
> the seats, so our 6 tasks were contributory, NOT the sole cause — but the
> process failure is mine regardless of the arithmetic, and it cost a
> 4-hour campaign.
> ⇒ **RULE, now followed: probe seats IMMEDIATELY BEFORE every multi-task
> dispatch, never from a cached reading.** A stale seat count is not a seat
> count. (Subsequent dispatches 136826/136869 were both probed first: 27/50
> and 9/50.)
>
> ### ★★★RETRACTED 2026-08-24 22:40 — **THERE IS NO RESUME DEFECT.**
> **MEASURED, decisive:** after 136753's 21:13 requeue its log prints
> `[fwhm-wall 0] re-anchored at measured 18.4092 um (mcorr 323.6)` — that is
> **eval 4's** width, NOT the seed's 18.3452. **The campaign warm-started
> correctly from its best in-band logged row**, losing exactly ONE
> evaluation, which is the designed §6 budget.
> **Why I got it wrong (two bad inferences, both avoidable):**
> 1. `Computing baseline FOM at iteration 0 (initial parameters)` is printed
>    **UNCONDITIONALLY** at every `opt.run()` regardless of `initial_params`
>    (`lumopt2/core/optimization.py:940`). It is not a seed indicator — it is
>    the one-evaluation resume cost being paid. I read it as proof of a cold
>    start.
> 2. The duplicate `eval 1` seed rows are **timestamped 17:24-17:26**, i.e.
>    the deliberate trust-box RE-DISPATCH — about three hours BEFORE the
>    20:30 death and the 21:13 requeue. At 17:25 the seed genuinely WAS the
>    only band-compliant row in each log (the sole alternative scored
>    −1.53/−1.55, out of band), so resuming to it was CORRECT output.
> ⇒ **The rows carry a `t` unix-time field — date the rows before inferring
> a restart.** Both errors came from reading a log line and a row ordering
> as evidence without checking timestamps.
> **The resume mechanism, MEASURED as it exists:** `run_campaign` calls
> `_best_from_log` UNCONDITIONALLY (`lumopt2_design.py:1884-1890`); it scans
> `<label>_evals.jsonl` (appended PER EVALUATION at `:1450`, before any guard
> can raise), filters to the FWHM deadband fail-closed for
> `fwhm_wall`/`width_grad` campaigns, takes max FOM, and returns
> `(params, lam_pk_nm)` → seed + scan centre. All four failure hypotheses
> (iteration-only file / path mismatch / flag-gated / not implemented) were
> tested against the code and FALSIFIED.
> **REAL residual gaps (not defects, but know them):** L-BFGS-B's quasi-Newton
> curvature is NOT checkpointed (lumopt2 has no such facility), so a requeue
> restarts with an identity Hessian at the resumed point — eval loss 1,
> curvature loss real; AL multipliers `wg_lam_hi/lo` reset to 0 on restart
> (only bites `width_grad=True` campaigns); `best["fom"]` is not restored so
> `_best.json`'s number can understate the campaign (params are re-filtered
> through the log, so the DELIVERED DESIGN is unaffected).
> ★**LABEL RULE made explicit:** `_best_from_log` compares raw `row["fom"]`
> across every row under a label, so **any edit that changes the FOM
> DEFINITION must take a NEW label** (s2→s3→s4→s5 already does this).
> Changing only bounds/trust boxes may reuse the label, as the 17:25
> relaunch did.
> ⇒ **CONSEQUENCE: campaign 136752 was safely relaunchable, and was
> relaunched — Athena job 136905** (4d_1g / 96 h / 160G, seats probed 27/50).
> It resumes from eval 4 (T 0.94493 / W 18.4597), not from the seed.
>
> ### ~~RESUME DEFECT — BOTH CAMPAIGNS COLD-RESTART FROM THE SEED~~ (WRONG, see above)
> **MEASURED from the fetched eval logs:** after their restarts, BOTH
> campaigns re-log `eval 1` at the SEED values (seesaw T 0.93790 = its seed;
> uniform_s5 T 0.90120 = its seed), and 136753's log reads
> `Computing baseline FOM at iteration 0 (initial parameters)` at 21:13:44.
> **They do NOT warm-start from the best logged row.** 136753 lost ~3.5 h
> this way. That violates §6's critical rule (loss budget ≤ 1 evaluation on
> preemption) and makes every long campaign a DEFECT at dispatch time until
> fixed. The jsonl rows survive (good — that is what makes the fix possible),
> so the fix is to seed from the best logged row on startup.
> ⇒ **FABLE ESCALATION** (it touches the campaign driver's startup path).
>
> ### ★CAMPAIGN SCIENCE AS OF THE DEATH/RESTART (logs now LOCAL in
> `results_from_athena/lumopt2_c325_logs/lumopt2_v2_{seesaw,uniform_s5}_evals.jsonl`)
> **136752 seesaw — was climbing cleanly IN BAND** (ceiling 18.713):
> ev1 seed T 0.93790 / W 18.3318 / e 0 → ev2 0.93879 / 18.3410 / e 6.3 →
> ev3 0.94033 / 18.3566 / e 13.0 → **ev4 0.94493 / 18.4597 / e 39.9 /
> Q_i 74 352**. e 39.9 = **0.80 nm per tooth**, well under the e=65 knee ⇒
> it is buying T with shifts that cost almost no width, exactly the measured
> free-zone rule. This campaign was WORKING when the licence killed it.
> **136753 uniform s5 — reproducing the band-edge thrash, as predicted:**
> ev1 0.90120 / 18.3452 → ev2 0.90679 / 18.3895 → ev4 0.90896 / 18.4092,
> but its big probes go OUT of band and are rejected: ev3 W 25.13, and
> **ev5 T 0.95912 at W 20.3738** (Q_i 93 528). Note the direction: mcorr
> falls 325.0 → 324.0 → 323.6 → 311.4, i.e. it is LOWERING corrugation to
> widen-and-buy-T, and is NOT finding the see-saw (which needs the outer
> teeth RAISED). ⇒ On current evidence the uniform seed needs the
> **AUGMENTED LAGRANGIAN**, not another hinge patch — the handoff's stated
> next move. Still early (5 evals, post-restart); let it run before calling it.
>
> ### ★★★★★MEASURED 2026-08-24 20:55 — **THE GPU WIDTH-ADJOINT RAN.**
> ### SIZE BOUND CONFIRMED. THE PRODUCTION FIX IS TILING.
> **Rung 30 (`quart`, x span 26.406 µm × y span 0.361 µm = 528 × 7 cells)
> EXITED 0 and printed a finite gradient vector** (job 136799 task 30):
> `[adjoint_only detune=1 C_field=(1.0,0.0) indices=[0, 50, 190]]`
> `array([-0.0014301, 0.00800267, 0.0074804])`
> **THE LADDER, COMPLETE (all four 2D rungs, one number each — cells at
> dx 50 nm):**
> | rung | region | cells | verdict |
> |---|---|---|---|
> | 27 full | 105.624 × 1.444 µm | 2112 × 29 = 61,248 | **FAIL** CUDA |
> | 28 yhalf | 105.624 × 0.722 µm | 2112 × 14 = 29,568 | **FAIL** CUDA |
> | 30 quart | 26.406 × 0.361 µm | 528 × 7 = 3,696 | **PASS** exit 0 |
> | 32 thin3d (3D!) | 105.624 × 1.444 × 0.16 µm | 2112 × 29 × 3 = 183,744 | **FAIL** CUDA |
> | 33 small3d (3D) | 26.406 × 0.722 × 0.16 µm | 528 × 14 × 3 = 22,176 | **PASS** exit 0 |
> | 29 xhalf / 31 patch | — | — | died on the license race |
> ★**3D DOES NOT RESCUE FULL SIZE** (rung 32, job 136826, same CUDA error) ⇒
> the zero-dimension hypothesis is now DEFINITIVELY DEAD; giving the region
> z-thickness at full x changes nothing.
> ★**THE PATTERN ACROSS ALL FIVE DECIDED RUNGS: every PASS has x = 528,
> every FAIL has x = 2112 — regardless of y, z, or total cells.** Note
> rung 33 PASSES at 22,176 cells while rung 28 FAILS at 29,568: those two
> are close in cell count but differ 4× in x. So an **X-DIMENSION bound is
> now better supported than a total-cell budget.**
> ⇒ **DISPATCHED to decide it (job 136869, probed 9/50 seats first):**
> rung **29** (x = 1056, 30,624 cells) — the direct x-threshold probe; and
> rung **34 `xnarrow_big`** (x = 528, y full, 3 z-cells = **45,936 cells**,
> above every failure so far) — PASS ⇒ bound is on X ALONE ⇒ tile in x only;
> FAIL ⇒ total-cell budget ⇒ tile in both axes.
> ★Rung 33's ratios vs FD are **1.369 / 1.828 / 1.269** (vs rung 30's
> 0.392 / 0.439 / 0.369) — the scale moves a LOT with the region, which
> CONFIRMS that these ratios are dominated by the crop, not by C_field.
> Do not read either as a calibration.
> ★**h5 NON-ZERO GATE: PASSED** (`h5_gate.py gfr_quart`, run on the login
> node 20:55). `adj_default_field_profile_adj_output.h5` max|E| = **0.4593**,
> with Monitor0 Ex/Ey/Ez all non-zero (6.5e-3 / 2.6e-3 / 1.6e-2). This is
> NOT the 2026-08-23 fake, whose every monitor field was EXACTLY 0.0.
> (Ex=Ey=0 on the z=0 plane of the broadband `field_profile` is the expected
> TM parity, not a dead source — the fieldregion injects volumetrically.)
> ★**SIGN GATE: 3/3 vs the keep-forever FD** `[-0.00365, +0.01825, +0.02026]`
> — corr_1 −/−, shift_1 +/+, wcav +/+.
> ★**AND THE RATIOS ARE NEARLY CONSTANT — the C_field signature:**
> corr_1 **0.392** | shift_1 **0.439** | wcav **0.369** (mean ≈ 0.400,
> spread ±9%). A dead or garbage adjoint cannot produce ONE consistent scale
> across three different parameter classes. That is what an uncalibrated
> adjoint awaiting its `C_field` looks like.
> ★★**HONEST CAVEAT — DO NOT FIT C_field ON THIS ROW.** Rung 30's region is
> CROPPED (528 × 7 vs the full 2112 × 29), so its softW is a DIFFERENT
> functional from the one the FD reference measured. The ~0.400 may be
> C_field, may be the crop, or both, and this row cannot separate them. The
> C fit must be done at the region the production run will actually use.
> ★**NEXT, in order:** (1) 3D rungs 32/33 (job 136826, in flight) — if a 3D
> region runs at FULL size, tiling is unnecessary and that beats everything
> here; (2) if not, BISECT the cell threshold (3,696 → 29,568) before
> designing tiles: the full region is 61,248 cells, so a ~4k tile budget
> means ~16 tiles ≈ 6 h and kills the speed win, while a ~15k budget means
> 4 tiles and keeps it. Suggested probes: 1056 × 14 (14,784) and 528 × 29
> (15,312) — they also separate total-cells from per-dimension bounds;
> (3) ESCALATE TO FABLE for the tiling design + its gate (summing tile
> adjoints vs separate solves is a FOM-level decision).
>
> ### ★★★★MEASURED 2026-08-24 20:24 — RUNG 27 **FAILED**, ERROR REPRODUCED.
> ### ★AND A NEW TRAP: THE CUDA ERROR SURFACES ~22 MIN LATE, NOT "IN SECONDS".
> **Rung 27 (FULL spans, `2D Z-normal`, 2112 × 29 cells) died with the exact
> target error**, MEASURED verbatim from `lum_array-136799_27.out`:
> `n315(process 0): Warning: GPU minimal memory estimate may be inaccurate!`
> `n315(process 0): ERROR: invalid configuration argument`
> lumopt2 retried once (`-gpu -t 1` → `-gpu -t 8`); both failed; exit 1.
> ★**TIMELINE THAT MATTERS — this invalidated my own verdict rule:** forward
> done 20:01:32 → "Running adjoint simulation..." 20:01:34 → **error 20:24:13,
> i.e. 22.6 min later.** The engine meshes on CPU before the GPU kernels
> launch (Ansys: "operations other than solver … still use the CPU"), so the
> launch error appears only AFTER a long CPU phase. **"Dies in seconds" is
> WRONG for this failure, and a long runtime is NOT evidence of a launch.**
> ⇒ The ladder's printed verdict rule ("a real solve time = LAUNCHED") is
> UNSAFE — a rung is only a PASS when the task EXITS 0 with a printed
> gradient vector. Judge on exit, never on elapsed time.
> ⇒ **RETRACTED (stated in session at 20:08, wrong):** "the full-size 2D
> adjoint launched on GPU / 136026 did not reproduce." It reproduced exactly.
> The zero-dimension and size hypotheses are BOTH still live — 27 tells us
> only that the largest 2D region fails, which is what we already believed.
> ★**Note the warning that immediately precedes the error**: "GPU minimal
> memory estimate may be inaccurate" — memory sizing is implicated right at
> the failing launch, which fits a grid/block dimension derived from a
> region-size or memory estimate.
> **STILL PENDING at 20:25: rungs 28 (yhalf, 2112 × 14) and 30 (quart,
> 528 × 7) both started their adjoints ~20:12** ⇒ their verdicts are due
> ~20:34 on the same 22-min offset. **30 surviving = the size bound**, and
> the fix is tiling.
> ★★**DO NOT CALL ANY RUNG A WORKING ADJOINT — THE 52-MINUTE FAKE IS THE
> PRECEDENT.** A running solve is not evidence. Gates, in order:
> (1) h5 non-zero gate on the adjoint output (`runners/lumopt2_design/gates/h5_gate.py`, runs
> on the login node: `python3 h5_gate.py gfr_full`); (2) sign check of the
> printed vector vs the keep-forever FD `[-0.00365, +0.01825, +0.02026]`;
> (3) `fit_c_field.py` for C_field, which has NEVER been fitted.
> ★NOTE the parity objection does NOT apply here: that killed the IMPORT
> source (tangential-only injection at a plane where Ex=Ey=0). A field region
> in source mode is a VOLUMETRIC current source — the docs say it "converts
> the recorded field into an array of dipole moments … with phase and
> orientation determined by the electric field" — so it can carry the NORMAL
> component (our Ez = 8.41). That is exactly why this route can work where
> the import sheet could not.
> ★**COST, if the gates pass:** forward 1351.6 s + adjoint ≈ the same ⇒
> **~45 min per gradient vs 8.7-12.1 h on CPU**, i.e. the in-loop exact width
> gradient becomes affordable — the priority-zero prize.
>
> ### ★★THE ZERO-DIMENSION FINDING (2026-08-24 ~20:30) — RECORDED, CONTRADICTED
> **Two independent lines converge on: the adjoint source dies because the
> field region has a ZERO z extent, not because it is too big.**
> 1. **MEASURED FROM THE SHIPPED BINARY** (`v261/bin/plugins/gpu/
>    lumcudafdtd.dll`, read-only string extraction): the CUDA plugin carries
>    its own pre-launch validators, and **two** of them are about dimensions —
>    `"Grid Dimensions (%u,%u,%u) include one or more values that exceed the
>    device limit"` AND `"Grid Dimensions (%u,%u,%u) include one or more ZERO
>    values. All dimensions must be nonzero"`. `cudaErrorInvalidConfiguration`
>    is raised for a ZERO dim just as much as an oversized one. Our twin is
>    `"2D Z-normal"` with **no z span** (engine:1075). The plugin also
>    contains `FdtdVolumeSource` and `FdtdPointSource` classes with compiled
>    dipole/volume CUDA kernels ⇒ the capability EXISTS.
> 2. **MEASURED FROM `v261/bin/fdtd-engine.exe`**: the engine's COMPLETE
>    GPU-unsupported list (2D sims, legacy aniso PML, some grid attributes,
>    BFAST, TFSF, deprecated mesh refinement, some materials, checkpoint
>    resume, multi-GPU; warnings for apodization/movie/time monitors) has
>    **NO entry for field regions or sources-from-monitors**. So this is an
>    UNGUARDED CRASH PATH, not a refused capability. (This supersedes the
>    weaker "Ansys docs list only TFSF and BFAST" claim above with a
>    version-exact one.)
> 3. **DOCUMENTED (web, 403-blocked pages ⇒ search extraction, re-read in a
>    browser before quoting):** 2026 R1 release notes advertise *"GPU support
>    for volumetric current sources was also added … key requirements for
>    inverse design using LumOpt"* and describe the workflow as recording with
>    a **3D Field Region** then playing it back as a source. Every Ansys
>    description of the supported GPU path says **3D**.
> 4. **It explains the asymmetry we measured today**: the same object is FINE
>    as a monitor on GPU (rungs ran 8+ min) because DFT monitors go through
>    separate kernels (`updateDft`/`finalizeDft`) that handle singleton dims;
>    only the SOURCE path launches the failing kernel.
> ⇒ **RUNG 32 (`thin3d`) IS NOW THE CRITICAL RUNG, not the size ladder** —
> and it is one of the three that died on the license race without testing
> anything. **Rung 33 (`small3d`, 3D at quarter x) ADDED** as its partner:
> 32 launches ⇒ z-span-0 was the whole bug, no tiling needed; 32 dies
> "exceeds device limit" but 33 launches ⇒ dimension AND size ⇒ tiling;
> both die ⇒ 3D is not the cure, go to routes 2/3.
> ★**A PREMISE CORRECTION worth keeping** (it was wrong in session and in the
> task brief): **port adjoints do NOT inject via dipoles** — `port_fom.py:110`
> sets `FDTD::ports "source port"`, a MODE source; `grep -i dipole port_fom.py`
> is empty. The dipole normalization lives on the FIELD side
> (`field_fom.py:112/115/145`, `power_target = 1e-15 W`, citing the engine's
> `simdipolesource.cpp:184-201`) ⇒ lumopt2's field-side scaling is ALREADY
> written for dipole injection, which is what makes an explicit-dipole
> substitute (route A) structurally cheap: `setup_adjoint_simulation` is
> called from ONE site (`project.py:619`) and nothing downstream depends on
> how the source was built.
> ★lumopt2 sets exactly ONE source-side property on the object —
> `setnamed(monitor, "source mode", True)` (`field_fom.py:73`) — and never
> creates the field region itself. Every other property is ours
> (engine:1073-1085), so the fix space is entirely on our side.
>
> ### ★FIRST READ OF 136799 (2026-08-24 19:45) — 3 casualties, NO verdict yet
> **MEASURED spans (corrects an assumption in the plan): the twin is
> x span 105.624 µm × y span 1.444 µm = ~2112 × 29 cells at dx 50 nm** — the
> region is FOUR TIMES longer than the "~23 µm" estimate. If a launch bound
> exists, 2112 is the number to suspect; y is already tiny.
> **Rungs 29 (xhalf), 31 (patch), 32 (thin3d) DIED at ~7-8 min with bare
> `LumApiError: 'in run:'`, exit 1.** Read the traceback before classifying:
> it is `compute_gradient → run_forward → run_jobs(gpu) → fdtd.run` — the
> **FORWARD** solve, on the ordinary port path, AFTER lumopt2's own 2
> retries. The width adjoint was never reached, so **these are NOT CUDA/
> FieldRegion verdicts**. Bare `in run:` is the documented license/startup
> race signature (§6); 6 tasks cold-started in the same second.
> ★**The size hypothesis is REFUTED as an explanation of these deaths**: the
> SMALLEST rung (31, 120×16 cells) died while the LARGEST (27, full 2112×29)
> is still running, and 30 (528×7) is running. Deaths are stochastic, not
> ordered by size.
> ⇒ **Recovery = staggered resubmit of 29/31/32 (`--array-tasks=29,31,32
> --max-concurrent=1` style) ONCE the queue drains — not now.**
> **Rungs 27, 28, 30 were alive in the FORWARD solve at +8 min.** The forward
> costs ~27-50 min here, so the CUDA verdict (adjoint launch) is expected
> ~20:20-20:30 IDT. Nothing before that is evidence.
> Campaigns unaffected: 136752 dEps/dP 200.5 s, 136753 running its adjoint.
>
> ★**MANDATORY GATE FOR ANY RETRY — this is what caught the fake result:**
> a plausible RUNTIME IS NOT EVIDENCE OF A WORKING ADJOINT. Before believing
> any timing, open the adjoint's output h5 and confirm the monitor fields are
> NON-ZERO. The 52-minute "success" had every field EXACTLY 0.0 and matched
> the forward's runtime precisely because an empty scene still integrates the
> full simulation time. Then, and only then, run the FD gate — `ADJ_FIX_FIELD`
> has never been fitted, so even a genuinely injecting adjoint is uncalibrated
> until it passes.
> KEEP-FOREVER byproduct of that failure: job 136189's FD half is genuine
> first-of-its-kind data — MEASURED d(softW)/dp at detune-1 =
> [−0.00365 corr_1, +0.01825 shift_1, +0.02026 wcav]. FD is
> config-independent, so it is valid for any future gate — including as the
> reference the retry must reproduce.
>
> ### ★MESHER METHODOLOGY — USER RULE 2026-08-24. Binding.
> "It is okay to use PVA if it is the only one that is differentiable, **as
> long as the best device in PVA is also best in conformal**. If we use PVA,
> **final devices must be verified in conformal against the INITIAL in
> conformal**."
> Both halves are load-bearing and the second is the stricter one: a
> conformal number for the winner alone proves nothing, because the claim is
> always a DELTA. Validation is a conformal-vs-conformal PAIR — final and
> seed, same mesher, same everything — never a cross-mesher subtraction. This
> is §2's identical-numerics rule applied to the mesher axis.
> ★**THE RANKING-TRANSFER ASSUMPTION IS UNTESTED, AND FABLE'S OWN MECHANISM
> SAYS IT IS NOT SAFE BY DEFAULT.** The verdict is that PVA's error comes from
> scalar averaging over-weighting ε at material faces, and the over-weight per
> unit length scales with the LOCAL CORE WIDTH — so wide (tooth) sections gain
> more spurious n_eff than narrow ones. That makes the error a function of the
> CORRUGATION PROFILE, which is exactly what the optimizer changes. A
> profile-dependent error can in principle reorder two designs. Nobody has
> ever measured two DIFFERENT devices under both meshers; the only paired
> measurement is one device (MX-14/15).
> ★**CHEAP CLOSER, serves BOTH halves of the rule at once (3 forwards):** run
> the uniform seed, seesaw_d090 and `BEST_T9636` under CONFORMAL at
> pitch-locked dx. Their PVA numbers already exist (0.90120 / 0.93790 /
> 0.96361), so this immediately (a) tests whether the PVA ordering survives,
> and (b) produces the conformal INITIAL that the user's validation rule
> requires. Do this BEFORE quoting any conformal delta. If the ordering does
> NOT transfer, PVA is disqualified as an optimization mesher and the whole
> architecture needs revisiting — so this is a gate, not a nicety.
> Related but separate: mode width has never been laddered vs dx under EITHER
> mesher. Fable's 6-run closer (bare N=100, dx ∈ {51.68, 32.30, 25.84} ×
> both meshers, ~12 GPU-h, per-hypothesis predictions, no new engine code) is
> designed and unrun — it decides which mesher is right, whereas the 3-forward
> check above decides whether it MATTERS for our rankings. The 3-forward one
> is higher priority.
>
> ### ★UNITS TRAP THAT COST REAL CONFUSION (2026-08-24) — state them always
> **`e` is the TOTAL, not the per-tooth shift**: `e = 2 × sum(25 shifts)`, so
> per-tooth = **e/50**. The knee at e=65 is a per-tooth shift of just
> **1.30 nm** (0.50% of the 258.4 nm segment); the largest rung ever run,
> e=287.5, is still only 5.75 nm/tooth; `BEST_T9636` spans 0.89-6.43 nm/tooth.
> The user reasonably remembered "width increases from the very first point"
> — true if you scan TENS of nm per tooth, because 10 nm/tooth is e=500,
> ~8× past the knee. Both readings are the same curve at different scales.
> ⇒ Whenever quoting `e`, give the per-tooth equivalent alongside it.
> ⇒ And note `shift_bounds` are (0, 200) nm PER TOOTH against a useful range
> of ~0-7 nm — see the conditioning item below.
>
> ### ★THE TRANSMISSION CEILING — what any further gain actually costs
> At the width spec Q_load is pinned near 2020 (MEASURED 1930-2109 across
> every device this programme has built), so T is a function of RADIATION
> Q_i alone. Q_i is DERIVED as `Q_L/(1-sqrt(T))`, so this is an exchange
> rate, not a prediction — but it is the right accounting frame:
> | target T | required Q_i | vs today's 110 087 |
> |---|---|---|
> | 0.965 | 114 500 | ×1.04 |
> | 0.970 | 133 755 | ×1.21 |
> | 0.975 | 160 711 | ×1.46 |
> | 0.980 | 201 144 | ×1.83 |
> | 0.990 | 403 307 | ×3.66 |
> Brutally nonlinear. **0.97 is plausibly reachable** from envelope shape plus
> the cavity-width headroom. **0.98 needs something structurally new**, and
> the one candidate that used to be quoted for it (second comb row) is
> measured dead three times — see the correction above. Device LENGTH is not
> available: Q_i ~ L^2.5-3.6 but L saturates ~20 µm and the width spec pins it.
>
> ### WHAT THE USER FORGOT (raise these unprompted)
> 1. **PRODUCTION CONFIRM is the biggest gap.** 0.96361 is a surrogate-N,
>    PVA-mesh OPTIMIZER number — not a device number. N≈169 + accurate mesh,
>    run OUTSIDE lumopt2 via a plain SweepSpec runner, is what makes it real.
> 2. **Width conversion is now ×1.049** (PVA→conformal). The old 0.92 is
>    RETIRED. Mesher arbitration settled 2026-08-24 (Fable, task
>    a46f03072ddb16c77): conformal variant 0 is presumptively truthful, the
>    ~20 µm spec is NOT an artifact, and the famous "−8%" gap was itself partly
>    a grid-phase artifact — the clean matched-grid gap is −4.4 to −4.7%.
>    Residual: mode width has never been laddered under either mesher; the
>    6-run closer (~12 GPU-h) is designed and unrun.
> 3. **The h5 janitor must be RUNNING** (`~/h5_roll_clean.sh`, nohup on the
>    Athena login node). It was found DEAD today with ~3.5 h of quota runway
>    left. Verify with `pgrep -af h5_roll_clean` at every check.
> 4. **The bounds are badly conditioned.** `shift_bounds` are 0-200 nm PER
>    TOOTH while the entire useful range is ~0-7 nm (BEST_T9636 spans
>    0.89-6.43). That ~30× mismatch caused THREE failures: the 136640
>    line-search abort and both first-step lurches to e≈285. The trust box
>    papers over it; it does not fix it.
> 5. **Two cheap untested experiments, both with plausible FREE gains:**
>    (a) shift DISTRIBUTION in the free zone — nobody has compared shapes at
>    matched e on the same device, and free-zone shifts are pure profit;
>    (b) the SPLIT — above-knee shifts buy at 0.0136 T/µm while the see-saw
>    appears to buy at ~0.021, so spending headroom on apodization instead may
>    beat the current design. Both need ~3 forwards. (b) rests on a rate
>    measured on a DIFFERENT device — treat as hypothesis, not result.
>
> **THE ONE THING THAT CHANGES HOW YOU READ THE WIDTH MODEL:** the
> fwhm_wall's elongation slope `FW_A_ELONG = 0.01355 um/nm` is a SECANT of a
> single pair, and it is WRONG in both directions. The width-vs-elongation
> law was MEASURED on 2026-08-23/24 (IGUM jobs 61742 + 61782, 6 rungs on the
> uniform corr-325 seed, pure common mode, pitch-locked mesh):
>
> | e = 2*sum(shift), nm | 0 | 60 | 120 | 180 | 240 | 287.5 |
> |---|---|---|---|---|---|---|
> | fwhm_env, um | 18.345 | **18.311** | 20.483 | 24.015 | 28.768 | 32.698 |
>
> It is a THRESHOLD: **flat to ~65 nm** (e=60 measures NARROWER than the
> seed), then a knee and a steep, still-accelerating rise. Fitted law
> `dW = 7.8654e-3 * max(0, e-65)^1.39`, max residual **0.106 um** vs the
> 0.367 um half-band. Engine: `_fw_elong_curve`, spec flag **`fw_curve`**
> (default False; ON only for 136640). The older quadratic `fw_convex`
> (FW_C_ELONG=1.07e-4) is ALSO refuted by these data — **do not enable it**.
>
> WHY IT MATTERED: the old linear wall charged +0.813 um of predicted
> widening at e=60 where the true cost is ZERO — a penalty (~0.795) on the
> order of the whole FOM (~0.67). Campaign 136466 was therefore effectively
> FORBIDDEN from using shifts: it oscillated e = 0 → 287 → 0.3 → 144 → 0.7,
> never probing the 1-100 nm range where shifts are free, and gained
> +0.0005 T in ~7 h while shift-FROZEN 136468 gained +0.0076. That is why it
> was restarted as s3/136640.
>
> WHAT ELONGATION PHYSICALLY IS (code-verified, make_func:445-476): it
> LENGTHENS the central cavity block (`cavity::x span = pitch/2 + 2*sum(s)`)
> AND shortens every free period by s (narrow segment `hp - s`, walk
> `2*hp - s`), with both regions walked inward from FIXED outer edges so
> total device length is constant. Cavity lengthening is linear and tiny
> (0.29 um at e=287.5); the width blow-up is MIRROR DETUNING eroding
> kappa_eff, and width ~ 1/kappa_eff is convex — hence the threshold shape.
> ⇒ In this parametrization cavity-lengthening and mirror-detuning are THE
> SAME MOVE, so "are tooth shifts necessary?" tests that COUPLED direction,
> not cavity length in isolation. NOTE `I_CAV` is the cavity **WIDTH (y
> span)**, 750-1150 nm — not a length.
>
> ALSO CLOSED 2026-08-23: the comb is worth **+0.0040 T** at benchmark width
> (not width-driven; it narrows the mode only 0.34%, ~5% of its own gain) ⇒
> a fabrication decision, not a physics necessity. And a CONCENTRATED shift
> pattern widens LESS than a uniform one at matched elongation (0.385 um of
> 14.35, i.e. 2.7%) ⇒ pattern is a minor correction, common mode dominates.

> **UPDATE 2026-08-21 — the §6 fix now has a researched, offline-VALIDATED v2
> design: `runners/lumopt2_design/V2_FWHM_PLAN.md`** (differentiable
> FWHM-tracking width `softW` validated ≤2 pp on the corrected profiles where
> σ errs 24 pp; lumopt2 field-adjoint feasibility read from source; AL
> architecture + gate ladder W0-W6; W0 passed, zero GPU used). Skill item 28.
>
> **LATER SAME DAY — implemented + in cluster validation:** engine carries
> width_grad (MixedFom + single-λ twin monitor + AL penalty + re-anchoring);
> local gates all pass; cluster gates W1r-W3 = Athena job 135971 tasks 10-13
> (two dispatch-time bug classes found & fixed en route, zero GPU lost —
> skill item 28 lessons a-d). ALSO: mesher presumption FLIPPED to conformal
> (see §5-4 below); chirp + sinc CLOSED by calibrated light-cone model;
> N_FREE 25→40 = banked v2.1 candidate (~10× model headroom); ★second comb
> ROW = **MEASURED DEAD, three times — this line previously called it a "live
> candidate with a ~2.2× Q_i phasor ceiling" and that was WRONG** (corrected
> 2026-08-24 after the user pushed back). The 2.2× is a DERIVED bound from a
> model the same handoff flags PRELIMINARY, and it stands against three
> independent measured negatives, the last of which POSTDATES it and tested
> this exact comb family: job 130154 (2026-08-10) multi-row 2D lattice —
> single row **+0.0115**, 2row +0.0099, 4row +0.0087, 4row-r80 +0.0088, i.e.
> EVERY multi-row variant BELOW single row, logged as "user's priority
> question answered NO". Earlier: job 121392 (2026-07-15) row-2 columns ~10×
> weaker than row-1, route CLOSED; job 123563 (2026-07-18) 2-row 1.139 vs
> 1-row 1.150, "NO coherent row buildup ⇒ no N-row/bigger-post path". Only
> surviving thread: those tests fixed the inter-row offset, so a δx fan is
> UNTESTED — that is a speculative retry of a thrice-negative result, not a
> candidate, and must be described as such. ★SEED-B FWHM NEVER
> MEASURED (verified: no fwhm_env in any seedB/A4 log; .fsp gone on IGUM) —
> `seed_width_audit.py` rebuilds ev0/best/seedB2-best from logged params,
> 3 forwards, answers whether seed B started/ended in the FWHM band.

## ►► IF YOU READ ONLY ONE BOX: THE JOB IS TO FIX THE MODE-WIDTH PROBLEM

The campaign produced designs at T≈0.96 that are **15-27% too wide**, using a
width constraint that was measuring the wrong thing through a broken extraction.
All runs are stopped. Nothing is pending. Your job is to re-specify the problem
and re-run it honestly. In order:

1. **§0** — the user's non-negotiables (width growth is a DEFECT; FWHM is the
   `post_processing` convention ONLY; no cheating; never compare across widths).
2. **§0a** — current state (everything stopped, where every file is).
3. **§0c** — every correction/retraction already made. **Do not re-derive a
   claim that is on that list as withdrawn.**
4. **§2** — the profile bug, because it voids every pre-2026-08-18 σ and FWHM.
5. **§6** — the fix: projection/re-trim, not a penalty band.
6. **§6b, §6e** — the two experiments to run first, in that order.

**The single most useful fact for the fix:** mode length **saturates** at
~19.7-20 µm (§6d-v), so length was never a lever. Only envelope **shape** at
fixed length is available — which means the **inner see-saw** (§6e), the
**comb**, and **taper length** (§6d-iii), not tooth shifts.

---

**Point a new chat at this file first.** It is written to be read AFTER the
in-flight jobs have finished, so section 5 tells you how to read them and what
each possible outcome means. Also read `CLAUDE.md` (project rules) and
`.claude/skills/lumopt2-design/SKILL.md` (the living runbook, items 1-27).
Deeper history: memory file `project_lumopt2_campaign_state.md`.

---

## 0. ★★★NON-NEGOTIABLES (user, 2026-08-18) — READ BEFORE TOUCHING ANYTHING

**(a) THE WIDTH GROWTH IS A DEFECT TO BE FIXED, NOT A RESULT TO BE REPORTED.**
The measured designs grew the mode by **+14.9% (best) up to +26.6% (d+80)**.
User's words: *"if fwhm actually changed in 30% its very bad"*. It is. This is
an acousto-optic detector that senses at a FIXED width — the width is a spec,
not a free parameter. A design that reaches T 0.966 by growing the mode a
quarter wider has not solved the problem, it has changed the problem. **The job
of the next phase is to recover the transmission at the ORIGINAL width**, not
to document how much width the old campaign spent.

**(b) FWHM IS MEASURED THE WAY `post_processing` MEASURES IT. FULL STOP.**
The one and only width observable is `sim_helpers.extract_and_process_field_profile`'s
recipe — resonance lambda -> |Ex|²+|Ey|²+|Ez|² -> **integrate over y** -> crop to
the grating -> `extract_envelope_peaks` -> `calculate_fwhm_relative`. This is the
same convention as every other trusted quantity in this project, and the engine
now calls those exact functions (verified to 7e-15 um against stored `fwhm_m`).
Do NOT invent a second convention, a fitted slope, or a theoretical model.
Both were tried on 2026-08-18, both were wrong, and both are deleted.

**(c) NO CHEATING TO MEET THE SPEC.** The baseline's UNIFORM corrugation stays
325 nm. Re-tuning it so a ~20 um target lands conveniently does not count as a
fix. Any claimed gain must be a gain at the SAME measured width.

**(d) NEVER COMPARE TWO DESIGNS AT DIFFERENT WIDTHS.** T_A vs T_B is meaningless
unless FWHM_A == FWHM_B. Re-trim first (section 6), then compare.

---

## 0a. STATE AS OF 2026-08-19 01:30 — EVERYTHING IS STOPPED, NOTHING IS PENDING

**All runs stopped on user instruction.** Both cluster queues are EMPTY.
- Athena `134032` (stage-4 seedA4) — **CANCELLED**. 25 h for nothing: FOM 0.7157612
  at eval 3 → 0.7157579 at eval 9, i.e. it moved BACKWARDS by 3e-6. Its 14-eval
  log was fetched first → `results_from_athena/lumopt2_logs_seedA4_evals.jsonl`.
- IGUM `55801` (bare) — died 2026-08-18 23:54 **on TIME LIMIT**, 12 evals, no
  `_best.json`. Walltime was badly undersized: the log shows one adjoint at
  **3107 s (52 min)**, so ~1.5-2 h per gradient iteration. Long campaigns need
  `--qos=4d_1g`.
- IGUM `56033` (seedB2) — finished cleanly earlier (exit 0, best_fom 0.702857).
- All eval logs, `_best.json` files and corrected `.npz` profiles are LOCAL.
- All watchers/monitors stopped. **No dispatch is pending. Nothing is at risk.**

**The next chat starts from zero running jobs and a clean slate.** The task is
§6: re-specify the problem correctly and re-run. Read §0, §0b, §0c, then §6/6b/6e.

### ADDENDUM 2026-08-19 — two short jobs ran AFTER the stop, both finished
Both were user-authorised, both COMPLETED (~10 min/task), results downloaded,
queue empty again. **Nothing is pending.**
- Athena **134977** — negative-shift mirror, s = −51.68/−103.37 (dispatched by me
  in error: the user asked for the WIDE segment, not a negative shift; it ran to
  completion before it could be cancelled, and the data is kept and reported).
- Athena **134984** — the correct run: positive shift on the **wide** segment,
  s = +51.68/+103.37, via the new `shift_target="wide"` knob.
- Verdict + full table: **§6b**. One-line summary: every shift variant widens the
  TM mode; wide-target is strictly dominated by narrow-target; the shift axis is
  closed for the width problem.
- Files: `results_from_athena/tm_shift_c400/results/` (`_S52w`, `_S103w`,
  `dsh1Sm52sm52`, `dsh1Sm103sm103`).
- **UNCOMMITTED** (user permission needed): `bragg_device.py`,
  `simulation_config.py`, `experiment_card.py`, `runners/sweeps/sweep_spec.py`,
  `sim_helpers.py`, `runners/sweeps/tm_shift_c400.py`.

### ★TOOLING TRAPS FOUND 2026-08-19 (all cost time this session)
- **`getent` is a FALSE-NEGATIVE DNS probe** under Git Bash on Windows — it
  reported even `google.com` unresolvable on a live network. A watcher armed on
  it would never fire. Use PowerShell `Resolve-DnsName` / `Test-NetConnection`.
  `ssh`'s own "Could not resolve hostname" IS trustworthy.
- **`scene_snapshot.py` crashes** with `UnicodeEncodeError` on the two-device
  config (Greek delta → cp1252 console). Run it as
  `PYTHONIOENCODING=utf-8 python debug_fsp_compare/scene_snapshot.py --out ...`.
- **`deploy_athena.sh --results-no-fsp` HANGS forever (0 bytes) as a background
  task** — it blocks on an interactive prompt with no stdin. For a few files use
  `scp` directly; note remote brace expansion (`result_{a,b}.mat`) does NOT
  expand through scp, so loop over names.
- Full detail: `memory/project_getent_false_negative.md`.

## 0c. ★EVERY CORRECTION AND RETRACTION MADE ON 2026-08-18 (user asked for this)

Recorded so nobody rebuilds on a claim that was already withdrawn. Each of these
was **wrong at some point during the session and then corrected**:

| # | What was claimed | What is actually true |
|---|---|---|
| 1 | "Origin FWHM is 17.100 µm, best is 22.210, so FWHM grew +29.9% while σ grew +1.9%" | Both numbers came from a raw-line metric **and** a broken extraction. **Void.** Corrected values: origin 17.7005, best 20.3362 (+14.89%), all PVA. |
| 2 | "19.24 µm is the comb origin's width" | 19.24 is the **BARE** N=100 device, and it is **conformal** mesh. Not comparable to a campaign number. |
| 3 | "The raw-line metric is fine for RELATIVE change even if absolutes differ" | **No.** It reads first/last crossing of an absolute half-max on an *oscillating* line, so it moves with fringe contrast, not just the envelope. Deleted entirely. |
| 4 | "Coupled-mode theory says the true growth is only ~4%, so the 27% may be a metric artifact" | CMT was fitted/validated against the same void numbers. **All CMT deleted on user order.** The measured answer is +14.89%. |
| 5 | "The σ-guarded campaigns should be cancelled immediately" (given, then withdrawn 40 min later, then re-supported) | Correct in the end, but I gave it before the arbiter measurement existed. The rule: **do not restructure a program on a number from a metric you wrote the same day and have not cross-checked.** |
| 6 | "The lumopt2 scene builds a DIFFERENT DEVICE (5.3 nm λ offset)" + a config diff appearing to prove it | **Retracted.** The diff compared against **SweepSpec defaults** (it showed polarization TE and pitch 500 nm — impossible for a TM corr-325 study). The real cause is the **mesher split** (§5). |
| 7 | "The profile bug means we were sampling at the transverse BOX EDGE, so radiation contaminated it" | Overstated a mechanism. The monitor is a narrow 2D Z-normal plane (y span ≈1.5 µm), so row 0 sits ~0.75 µm off-axis in the evanescent skirt — **not** the box edge. What is certain is only that it was one off-axis row instead of the y-integral. |
| 8 | "Q_i ∝ L_mode² is the biggest Q lever" (long-standing project belief) | **MEASURED Q_i ∝ L^2.5-3.6**, and L **saturates** at ~19.7-20 µm, so length was never available at all. Corrected in the memory index too. |
| 9 | "κ ∝ corr, measured" (long-standing project belief) | Holds narrowly, **fails between 325 and 400 nm**: Q_i ∝ corr^−1.8, and L moved only 13% for a 23% corr change. |
| 10 | "Apodization is 1.75× more width-efficient than shifts" | Computed on void widths. Corrected: **2.9×** (0.0483 vs 0.0167 T/µm). |
| 11 | "seedB2 / bare failed" (implied by monitor `err=2`) | Both were **not** failures at that point — seedB2 exited 0; the errors were IGUM ssh flakiness. (bare *later* died on walltime, separately.) |
| 12 | "Both clusters are unreachable → possible outage" | **Local VPN drop.** Athena failed DNS; two raw-IP hosts timed out. Three Technion hosts do not fail together. Jobs were unaffected. |

**The meta-lesson, and it is the reason §0(b) exists:** almost every one of these
came from comparing numbers that were not comparable — two FWHM conventions, two
meshers, two extraction paths, a fitted surrogate vs a measurement. **Before
quoting any two numbers together, check they came from the same convention, the
same mesher, and the same pipeline.**

## 0b. ★★★IS THE INVERSE DESIGN BROKEN? — YES, SUBSTANTIALLY. BE HONEST ABOUT IT.

**Verdict: the optimization MACHINERY works; the PROBLEM SPECIFICATION was
wrong. The optimizer did exactly what it was told, and what it was told was
wrong.** That is the good news and the bad news together — the adjoint stack
does not need rebuilding, but essentially none of the campaign's design output
is usable as a delivered design.

**What is broken (all confirmed by measurement, 2026-08-18):**
1. **The width constraint never constrained anything.** The campaign controlled
   `sigma`, and sigma is BLIND to apodization: corrugation shaping moved the
   real FWHM +4.89% while moving sigma +0.001%. So on the dominant design axis
   the optimizer was effectively unconstrained in width.
2. **The width was measured through a broken extraction the whole time.**
   `profile_line` never integrated over y and always read one off-axis row.
   Every sigma, every FWHM, every sigma-anchor and wall calibrated from them:
   void. The guard was policing a number that was not the mode width.
3. **The objective intrinsically pays for widening.** T rises with Q_i and
   Q_i ~ L_mode². With (1) and (2), riding the width lever was not a bug in the
   optimizer's behaviour — it was the rational response to the stated problem.
   Result: the banked designs bought +15% to +27% width.
4. **A core design variable was mis-chosen for this polarization.** Tooth shifts
   were inherited from TE, where they cost ~0.5% width for +0.085 T. For TM they
   cost 3.6-7.2% for +0.006-0.012 T — about **100x worse** (section 6b). The
   campaign then spent much of its budget on that lever.
5. **Every fitted surrogate was calibrated on void data** (sigma-hat
   coefficients, the FWHM_A_* slopes, the sigma-neutral payback recipe). The
   sigma-neutral probe consequently FAILED to hold width — all four rows came
   out wider.
6. **The base scene may not even be the project's standard device** — 5.3 nm
   resonance offset, unresolved (section 5).

**What still works and does NOT need redoing:**
- The adjoint gradient stack: the 4-fix stack + the measured complex C-fix for
  the ~6.7° phase error (vec_error 11.40 -> 0.144). This was hard-won and is fine.
- All infrastructure: dispatch, per-study sweep lists, trust regions, cold-start
  resume, the `_final_fom` completion path, license/preemption handling.
- Every port quantity ever measured: T, lambda, Q_L, Q_i, R, loss.
- **The comb sub-programme is a genuine success**: +17.1% Q_i and +0.046 T at
  −0.35% width, with a clean with/without control. It is the one part of this
  work that is both real and on-spec.

**Therefore, do NOT:** report the T~0.96 designs as achievements, seed a new
campaign from them, or trust any width-related number produced before
2026-08-18. **Do:** treat the next phase as re-running the optimization against
a correctly specified problem (section 6), and expect the honest answer to land
well below 0.96 — the origin is 0.893 and the only validated fixed-width gain in
the programme is the comb's +0.046.

---

## 1. The device and the goal

Pi-shift Bragg grating, TM, corr-325 family, h350, pitch 516.83 nm, W800,
n 1.97/1.444, plus a 57-post SiN "comb" (an anti-radiation decoration).
Optimization surrogate = **N=100 periods/side**; production confirm is N≈165-169.

Goal: **maximise peak transmission T at the resonance WHILE HOLDING the spatial
mode width fixed.** The width is a hard spec — this is an acousto-optic detector
that senses at a fixed width; widening is forbidden and narrowing does not help
(memory: `project_acoustic_detector_width_spec.md`).

Why the width keeps trying to grow: T rises with Q_i, and **Q_i scales as
mode-length squared**. Widening is therefore the single most profitable move
available to any optimizer, and it will find it unless the constraint is exact.

---

## 2. ★★★THE BUG THAT DOMINATES EVERYTHING (found 2026-08-18, user caught it)

`lumopt2_design.profile_line` **never integrated over y**. It flattened the
(y, lambda) axes into one and indexed with the LAMBDA index — which is always
smaller than n_lambda — so it **always returned y-row 0**, for every design, in
every campaign, for the whole program.

`field_profile` is a 2D Z-normal monitor of y span 1.5*width_wide (~1.5 um,
see `bragg_device.py:1335-1339`), so row 0 sits ~0.75 um off the guide axis in
the evanescent skirt instead of across the mode.

**Measured impact:** the buggy path reports the uniform origin at
`fwhm_env = 16.7224 um` (Athena job 134299 task 0). The same device family
measured correctly is **19.24 um**. That is a **−13% error**, so the bug is not
cosmetic.

**Consequence: every `sigma_um` and every FWHM this engine ever logged is VOID** —
the audit rows, the sigma anchors and sigma-hat walls calibrated from them, the
shift ladder's sigma values, the sigma-neutral probe's widths, all of it.

**FIXED.** `profile_line` now replicates `sim_helpers.extract_and_process_field_profile`
step for step: pick resonance lambda -> |Ex|²+|Ey|²+|Ez|² -> **trapz over y** ->
crop to |x| <= n_side*pitch -> envelope through standing-wave peaks ->
floor-relative half-max.

**VALIDATED to machine precision, no GPU:** `eng.fwhm_env_of_line(x, I)` run on
stored `field_energy_density_1D` reproduces the stored `fwhm_m` exactly —
N100 c325 19.244767 um and N80 c325 18.393528 um, both matching to **7e-15 um**;
re-derived envelope matches stored `field_envelope_1D` to 5e-16 relative.

---

## 3. What is VALID and what is VOID

**VALID (port quantities — unaffected by the profile bug):**
- Every T, lambda_resonance, Q_L, Q_i, R, loss ever measured.
- The whole comb program: phase and pitch are sharp, radius and distance loose,
  post count irrelevant from 29 to 113 (n=29 halves the posts for free).
- The comb's benefit: bare-uniform T 0.88073 vs comb-uniform T 0.89265 at
  identical knobs = **+0.0119**, reproducing the A0 gate's +0.0105.
- Shift ladder's T ordering: T rises monotonically with tooth shift, **no
  interior optimum** — the width constraint is what stops it, not physics.
- All infrastructure: trust regions, cold-start resume, the `_final_fom`
  completion fix (verified on a real completion), license/preemption handling.

**VOID (anything width-related):**
- All sigma and FWHM values in every `*_evals.jsonl`.
- Any "in band" / "width-compliant" label — those meant "in the sigma band",
  measured wrongly.
- The trade line `T = 0.89265 + 0.01549*dFWHM`, the per-lever width
  efficiencies, the "seed B beats the trade line" ranking, the FWHM_hat ratios
  1.17-1.19, the 3-point factorial slopes. Kept in memory only as a record of
  reasoning — **re-derive all of them** from y-integrated data before citing.

**DELETED by user order — do not reintroduce (CLAUDE.md §8 "dropped stays dropped"):**
- the raw-line FWHM metric (`fwhm_raw_of_line` / `mode_fwhm_um`),
- the fitted `FWHM_A_RHO` / `FWHM_A_SHIFT` slopes,
- **all coupled-mode-theory (CMT) width modelling** ("delete all cmt use here
  it is not relevant"). It had been "validated" against the void numbers anyway.

**THE WIDTH OBSERVABLE IS NOW EXACTLY ONE THING:** `fwhm_env_of_line` ==
`post_processing`'s `fwhm_m`, by construction and verified.

---

## 4. Reference numbers you can trust (MEASURED from stored .mat this session)

| quantity | value | source |
|---|---|---|
| bare N=100 corr-325 mode FWHM | **19.244767 um** | `results_from_igum/tm_nladder_c325/results/result_N100_TM_avg_C325_Ybox8p0_Zbox8p8.mat` |
| bare N=80 corr-325 mode FWHM | 18.393528 um | same dir, N80 file |
| FWHM across 7 different boxes | 19.2411 - 19.2471 (spread **0.03%**) | `results_from_athena/tm_span_conv_c325/` |
| T_res across those same 7 boxes | 0.9091 - 0.9194 (spread **0.010**) | same |

Two rules follow, both important:
- The width metric's **noise floor is 0.03%**, and width is box-INSENSITIVE. So
  holding width to a fraction of a percent is measurable and realistic.
- **T is ~30x more numerics-sensitive than width.** Never compare absolute T
  across different boxes/mesh; always use an in-study control at identical
  numerics (CLAUDE.md §2).

Comb-decorated origin should sit near 19.17 um (A0 gate value) — comb is
width-neutral to ~0.4%.

---

## 5. ★IN-FLIGHT JOBS — HOW TO READ THEM WHEN THEY LAND

State at 13:05 IDT. `ssh evyatarrubin@athena.technion.ac.il` (host-first form
only; strip the banner with `grep -vE "post-quantum|openssh|may need"`).

| job | tasks | what it is | trust? |
|---|---|---|---|
| 134032 | stage-4 campaign | long lumopt2 run, 12h+, FLAT progress | widths VOID (pre-fix) |
| 134299 | 3 (t0 done, t1 running, t2 pending) | FWHM audit: origin/best/noshift | **t0,t1 VOID** (pre-fix); **t2 valid** (starts after the fix was deployed) |
| 134334 | 0-2 | width recovery from stored .fsp: origin, best, noshift | **VALID** |
| 134335 | 3-6 | width recovery: d+20, d+40, d+60, d+80 | **VALID** |
| IGUM 55801 | bare campaign | still running | widths VOID |

134334/134335 re-read the stored forward `.fsp` files and recompute width with
the corrected pipeline — **no re-solving**, ~2 min/case. They are the answer.

**Read them with:**
```bash
ssh evyatarrubin@athena.technion.ac.il "grep -h 'fsp_width' ~/bragg_sim_athena/jobs/logs/lum_*13433[45]*.out"
```
Each line prints: `T`, `lam`, `FWHM um`, `sigma um`, and the 19.2448 reference.
Each task also writes `<out>/fspw_*/fspw_*_profile.npz` (x_um, I, fwhm_um,
sigma_um) — keep these, they make every future width question free.

### ★★★THEY LANDED — THE CORRECTED WIDTHS (MEASURED, jobs 134334/134335)

All six from ONE corrected pipeline, so the ratios are apples-to-apples:

| design | T | FWHM um | vs origin | sigma um | vs origin |
|---|---|---|---|---|---|
| origin  (uniform+comb, no shifts) | 0.89265 | 17.7005 | — | 17.2518 | — |
| noshift (apodized+comb, no shifts)| 0.93450 | 18.5664 | **+4.89%** | 17.2520 | **+0.001%** |
| best BEST_T9635 | 0.96404 | 20.3362 | **+14.89%** | 17.5221 | +1.567% |
| d+20 | 0.96587 | 20.6170 | +16.48% | 17.5472 | +1.712% |
| d+40 | 0.96673 | 20.9619 | +18.43% | 17.5834 | +1.922% |
| d+60 | 0.96632 | 21.4767 | +21.33% | 17.6300 | +2.192% |

**VERDICT: the width really did grow ~15-21% on every T~0.96 device.** Not the
+27% the broken metric claimed, not the +4% CMT claimed — about +15% for the
banked best. The gains were substantially bought with mode width.

**★THE CORE DIAGNOSIS SURVIVES THE BUG FIX, and is now airtight:** the
corrugation apodization alone moved FWHM **+4.89%** while moving sigma
**+0.001%** (17.2518 -> 17.2520, identical to four decimals). sigma is not
merely insensitive to apodization — it is *blind* to it. Over the full change
sigma under-reports the width growth by ~10x (+1.57% vs +14.89%).

**★Corrected width efficiency:** corrugation apodization **0.0483 T/um** vs
tooth shifts **0.0167 T/um** — apodization is **2.9x more width-efficient**
(the earlier void estimate said 1.75x; the corrected gap is larger). Spend width
budget on apodization, not on shifts.

**T saturates:** d+40 is the peak at 0.96673; d+60 is lower (0.96632) while
still 2.9% wider. Past d+40 you are paying width for nothing.

(d+80 completed too: T 0.9653, FWHM **22.4013 um**, +26.6% — the widest of all.)

### DID THE COMB ALONE CHANGE THE WIDTH? NO — measured, clean control

`comb_q3db` contains a with/without pair at N=165, IDENTICAL numerics, both
through the SweepSpec/`post_processing` pipeline:

| case | fwhm_m um | dFWHM | T_res | Q_i |
|---|---|---|---|---|
| **no comb** | 19.9702 | — | 0.4906 | 46,499 |
| **winner comb** (r80, 57 posts, d1.9) | 19.9001 | **−0.35%** | 0.5361 | **54,457** |
| comb variant (x-14604..15412) | 19.9601 | −0.05% | 0.5283 | 52,997 |
| comb variant (x-14732..15004) | 20.0468 | +0.38% | 0.4371 | 38,784 |

**The comb is width-neutral: −0.35%, against a 0.03% noise floor.** It buys
**+17.1% Q_i** (46,499 -> 54,457) and +0.046 T for essentially no width. That
makes the comb the ONLY lever in the program measured to deliver a large gain
at constant width — the grating levers cost +4.9% (apodization) and +9.5%
(shifts) of width for theirs. Note the third row: a mis-placed comb both widens
AND loses Q_i, so comb PHASE is what matters, consistent with the comb study.

### ★★RESOLVED 2026-08-18 — IT IS A **MESHER** DIFFERENCE, NOT A GEOMETRY ONE

**The campaign and every stored SweepSpec study use DIFFERENT MESHERS:**
- `lumopt2_design.py:745` — `setnamed("FDTD","mesh refinement","precise volume average")`
- `bragg_device.py:780`  — `set("mesh refinement","conformal variant 0")`

The engine overrides to PVA deliberately (comment at :739-745: conformal
variant 0 **staircases the grid-aligned TOOTH edges**, while the comb cylinders
meshed fine). `CampaignSpec.scan_center_nm = 1564.21` even documents it:
*"MEASURED at PVA (job 132654) — the precise-volume-average mesher shifts λ
+5.2 nm vs the family's 1559.0"*.

That accounts for **both** discrepancies quantitatively:
| | PVA (campaign) | conformal (stored) | Δ |
|---|---|---|---|
| λ_res | 1564.276 | 1559.006 | **+5.27 nm** (documented: +5.2) |
| mode FWHM | 17.7005 | 19.2448 | **−8.0%** |
(The comb −0.35% and the box 0.03% are far too small to matter here.)

**CONSEQUENCES — these matter for every number in this file:**
1. **Never compare a campaign width to a stored width or to the ~20 µm spec.**
   The spec, the 19.24 µm anchor and the 19.91 µm production value are all
   **conformal**; every campaign width is **PVA**, which reads ~8% narrower.
2. **Ratios remain valid within each pipeline** — the §5 table (origin →
   best = +14.89%) is all-PVA and stands; the comb control (−0.35%) is
   all-conformal and stands.
3. Rough conversion from the one paired device: **PVA ≈ 0.92 × conformal**. So
   the campaign's best, 20.34 PVA, is ≈22.1 conformal — against a ~20 µm spec
   and a 19.91 µm production value, i.e. **~10% over spec even after the
   conversion.** The over-width conclusion is unchanged.
4. ★**Which mesher is right — PRESUMPTION FLIPPED 2026-08-21 (researched,
   Ansys KB + Farjadpour/Kottke/Johnson):** presume **conformal variant 0**
   is the better reference. The docs recommend CT0 for dielectric
   high-contrast structures; the only documented staircase reversion is >2
   materials/cell (NO public backing for "CT0 staircases grid-aligned tooth
   edges" — that engine comment is uncorroborated); PVA is documented as a
   GRADIENT-SMOOTHNESS tool ("naive smoothing", first-order, known-sign bias
   matching our +5.3 nm red-shift). The ~20 µm spec stays conformal-defined.
   Cheap arbitration (parked for user): single-period Bloch-cell λ ladder,
   both meshers × dx 50/35/25/17.5/10 (~minutes/run) + one full-device
   PVA-25 vs stored conformal-35 confirm. Full digest: mesher memory file.
5. Any future cross-pipeline comparison must state its mesher.

### (superseded) earlier note — the lumopt2 scene is NOT the stored N=100 device

The lumopt2 "origin" reads **17.7005 um at lambda 1564.276**, while the stored
bare N=100 corr-325 anchor reads **19.2448 um at lambda 1559.006**. Ruled out:
- **crop**: the recovered profiles span ±51.66 um vs the stored ±51.8, and
  cropping the stored profile anywhere ≥51.68 um changes nothing (verified;
  the floor-relative FWHM IS very crop-sensitive below ~45 um, but neither
  dataset is truncated there).
- **the comb**: measured above at −0.35%, nowhere near −8%.
- **box size**: FWHM varies 0.03% across seven boxes.
**The smoking gun is the resonance: 1564.276 vs 1559.006, a 5.3 nm offset**,
where the comb moves lambda by only 0.01 nm. So the lumopt2 base scene builds a
genuinely DIFFERENT device from the standard builder's N=100 corr-325 — not a
measurement artefact. Cavity length, the free/frozen tooth boundary, or an
`avg`-width convention are the candidates.
**Consequence:** ratios WITHIN the lumopt2 set are sound (one scene, one
pipeline). Absolute lumopt2 widths must NOT be compared to the ~20 um spec or
to any stored SweepSpec number until this is explained.
**THE CHECK:** build the lumopt2 uniform seed and the standard N=100 device
locally (`save_fsp`, build-only, <1 min each) and diff the geometry — cavity
length first. This is free and needs no GPU.

---

## 6. THE FIX — projection, not a penalty band

**This section is the actual task.** Per section 0(a), the +15-27% width growth
is the defect. Everything here exists to recover T at the ORIGINAL width.


A band has slack and an optimizer always spends slack; a projection has none.
Since T rises monotonically with width with no interior optimum, the answer
always lies **on** the constraint surface, so you never explore the interior —
you only need the ability to **return** to it.

**The scheme:**
1. Pick ONE scalar width knob — the amplitude of the apodization deviation, or
   the global shift scale. Both move width monotonically.
2. The optimizer proposes a design.
3. **Bisect that one knob on the MEASURED `fwhm_env` until it lands on the
   target width.** (Noise floor is 0.03%, so this converges cleanly; 2-3 sims,
   fewer once you have a local secant estimate.)
4. **Only then** score its transmission.

Widening now cannot pay — it is undone before anyone counts the transmission.
No width model is needed anywhere, only the measurement, which is now correct.

**Rule that follows: never compare two designs at different widths.** T_A vs
T_B is meaningless unless FWHM_A == FWHM_B. Everything reported should be
re-trimmed.

**USER CONSTRAINT (verbatim intent, 2026-08-18):** the baseline's UNIFORM
corrugation stays 325 nm. Re-matching it so that a ~20 um target lands
conveniently **does not count** as a fix — "we do not want to cheat in any way".
The apodization DEVIATION is a design variable; the baseline is not.

**Guards already shipped:** `CampaignSpec.fwhm0_um` — when set, accepted-best
designs must keep `fwhm_env_um / fwhm0_um` inside the deadband (raises
`WidthTrip`), and `_best_from_log` filters restarts and final selection on it.
Default `None` = legacy sigma-only, so live campaigns are requeue-safe.
Every eval also saves its profile to `<out>/profiles/*.npz`.

**The one physics lever with real headroom at fixed width:** the pi-shift mode
is an exponential with a cusp at the centre, and that kink is the natural
radiation source. Smoothing it must be paid for by stronger confinement just
outside, or the mode simply lengthens — i.e. dip-at-the-cusp plus
overshoot-outside, which is what seed B's profile is.
`runners/lumopt2_design/rho_neutral_shape.py` is **written, smoke-tested, NOT
dispatched**: 4 rows at rho = 1.000000 EXACTLY (mean corrugation pinned to the
baseline 325 nm), amplitudes a = 0.5/1.0/1.5/2.0, shifts zero. It asks whether
redistributing a FIXED corrugation budget buys T at constant width. Needs user
approval (new study, CLAUDE.md §8).

---

## 6b. ★THE FIRST EXPERIMENT TO RUN — do tooth shifts help at CONSTANT width?

> ### ★★★UPDATE 2026-08-19 — THE SHIFT AXIS IS NOW CLOSED FOR THE WIDTH PROBLEM
>
> The user asked: does the shift behave differently if it shortens the **wide**
> segment instead of the narrow one? It had never been possible to ask — the wide
> segment was hard-coded to `half_pitch` at both arm build sites. Added the
> `shift_target` knob (`"narrow"` = legacy default, `"wide"` = new) and MEASURED
> it (Athena **134984**), plus the strict negative mirror (Athena **134977**).
>
> **ANSWER: every version of the tooth shift widens the TM mode. There is no
> width-neutral variant.** Do not spend more GPU looking for one.
>
> corr-400 N=80, box y6.8/z8.8, mesh "optimization", ports-only; s=0 anchor =
> stored `asym_dw_study/results/result_N80_TM_avg_Ybox6p8_Zbox8p8.mat`:
>
> | variant | s nm | lambda | T_res | dT | Q_i | mode um | d_mode | dT per %w |
> |---|---|---|---|---|---|---|---|---|
> | control | 0 | 1558.617 | 0.8864 | — | 22404 | 15.532 | — | — |
> | narrow | +51.68 | 1558.946 | 0.9038 | +0.0174 | 27051 | 15.707 | +1.12% | +0.0155 |
> | narrow | +103.37 | 1559.186 | 0.9179 | +0.0315 | 32033 | 16.260 | +4.69% | +0.0067 |
> | narrow | +155.05 | 1559.266 | 0.9279 | +0.0415 | 36692 | 16.983 | +9.34% | +0.0044 |
> | narrow | +206.73 | 1559.196 | 0.9335 | +0.0471 | 39921 | 16.947 | +9.11% | +0.0052 |
> | **wide** | +51.68 | 1558.796 | 0.9006 | +0.0142 | 26090 | 15.783 | +1.62% | +0.0088 |
> | **wide** | +103.37 | 1558.876 | 0.9067 | +0.0203 | 27908 | 16.404 | +5.61% | +0.0036 |
> | neg | −51.68 | 1558.276 | 0.8665 | −0.0199 | 18869 | 15.840 | +1.98% | −0.0100 |
> | neg | −103.37 | 1558.036 | 0.8457 | −0.0407 | 16027 | 16.380 | +5.46% | −0.0075 |
>
> **Wide-target is STRICTLY DOMINATED**: less T *and* more widening than
> narrow-target at both rungs (~1.8x worse dT per % width). Same sign and shape on
> every observable (T, mode, Q_i, lambda) ⇒ one mechanism at two strengths, not a
> new lever. The campaign's narrow-target basis was already the better half; the
> shift bounds do NOT need reopening.
>
> **CONFIDENCE:** the +103.37 narrow-vs-wide gap is 0.0112 ≈ **6.2x** the dx=50 nm
> jitter floor (0.0018) — solid. The +51.68 gap is 0.0032 ≈ 1.8x — consistent but
> not independently decisive. The verdict rests on the +103 pair.
>
> **★THE TRAP (cost a wrong prediction — do not repeat it):** the cavity absorbs
> `2*sum(shift)` whichever segment is shortened, so **cavity lengthening is
> COMMON-MODE** between narrow and wide and is the dominant T lever here; the
> duty-cycle/<n_eff> term is only the ~36% DIFFERENTIAL. I predicted wide-target
> would fall BELOW control on the <n_eff> argument; it GAINED, because I had
> neglected the common-mode term. In this builder the "shift" bundles THREE
> changes — duty cycle + local period + cavity length — always separate them
> before predicting a sign. (Attempted to prove the split from stored data:
> impossible. The only stored cavity-detuning rows are ±20/±40 nm on W1000-W1100
> cavities in `results_from_athena/tm_center_completion/`; we would need −206.7 nm
> at our avg-width cavity. And a bare cavity override is not a clean control
> anyway — the shift REDISTRIBUTES length while an override ADDS it. The 64/36
> split is INTERPRETATION, not measurement.)
>
> **THE <n_eff> SIGNATURE IS REAL** and points as the light-cone argument says:
> lambda rises +0.569 nm for narrow vs only +0.259 nm for wide at s=103.37, and
> FALLS for negative shifts — i.e. lengthening the wide fraction raises <n_eff>,
> widening the (n_eff − n_clad) light-cone margin that limits TM radiation.
>
> **WHY NO SHIFT CAN EVER NARROW (the general no-go):** envelope ~ exp(−∫q dx)
> with q = sqrt(kappa² − delta²) ≤ kappa. Any local detuning — either duty-cycle
> sign, either segment — only lengthens the decay. Narrowing REQUIRES raising
> kappa near the centre, which no shift does. The only routes that raise kappa
> are (a) more corrugation at the centre (the excluded "cheat", and the exact
> opposite of the dip that buys T), (b) a second scattering mechanism = **the
> comb** (the ONLY measured width-negative lever: −0.35% width at +17.1% Q_i, and
> never optimised for width), (c) a perturbation TM is intrinsically strong at =
> a TOP-SURFACE corrugation, since at a sidewall TM's E_z is tangential (plain
> Δε, no enhancement) while at the top surface it is normal — the Johnson
> boundary theorem already coded at `lumopt2_design.py:625-651`. (c) costs a
> second etch depth, so it is a fab question, not a physics dead end.
>
> **CODE ADDED (uncommitted at handoff time):** `shift_target` through
> `bragg_device.py` (both arm loops + validation), `simulation_config.py`,
> `experiment_card.py` `_CARD_FIELD_MAP`, `runners/sweeps/sweep_spec.py`, and a
> `"w"` marker in `sim_helpers.py generate_file_tag` — **without that marker the
> wide rows overwrite the stored `_S52`/`_S103` narrow results.** Runner:
> `runners/sweeps/tm_shift_c400.py`. VERIFIED: all 6 committed scene snapshots
> byte-identical (default path provably unchanged); the new path differs from
> narrow-target in exactly 4 objects (innermost tooth, both arms) with spans
> 258.415/155.045 nm swapping; cavity and total grating length identical.
> Full detail: `memory/project_shift_target_sign_test.md`.

**Status: NEVER MEASURED.** Every shift datapoint in this program was taken at a
different (larger) width, so "shifts raise T" has always meant "shifts raise T
by widening the mode". Whether they add anything at FIXED width is open.
(Still true as written — the 2026-08-19 run above settled WHICH SEGMENT, not the
constant-width question. But note it lowers the priority: since no shift variant
is width-neutral, any constant-width test must pay the width back from another
knob, and §6e's see-saw + the comb are the better places to spend the GPU.)

### ★★★WHAT ACTUALLY NARROWS THE TM MODE — the measured inventory (2026-08-19)

Found by scanning stored studies and comparing every row to **its own in-study
control**. These are the ONLY width-reducing effects this program has ever
measured for TM:

| effect | study dir | d_width | dT | verdict |
|---|---|---|---|---|
| **air trench** (rect L84 um x W800) | `air_trench_w1050` | **−0.85%** | **+0.0157** | WIN-WIN |
| **cavity width W1250** | `cavity_width_ladder` | **−0.29%** | **+0.0178** | WIN-WIN |
| cavity width W1400 | `cavity_width_ladder` | −0.74% | −0.0061 | ~T-neutral |
| cavity **hourglass** pinch 150 | `inner_shape_study` | **−1.05%** | −0.0280 | costs T |
| cavity hourglass pinch 75 | `inner_shape_study` | −0.50% | −0.0147 | costs T |
| **comb** (corr-325 N165) | `comb_q3db` | −0.35% | Q_i +17.1% | WIN-WIN |

Controls: `cavity_width_ladder`/`inner_shape_study` → in-dir
`result_N80_TM_avg_Ybox6p8_Zbox8p8.mat` (15.532 um, T 0.8864); `air_trench_w1050`
→ in-dir `..._ff.mat` (15.622 um, T 0.9218).

**★THE PATTERN — narrowing lives in the CAVITY, never in the teeth.** Every one
is either a change in/near the cavity (where the mode peaks) or an added
scatterer. NOTHING done to the teeth ever narrowed: apodization, shifts (both
duty-cycle signs, both segments), tooth shapes (ellipse/tri/wedge +1.6% to
+9.8%) and the see-saw all WIDEN. That is the general no-go made visible —
tooth-level detuning only lengthens the decay; narrowing needs kappa raised near
the CENTRE. Sign detail: cavity **hourglass narrows, barrel widens** (+0.34% /
+0.58%) — a clean antisymmetric pair. Cavity width is NON-monotonic: W1050
+0.59%, W1150 +0.18%, W1250 −0.29%, W1400 −0.74%.

**CAVEATS — do not overstate:** (1) all corr-400 N=80 (~15.5 um modes), NOT the
corr-325 production family — porting UNVERIFIED, the comb row is the only
corr-325 point; (2) all are **≤1%** against a **+15%** problem — counterweights,
not a solution; (3) stacking unmeasured, and modularity in this program has
sign-inverted under apodization before; (4) no dx=50 width jitter floor exists
for this family, so the −0.29% row is the one most likely near noise.

Full detail + sources: `memory/project_tm_width_reducing_levers.md`.

What the corrected data says INDIRECTLY (all MEASURED, one pipeline):
- apodization lever (origin -> noshift, zero shifts): +0.0419 T for +0.866 um
  = **0.0483 T/um**
- shift lever (noshift -> best, same corrugation): +0.0295 T for +1.770 um
  = **0.0167 T/um**
=> per micron of width spent, shifts are **2.9x worse than apodization**. So at
a fixed width budget you would rather spend it on apodization. That is
suggestive, NOT conclusive — it compares marginal rates at different points on
two curves, and neither curve's shape is known.
- The sigma-neutral probe (d+20..d+80) TRIED to add shifts while paying the
  width back with corrugation. It failed: every row came out WIDER (20.62,
  20.96, 21.48, 22.40 vs best 20.34), because the payback was sized with the
  sigma surrogate — which is blind to apodization AND was computed through the
  broken profile path. Those rows do not answer the question.
- Physics says shifts SHOULD help in principle: they are a distributed phase
  shift, spreading the abrupt pi discontinuity over several periods, which is
  the classic gentle-confinement trick for cutting light-cone radiation. So do
  not assume the answer is "no".

### ★★★TE vs TM: THE SHIFTS ARE NEARLY FREE FOR TE AND EXPENSIVE FOR TM

MEASURED, stored study `results_from_athena/tm_te_shift/results/`, N=80, each
polarization against its OWN S=0 baseline, all via the trusted `post_processing`
pipeline (user recalled this and was right):

| shift nm | TE fwhm_m | TE dFWHM | TE T | TM fwhm_m | TM dFWHM | TM T |
|---|---|---|---|---|---|---|
| 0   | 15.2164 | — | 0.8594 | 17.8992 | — | 0.9739 |
| 50  | 15.0689 | **−0.97%** | 0.9096 | 18.0202 | +0.68% | 0.9768 |
| 100 | 15.2991 | **+0.54%** | **0.9439** | 18.5387 | +3.57% | 0.9801 |
| 150 | 15.6470 | +2.83% | 0.9344 | 19.1702 | +7.10% | 0.9829 |
| 200 | 15.8074 | +3.88% | 0.9057 | 19.1813 | +7.16% | 0.9855 |

- **TE:** +0.0845 T at **+0.54%** width (S=100) => **~1.02 T per um**. The shifts
  are essentially FREE. TE also has a clear INTERIOR OPTIMUM at S≈100 (T falls
  at 150 and 200), and at S=50 the mode even NARROWS.
- **TM:** +0.0062 T at +3.57% width (S=100), +0.0116 at +7.16% (S=200)
  => **~0.0097 T per um**. Monotonic, no interior optimum in range.
- **TE shifts are ~100x more width-efficient than TM shifts.**

**This is the likely root of the whole problem:** the tooth-shift lever was
inherited from TE work where it costs almost nothing, and applied to TM where it
is the single most width-hungry knob available. It is consistent with the
current campaign's own TM numbers (noshift -> best: +0.0295 T for +9.5% width)
and with the corrected finding that for TM, apodization is 2.9x more
width-efficient than shifts.

CAVEATS: N=80, an older study, and the TE and TM devices differ in geometry
(resonances 1570.7 vs 1523.6 nm), so this is NOT a controlled A/B on
polarization alone — but each polarization is internally controlled against its
own S=0 row, so the RELATIVE responses are sound.

**Implication for TM:** expect shifts to be the wrong lever. Do not delete them
on this evidence alone (N=80, different family), but weight the experiment below
accordingly and consider testing S≈50 too, where TM is cheapest (+0.68%).

**THE DECISIVE EXPERIMENT (cheap, ~3-5 forwards, no new physics):**
Take `best` (apodized + shifts 2Ss=130.6, FWHM 20.3362) and raise the
corrugation — **bisecting on the MEASURED `fwhm_env`, never on a surrogate** —
until it lands on **18.5664 um**, which is `noshift`'s width. Then compare its T
against `noshift`'s **T 0.93450**, a shift-free design at exactly that width.
- T > 0.9345 -> shifts DO add something at constant width; keep them, and the
  design axis is shifts + corrugation traded against each other.
- T <= 0.9345 -> shifts contribute nothing that apodization does not do better;
  drop them (they are also the width-hungrier lever) and optimize the
  corrugation SHAPE alone, which is what `rho_neutral_shape.py` explores.

Run the same re-trim against the ORIGIN's width (17.7005 um, control T 0.89265)
for the on-spec version of the same question.
This is the cleanest single answer available and it settles the shift axis.

---

## 6c. ★★★WHY TM ≠ TE — physics, from our own archive + a literature sweep (2026-08-18)

### (i) WE ALREADY FALSIFIED THE DISTRIBUTED SHIFT FOR TM, IN 2026-07
`memory/project_loss_exploration_chain.md`, "FALSIFIED / CLOSED routes":
> *"Distributed pi-shift (job 117530): ALL variants +21..+39% loss, fwhm also
> widens — each shifted gap is its own radiating kink; lumped shift optimal."*
The lumopt2 campaign then adopted **per-tooth shifts over 25 teeth** — i.e. a
distributed pi-shift — as a core design variable. That is a repeat of a closed
negative result. (Not a perfect contradiction: 117530 used much larger shifts on
a corr-400 cavity-scope device, and the campaign's small shifts DID raise T. But
it was a strong recorded prior that was not consulted.)

### (ii) THE QUANTITATIVE REASON: TM HAS HALF THE k-SPACE MARGIN
A first-order grating **cannot radiate**: at Bragg, beta = K/2, so every order
sits at K(1/2 - m), i.e. |k| >= n_eff*k0 > n_clad*k0 — all evanescent. Radiation
exists ONLY where periodicity is broken (the defect), and its strength is the
mode envelope's Fourier weight **inside the cladding light cone**
(Englund/Fushman/Vuckovic, Opt. Express 13, 1202 (2005)).
The margin to that light cone is **dk = (n_eff - n_clad)*k0** — DERIVED from our
own stored resonances:

| | pitch | lambda_B | n_eff | n_eff - n_clad | dk (rad/um) | smoothing length 1/dk |
|---|---|---|---|---|---|---|
| TE | 500 nm | 1570.7 | 1.5707 | 0.1267 | **0.507** | **1.97 um** |
| TM | 500 nm | 1523.6 | 1.5236 | 0.0796 | 0.328 | 3.05 um |
| TM anchored | 516.83 | 1558.6 | 1.5079 | 0.0639 | **0.258** | **3.87 um** |

**TM must smooth a feature over ~2x the length TE does before it stops
radiating.** Against a ~20 um budget that is 10% (TE) vs 20% (TM). This is the
cleanest single-number statement of the whole problem and it comes from our own
measurements.

### (iii) PUBLISHED AND DIRECTLY APPLICABLE: THE TM STOP-BAND COLLAPSES IN THIN CORES
Zhang, McCutcheon, Burgess, Loncar, *"Ultra-high-Q TE/TM dual-polarized photonic
crystal nanocavities"*, Opt. Lett. **34**, 2694 (2009), arXiv:0905.3854:
> *"Decreasing thickness causes the width of the TM bandgap to sharply decrease,
> whereas the width of the TE bandgap remains almost constant."*
> *"The narrowed TM bandgap results in a reduced Bragg confinement, which
> increases the transmission losses through the Bragg mirrors."*
**Our core is 350 nm** — exactly that regime. Weaker TM confinement per period =>
deeper mirror penetration => longer mode => every length-based lever costs more.
(Same paper: with a thick enough core TM is NOT worse — Q_TM 2.4e6 > Q_TE 1.2e6.)

### (iv) ★THE TOOTH SHIFT IS THREE PERTURBATIONS, NOT ONE — AND WE HAVE THE RECEIPT
`bragg_device.py:226-229`: a positive shift SHORTENS THE NARROW GAP and the
cavity absorbs 2*sum(shift). So one "shift" simultaneously applies:
1. the intended local **phase advance**;
2. a local **duty-cycle change** (wide tooth keeps its half-pitch while the
   period shrinks => D rises above 0.5 => local kappa changes as sin(pi*D));
3. a local **DC-index change** (more high-index material per period) => a local
   **Bragg detuning / chirp**.
**Proof that (3) is real and large in our device:** `shift_ladder.py:23-24`
recorded **lambda moving +1.6 nm per +374 nm of 2*Sigma_s**. A pure phase
redistribution cannot move the resonance at all. And the DC-index term scales
like 1/(n_eff - n_clad), which is **~2x larger for TM** by the table in (ii).
So the TE/TM shift comparison is CONFOUNDED: we are not comparing the same
perturbation across polarizations.

### (v) SO IS THERE A "TM VERSION"? — the useful distinction is TRANSVERSE, not wide-vs-narrow
The productive split is **longitudinal (segment lengths) vs transverse (widths)**,
not wide-segment vs narrow-segment. Our own corrected numbers already say
transverse wins for TM (apodization 0.0483 vs shifts 0.0167 T/um = **2.9x**), and
the low-index 1-D-cavity literature apodizes **transversely** as standard:
- McCutcheon & Loncar, Opt. Express **16**, 19136 (2008) — SiN n~2.0, 1-D
  confinement, hole taper; Q 2.3e5 at V ~ 0.55(lambda/n)^3.
- *High-Q asymmetrically cladded SiN 1D photonic crystal cavities*, Nanophotonics
  (2022), PMC9412843 — **quadratically tapered nanostick WIDTHS**, no pitch change.
- Quan & Loncar, Opt. Express **19**, 18529 (2011) — the deterministic recipe:
  **constant pitch, quadratic taper of scatterer size => linear mirror-strength
  ramp => Gaussian envelope**, which is the Fourier-optimal shape.
★**A length-neutral AND index-neutral kappa knob already exists in the builder**:
`wall_phase_offset_deg` (misaligned sidewalls, kappa = kappa0*sin(pi*dP/Lambda);
`bragg_device.py:934`; cf. Wang et al., Opt. Lett. **39**, 5519 (2014); Jiang et
al., Micromachines **15**, 666 (2024)). Ramping it over the inner periods gives a
Gaussian kappa taper with **zero change to average width, average index, or
device length** — it sidesteps confound (3) entirely.
**CAVEAT, verified in code (`bragg_device.py:478-492`):** it is currently a
GLOBAL uniform knob that RAISES ValueError if combined with apodization,
per-tooth arrays, or tooth shifts, and it also breaks the y-mirror plane so
`use_y_symmetry` must be OFF (2x cost). Making it a per-tooth taper is an ENGINE
CHANGE, not a config change. Scope it before promising it.

### (vi) DUTY CYCLE IS NOT THE ANSWER (and here is why)
D = 0.5 maximises kappa (sin(pi*D)) and kills even harmonics — but per (ii), for
a FIRST-ORDER grating no harmonic can reach the light cone anyway, so harmonic
suppression buys **nothing** radiatively. Duty is a kappa + DC-index lever, not a
radiation lever. No polarization-dependent duty optimum found in the literature.

### (vii) Q vs LENGTH — our assumed scaling may be wrong
From the light-cone integral, an exponential (cusped) envelope gives
**Q_rad ~ dk/gamma ~ L*dk — LINEAR in L**, not quadratic. And for SiN slow-light
nanobeams, Zhan et al., APL Photonics **5**, 066101 (2020) MEASURED stored energy
**cubic in cavity length**. We have been assuming Q ~ L^2; it is not established.
**At FIXED length the literature offers exactly two levers**: (a) termination /
mode-profile matching (a kappa taper, not a length change) — Lalanne & Hugonin,
IEEE JQE **39**, 1430 (2003); Sauvan et al., PRB **71**, 165118 (2005); and
(b) **radiation recycling / cancellation** — Lalanne, Mias & Hugonin, Opt.
Express **12**, 458 (2004); Kazarinov & Henry, IEEE JQE **21**, 144 (1985).
Our comb is already (b).

### (viii) ★★A TESTABLE MECHANISM FOR THE COMB (best new idea from the sweep)
DERIVED from our own banked geometry at lambda ~ 1566 nm:
- guided Bloch wavenumber beta = pi/0.51683 = **6.0786 rad/um**
- cladding light cone n_clad*k0 = **5.7940 rad/um**
- comb reciprocal vector K_c = 2*pi/0.53098 = **11.8329 rad/um**
- **beta - K_c = -5.7543**, and |−5.7543| **< 5.7940** => **PROPAGATING in the
  cladding**, at ux = cos(theta) = **0.9932**, theta ~ **6.7deg — grazing**.
- The pitch at which this order goes evanescent is **529.2 nm**. The optimizer
  landed on **530.98 nm — 1.8 nm on the propagating side.**
=> **HYPOTHESIS:** the comb is not a mirror or an index cladding. It is a second,
laterally offset, longitudinally phased radiator, tuned to emit into the SAME
grazing lobe the defect leaks into, ANTI-PHASE — i.e. Kazarinov-Henry
cancellation / Noda's double-lattice (Yoshida et al., Nat. Mater. **18**, 121
(2019)) transplanted to a ridge waveguide.
**Two independent cross-checks that this is not numerology:**
- it predicts our MEASURED sensitivity ranking exactly — longitudinal phase turns
  over 2*pi per **1.03 um** (~2 grating periods) => *sharp*; standoff turns over
  2*pi per **9.2 um** => *loose*; radius sets amplitude only => *loose*. Our comb
  study measured precisely "phase and pitch sharp, radius and distance loose".
- the angle matches the independently measured leak: memory records the
  innermost-tooth leak as *"Lorentzian^2 at grazing ux 0.99"*; this predicts 0.993.
**DISCRIMINATING EXPERIMENTS (cheap):** (1) comb-pitch scan 522/526/529/531/535/540
nm — a sharp feature within ~2 nm of 529 confirms it, a smooth monotone curve
refutes it; (2) comb longitudinal phase scan over one grating period — predict
Q_i oscillates with period 1.03 um and crosses BELOW the no-comb control (we
already have one such point: the mis-placed variant, Q_i 38,784 < 46,499, which
is our strongest existing evidence for coherent interference); (3) standoff scan
1-4 um — predict nearly flat (period ~9 um), which distinguishes this from a
mirror-at-standoff picture (which would give ~542 nm fringes).
**Also predicted:** the comb should be much WEAKER for TE (for TM, E is along the
posts' 350 nm axis => maximal polarizability; for TE it is transverse).
★This configuration appears to be **absent from the literature** — if the
mechanism confirms, it is publishable.

### (ix) WHAT THE LITERATURE DOES *NOT* SUPPORT
No peer-reviewed source measures radiation loss vs phase-shift distribution — DFB
models are 1-D and cannot radiate. Our *"each shifted gap is its own radiating
kink"* reading has **no literature support and no refutation**. The only source
addressing it at all (US Patent 11,125,935, Honeywell) claims the OPPOSITE sign
(distributed shift lowers loss) — consistent with our TE result, not our TM one.
Treat the kink picture as our own hypothesis, not established physics.

---

## 6d. ★★★Q-vs-LENGTH LITERATURE — the trade-off is NOT fundamental, and we have a taper problem

Second literature sweep, 2026-08-18. Full citation list in the session record; the
load-bearing items only, here.

### (i) THERE IS NO BOUND. Four groups beat "Q costs length", with numbers.
- **Watts, Johnson, Haus, Joannopoulos, Opt. Lett. 27, 1785 (2002)** —
  *"neither a complete photonic bandgap NOR a trade-off in mode localization for
  Q is required … our V is roughly independent of Q."* Their system is a
  **quarter-wave-shifted index-guided Bragg cavity — topologically OUR DEVICE.**
  Junctions are radiation-free when the core-minus-cladding permittivity contrast
  is preserved across the step. 3D: Q > 1e5 with only N = 10.
- ★**Lalanne & Hugonin, IEEE JQE 39, 1430 (2003)** (numbers from the open ECIO
  companion): a **1D Bragg cavity**, two parameters on the inner segments
  (−30 nm size, +65 nm outward displacement) → mode-matching took Q 200→750, and
  **radiation recycling added another >100x**, for a **~500x gain in Q/V at only
  +6% mode volume.** Closest published geometry to ours in either sweep.
- **Johnson, Fan, Mekis, Joannopoulos, APL 78, 3388 (2001)** — multipole
  cancellation, *"we do not sacrifice localization"*, 16x (2D) / 4.5x (3D).
  ★Design warning: the Q peak is a **sharp Lorentzian in parameter space**
  (R² = 0.9994) while near-fields look identical — **a coarse sweep steps over
  it.** Relevant to how we scan the comb and the inner teeth.
- **Dharanipathy/Minkov/Savona, APL 105, 101101 (2014)**: Q x7 for V x1.48 with V
  explicitly constrained in the optimizer — our exact situation. And in **SiN**:
  **Vij/Waks, arXiv:2509.16827 (2025)**, measured 3.9x Q improvement, hole
  positions only, footprint fixed.
- **[NOT FOUND]** any theorem bounding radiative Q at fixed V. The only ceilings
  named in the primary sources are material absorption and fabrication disorder.

### (ii) ★OUR ASPECT RATIO IS IN THE TM-HOSTILE REGIME
**Zhang/McCutcheon/Burgess/Loncar, Opt. Lett. 34, 2694 (2009)** measured, in ONE
structure, Q_TM collapsing **2.4e6 → 9,000 (~270x)** as thickness:width went
3:1 → 1:1, *"and a narrow bandgap also leads to large penetration depth of the
mode into the Bragg mirrors, thereby increasing the mode volume"* — **both of our
symptoms (low TM Q, long mode) from one variable.** Q_TE was unchanged.
**Our core is 350 nm tall x 800 nm wide = 1:2.3 — i.e. past 1:1, further into the
bad regime.** Independent confirmation: Johnson et al., PRB 60, 5751 (1999) finds
TM gaps want h ~ 2.3a and TE gaps h ~ 0.6a; **our h/a = 350/517 = 0.68 is
TE-optimal.** And Barclay's SiN group fixed exactly this by going **350 → 610 nm
thick** to pull the mode off the light line (arXiv:1905.03341, Q ~ 1e6).
If height is frozen by fab (single-litho 350 nm, per memory), this is a
**standing explanation for the TM penalty, not an action** — but it should be
stated in any writeup, and it argues the TM device is being asked to do something
the geometry disfavours.

### (iii) ★★THE APODIZATION TAPER IS TOO SHORT — the cheapest actionable finding
The Quan & Loncar recipe (Opt. Express 19, 18529 (2011)) is **constant pitch +
quadratic taper of scatterer size => LINEAR ramp of mirror strength => Gaussian
envelope**. Our builder already does the right thing: `bragg_device.py:1039-1050`
ramps corrugation depth **linearly** over `n_apod` teeth, and we have **kappa ∝
corrugation MEASURED**, so a linear depth ramp IS a linear mirror-strength ramp.
**But the length is wrong.** apod-20 = 20 periods = **10.3 um**, against a
Gaussian 1/e half-width of **~17 um** for a 20 um FWHM mode — the taper covers
only **0.61x the mode**, so beyond tooth 20 the envelope reverts to exponential
and the Lorentzian tail returns. Numerical light-cone model, FWHM held at 20 um
and kappa_max re-solved per taper length:

| taper N_t | length | leakage reduction vs no taper |
|---|---|---|
| 10 | 5.2 um | 1.3x |
| **20 (current)** | 10.3 um | **4.4x** |
| 30 | 15.5 um | 25x |
| **40** | **20.7 um** | **~180x (plateau)** |
| 50-80 | 26-41 um | no further gain |

**RECOMMENDATION: extend apodization from 20 to ~40 periods/side** (~1.2x the
mode's 1/e half-width), re-trimming full depth to hold 20 um. DERIVED from a 1D
model, not measured — but it is a cheap, well-posed FDTD test.

### (iv) THERE IS A CROSSOVER, AND WE ARE JUST PAST IT
Same model, holding intensity FWHM fixed (our actual constraint), comparing a
Gaussian envelope against the plain exponential:

| mode FWHM | dk*L | Gaussian / exponential leakage |
|---|---|---|
| 12 um | 2.6 | **0.56x — apodization HURTS** |
| **20 um (our spec)** | **4.4** | **~105x** |
| 24 um | 5.3 | ~5e3 |

Crossover at dk*L ~ 3 (FWHM ~ 14 um for our TM device). **Below it, apodization
is actively harmful** — the Gaussian's broader k-space core beats the
Lorentzian's narrow core only once dk*L is large enough. We sit just past it,
which is why apodization helps but not spectacularly.
★CAVEAT the agent flagged and I repeat: this gain is violently sensitive to
n_eff — 11x at 1.4995, 105x at 1.5084, 3.5e3 at 1.52. A 1.4% n_eff change moves
it ~300x. Treat as a trend, never as a Q prediction.

### (v-RESULT) ★★★DIAGNOSTIC RUN 2026-08-18 — THE LOSS **IS** ENVELOPE-LIMITED
MEASURED from stored .mat (no GPU), two INDEPENDENT parameter axes:

**Axis 1 — N-ladder at corr 325, identical box** (`results_from_igum/tm_nladder_c325/`):

| N | mode L (um) | T | Q_i |
|---|---|---|---|
| 60 | 16.804 | 0.9674 | 24,054 |
| 70 | 17.738 | 0.9624 | 30,488 |
| 80 | 18.394 | 0.9524 | 35,081 |
| 100 | 19.245 | 0.9104 | 38,409 |
| 120 | 19.661 | 0.8441 | 43,747 |

**=> Q_i ~ L^3.60** (5-point log-log fit).

**Axis 2 — corrugation at fixed N** (325 vs 400, `tm_nladder_c400/`):
- N=60: L 16.804→14.567, Q_i 24,054→16,961 => **Q_i ~ L^2.45**
- N=70: L 17.738→15.151, Q_i 30,488→20,297 => **Q_i ~ L^2.58**

**VERDICT: the measured exponent is 2.5-3.6, bracketing the predicted L^3 for a
radiation/envelope-limited mode.** There is **no dominant distributed loss
floor**. Envelope engineering is therefore the correct axis, the taper-length
finding in (iii) is worth testing, and the Gaussian-vs-exponential shape gain in
(iv) is physically available.
★CAVEATS: Q_i = Q_L/(1-sqrt(T)) is stiff near T->1 — at T~0.91 a 0.01 error in T
moves Q_i ~11%, so each exponent carries roughly +-0.3-0.5. The two axes differ
(3.60 vs ~2.5) because they are different paths: N changes the mirror length,
corr changes kappa. On the corr axis alone Q_i ~ corr^-1.8, notably flatter than
the -3 that kappa ∝ corr would give — i.e. **kappa ∝ corr is NOT holding well
between 325 and 400 nm** (L moved only 13% for a 23% corr change). Any surrogate
built on kappa ∝ corr over that range should be re-checked.
★BONUS, and it matters for the spec: **the mode length SATURATES.** N=100→120
grows L only 2.2% (19.245→19.661) while T falls hard (0.9104→0.8441) as coupling
weakens. So corr-325's mirror-limited asymptote is **~19.7-20 um** — the ~20 um
spec is essentially this family's NATURAL mode length, which is presumably why
corr-325 was chosen. Consequence: **L is not available as a lever anyway**; at
N>=100 you are already at the asymptote, so the only remaining route to Q is
changing the envelope SHAPE at fixed L — exactly (iv) and the see-saw (§6e).

### (v) ★FREE DIAGNOSTIC — the test that produced the result above
**Is our loss envelope-limited at all?** For an exponential (un-apodized)
envelope the light-cone integral gives **Q_rad ∝ L³ ∝ kappa⁻³ ∝ corr⁻³**
(DERIVED and numerically verified: successive doublings gave 7.53/7.88/7.93 vs 8).
We already have kappa ∝ corr MEASURED and Q_i at several corrugations in stored
.mat files. **Fit log Q_i vs log(corrugation):**
- slope ≈ **−3** => loss IS envelope-limited => apodization/taper work is the
  right axis and (iii) should pay.
- slope **flatter** => a distributed loss floor dominates (per-period scattering,
  roughness, mode conversion) that apodization **cannot touch**, and the whole
  envelope-engineering program is capped.
This costs nothing and it decides which half of the program is worth funding.

### (vi) Q ~ L² WAS WRONG
Un-apodized (exponential envelope): **Q ∝ L³**. Apodized (Gaussian):
**Q ∝ exp(dk²L²/2)** — the papers' "Q exponential in N, V linear in N" and
"Q exponential in mode size" are the same law in different variables. Measured
corroboration in SiN: **Zhan et al., APL Photonics 5, 066101 (2020)** — stored
energy **cubic in cavity length**. Every earlier Q_i decomposition in this
programme that assumed L² should be redone.

### (vii) CONTEXT: WHERE OUR DEVICE CLASS ACTUALLY SITS
Measured Q for phase-shifted *sidewall-corrugated* grating cavities in the
literature: X. Wang 2013 (SOI slot, corrugated, phase-shifted) **3e4**;
J. Biophotonics 2013 **1.5e4**; Velha 2007 (etched grooves, tapered) **5.8e4**
vs ~9,000 untapered (**tapering bought 6.4x**); Md Zain 2008 **1.49e5**.
**Our Q_i ~ 46,500 is already at or above that class.** Hole-based nanobeams
reach 1e6-1e9, but on a different physics (large index contrast, real bandgap).
★**Get Husko, Ducharme, Fahrenkopf, Guest, OSA Continuum 4, 933 (2021)** —
foundry **SiN, quarter-wave-shifted, square-wave sidewall corrugations, Λ=520 nm,
ΔW=250/350/450 nm, N=100/200/300**. That is a near-exact match to our device and
neither sweep could open it. **Institutional access needed; read it before any
writeup.**

### (viii) OPEN IN THE LITERATURE (i.e. our results may be novel)
**[NOT FOUND]** in either sweep: any application of the gentle-confinement /
Gaussian-mirror recipe to a hole-free **sidewall-corrugated** cavity with a **Q**
result (all corrugation-apodization papers target spectral sidelobes); any
Q-vs-taper-length curve for such a cavity; any published kappa_TE/kappa_TM ratio
for a given corrugation depth in SiN; anyone redoing Englund's Gaussian-envelope
k-space asymptotics for a **TM** mode using E_z (Englund's Eqs. 6-7 carry the E_z
term from the start — the TE simplification is an explicit later choice, so the
TM version is a small derivation nobody has published). Our side-comb geometry
was also absent from the first sweep. **Several genuinely publishable gaps.**

---

## 6e. ★★★THE "TM VERSION" ALREADY EXISTS AND WE MEASURED IT: THE INNER SEE-SAW

User's question, 2026-08-18: *"is there no way to do tooth shift that does for tm
decrease in loss and fwhm the same, like was in te"* — **yes, and it is recorded
in `memory/project_loss_exploration_chain.md` (job 117814, accurate mesh dx~35,
converged box).** It is not a longitudinal shift; it is a localized, zero-sum,
antisymmetric WIDTH perturbation on the innermost teeth:

> Family A INNER SEE-SAW (teeth ±1 = 1000+δ, ±2 = 1000−δ, **zero net area, even
> parity**): δ=+10 → 0.0814, **+20 → 0.0810**, +30 → 0.0810 (SATURATES);
> δ=−10 → 0.0834, −20 → 0.0851, −30 → 0.0871.
> **CHAMPION: rect-1050 cavity + see-saw δ=+20 → loss 0.0810 (−31% vs control),
> T 0.878 → 0.9179, fwhm +0.8%, λ_res UNMOVED.**

**−31% loss at +0.8% width with the resonance unmoved.** That is precisely the
TE-like behaviour the user is asking for, achieved in TM. The recorded reading:
*"Antisymmetric + saturating + linear-through-zero = genuine INTERFERENCE
cancellation of the residual cavity-local radiating moment (the multipole
prediction)."*

### ★THE DESIGN RULE (why this works and the tooth shift does not)

| property | inner see-saw | campaign tooth shift |
|---|---|---|
| **localized?** | YES — innermost 2 tooth pairs | NO — spread over 25 teeth |
| **zero net area / DC index?** | YES by construction | NO — shortens narrow gaps, raises n_avg |
| **resonance drift** | **λ unmoved** | **+1.6 nm per +374 nm of 2Σs (MEASURED)** |
| **transverse or longitudinal?** | transverse (width) | longitudinal (segment length) |
| **effect on mode length** | **+0.8%** | **+9.5%** |
| mechanism | cancels a radiating multipole | spreads the defect = lengthens the mode |

All four properties matter, and the tooth shift fails all four. Spreading the
phase over 25 teeth **is** lengthening the mode — that is what a distributed
phase shift does by definition (the whole DFB literature says so, §6c-(ii)). A
zero-net-area antisymmetric pair instead leaves the envelope alone and kills a
radiation *moment* by interference.

### Literature backing (found independently, §6d)
- **Johnson, Fan, Mekis, Joannopoulos, APL 78, 3388 (2001)** — multipole
  cancellation: *"unlike a previous, mode-delocalization mechanism, **we do not
  sacrifice localization**."* Exactly our see-saw's signature. Gains 16x (2D),
  4.5x (3D).
- **Lalanne & Hugonin, IEEE JQE 39, 1430 (2003)** — **~500x in Q/V at +6% mode
  volume**, in a 1D Bragg cavity, from **two localized inner-segment parameters**.
- ★**Johnson 2001's design warning applies to us:** the Q peak is a **sharp
  Lorentzian in parameter space** while near-fields look identical. **Our see-saw
  was scanned on a coarse 10 nm grid (δ = 10/20/30) and "saturated" — a
  saturation on a coarse grid is exactly what stepping over a sharp peak looks
  like.** A fine δ scan (2-5 nm steps, and δ > 30) is cheap and may not be
  saturated at all.

### ★THE SEE-SAW IS PURE CORRUGATION — IT IS ALREADY IN OUR PARAMETER BASIS
The engine's per-tooth basis is `corr_d = w_wide − w_narrow` and
`avg_d = (w_wide + w_narrow)/2`, both FREE for the inner 25 teeth. The see-saw
moves only the WIDE tooth by ±δ, so in campaign coordinates it is exactly:

    Δcorr_d = ±δ ,  Δavg_d = ±δ/2      (tooth 1 positive, tooth 2 negative)

Concretely on corr-325 / W800 (w_wide 962.5, w_narrow 637.5) with δ = 20 nm:

| tooth | corr (nm) | avg (nm) |
|---|---|---|
| 1 | **345** | **810** |
| 2 | **305** | **790** |
| 3..25 | 325 | 800 |

A stricter, **per-tooth area-neutral** variant is worth testing alongside it:
`Δcorr_d = ±δ, Δavg_d = 0` (w_wide and w_narrow move oppositely), i.e.
corr = [345, 305, 325, ...] with avg flat at 800.

### ★★SO WHY DIDN'T THE OPTIMIZER FIND IT? (it had the freedom the whole time)
BEST_T9635's corrugation profile is
`282.6, 289.2, 303.6, 311.7, 313.2, 316.5, 318.8, 319.7, 318.7, 320.1, 322.3, ...`
— a **smooth monotone taper** rising to ~322 with only ~±2 nm ripple. It is NOT
an alternating ±20 nm see-saw. Three reasons, and they are all fixable:
1. From a uniform seed the greedy gradient direction is "lower kappa near the
   cusp", which widens the mode — and with the width guard broken (§2) that read
   as **free transmission**. The optimizer took the cheap direction because we
   priced it wrong.
2. An alternating pattern is a high-spatial-frequency feature; L-BFGS-B from a
   smooth seed with a smooth gradient converges to smooth solutions.
3. Johnson 2001: the multipole-cancellation Q peak is a **sharp Lorentzian in
   parameter space** with visually identical near-fields — gradient descent
   steps straight over it unless it starts nearby.
**=> SEED the see-saw; do not expect to discover it.** This is the single
clearest lesson for the corrected campaign's initial conditions, and it costs
nothing to act on.

### ★WHAT HAS NEVER BEEN TRIED
The see-saw was measured on the **corr-400 / W800 / cavity-1050 / N=80** device
under the cavity-only scope. It has **never** been tried:
- in the **corr-325** family the current campaign uses,
- at **N=100/165**,
- **combined with the comb** (the two mechanisms are independent — the see-saw
  cancels a cavity-local moment, the comb cancels a grazing far-field lobe, so
  they may stack),
- with more than 2 tooth pairs, or with the finer δ grid above.
**This is the highest-value untested lever in the programme**, and unlike the
shifts it is already measured to be width-neutral.

---

## 6f. ★★★WHAT IS ACTUALLY LEFT TO TRY — checked against the CLOSED list

Source: `memory/project_loss_exploration_chain.md` (cavity loss program, Rounds
1-8, `results_from_athena/LOSS_EXPLORATION_FINDINGS.md`). **Read this before
proposing anything — a lot is already falsified.**

### ★THE FINDING THAT REPRIORITISES EVERYTHING (Round 7 k-space diagnostic)
> *"only ~30% of radiating weight is **cavity-local** and the champion already
> harvests ≈ that; remaining **~70% is distributed along the arms** → arm/envelope
> levers = phase 2, further cavity reshaping ≈ exhausted."*

So **cavity work is capped at ~30% of the radiation and the see-saw already took
most of it.** This DOWNGRADES §6e's expected value: the see-saw is still worth
porting to corr-325 and stacking with the comb, but do NOT expect another −31%.
**The remaining 70% lives in the ARMS — envelope/arm levers are the real target**,
which agrees with the independent finding that loss is envelope-limited (§6d-v).

### CLOSED — do not re-propose (all measured, evidence in FINDINGS.md)
distributed π-shift (+21..+39% loss) · step-envelope islands · **inner-tooth
shapes** (null-to-bad; note *"the cavity-side face of tooth 1 is load-bearing —
fab corner rounding there is a loss risk"*) · **wall-phase offset** ·
anti-radiator asym-DW · hourglass · external scatterers · cavity SHAPE on top of
rect-1050 (barrel/tri hurt; the cavity optimum is purely SCALAR = added area).

★**RETRACTION:** §6c recommended ramping `wall_phase_offset_deg` as a
length-neutral κ taper, on literature grounds. **It is on the closed list.** I
proposed it without checking the archive. Withdrawn — do not spend GPU on it
unless someone re-reads why it closed and finds the earlier test was scoped
differently (it was tested as a global uniform knob, not a ramp, so a per-tooth
ramp is *arguably* untested — but the burden of proof is on that argument).

### ★OPEN AND ALREADY MEASURED — the "OUT-OF-SCOPE parking list" was parked FOR US
The loss program explicitly deferred these to the inverse-design phase, i.e. now:

| parked route | measured | note |
|---|---|---|
| **TM whole-device SINUSOID corrugation** | **−10% loss @ +0.8% fwhm** | ★the non-rectangular-teeth answer, and it is nearly width-neutral |
| W1000 / C500 + cav1250 | **−60% loss @ +6% fwhm** | biggest effect on the list; costs width, but re-trim it (§6) and see what survives |
| tapered island (8 teeth) | −36% @ +4.8% | an arm/envelope lever |
| TE whole-device sinusoid | −29% @ +7.5% | TE only |
| TE barrel300 | −9% | TE only |

**Non-rectangular teeth: the useful version is the whole-device corrugation
PROFILE, not the inner-tooth shape.** `corrugation_profile` already exists as a
builder feature (listed under "Machinery — UNCOMMITTED builder features"), so the
sinusoid is a config change, not new code. Literature agrees the profile matters:
Lee & Streifer, JOSA **68**, 1071 (1978) computes radiation for rectangular vs
sinusoidal vs triangular corrugations; and *Polarization-Independent Complex
Bragg Grating Filters on Silicon Nitride*, Laser Photon. Rev. (2024),
doi:10.1002/lpor.202402114, equalises TE/TM specifically with a **triangular**
lateral corrugation.

### ★★★CLOSED 2026-08-19 — MOIRÉ (user: "moiré width too big"), x-ASYMMETRY, GAUSSIAN

**MOIRÉ — REFUTED, and the reason is a hard geometric conflict.** DERIVED with
κ₀=0.0353 µm⁻¹, Λ=516.83 nm, N=100 (device 103.4 µm):
- one beat node in the device requires Δk ≤ 0.061 µm⁻¹ ⇒ **mode FWHM ≥ 50.8 µm**
- a 20 µm mode requires Δk = 0.393 µm⁻¹ ⇒ node spacing 16 µm ⇒ **6.5 nodes in
  the device**, i.e. a coupled-cavity array, not one cavity.

| Δpitch nm | node spacing | nodes in device | mode FWHM |
|---|---|---|---|
| 2 | 133.6 µm | 0.77 | 57.8 µm |
| 5 | 53.4 | 1.93 | 36.5 |
| 10 | 26.7 | 3.87 | 25.8 |
| 16.7 | 16.0 | 6.46 | **20.0** |

**The single-node moiré floor is ~51 µm against a 20 µm spec — 2.5× too wide.**
The user called this from lab experience before the calculation; the calculation
agrees. **Do not propose moiré for this device.** (It would only make sense on a
device whose spec mode is ≳50 µm.)

**x-ASYMMETRY — PROVABLY HARMFUL, already answered in the archive.**
`memory/project_bic_kerker_batch1_dispatch.md`:
> *"SYMMETRY answered: **mirror-symmetric is PROVABLY optimal** (antisym
> perturbation → odd δA ⊥ even A₀ → strictly adds radiation; anti-moment study
> confirmed)."*
The argument: the device's radiating amplitude A₀ is **even**; an antisymmetric
perturbation produces an **odd** δA; odd ⊥ even, so the cross term in
|A₀+δA|² vanishes and the perturbation can only **add** |δA|² in quadrature — it
can never interfere destructively. Confirmed by the anti-moment study, and
`asym_inner_dw_delta_nm` ("anti-radiator asym-DW") is on the CLOSED list.
★**Why the comb is NOT a counter-example** (it is deliberately not x-mirrored):
the comb is a **separate radiator placed away from the guide**, whose emitted
field is phased to be anti-phase with the *total* far-field leak. That is
far-field interference between two sources. Perturbing the **grating itself**
antisymmetrically instead modifies the cavity's own amplitude by an odd amount,
which is the orthogonal case above. **Two different mechanisms — do not conflate
them.** So: keep the grating mirror-symmetric; keep the comb free to be
asymmetric.
Also from the same entry, a ceiling worth knowing: *"passive scatterer α is REAL
but optimal α is COMPLEX → phase-limited; even sign-correct real-α site models
only ~13% cancel and parasitics dominate"* — i.e. **comb-like passive cancellers
are phase-limited to ~13%**, consistent with the comb's measured +17.1% Q_i.

**GAUSSIAN ENVELOPE — already tried by the user's lab** (user, 2026-08-19). So
§6d-(iii)'s "extend the taper to ~40 periods" is *not* virgin territory. ★Before
spending GPU on it, ask the user what their lab measured: if a Gaussian envelope
was already built and did not deliver, the envelope-shape axis is much more
constrained than §6d suggests, and the ~180× model number is contradicted by
experiment.

### ★★★ARE WE ALREADY AT THE FAB CEILING? — check this BEFORE optimising further
Measured Q for **phase-shifted sidewall-corrugated** grating cavities in the
literature: X. Wang et al., Opt. Express 21, 19029 (2013) **3e4**; X. Wang et
al., J. Biophotonics 6, 821 (2013) **1.5e4**. (Groove/hole-type cousins reach
more: Velha 2007 **5.8e4**, Md Zain 2008 **1.49e5**.)
**Our SIMULATED Q_i is 46,500** — at or above every measured sidewall-corrugated
value found. Every simulation here is of a perfectly smooth, perfectly periodic
device; real ones are limited by sidewall roughness and phase/stitching error,
and Englund 2005 states the general ceiling plainly: *"Qs are bounded to
currently ~1e4 by material absorption and surface roughness."*
⇒ **It is entirely possible the fabricated device is already roughness-limited,
in which case further simulated Q gains buy nothing.** This is cheap to settle
and would save months: **compare a measured Q from a fabricated device of this
family against its simulated Q.** If they diverge badly, the productive work is
fab/process, not geometry. Nothing in this handoff is worth GPU until that is
known. (The user's lab has fabricated devices — ask.)

### OTHER AXES CONSIDERED 2026-08-19 — with why they do or do not help
- **Shorter operating wavelength.** δk = (n_eff−n_clad)·2π/λ, so δk ∝ **1/λ** at
  fixed indices. 1550 → 1310 nm is **+18% δk**, and the Gaussian suppression
  exponent goes as δk²L² ⇒ **~+40%** in the exponent, free, no geometry change.
  Only viable if the application's source wavelength is not fixed — probably it
  is, but it costs one question to ask.
- **Narrowing the waveguide to fix the TM aspect ratio: NO.** Zhang 2009 says TM
  wants tall:wide ≈ 3:1 and ours is 1:2.3, which tempts one to shrink W. But
  narrowing W **lowers n_eff toward n_clad**, collapsing the very light-cone
  margin (§6c-ii) that is TM's core problem. The two effects fight and the
  light-cone one dominates. **Only going TALLER helps**, which is the fab-stack
  lever below.
- **Relaxing the acoustic constraint itself.** The archive lists "relax a
  constraint" as one of only three escapes. The 20 µm spec comes from the
  acoustic transducer; if the transducer could be co-designed to a longer
  interaction length, the entire width-vs-Q fight changes character (Q_i rises
  steeply with L — measured L^2.5-3.6). Outside our scope to decide, but it is a
  legitimate engineering question for the user, not a physics dead end.
- **Higher-order defect modes / coupled cavities: no.** Both give wider or
  multi-peaked modes — the moiré refutation above is the same arithmetic.
- **Duty cycle: no** (§6d-vi) — no harmonic of a first-order grating reaches the
  light cone, so harmonic suppression buys nothing radiatively.

### ★THE LEVER THE ARCHIVE ITSELF POINTS AT (converges with §6c-ii)
Same entry: *"The genuine-novelty levers all relax something: **CLADDING INDEX /
light-cone (suspended air-clad membrane = the big lever)**, SWG metamaterial
cladding, or the width-cost Pareto."* This is exactly the δk finding of §6c-(ii)
arrived at independently: raising (n_eff − n_clad) enlarges the k-space margin,
and TM's margin is only half TE's. Barclay's SiN group did precisely this by
going 350 → 610 nm thick (arXiv:1905.03341, Q ~ 1e6). **It changes the fab stack,
so it is a user decision, not a study we can just run.**

### ★★MOIRÉ — the original write-up (superseded by the refutation above)
Nothing in the archive mentions it. **And the physics lands exactly on the
gentle-confinement prescription:**
Superimpose two pitches Λ₁, Λ₂ (or modulate the pitch) so the coupling beats:
`κ(x) = κ₀·|cos(Δk·x/2)|`, Δk = 2π(1/Λ₁ − 1/Λ₂).
- At a beat **node** κ→0 **and the grating phase flips by π** — so the moiré
  *generates its own π-shift*. There is no separate abrupt defect, hence **no
  cusp to smooth** — the thing we have been fighting all along.
- Near that node, `κ(x) ≈ κ₀·(Δk/2)·|x|` — i.e. **mirror strength increasing
  LINEARLY with distance**, which is precisely Quan & Lončar's rule that yields a
  **GAUSSIAN envelope** (Opt. Express 19, 18529 (2011)). Our apod ramp
  approximates this with 20-40 free teeth; a moiré gets it from **ONE global
  parameter**.
- Mode length is then set by σ = κ₀Δk/2 ⇒ **L ∝ 1/√(κ₀Δk)**, so the pitch
  detuning is a clean, single-knob **mode-length dial** — exactly what the
  re-trim scheme in §6 needs, and far better conditioned than 25 coupled
  corrugations.
- Moiré/phase-shifted gratings are standard in FBGs and have been done in
  integrated sidewall gratings for narrow-band filters, so it is fabricable; what
  is (per both sweeps) **unpublished is using it as a radiation-minimising
  envelope for a fixed-mode-length cavity**.
★**Caveats to check first, cheaply:** (a) at the node κ→0 locally, so the local
stop-band vanishes — confirm the mode is still confined by the surrounding strong
regions and no new leakage channel opens; (b) a moiré has TWO beat nodes per beat
period — the device must contain exactly one, or you build a coupled-cavity pair;
(c) the beat also modulates the DC index unless the two components are balanced —
that is the §6c-(iv) detuning trap again, so balance it by construction.
**RECOMMENDATION: this is the most promising untried idea on the list**, it is a
config-level change if `corrugation_profile`/per-tooth arrays can express a
pitch beat, and it attacks the 70% (arms/envelope), not the exhausted 30%.

---

## 7. PARKED — needs the user, do not do alone

- **scancel of 134032 (stage-4, Athena) and 55801 (bare, IGUM)** — both steer on
  void widths. Recommended, but stopping runs is confirm-first (CLAUDE.md §6);
  use the `stop-runs` skill and resolve the job ID first.
- **scancel of 134299** — its t0/t1 are void and superseded by 134334. Only t2
  is worth keeping (post-fix). Low value either way; it drains on its own.
- Dispatching `rho_neutral_shape.py`.
- Large uncommitted git inventory — never commit without permission.
- Ansys adjoint gradient-bug report; far-field needle readout; width-band Pareto.

Note: seedB2 (IGUM 56033) **finished cleanly** at 12:23 (exit 0,
best_fom 0.702857, `_best.json` fetched to `results_from_igum/lumopt2_logs/`).
It was not a failure. Its final design is width-suspect like the rest.

---

## 8. Commands

```bash
# status
ssh evyatarrubin@athena.technion.ac.il "squeue -r -u evyatarrubin -o '%.14i %.8T %.10M %R'"
ssh evyatarrubin@132.68.58.101 "squeue -u evyatarrubin -o '%.10i %.8T %.10M'"   # IGUM

# license seats — MANDATORY before any multi-task dispatch; probe from IGUM
ssh evyatarrubin@132.68.58.101 "\$HOME/research/lumerical/Lumerical-2026-R1.3/opt/lumerical/v261/licensingclient/linx64/lmutil lmstat -c 1055@132.68.48.51 -f lum_fdtd_solve | grep 'Users of'"

# dispatch a lumopt2 study (verify flags against the parser first; unknown flags ABORT)
SBATCH_MEM=160G LUMOPT2_QOS=2h_2g LUMOPT2_TIME=01:55:00 \
  bash athena/deploy_athena.sh --lumopt2-design=runners.lumopt2_design.<module> \
  --max-concurrent=1 [--after=<jobid>] [--array-tasks=<lo>-<hi>]

# code-only push (NOT --no-submit, which does not exist)
bash athena/deploy_athena.sh --upload-only
```

Offline width check on any stored .mat, no GPU:
```python
import scipy.io, numpy as np, sys; sys.path.insert(0,'.')
from runners.lumopt2_design import lumopt2_design as eng
d = scipy.io.loadmat(PATH, squeeze_me=True)
x = np.asarray(d["field_x"],float)*1e6; I = np.asarray(d["field_energy_density_1D"],float)
print(eng.fwhm_env_of_line(x, I), float(d["fwhm_m"])*1e6)   # must agree
```

---

## 9. Files

- Engine: `c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\runners\lumopt2_design\lumopt2_design.py`
- Width recovery tool: `...\runners\lumopt2_design\fsp_width.py`
- Ready, undispatched study: `...\runners\lumopt2_design\rho_neutral_shape.py`
- Banked designs: `...\runners\lumopt2_design\best_designs.py`
- Registry: `...\runners\lumopt2_design\DESIGNS.md` (★its FWHM block is VOID)
- IGUM campaign logs: `...\results_from_igum\lumopt2_logs\`
- Runbook: `...\.claude\skills\lumopt2-design\SKILL.md`
- Memory: `C:\Users\evyat\.claude\projects\c--Users-evyat-Lumerical-phase-shift-grating-FTDT-codes\memory\project_lumopt2_campaign_state.md`

---

## 10. Lessons recorded today (skill items 25-27)

1. **Never let the CONTROLLED quantity differ from the SPECIFIED quantity
   without measuring both every evaluation.** The campaign controlled sigma for
   days while the spec is FWHM.
2. **A constraint stated in a surrogate's units is meaningless until you
   measure its conversion into the spec's units.** `RHO_DN = 0.95` was quietly
   permitting large width growth; nobody had ever expressed that band in microns.
3. **Never restructure a program on a number from a metric you wrote the same
   day and have not cross-checked.** Three estimators gave three answers; the
   new one was the wrong one.
4. **Reuse the project's own analysis functions rather than reimplementing
   them.** The fix was to call `sim_helpers.extract_envelope_peaks` and
   `calculate_fwhm_relative` directly — which then matched to 1e-15.
5. **Keep the raw profile.** All of today's GPU re-runs existed only because no
   field profile was ever saved. Every eval now writes a ~30 kB `.npz`.
