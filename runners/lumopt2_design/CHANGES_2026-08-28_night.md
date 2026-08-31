# Changes since CHANGES_2026-08-28.md (same day, evening → night)

Commit: `62a1b1a`. Everything below happened after the previous summary was written.

## Two live bugs found and fixed (job 137853)
1. **Control task misfired again** — index 34 is also inside `_GFR_RUNGS` (it spans 27–36, not 27–32); 137853_34 ran a GPU probe rung (50 min, exit 0, wrong experiment). Structural fix: control → task **46**, pipeline smoke → task **47**, `N_TASKS` 46→48, and the predispatch gate now **audits index reachability programmatically** (literal count + membership + range). The first version of that check was dead code behind the gate's `sys.exit` — caught and spliced before the exit.
2. **λ-chain skipped on hardware** — iterate 0 of the toy printed `λ-CHAIN SKIPPED (dTp +0.394)`: Lumerical stores spectra frequency-ascending = **λ-descending**, so the stencil's "hi" index sat on the blue side and the ascending-convention guard refused a genuine maximum. Fix: `|dl|` for the stencil width + canonicalizing swap so `wl[i_lo] < wl[i_hi]` always. Math gate extended: gLam invariant under grid reversal to 1e-9, plus a teeth check that the unswapped recipe still refuses the reversed grid.

## Smoke tier hardened
- Moved to N=60 (N=40's resonance risked clipping the 10 nm window) and given a **semantic exit**: the task fails (nonzero) unless the λ-chain actually executed — so an `afterok` dependency genuinely gates the expensive jobs (exit-0-with-skip looked exactly like success).

## Tonight's dispatch (using the sleep hours)
- 137853_41 (chain-skipped, degenerated to a control replica) **cancelled** with approval — its information gets re-measured by the fresh control under final code.
- **Job 137868** = pipeline smoke (task 47, ~2 h) → **afterok** → **job 137869** = corrected toy (task 41) + fresh control (task 46), 12 h lane. Verdict expected ~14:00; auto-proceed to the 96 h campaign stands if criteria pass.

## Routine-ops rules added
- **Quota watch** folded into the monitor's existing ssh poll (alerts ≥250 G; purges stay user-approved; memory `feedback_quota_watch_routine`).
- **Skill item 40**: while hardware runs, an Opus deep-check every ~2 h (progress cadence, fresh-eyes log read, quota/janitor, jsonl pulled local); Fable woken only on anomaly.
