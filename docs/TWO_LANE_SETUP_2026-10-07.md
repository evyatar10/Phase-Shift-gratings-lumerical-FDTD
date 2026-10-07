# Two-lane TE inverse design: Claude lane and Codex lane

Written 2026-10-07 by Claude (Opus 5.5) at the user's request. Audience: the
GPT/Codex agent and the user. This file is the entry point for both lanes.

## 0. Message to Codex (read this first)

1. **You now work ONLY in** `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes_codex`
   (branch `codex-te`). Your Athena base is `~/bragg_sim_codex` and your IGUM base
   is `~/research/bragg_sim_igum_codex`; both are already set in your
   `athena.conf`, `igum.conf` and job scripts. Never edit, deploy to or cancel
   anything in Claude's lane (§2), including its job 172063.
2. **Your previous pause is lifted. LENGTH IS CONSTANT: N = 98 periods/side
   for BOTH seeds, 60 free/side (user decision 2026-10-08, final).** Do not
   use N80 or any other length. A short N80 test (Claude job 171897, cancelled)
   MEASURED that N80 cuts S1's mode width by 4.4 % (19.385 → 18.528 µm) and
   S2's by 0.9 %, which is why the user fixed N at 98. Goal: raise peak T at
   spatial envelope FWHM ≈ 20 µm.
3. **Your main mistakes on 2026-10-07 (§6b), to avoid repeating:**
   - The S2 finite-difference test used ±2 nm legs that moved a 0.20 nm-wide
     resonance by ~0.18 nm. That is not a local derivative, so its "failure"
     does not show the adjoint is wrong. Its small-step legs sat at the
     1.3e-5 noise floor.
   - You then spent ~40 solves on forensics of it.
   - You blocked all TE work, S1 included, on that test, so zero optimizer
     iterates ran for a day.
4. **Engine choices you introduced that Claude reverted** (§6c): crash on a
   missing gλ, FAILED exit at budget end, diagnostics sharing the width
   measurement's try block, and NaN raising in log rows. We recommend you
   adopt the reverts.
5. **What you got right and Claude kept** (§6a): the λ-normalization fix, the
   persistence/resume hardening, the persistent forward budget, the measured
   width contract and the seat rule.
6. **Follow the main-goal checkpoint** (§3.5) at every milestone. When
   acceptance is on measured T and W, a gradient error costs rejected steps,
   not wrong results. The per-iterate gain ratio (measured ΔFOM / predicted)
   is the live calibration check. Run a separate FD calibration only when that
   ratio goes bad.
7. **Pick a route that DIFFERS from Claude's lane** (§5), so that the
   comparison teaches something. Report using the §7 table, and log any change
   worth sharing in §9.

## 1. The decision

The user wants two independent TE inverse-design efforts, one by Claude and
one by Codex, so that each can learn from the other. They are separated by
**git branch + worktree + server folder**, not by copying code into a second
folder of one tree. A folder copy would still share one server tree, where
deploys overwrite each other. The two lanes share history up to commit
`9a8d440` ("Baseline before two-lane split").

## 2. Layout

| | Claude lane | Codex lane |
|---|---|---|
| Git branch | `claude-te` | `codex-te` |
| Local folder | `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes` | `C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes_codex` |
| Athena server base | `/home/evyatarrubin/bragg_sim_athena` | `/home/evyatarrubin/bragg_sim_codex` (created by the first deploy) |
| IGUM server base | `/home/evyatarrubin/research/bragg_sim_igum` | `/home/evyatarrubin/research/bragg_sim_igum_codex` |
| Entry docs | `CLAUDE.md`, this file | `AGENTS.md`, this file |
| Skills | `.claude/skills/` | `.agents/skills/`, `~/.codex/skills/pi-shift-*` (repo/server paths repointed to the Codex lane; originals kept as `*.pre_split_20261007.bak`) |
| Campaign labels | `lumopt2_te_s1_n98`, `lumopt2_te_s2_n98` | use a `_codex` suffix, e.g. `lumopt2_te_s2_n98_codex` |

The Codex worktree received copies of its untracked evidence
(`docs/readiness_evidence_20261006/`, `docs/readiness_evidence_20261007/`) and
TE result folders (`results_from_athena/{campaign_te_s1, recover_te_s2_20261006,
te_readiness_20261007, validate_te}`, `results_from_igum/te_readiness_20261007`).
The originals were not moved or deleted.

## 3. Rules of engagement (both lanes)

1. Edit, commit, deploy and cancel only within your own lane: its branch, its
   folder, its server base and its own job IDs. Never touch the other lane's.
2. **These resources are SHARED; check them before every dispatch:**
   - Lumerical license pool: 50 seats on 132.68.48.51; one GPU solve takes
     ceil(SMs/16) seats (A100 7, H200 9, L40S 9).
   - Athena home quota: both server bases sit under the same 300 G home.
   - Per-user QOS limits: both lanes submit as the same user `evyatarrubin`.
     `squeue -u evyatarrubin` shows both lanes' jobs. Identify your own by
     job ID, or by the log path under your own base.
3. Old results under `~/bragg_sim_athena/results/` (all pre-split runs,
   including Codex's `/work/results/s2_*` outputs) now belong to the Claude
   lane's tree. Codex may READ them and must never write or delete there.
4. Learning crosses lanes only by **deliberate merge or cherry-pick** of a
   change that has shown a benefit on hardware, with a note in this file's
   log (§9). Never rsync or copy files between the folders.
5. Project physics conventions (CLAUDE.md §2) bind both lanes:
   - built-in resonance finder only;
   - spectral FWHM vs spatial FWHM kept apart;
   - the measured spatial envelope FWHM (`fwhm_env_um`) is the width authority;
   - **MAIN-GOAL CHECKPOINT**: at every milestone, and before any new
     diagnostic, write three lines: the goal, what actually blocks it now,
     and whether the next action removes that blocker. Drop actions that
     don't (user rule 2026-10-07).

## 4. The shared problem

| Item | Value | Label |
|---|---|---|
| Goal (user) | raise peak transmission T of the TE pi-shift Bragg grating while holding the spatial envelope FWHM near 20 µm | requirement |
| Device | TE, h 350 nm, n 1.97/1.444, bare (no posts), box 6.8 × 6.81 µm, PVA mesh, dx 50 nm (S2 49.106 nm) | spec |
| Length | **N = 98 periods/side, CONSTANT for both seeds** (user 2026-10-08), 60 free/apodized per side | spec |
| Why not N80 | N80 cut S1's width 4.4 % (19.385 → 18.528 µm, Q_L 1550 → 757) and S2's 0.9 % | MEASURED (171897, IGUM 101434) |
| Variables | 296-slot vector, 181 active: 60 corr, 60 avg width, 60 shift, cavity width | code |
| S1 seed | uniform, corr 250 / pitch 500 / W800 | spec |
| S2 seed | Itai's Nt60 overshoot apodization, pitch 491.06, bulk 494.2/1000, cavity 951.4 | spec |
| Width bands | S1 19.121 µm ±2 %; S2 19.636 µm ±2 % (each seed's own N98 anchor) | spec |
| S1 best so far | t_pk 0.95276, W 19.385 µm, λ 1561.044 nm, at **N98** (Athena 170253, eval 3) | MEASURED |
| S2 seed at N98 | T 0.9731, W 19.636 µm, Q_L 7694 | MEASURED (168530_11) |
| Engine | lumopt2 adjoint + v3 bounded QP step (`v3_step.py`), peak3 objective, separate port and width adjoints, accept/reject on measured T and W | code |
| S1 cost per iterate | ~2.4 GPU-h at N98 on an A100 | MEASURED (170253) |

## 5. Lane status at the split

**Claude lane:**
- Athena job **172063** (4d_1g, 96 h, 256 G), started 2026-10-08 ~00:45:
  - task 0: S1 at N98 (label `lumopt2_te_s1_n98`), starting from the N98 best 0.9528;
  - task 1: S2 at N98 (label `lumopt2_te_s2_n98`), starting from the seed.
- Each task stops cleanly after 8 forward evaluations; it is extended later by raising `MAX_FEVAL` and redeploying, which resumes from the accepted best.
- Runner: `runners/lumopt2_design/campaign_te_n98.py`. The N80 runner and its jobs (171897, 172053) are retired; the runner is archived in `runners/archive/te_n80_20261007/`.
- Calibration: S1 reuses its N98 C factors. S2 uses its own field C and port C = (1, 0), uncalibrated.
- Live gate: the per-iterate gain ratio, measured ΔFOM / `dT_pred` in `<label>_proj.jsonl` (S1 history at N98: 0.84–1.01).

**Codex lane:** stopped at its handoff (`docs/TE_NEW_CHAT_HANDOFF_2026-10-07.md`), whose next step was a bounded S1 restart smoke at N98. That step is superseded for the Claude lane by job 172063. Codex is free to choose its own route, at N98. Suggested routes that would be informative *because they differ*:
- the augmented-Lagrangian formulation (THEORY.md §10);
- the S2 lane with its own calibration philosophy;
- a different step metric (shift weighting).

Codex should not run a second copy of the same N98 v3 lanes.

## 6. Review of Codex's 2026-10-06/07 work (by Claude)

### 6a. What Codex found or fixed

| Codex item | Verdict |
|---|---|
| Wavelength-gradient normalization mixed a T_out adjoint numerator with a modal-power spectral slope (~5 % scale error) | Real but minor: it affects only the λ trust row. Kept in both lanes. |
| Restart/persistence: unique eval IDs, profile binding, accepted/rejected state on resume, interrupted-JSONL repair, atomic state, persistent forward budget | Mostly real hardening. Valuable because every Athena partition requeues on preemption. Kept. A review found no crash bugs on the fresh-start → cold-resume path; the budget counting is correct. Not verified: whether each original bug actually existed. |
| "S2 port-gradient calibration FAILED" (fit residuals +16.7 / +162 / −8.3 / +13.3 / +0.6 / −32 %) | **The measurement was misread; it does not show a code bug.** Its ±2 nm finite-difference legs moved a 0.20 nm-wide resonance by ~0.18 nm, almost a full linewidth, so they did not measure a local derivative. Its small-step legs sat at the 1.3e-5 noise floor. It does not show the adjoint is wrong. |
| Q3dB predictor audit fixes (197d343) | Real small fixes to the prediction tool. Kept; not on the optimizer's path. |
| Mesh / monitor / source-phase audits | Codex itself found no defect. |
| Seat rule: each solve takes ceil(SMs/16) seats | Correct; matches our 2026-10-04 measurement. |
| N98 → N80 IGUM length pair | Good measurement (S2 width −0.9 %); the user then fixed N = 98. |

### 6b. Mistakes

| Mistake | Cost |
|---|---|
| Treated a non-local FD comparison as a calibration failure, then chased it with temporal, source-phase and step-ladder forensics | ~40 solves, one day |
| Blocked all TE work on S2 (later called "too broad" in its own notes) | zero optimizer iterates on either seed for a day |
| Heavy process overhead (hash-pinned release markers for every job); several of its own dispatches failed on tooling: 171516 used the wrong API path, an operator lacked `LUMAPI_PATH`, and a wrapper-path refusal | cycles lost; no GPU time lost |
| Committed ~250k lines of evidence dumps and ~15 single-use scripts into live folders (against CLAUDE.md §7 and §10) | repo clutter |

### 6c. Engine choices Claude reverted on `claude-te` only

Codex introduced all four engine behaviours below in its integration
(1b2d75e). None has fired yet; each can stop a run that would otherwise
continue. All 11 offline gates in the Claude lane are green after the reverts.

| Item | Before Codex (a3fab34) | Codex changed it to | claude-te now | Who was at fault |
|---|---|---|---|---|
| Missing or nonfinite gλ | Degraded step with NO λ limit; in smoke 169002 the resonance moved +3.75 nm in one step | Raise and kill the campaign | Degraded step at ¼ of the trust radius, no λ row, logged loudly. Gate V23 checks the ¼ bound, and a test confirms that without the shrink the step exceeds it. | **Both.** Codex correctly found a real weakness in our code; crashing was too drastic, because a resume hits the same point again. |
| Budget exhaustion | No budget system; runs ended normally at max_feval | The new persistent budget (good, kept) ends the job as FAILED with no `_best.json`, and afterok chains don't fire | Clean stop: `_best.json` written, exit 0 | **Codex** |
| Port-diagnostic readouts | Did not exist | Added inside the same try block as the width measurement, so a diagnostic failure loses the eval's width (fatal at iterate 0) | Diagnostics in their own try | **Codex** (new risk) |
| NaN in a journal row | Written silently, harmless | `allow_nan=False` raises outside any try and kills the run | NaN/inf written as null (the optimizer state file stays strict) | **Codex** (new risk) |
| S2 calibration scripts and their 2 gates | Did not exist | ~15 one-off scripts in `runners/lumopt2_design/` | Moved unedited to `runners/archive/te_s2_calibration_codex_20261007/` (still live on `codex-te`) | **Codex** (CLAUDE.md §10 accumulation) |

Recommendation to Codex: adopt the first four reverts on `codex-te` too,
by cherry-picking the engine part of claude-te commit `0a27c0e`.

Kept from Codex in the Claude lane:
- the λ-normalization fix;
- persistence and the forward budget;
- the measured-width contract;
- the requirement that only accepted evals ratchet `fom_best`;
- the GPU-thread readback check, which passed on hardware (171897).

## 7. Comparison protocol (how the user compares the lanes)

- Same device, seeds, **N = 98**, and each seed's own ±2 % width band.
- Report per lane, per seed, for every iterate:

  | iterate | eval | t_pk (modal) | FOM | W (µm) | λ (nm) | accepted | dT_pred | gain ratio | GPU-h cumulative |
  |---|---|---|---|---|---|---|---|---|---|

- The headline metric is **best accepted t_pk with W inside the band, against GPU-hours spent**.
- Numbers are labeled MEASURED / DERIVED / EXPECTED, and results compare only within identical numerics (engine R1.3 build 4572, PVA, same window/mesh/box).
- A lane that ends with a design outside the width band has no result.

## 8. How learning crosses lanes

When one lane shows a hardware-verified benefit (an algorithm change that
raised accepted t_pk per GPU-h, or a fixed real bug), it writes a one-line
entry in §9 with the commit hash. The other lane may then `git cherry-pick`
that commit onto its own branch, run its own gates, and record the outcome.
Neither lane edits the other's branch.

## 9. Cross-lane log

| Date | Lane | Commit | What | Adopted by other lane? |
|---|---|---|---|---|
| 2026-10-07 | both | 9a8d440 | shared baseline at the split | — |
| 2026-10-07 | claude | 0a27c0e | engine reverts of §6c | — |
| 2026-10-08 | claude | (this branch) | N = 98 constant for both seeds; `campaign_te_n98.py`, job 172063 | — |

## 10. Open items for the user

- Shift weighting in the step metric: shifts are currently held back ~545× relative to corrugations. Pending your yes/no.
- (Resolved 2026-10-08) Length: N = 98 for everything.
