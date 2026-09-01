# Code-compaction plan — audited 2026-09-01 (read-only audit, user-requested)

Goal (user): the study code compact and human-understandable per CLAUDE.md
§10/§11 ("ponytail" ladder). NOTHING here may touch the deployed tree while
campaigns run — every item is marked SAFE-NOW (local/doc only) or
BOUNDARY-ONLY (needs d1/d1u finished + one gated redeploy). Deletes need
explicit user approval, always.

## 30-second summary

The engine works but has three sediment layers: a 3309-line god-module
(~330 lines legacy penalty code + ~90 lines never-adopted RGP),
a 54-task validation file with only ~14 live tasks (tasks 27-36 are dead
GPU-ladder rungs that already caused two index-shadowing bugs), and 13 of
16 campaign files that are museum pieces. The live lanes already use the
right parameterized pattern (dataclasses.replace) — no new abstraction
needed. Done today: stale "task 35 = smoke" doc pointer fixed (was aiming
at a CUDA probe), gates/run_all_gates.py added, b1's seed-lesson lifted
into THEORY.md §6d.

## Action list (priority order)

| # | What | When | Effort | Status |
|---|---|---|---|---|
| 1 | Fix stale "task 35 = smoke" → task 47/50 (CLAUDE.md §5 + skill item 39) | SAFE-NOW | S | ★DONE 2026-09-01 |
| 2 | Delete scratch_s5vec.txt (banked as UNIFORM_S5_FAST_BEST) | needs approval | S | PARKED |
| 3 | gates/run_all_gates.py — one command, four gates, fail fast | SAFE-NOW | S | ★DONE 2026-09-01 (verified green) |
| 4 | Compact HANDOFF.md (3331 lines → live box + HANDOFF_HISTORY.md, unedited moves); archive V2_FWHM_PLAN.md + HANDOFF_2026-08-30.md; merge CHANGES_*.md | SAFE-NOW but SINGLE-SESSION ONLY (a parallel chat appends to HANDOFF — deferred) | M | PARKED (concurrent session active) |
| 5 | Decompose lumopt2_design.py → 8-module package behind a re-export facade (spec/widths/scene/shim/stepper/driver/legacy_penalty/entries; lumopt2_design.py keeps every `import ... as eng` working — 26 call sites unchanged). Gate: compileall + all 4 gates + task-47 smoke on first redeploy | BOUNDARY-ONLY | L | PARKED |
| 6 | Split validate_c325.py: ~14 LIVE tasks stay; CLOSED/DEAD branches → archive/validate_c325_closed.py with IDS FROZEN (renumbering = the shadowing trap; predispatch gate re-audits) | BOUNDARY-ONLY | M | PARKED |
| 7 | Archive museum campaigns (7× campaign_c325_*, campaign_v2_projection.py [imported by NOTHING, dangerously similar name], _seesaw, _noshift, later _best [repoint export_gds.py:20 first]) + ~15 closed probe/ladder scripts → runners/archive/lumopt2_design/; inline campaign_v2_uniform constants into campaign_v2_proj first (proj imports them at :47) | BOUNDARY-ONLY | M | PARKED |
| 8 | Delete campaign_v2_proj_b2.py (rejected pivot, lessons recorded) + optionally RGP code (_rgp_step ~90 lines, wgp_rgp fields, notes_relaxed_projection.md, gate section 7 — they go together or not at all) | needs approval | S | PARKED |
| 9 | Lift b1 seed-lesson docstring → THEORY.md §6d | SAFE-NOW | S | ★DONE 2026-09-01 |

## Task classification (validate_c325.py, ids frozen)

- LIVE (~14): 0, 1, 2, 38, 39, 42, 43, 45, 47, 50, 51, 52, 53.
- CLOSED-KEEP (~30): 3-9, 10-13, 14-18, 19-22, 23-26, 41, 44, 46, 48, 49.
- DEAD: 27-36 (_GFR_RUNGS ladder; verdict = 4-tile/1024-cell bound; the
  reusable diagnostic is gpu_probe.py).

## Keep-live tools (never archive)

best_designs.py (data) · fit_c_field.py (re-fit at every numerics change) ·
fsp_width.py (HANDOFF recovery tool) · export_gds.py (deliverable) ·
gpu_probe.py (CLAUDE.md §5) · gates/ (all six + run_all_gates.py) ·
COMB_HANDOFF.md must keep its path (r=80 provenance citations).

## Live-lane import chains (why BOUNDARY-ONLY is strict)

campaign_v2_proj.py ← imported by d1, d1u, both gate files, 7 validate
tasks; it imports campaign_v2_uniform (:47). export_gds.py imports
campaign_v2_proj_best. Every Athena partition REQUEUEs — a requeued task
re-imports the deployed tree at any moment, so no deploy of moved/renamed
modules while 139225/139226 live.
