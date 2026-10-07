# Budget-aware manager / worker / Claude workflow — 2026-10-07

The requested reusable skill is installed at
`C:/Users/evyat/.codex/skills/budget-aware-research/SKILL.md`.
Astra manages focused decisions, synthesis and review. Authorized Sol workers
own bounded inspections, patches, tests and integration execution, with one
owner per file/patch. Claude supplies a text-only second opinion through the
already tested `ask-claude` skill. No project source, AGENTS instructions,
server state, model configuration or account authentication was changed by
this work.

The role split reflects the user's chosen workflow. Official OpenAI
documentation describes Astra as suited to demanding reasoning and research,
and GPT-6.1 Sol as offering comparable capability at lower cost; it does not
establish independent subscription pools or free Sol work. See
[Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) and
[Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
Plan usage depends on model, task, reasoning, tools and caching; prompt length
alone is not a reliable allowance estimate. Actual dashboard resets take
precedence over expected dates. See
[official usage guidance](https://learn.chatgpt.com/docs/pricing).

## Measured budget, not token estimates

A direct `get_usage_limits` snapshot during this work returned one
account-wide `codex` bucket, primary `usedPercent=82`,
`windowDurationMins=10080`, `resetsAt=1791791049`. Derived remaining is 18%.
The measured reset timestamp converts to 2026-10-12 07:44:09 UTC, or
2026-10-12 10:44:09 Asia/Jerusalem. Secondary and individual/model limits were
null; no Astra-specific remaining allowance was exposed. Ordinary usage was
allowed. Paid balance was zero; one reset credit was available but was not
redeemed and is not authorization to redeem it. Account and credit identifiers
are omitted.

The skill's <=20%, <=10% and <=5% thresholds are explicit workflow heuristics:
checkpoint/conserve, reserve Codex for integration and verification while
consulting authorized Claude, and prioritize continuity before depletion.
They are not provider limits. Unknown values remain unknown. No token budget,
number of calls left, tomorrow reset, automatic purchase or automatic reset
is inferred. Claude's remaining subscription allowance is also unknown.

The reusable `scripts/summarize_usage.py` accepts raw or MCP-wrapped usage
JSON, prefers a populated bucket map over the legacy view, retains each actual
bucket/window/reset, computes clamped remaining values, and preserves unknown
windows. It outputs no account or credit identifiers. Explicit provider blocks
remain a separate boolean that overrides an optimistic heuristic band; the
minimum across reported windows does not identify model applicability.
It makes no network calls or state changes.

```powershell
python -X utf8 "C:/Users/evyat/.codex/skills/budget-aware-research/scripts/summarize_usage.py" "C:/path/to/usage_snapshot.json"
```

## Exact Claude Opus 5.5 test

The installed Claude Code 2.1.289 native executable contains the exact model
identifier `claude-opus-5-5`. One bounded consultation used that identifier
without substituting an alias or default. The call retained the existing
helper's empty tool set, empty strict MCP configuration, safe/restricted
mode, no hooks/plugins/project discovery, no session persistence, fresh scratch
directory, 180-second deadline and $0.50 CLI budget cap.

Measured result: helper exit 0, CLI exit 0, success response, one model turn,
no permission denials, and `modelUsage` exactly `claude-opus-5-5`.
Reported duration was 11552 ms and CLI-estimated usage cost $0.030236; neither
figure establishes subscription charge or remaining Claude allowance. The
response was 437 whitespace-separated words despite a requested 300-word
maximum; the call was not repeated solely to shorten it.

The supplied question asked for independent acceptance/validation advice
under known T_out versus modal-S normalization differences, unchanged
calibration gates, measured non-growing spatial FWHM, approximate wavelength
gradient and bounded smoke authorization. Claude recommended T_out peak3
improvement with measured width and trust-region checks, separate modal-S
reporting, and preserving the existing rank/sign/10% calibration gate. It
identified unspecified modal-transmission acceptance policy, width reference/
noise tolerance and trust-ratio thresholds. These are questions about the
supplied brief, not proof that current code or the user has left them open.
Root must reconcile them with actual instructions and code. The review is
unvalidated input and does not pass any scientific or hardware gate.

Prompt:
`C:/Users/evyat/AppData/Local/Temp/budget_workflow_lj4gl_ho/opus_review_prompt.txt`

Exact command/result evidence:
`C:/Users/evyat/AppData/Local/Temp/ask_claude_zudwptuk/command.json`
and `C:/Users/evyat/AppData/Local/Temp/ask_claude_zudwptuk/response.json`.

## Validation and continuity

Skill frontmatter/scaffold validation passed with exit 0. Ten offline utility
tests passed with exit 0: actual schema/reset conversion, missing values,
map precedence, legacy fallback, MCP wrapper, tool error, nonfinite/type/bound
handling, exact threshold boundaries, explicit usage block and multiple
buckets without invented model scope. The CLI was also exercised on a
sanitized copy of the measured fields. Tests make no provider or server calls.

Tests and sanitized snapshot are preserved under:
`C:/Users/evyat/AppData/Local/Temp/budget_workflow_lj4gl_ho/`.

The protocol requires bounded fresh prompts or small history forks, one patch
owner and relevant tests. It retains all calibration, offline, hidden-CAD,
hardware-smoke and measured acceptance gates. Budget pressure gives no
permission for scientific shortcuts or expanded dispatch scope.

Before depletion or a long wait, checkpoint current authorization, results,
job/code identities and next actions, then verify the existing supported
automation's enabled state, cadence, target and execution evidence. Preserve
existing schedules rather than creating duplicates. This skill does not
create an automation or guarantee a wakeup; that belongs to the parent task's
authorized scheduler work. A shell watcher alone does not establish future
agent execution. Active cluster jobs remain intact.

## Root integration

Root added the portable preferences to `C:/Users/evyat/.codex/AGENTS.md`
and the project's AGENTS.md. The following task context loaded the global
preferences together with project instructions, confirming discovery here.
The usage-summary regression tests are now retained inside the skill as
`scripts/test_budget_summary.py` and all ten pass there. Root also ran the
utility on `readiness_evidence_20261007/usage_snapshot_0106.json` successfully.

The existing `s2-overnight-inverse-design-readiness` heartbeat was updated in
place and the native tool confirmed ACTIVE at a 30-minute cadence. Its prompt
includes budget routing and the readiness gates; no duplicate was created.
This confirms scheduling configuration, not guaranteed future execution.
Global instruction discovery follows the official
[AGENTS.md guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

At 01:19 Israel time, a fresh stage-boundary query reported15% remaining in
the same weekly bucket, with ordinary usage allowed and the same reset time.
This is an account-wide observation, not a per-task cost measurement. The
tested candidate is checkpointed and further model work should remain focused
on incoming calibration evidence and its necessary validation.
