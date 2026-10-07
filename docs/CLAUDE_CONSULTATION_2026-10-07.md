# Local Claude consultation skill — 2026-10-07

The user authorized consulting local Claude if stuck during the overnight
workflow. A bounded text-only consultation was tested successfully. No project
code, server state, authentication, or configuration was intentionally changed
by the consultation. This test did not run CAD or FDTD and does not validate
the TE calibration or optimizer physics.

## Availability measured this session

`claude` is absent from the current PowerShell PATH. The installed native CLI is:

`C:/Users/evyat/AppData/Local/Packages/Claude_pzs8sxrjxfjjc/LocalCache/Roaming/Claude/claude-code/2.1.289/e1f0154146bb/claude.exe`

Bounded `--version` and `--help` commands returned exit 0. Version output:
`2.1.289 (Claude Code)`. `auth status --help` confirmed the status syntax;
`auth status --json` returned exit 0 and reported `loggedIn=true`,
`authMethod=claude.ai`, `apiProvider=firstParty`, `subscriptionType=max`.
Only these non-secret status fields were displayed. No credential file was
opened and no login, token, install, update, or configuration command was run.

## Installed reusable skill

- Skill: `C:/Users/evyat/.codex/skills/ask-claude/SKILL.md`
- Helper: `C:/Users/evyat/.codex/skills/ask-claude/scripts/ask_claude.py`

The existing `C:/Users/evyat/.agents/skills/ask-gpt/` was inspected as a
reference and preserved. Its repository-reading agent loop was not copied:
this Claude helper has no tools and receives explicit excerpts instead.

Invocation:

```powershell
python -X utf8 "C:/Users/evyat/.codex/skills/ask-claude/scripts/ask_claude.py" --prompt-file "C:/path/to/focused_prompt.txt"
```

The helper resolves the native CLI from PATH or known desktop package paths
and checks its supported safety flags before each consultation. It creates
a fresh preserved scratch directory and supplies the prompt on stdin using
`subprocess.run` with an argument list, without shell interpolation.

The actual call uses `--print --output-format json --safe-mode --restricted
--tools "" --strict-mcp-config --mcp-config '{"mcpServers":{}}'
--disable-slash-commands --no-chrome --permission-mode dontAsk
--permission-prompts none --no-session-persistence --max-budget-usd 0.5
--effort low`, plus a short text-review system prompt. The user's default
Claude model is retained unless a specific model is requested. All these
flags were verified in this executable's help output.

Defaults: 180-second wall-clock limit and $0.50 CLI budget cap; allowed
overrides are 10–300 seconds and $0.01–$2.00. No automatic retries, resume,
write mode, permission bypass, external MCPs, or read-capable tools exist.
The helper rejects `BENCH_NO_CROSS_AI` environments.

## Small live test

The supplied fictional example had evaluation A, FOM 0.80, with committed
acceptance A; evaluation B, FOM 0.90, lacked acceptance after a crash. The
question explicitly stipulated that matching acceptance IDs determine
eligibility and requested at most three sentences. It supplied no project
files or secrets.

Measured outcome: helper process exit 0, Claude CLI exit 0,
`subtype=success`, `is_error=false`, `num_turns=1`, no permission denials.
Claude selected A and explained that B lacked the required commit. This
shows a working authenticated text response, not a new code audit or an
independent derivation of the persistence design.

Response metadata reported model `claude-sonnet-5-5`, duration 1928 ms and
`total_cost_usd=0.004`. The dollar figure is the CLI's reported usage estimate,
not a verified subscription charge or remaining quota.

Preserved live evidence:

`C:/Users/evyat/AppData/Local/Temp/ask_claude_iiwnk3ar/`

This directory contains `prompt.txt`, `command.json`, `response.json` and
`cli_stderr.txt`. The helper prints its final answer on stdout and evidence
paths/exit status on stderr.

## Validation and limits

`skill-creator/scripts/quick_validate.py` returned exit 0: `Skill is valid!`.
Five offline helper tests passed with exit 0, covering the safe command and
isolated working directory, explicit timeout/partial-output preservation,
unsuccessful API response, missing safety flag refusal before consultation,
and benchmark refusal before any CLI call. Tests mock the subprocess
boundary and use `offline.invalid`; no guard was tested against a live
server. Test script:

`C:/Users/evyat/AppData/Local/Temp/ask_claude_gate_lzpq88fw/test_helper.py`

This is an application-level no-tool consultation, not an OS sandbox. The
Claude executable still contacts its model service and may maintain its own
operational metadata. Prompt text leaves the machine for that service and
consumes Claude quota. Safe mode disables custom hooks/plugins/project
discovery while retaining normal authentication. The helper deliberately
does not use `--bare`, whose help says it disables OAuth authentication.

Claude cannot follow file paths or inspect the repository under these flags;
the caller must supply minimal relevant excerpts. Advice is input only:
verify claims that change decisions against code/data and retain the current
authorization limits. A successful historical call does not promise future
service availability; report failures and preserve evidence without relaxing
restrictions. The native binary may move after desktop updates; the helper's
discovery and safety-capability check fail visibly if availability changes.
