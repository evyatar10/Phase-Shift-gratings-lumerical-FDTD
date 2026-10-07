# Claude skill transfer to Codex

## October 7: self-contained project runbooks

All 13 project Codex entrypoints now link to a package-local
`references/claude-original.md`, copied byte-for-byte from the corresponding
Claude runbook. Reading the detailed runbooks no longer requires the
`.claude/skills/` tree. Each entrypoint identifies the copy as historical
evidence and gives current user instructions, AGENTS.md, the Codex adaptation
and live code precedence. Historical repository paths are interpreted from
the repository root; memory and installed runtime references remain external.
No memories were imported.

The work-alone package also includes a byte-preserved
`references/campaign_monitor_template.sh`; its existing top-level companion
remains unchanged. The template requires inspection before use. No historical
script was executed during this transfer.

All 13 entrypoints passed the skill-creator validator, their Markdown file
links and shared context links resolve, and every local runbook and companion
matches its Claude source. Existing Codex adaptation text is preserved apart
from the link replacement and added provenance notes. All original source
hashes are unchanged.

The 18 personal packages were checked again against their complete source
trees: all 306 source files are present at their Codex destinations, including
288 support files. Eighteen files intentionally differ between Claude and
Codex (14 entrypoints and four Google Workspace support files); all existing
Codex files were preserved. Completeness and integrity checks do not establish
that historical tools or recipes work in the current runtime.

New evidence:

- [Self-contained inventory](readiness_evidence_20261007/claude_skill_selfcontained_inventory_20261007.json): all project runbook/companion hashes, before/after entrypoint hashes, and every personal source/destination file hash.
- [Validation results](readiness_evidence_20261007/claude_skill_selfcontained_validation_20261007.json): 13 validator exits, resolved links and source/personal preservation checks.
- Pre-edit archive: `C:/Users/evyat/AppData/Local/Temp/skills_selfcontained_transfer_24_y6isc/before_transfer.zip`, SHA256 `49cf057d25f411b989aeb8f68759df69f980f2654b7e423b1db9a7954fcb6104`. It preserves the project Claude/Codex skill trees and this transfer document as they stood before these changes.

## Earlier October 7 adaptation record

Verified access to project `.claude/skills/` and
`C:/Users/evyat/.claude/skills/`. The project Claude memory directory is also
readable at `C:/Users/evyat/.claude/projects/c--Users-evyat-Lumerical-phase-shift-grating-FTDT-codes/memory/`.

Found13 project skills and18 active personal skill packages. All31 already
had Codex-discoverable copies. The project copies were incomplete adaptations:
some still assumed Claude hooks/tools, obsolete physics and old server flags.
Those13 copies were adapted in `.agents/skills/`; the originals remain intact.

| Project skill | Main correction |
|---|---|
| add-study | Current per-study manifest/discovery contract and shared-code freeze |
| athena-preflight | Actual seat accounting, quota preservation, memory/QOS checks |
| athena-status | Queue/accounting/output evidence; explicit unreachable states |
| check-result | Current envelope width and modal versus T_out definitions |
| dispatch-study | Current lumopt2 flag, resources, hashes and authorization scope |
| fetch-results | Preserve partial/conflicting files; verified targeted transfers |
| lock-target | Current predictor over obsolete length extrapolation; matched numerics |
| lumopt2-design | TE v3,296 slots, separate seed anchors and calibrated gradients |
| predict-q3db | Current predictor/tool routing and model validity limits |
| safe-compact | Repository handoff instead of assumed Claude memory/TodoWrite APIs |
| stop-runs | Explicit user scope; no assumed automatic scancel permission prompt |
| update-container | Preserved staging/images, explicit cluster scope, no auto-cleanup |
| work-alone | Continuous authorized work; honest watcher/wakeup limits |

Shared runtime guidance is `.agents/skills/project-context.md`. Each entrypoint
links to its preserved original runbook for detailed historical evidence.
No automatic Claude model routing was carried over. Model orchestration was
discussed separately; no model setting or delegation policy was changed.

The18 personal packages and all their supporting files already exist under
`C:/Users/evyat/.agents/skills/`, including ask-gpt, research-deck,
study-narrator and the synced browser, research, document and presentation
skills. They were inventoried and preserved without overwriting their existing
Codex adaptations. Two entries in Claude's `.trash` were excluded deliberately.
Skill files alone do not supply Claude desktop tools or connector access;
actual tools available in the current Codex session remain authoritative.

Validation: all13 project skills passed the skill validator; all entrypoint
file links resolved; original Claude files were byte-compared against the
pre-edit archive and are unchanged. This checks packaging/integrity, not every
historical recipe against live servers. No server changes were made for this
transfer, and no files were deleted.

Evidence:
- `readiness_evidence_20261007/claude_skill_inventory.json`: source/destination
  paths, hashes, actions and companion-file completeness.
- `readiness_evidence_20261007/skill_validation.json`: validation exits.
- `readiness_evidence_20261007/skills_before_transfer_20261007.zip`: original
  project Claude/Codex skill trees; SHA256
  `2364aaaeaa8dcb2ad094f43453f2e8eea11ca1c5938e85cf7259959c96edfb93`.

Parent review also corrected two inherited entrypoint statements: stop-runs no
longer claims scancel supplies a Codex confirmation prompt; safe-compact allows
focused local commits when the user has authorized them. Both validators passed
again. Historical runbooks stay byte-preserved. The original transfer inventory
precedes these two documented refinements; final hashes are recorded in
`readiness_evidence_20261007/claude_skill_transfer_parent_refinements_1512.json`.
