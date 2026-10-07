# Local integration of the tested restart/modal candidate

The consolidated candidate is integrated locally. **All thirteen offline
suites pass from the actual integrated tree, aggregate exit0.** No calibration
was installed, no S1/S2 opt-in flags were activated, and no CAD, solve, upload,
deployment, staging, commit or branch operation occurred.

The recovery chain was terminal before integration: all FD legs and
170752_0/1 completed, while 170754_2 failed its scientific calibration gate.
The parent owns that diagnosis. Hardware validation remains blocked; an
offline pass does not override the failed fit.

## Scope and preservation

Verified every manifest baseline hash/new-file absence, candidate-file hash,
seven shared-dependency hashes, consolidated patch hash, candidate archive
integrity/content and the original frozen-recovery archive checksum before
writing source. No mismatches occurred. Only
`docs/te_modal_guard_candidate_20261007.patch` was applied, exactly once;
the old restart patch and temporary delta were not applied.

The seven changed paths are the engine, projection/v3/persistence gates,
aggregate runner and new campaign-budget/modal gates. Shared helpers remain
byte-identical to the reviewed manifest. S1/S2 configuration bytes match the
preserved candidate archive. Defaults remain `wgp_modal_guard=False` and
`best_metric='fom'`; S2 `ADJ_FIX_PORT` and smoke calibration SHA remain None.

The frozen archive remains intact. An additional pre-integration snapshot is
preserved as `integration_before_sources.zip` in the evidence directory.
Six integrated files match candidate source after normalizing line endings;
actual hashes differ where Git applied Windows line-ending conversion.

## Integration defect found and repaired

The first integrated aggregate passed twelve suites, then the modal gate failed
to import `runners`. The isolated bootstrap had supplied ROOT on `sys.path`,
masking the direct-script entrypoint defect. Added only `import sys` and the
standard ROOT `sys.path` insertion to `gate_modal_guard.py`. No test assertions
or physics changed. The failed log/record remain preserved, and the second
complete integrated aggregate passed all thirteen suites, including ten modal
guard tests. Its exact child commands and inferred exit0 results are recorded;
the aggregate fails immediately if any child returns nonzero.

Command from ROOT:

```powershell
python -X utf8 runners/lumopt2_design/gates/run_all_gates.py
```

It was executed through the evidence capture script using the installed
Python executable; the full command is in the gate record below.

## Exact integrated identities

| File under `runners/lumopt2_design/` | SHA256 |
|---|---|
| `lumopt2_design.py` | `53108bbd368dd699179ab65029628aa4c4b2d8ad687777f88b82f404a0952569` |
| `gates/gate_projection_local.py` | `36c6fe8b520b72882162e8142cf1d4e144bf990c078d2cf0409ef859fdab5cf4` |
| `gates/gate_v3_local.py` | `1d76af190376fcd1d7157ff9ab3fb67d50b31219d14b33b662402b476a362ac2` |
| `gates/gate_persistence.py` | `9abef8e0da93a62285e8750fee465b44a958d408d0d95fa2473cc49be8eefd25` |
| `gates/run_all_gates.py` | `919f620de875a9240b34c4b57763eeca20b1dd131ce5a6286775d1fc5e24310b` |
| `gates/gate_campaign_budget.py` | `9f52ead7cb2561ee1dbe23b42cb86e209e03648cbf3416e296446f1cf94beb1e` |
| `gates/gate_modal_guard.py` | `5c6cb5a538ba23744e46d76573fc8d1840e15e4c746ebbf69ec6205a298d56d9` |

Use these integrated byte hashes for later source identity, rather than the
temporary candidate's byte hashes. The original manifest and candidate archive
remain historical evidence, not a second patch to apply.

Evidence directory:
`docs/readiness_evidence_20261007/integration_20261007_705c8ab2/`.

- [Prechecks and apply-once record](readiness_evidence_20261007/integration_20261007_705c8ab2/integration_apply_record.json).
- [Successful integrated gate record](readiness_evidence_20261007/integration_20261007_705c8ab2/integration_gate_record_attempt2.json).
- [Complete successful log](readiness_evidence_20261007/integration_20261007_705c8ab2/integrated_all13_attempt2.log).
- [Actual integrated identity and shared-helper verification](readiness_evidence_20261007/integration_20261007_705c8ab2/integration_final_identity.json).
- [Validated integrated source archive](readiness_evidence_20261007/integration_20261007_705c8ab2/integrated_validated_sources.zip).

Integrated archive integrity checked; SHA256
`925644733de2449d9c19c8b12a9f274562e92be1f040efc32fb0a76e1944ece4`.
The evidence directory also preserves the failed first log and the two-line
standalone-gate repair patch. Ownership is released to the parent for review
and its separately authorized commit; scientific qualification and hardware
work remain pending.
