# Opt-in measured-modal pilot guard — isolated candidate

The new candidate preserves the calibrated peak3(abs(T_out)) search objective
and its gradient, while allowing a pilot to require measured modal peak
transmission to improve or hold. This is a measured acceptance/selection guard,
not a derived modal-transmission adjoint or an optimality certificate.

No production engine, S1/S2 configuration, original tested candidate, server,
CAD scene or job was changed. Defaults remain `wgp_modal_guard=False` and
`best_metric='fom'`. No Git staging, commit or branch mutation occurred.

## Candidate behavior

For an explicitly configured pilot use:

```python
dataclasses.replace(pilot_spec,
                    wgp_modal_guard=True,
                    best_metric="modal_peak3")
```

- The guard compares each candidate's measured `modal_peak3` with the best
  accepted row inside the measured **hard spatial-FWHM band**, at zero modal
  slack. Carrier-FOM ascent cannot buy a modal decline. Carrier noise slack is
  not transferred to the modal observable.
- A lower-modal step may restore a physically infeasible width when the hard
  width violation decreases. An initially infeasible seed may be evaluated and
  restored. Infeasible rows never establish the feasible modal reference;
  restoration cannot lower an existing reference.
- The reference increases only after acceptance and gradient work complete.
  Rejected width/wavelength/mode trials and unclassified callback rows cannot
  raise it. Cold restart reconstructs it from the journal and committed
  `accepted_eval_ids`; a stale scalar `modal_best` in optstate cannot override
  the verified row history. The scalar is persisted for diagnostics.
- Missing/nonfinite live modal measurements fail before acceptance/gradient
  work. Selection excludes missing/nonfinite measurements, rows outside the
  hard width band, ineligible geometry identities and uncommitted evaluations.
  For opted-in modal selection/reference, legacy rows without committed IDs
  are excluded: legacy seed geometry must be remeasured and committed.
- `best_metric='modal_peak3'` ranks eligible rows by measured modal peak3.
  Fallback carrier FOM is never treated as a modal score; fallback supplies
  starting geometry only. A completed campaign still requires an eligible
  measured row.
- Final returned `fom` remains the selected row's actual carrier FOM, accompanied
  by its `modal_peak3`. The best JSON records paired `best_fom`,
  `best_modal_peak3`, `best_metric`, `best_eval_id` and matching parameters.
  Historical carrier-based callback snapshots remain diagnostics, not the
  certified modal selection.

The local wavelength trust bound, width/MAC checks, calibration requirements
and solve budget remain in force. Nothing activates these options in S1/S2.

## Verification

All **13 offline suites passed, exit 0**: the previous twelve candidate suites
plus `gate_modal_guard.py` (10 focused tests). Tests exercise the real
`run_projected` and final `run_campaign` paths with offline project doubles:
carrier-up/modal-down rejection, rejected high-modal poisoning, successive
small dips, committed feasible cold resume, stale-scalar disregard, infeasible
restoration, strict-width/committed-ID/modal ranking, missing/nonfinite metrics,
paired final reporting, and unchanged default selection.

The candidate aggregate now lists all thirteen suites. After the full suite
run, only its gate-list/documentation registration changed; AST list validation
and syntax checks passed without repeating numerical tests.

`git apply --check` passed for the **new consolidated patch against the current
ROOT baseline**, without applying it. An initial Windows CRLF patch-generation
attempt failed the check; that attempt remains preserved in the new temporary
candidate. The corrected patch uses LF. Some inherited candidate gate files
have CRLF/mixed endings; the text patch normalizes them to LF, while the archive
preserves exact tested file bytes. Rerun all thirteen suites from the actual
integrated tree before upload. These checks establish offline behavior only;
the authorized hardware smoke remains pending.

## Exact artifacts and integration rule

New temporary candidate:
`C:/Users/evyat/AppData/Local/Temp/te_modal_guard_candidate_20261007_52ed8b9e/`.
Use its `run_validation.py` for the recorded candidate bootstrap; it imports
the isolated engine and reads unchanged shared dependencies from ROOT.

- Manifest: [te_modal_guard_candidate_location.json](te_modal_guard_candidate_location.json).
- Consolidated patch: [te_modal_guard_candidate_20261007.patch](te_modal_guard_candidate_20261007.patch).
- Gate record: [modal_guard_gate_results_20261007_52ed8b9e.json](readiness_evidence_20261007/modal_guard_gate_results_20261007_52ed8b9e.json).
- Durable byte-preserving archive: [modal_guard_candidate_20261007_52ed8b9e.zip](readiness_evidence_20261007/modal_guard_candidate_20261007_52ed8b9e.zip).

**Apply only this new consolidated patch. Do not also apply the original
restart patch or the temporary modal delta.** It includes inherited persistence,
width, wavelength, acceptance and evaluation-budget fixes, plus this optional
guard/selector. Recheck the new manifest's baseline and shared-dependency hashes
after the frozen calibration dependencies finish and before integration.
The original patch, manifest and archive remain preserved.

SHA256 identities:

| Artifact | SHA256 |
|---|---|
| New candidate engine | `d11f34db9777dd1d9deba6ab19372238c046e8d705a5c9c8c4281702e88b9559` |
| New consolidated patch | `364c93ac245ecfa242313755d60c150a6a72019c9e5d0b097126236024f71cb7` |
| New durable archive | `82ea60faa70780021183be2b8c8c04296b4baca48585dcee34eda45e84b83187` |
| Preserved previous candidate engine | `2558954f23df132a57f6de259e2a60ee43bf62012ce87617913b288117a38587` |
| Unchanged frozen production engine | `8f09350583a53146091faf590c2c93e8d93ac01a46b48006890db62c72a295e5` |

The gate record contains every command, exit code, source hash and full log.
The archive holds seven candidate files, shared dependencies, snapshots,
bootstrap scripts, the consolidated patch, manifest and validation logs.
