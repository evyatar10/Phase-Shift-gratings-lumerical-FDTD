# N98 S2 pilot preparation — blocked skeleton, not hardware validation

Prepared offline using the final isolated modal candidate and live S2/
`validate_te._campaign_spec` definitions. No production/config/helper edits,
CAD, FDTD, upload, server action or Git operation occurred. The parent reported
8/12 calibration FD legs complete and 170750_9 active at 02:40; this preparation
does not poll, duplicate or alter that recovery.

## Exact intended pilot

| Setting | Prepared value |
|---|---|
| Device | TE, N98 per side, 60 free periods per side, bare/no surrounding posts |
| Seed | Exact live `campaign_te_s2.SEED`: 296 slots, first corrugation 0 nm, all shifts 0; not the perturbed calibration `te_point()` |
| Geometry/numerics | Pitch 491.06 nm, core height 350 nm, precise volume average, existing S2 bounds and mesh/domain settings |
| Scan | Center 1560.464 nm, 2 nm width, 501 frequency-uniform samples |
| Measured spatial-FWHM band | 19.24328–20.02872 µm, retained ±2% of S2's 19.636 µm anchor |
| Local wavelength trust | ±0.05 nm per step; cumulative drift allowed |
| Objective/selection | Calibrated peak3(abs(T_out)) carrier; `wgp_modal_guard=True`, `best_metric='modal_peak3'` |
| Derivatives | Fresh width row: `wgp_reuse_k=0`; existing port selector/moving-width chain |
| Bounds | `max_iter=2`, `max_feval=4`, intended `MAX_RESTARTS=0` |
| Identity | `lumopt2_te_s2_n98_pilot_preparation_20261007_2a7c49d0` |

Two loop evaluations mean the initial measurement and at most one measured
proposal, not two demonstrated improvements. Four is the persistent total of
forward batches, including uncached gradient forwards and interrupted reserved
attempts. It is not four adjoints or four allowed retries per restart.

## Fail-closed packet

Scratch directory:
`C:/Users/evyat/AppData/Local/Temp/te_n98_pilot_preparation_20261007_2a7c49d0/`.
`n98_pilot_packet.py` is `IS_HELPER=True`, with only task0 declared. It is outside
the production runner tree and is not advertised in Athena/IGUM menus.
No deployable study was registered.

Its calibration SHA is **None** and the recorded port coefficients are null.
The loader reuses the prepared N70 smoke's exact-hash/objective/rank/sign and
recomputed 10% fit/held-out checks, additionally requiring the retained S2 N98
family identity. No S1/softmax fallback exists. Even supplying a passing port
fit cannot unlock `main`: it explicitly raises preparation-only before any
CAD/solve. This avoids inventing a second field-approval schema.

The stored field factor `(0.983623,-0.088013)` remains marked **qualified**:
the recorded shift_1 residual is +23.8%, although the vector residual is 1.25%.
No test fixture or preparation status certifies this factor. Synthetic port
coefficients are used solely inside offline tests/geometry derivation, then
removed from the saved packet.

There is also a **calibration-to-seed transfer gap**. The recovered port fit
uses `validate_te.te_point(recover_te_s2.SPEC)`, with first corrugation lifted
to 5 nm, shifts1/30 set to 5 nm and cavity width increased by 10 nm. Its scan
is re-centered through `validate_te.GATE_LAM_NM[1]=1560.878859628509` nm.
The packet intentionally retains the unmodified campaign SEED and its
1560.464 nm scan center. A passing fit at that perturbed calibration point
does not prove gradients at the packet seed. Before launching, require local
directional qualification at the actual seed or explicitly review an
intentional re-centered starting geometry. Do not silently replace the seed
with the calibration point or treat their different centers as equivalent.

## Evidence collection after a reviewed implementation

Preserve every `_evals.jsonl` row, immutable `eval_id`, parameters, production
FWHM/profile path, modal peak3, T_out peak3/FOM, resonance, twin wavelength and
cW quality. Classify accepted rows through committed `accepted_eval_ids` and
ineligible geometry history; rejected/unclassified rows stay visible and
cannot become the certified best. Preserve `_proj.jsonl`, `_optstate.json`,
state history, forward budget, gradient NPZs, solver logs, FSP and companion HDF5.

Compare each actually evaluated parameter displacement from its accepted base,
including rejected trials. `v3_pred_rows[0]` gives the moving-width prediction;
alternatively reconstruct `(gW_eff+cW*gLambda)·dp` from the base gradient artifact
and recorded cW. **`dw_pred` alone is the fixed-wavelength row under v3**, so it
must not be compared as the total moving-resonance width prediction.
Use `gLambda·dp`/`dlam_pred_nm` for wavelength, and compare with measured changes.
Gradient NPZs contain p, gT, gW, gW_eff and gLam. Do not score the unevaluated
next proposal emitted after the final loop evaluation. Cold resumes repeat
integer eval numbers; match immutable IDs, parameters and provenance, not `it`
alone. Report absolute errors when changes are too small for useful ratios.

## Remaining qualification and launch work

Finish the existing S2 peak3 fit at unchanged per-class/held-out gates. Then
review/integrate the consolidated modal candidate, rerun its actual integrated
13 suites and inspect the corrected N70 hardware smoke. N70 plumbing evidence
does not establish N98 spatial-width or derivative accuracy.

Resolve the qualified field component and validate a held-out actual N98 step
direction, including carrier/modal directional response, gLambda and the total
moving-width prediction; include restoration if that branch will be relied on.
Central FD must not leave the seed's active corrugation/shift bounds: select an
appropriate feasible/interior qualification point or a justified one-sided
test, with measured signal and existing acceptance tolerances. This two-eval
pilot itself cannot establish full directional calibration or stationarity.
Do not relax a failed scientific gate to make the packet launchable.

After these gates, a reviewed production study must replace this skeleton,
retain task0 only, pass discovery/task-list and exact source-identity checks,
establish/reuse matching hidden N98 build evidence, and run Athena preflight.
Shared recovery sources stay frozen until dependent tasks finish. This packet
does not authorize a full campaign.

## Offline verification and preserved files

Six focused tests passed, **exit0**: missing calibration before CAD/solve;
passing port fit still cannot launch; exact seed/bounds/family/modal flags;
hash/objective/family/failed-fit rejection; persistent four-forward budget;
task membership and pinned candidate. No 13-suite rerun or broad review was done.

- [Exact spec and seed/bounds](readiness_evidence_20261007/n98_pilot_spec_20261007_2a7c49d0.json).
- [Commands, exit and hashes](readiness_evidence_20261007/n98_pilot_gate_results_20261007_2a7c49d0.json).
- [Preparation archive, integrity checked](readiness_evidence_20261007/n98_pilot_preparation_20261007_2a7c49d0.zip).

Packet helper SHA256: `faa2d328061f9278c1c4e76fa7465053a6fe2f1297028ee078ba8e097baa495e`.
Archive SHA256: `b35cf728d62f462a5da29327d48c10a0b06c3131830c4e30073fb7e13fe5f979`.
Seed JSON SHA256: `b1e0a9453806876f64fe43c145dc3239f8a0ba669cf25965d60b5ed71c4c2e7e`
(compact JSON list as described in `n98_pilot_preserve.py`).
Pinned modal engine: `d11f34db9777dd1d9deba6ab19372238c046e8d705a5c9c8c4281702e88b9559`.
