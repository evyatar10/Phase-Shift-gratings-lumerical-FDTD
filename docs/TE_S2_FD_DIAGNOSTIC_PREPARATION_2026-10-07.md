# Forward-only S2 nominal/FD-step diagnostic preparation

Prepared a new isolated-result runner and six-check offline gate. No CAD,
FDTD, server action, upload or Git operation was performed here. Existing
recovery sources, globals and results were not changed. The failed overall
10% gradient calibration gate remains failed; this diagnostic cannot mark
it passed.

## Exact study

Runner: [diagnose_te_s2_fdstep.py](../runners/lumopt2_design/diagnose_te_s2_fdstep.py).
It clones live `recover_te_s2.SPEC` with label
`lumopt2_te_s2_fdstep_diagnostic_20261007` and uses the original recovered
`validate_te.te_point()`, not the campaign seed. Family: TE N98/free60, bare,
491.06 nm pitch, 350 nm height, PVA, 1560.878859628509 nm scan center,
2 nm/501-point frequency grid and unchanged mesh/domain settings.

| Task | Point |
|---|---|
| 0 | Exact nominal recovered calibration point |
| 1 | That same point, corr30/index29 +0.5 nm |
| 2 | That same point, corr30/index29 −0.5 nm |

Each task runs one forward only. There is no optimizer, adjoint, gradient
assembly or calibrated-coefficient requirement. It copies the proven FD-only
setup pattern: lock the grid with `set_fwd`, disable only `optimization_dft`,
assert exact equality of x/y/z mesh arrays, save inside a unique attempt,
run forward and require a successful solver log plus companion HDF5. Ports
NPZ, FSP, HDF5 and every attempt are retained. `KEEP_H5=1` is required before
CAD/solve. Checkpoints are immutable and identity checked; completed tasks
resume without a new solve if their artifacts remain present.

Identity captures the integrated engine and relevant geometry/scoring sources,
the exact parameter vector, scan/family, task sign and reference provenance.
The old reference identity's source hash need not equal the new engine: the
purpose of task0 is to test nominal reproducibility after that integration.

## Nominal comparison and continuation gate

Reference is the old recovery export `task_00.json` containing `result.s2_gate`,
SHA256 `732ca5752ff2b0449a892ebafe7ba6a87a5ee1fce63bd929f8f7343bf50df13f`.
This pins the **raw remote file bytes**, now preserved locally as
[s2_reference_task00_remote_original.json](readiness_evidence_20261007/s2_reference_task00_remote_original.json).
The parent's raw fetch and local verification establish semantic JSON equality
with the watcher's pretty-reserialized copy, whose different byte hash is
`90b1b4a2423c4f37c8299611b5fb56478668c0d836a8f119da48be7a21ce37c2`.
The default initially pinned that watcher copy; pre-dispatch checking caught
the byte difference before upload/solve. The default is corrected to the remote
hash, and six checks/task-list generation were rerun, exit0. The exact raw
reference also passed the runner's hash/point/family check. Tests use explicit
hash overrides for their local fixtures, never assume reserialized bytes match.
The default runtime location is:

`config.BASE_SAVE_DIR/recover_te_s2/results/lumopt2_te_s2_cport_recovery_20261006/task_00.json`.

Optional `S2_FDSTEP_REFERENCE_CHECKPOINT` and
`S2_FDSTEP_REFERENCE_SHA256` allow an explicitly pinned checkpoint at another
stable path; no bulk old results need copying. Keep that provenance stable
between task0 and the later legs. A missing/mismatched reference fails before CAD.

Task0 records old/new softmax FOM, T_out peak3, modal peak3, modal resonance,
sampled modal peak, **spectral** FWHM, T_out at modal peak and source-port T_in.
It reports signed, absolute and relative differences, with old source/checkpoint
provenance. Its status is `REVIEW_REQUIRED_NO_AUTOMATIC_PHYSICS_PASS`; no guessed
reproducibility/noise threshold was added.

Task1/2 cannot execute until the study output directory contains
`center_reviewed.sha256` holding the exact SHA256 of its **new** `task_00.json`.
The new center's current source identity and FSP/HDF5/ports artifacts are also
checked. This is an explicit continuation-review gate, not inferred physics
approval. Later results retain that reviewed checkpoint hash and compare with
both the old and new center. Do not enable ±legs solely because task0 finished.

## Offline checks and discovery

[gate_s2_fdstep_diagnostic.py](../runners/lumopt2_design/gates/gate_s2_fdstep_diagnostic.py)
passed six tests, exit0: exact point/deltas/bounds/fresh identity/no global
mutation; missing-center blocking before CAD; fake forward/artifact preservation
and no-solve resume; exact review-marker/source identity and checkpoint
no-overwrite; reference-hash/mesh-change rejection; task membership/HDF5 guard.

Both Athena and IGUM explicit task-list generators returned exactly `0,1,2`,
exit0. `IS_HELPER=True` remains set, so this preparation is excluded from menu
discovery. Explicit `--lumopt2-design=<module>` imports SPEC/main/N_TASKS and
accepts the helper; no parser or registration change is required for an
explicit reviewed task0 launch.

Suggested flags **after parent review, required hidden-build evidence and
preflight**, with a compatible memory/QOS/walltime selected by the parent:

```bash
SBATCH_MEM=256G LUMOPT2_QOS=12h_4g LUMOPT2_TIME=12:00:00 \
bash athena/deploy_athena.sh \
--lumopt2-design=runners.lumopt2_design.diagnose_te_s2_fdstep \
--array-tasks=0 --max-concurrent=1 --keep-h5 --gpu=a100
```

The parent verified that 12h_4g supports the 256G request. This preparation
did not dispatch it.
Only task0 is intended initially; tasks1/2 remain behind the reviewed-center
marker. Hidden CAD and actual nominal agreement are pending. Failed calibration
continues to block optimizer smoke/pilot/campaign work.

Evidence: [final commands, exit codes and hashes](readiness_evidence_20261007/fdstep_preparation_20261007_6f402dc1/fdstep_preparation_checks_raw_reference.json).
Its directory preserves the complete six-check log and both generated task
lists/metadata. No full thirteen-suite rerun was needed for these new files.

The earlier watcher-hash check record and logs remain preserved as superseded
preparation evidence.

Runner SHA256: `3bf24e3ea47600f1c08f13803c7ab29bc6cf8b95c08fef63152168b1069ee361`.
Gate SHA256: `dfcec332d7dcfc7ed656866631a2d914f7d36628d11ede07a31f08c3ad56113b`.
Integrated engine remains `53108bbd368dd699179ab65029628aa4c4b2d8ad687777f88b82f404a0952569`.
