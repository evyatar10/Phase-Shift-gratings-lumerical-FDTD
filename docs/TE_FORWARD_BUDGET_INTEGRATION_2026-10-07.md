# Global forward-evaluation budget helper

`runners/lumopt2_design/evaluation_budget.py` implements a single-writer budget
for one campaign label in one output directory. The helper is now wired to
the isolated restart candidate recorded in `te_restart_candidate_location.json`;
the production engine remains frozen. Its standalone offline gate is
`python -X utf8 runners/lumopt2_design/gates/gate_evaluation_budget.py`.

## Meaning of a budget unit

One unit is a **geometry forward-evaluation batch**, the project's
`_queue_forward_jobs()` invocation. It is the `max_feval` quantity, not a total
GPU-solve allowance. A width-adjoint twin or multiple configuration forward
files can add engine jobs within that batch. Adjoint jobs and gradient calls
that reuse a cached forward are not new forward-evaluation batches. Vendor
retries of jobs already inside a batch are not additional units. Budgeting
physical engine attempts separately would require a lower-level runner policy.

Reserve before any forward jobs are queued. A failed/interrupted batch remains
spent, including interruptions before the solver or callback starts. No refund
or automatic reset exists. An explicit increase in configured `max_feval`
increases the total limit; the already consumed amount remains. A lower stale
configuration fails rather than silently changing the ledger.

## Engine integration

Create one ledger in `run_campaign`, before the recenter/width-trip attempt loop:

```python
from runners.lumopt2_design.evaluation_budget import (
    ForwardEvaluationBudget, BudgetExhausted, find_budget_exhausted)

budget = ForwardEvaluationBudget(out_dir, spec.label, spec.max_feval)
```

After each `make_project` call, attach the same ledger to that newly created
project. The installed Project API was inspected: `compute_fom` calls
`run_forward`, whereas the uncached concurrent-adjoint `compute_gradient` path
calls `_queue_forward_jobs` directly. Consequently wrapping only `compute_fom`
or only `run_forward` misses an actual path. The shared queue hook covers both:

```python
queue_forward = project._queue_forward_jobs

def queue_forward_with_budget(_queue=queue_forward):
    reservation_id = budget.reserve()
    project._forward_reservation_id = reservation_id
    return _queue()

project._queue_forward_jobs = queue_forward_with_budget
```

Attach once per new project, including each automatic recenter retry. Do not
attach to the width/port adjoint queue. The instance-private method dependency
should be verified after any lumopt2 upgrade. The current signature takes no
arguments; a signature change must fail clearly, not bypass the guard.

In the campaign exception handler, recognize exhaustion **before** recenter,
width restoration, or generic error interpretation:

```python
except BudgetExhausted:
    raise
except (RecenterNeeded, WidthTrip, RuntimeError) as error:
    terminal = find_budget_exhausted(error)
    if terminal is not None:
        raise terminal from error
    # Existing guard-chain processing follows.
```

The explicit search walks both exception cause and context because installed
vendor optimization code can wrap exceptions. A caller may report the latest
eligible design when catching `BudgetExhausted`, but must label the termination
as budget exhaustion, never convergence. The optimizer's per-invocation
`max_feval` may remain as a secondary limit; it does not replace this ledger.

Add the standalone gate to the aggregate gate runner after engine integration,
and add driver tests for recenter retry and cold resume with the real hook.
The candidate aggregate now includes width-contract, standalone-budget and
real-campaign-budget gates, for twelve suites. All twelve passed through the
candidate `run_gate.py` with candidate gate files preferred and workspace gates
used otherwise. Exact paths, hashes, exit codes and logs are recorded in
`readiness_evidence_20261007/budget_integration_gate_results.json`. Candidate
engine integration and the persistence fixture were edited; production engine
and existing persistence helper were preserved.

## Persistence and historical bootstrap

The active state is `<label>_forward_budget.json`. Existing
`persistence.save_state` writes immutable versions under `state_history`, fsyncs
them and atomically replaces the active file. Budget reads validate their own
schema, numerical fields and immutable identity. Only files matching this
specific budget-state stem participate; unrelated optimizer history is ignored.

For matching snapshots, consumption and configured limit are monotonic:
the reader conservatively takes their maximum. This deliberately counts a
reservation whose history snapshot was flushed but whose active-file
publication failed. Matching immutable identity/bootstrap provenance and
nonconflicting reservation IDs are required. Corrupt active state or history
fails closed. This is a narrow high-water rule for an append-only resource
count, not a generic repair of optimizer state.

When no budget history exists, the helper bootstraps every complete evaluation
row in `<label>_evals.jsonl`, preserving all original bytes. Repeated `eval=0`
rows are separate evaluations, not duplicates to discard. A tolerated malformed
unterminated callback tail counts as one additional uncertain attempt. Missing
logs mean zero known historical evaluations. Old failed attempts that never
logged anything cannot be reconstructed; this limitation is warned and remains
in the bootstrap provenance. A historical consumed count already beyond a new
cap is valid evidence but allows no new forward.

Do not delete or rename the ledger/history to refill a campaign. Do not share
one output directory/label between concurrent writers. Additional history is
intentionally retained, including failed publication snapshots and temp files.

Offline tests cover exhaustion across retries/resume, reserved interrupted
forwards, interrupted atomic publication, absent active state with own valid
history, corrupt/invalid states, history identity conflicts, explicit extension
and stale config, legacy repeated rows/partial tail/overspend, cached-adjoint
exclusion, and wrapped terminal exceptions. These are resource-control tests;
they do not establish FDTD physics or hardware integration.
