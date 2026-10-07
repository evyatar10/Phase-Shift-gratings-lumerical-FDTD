"""Single-writer forward budget across recentering, interruptions and resumes.

One unit is a geometry forward-evaluation batch, not every engine solve in
that batch (the width twin can add jobs). Reserve before queuing forward jobs.
Never reserve for an adjoint or a cached gradient. Reservations are never
refunded. Atomic state/history
use persistence.py; matching budget history supplies a conservative high-water
count if publication was interrupted after its snapshot was flushed.
"""
import hashlib
import json
import math
from pathlib import Path
import warnings

from runners.lumopt2_design import persistence as io


class BudgetExhausted(Exception):
    """Terminal campaign outcome, distinct from recentering or convergence."""


class BudgetStateError(RuntimeError):
    """Preserve the ledger and inspect its history before recovery."""


def find_budget_exhausted(error):
    """Find the terminal outcome even when vendor code wraps its exception."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BudgetExhausted):
            return current
        pending.extend(e for e in (current.__cause__, current.__context__)
                       if e is not None)
    return None


class ForwardEvaluationBudget:
    def __init__(self, out_dir, label, max_feval):
        if not isinstance(label, str) or not label or any(c in label for c in '/\\'):
            raise ValueError('budget label must be a filename stem')
        if type(max_feval) is not int or max_feval < 0:
            raise ValueError('max_feval must be a nonnegative integer')
        self.label, self.max_feval = label, max_feval
        self.path = Path(out_dir)/f'{label}_forward_budget.json'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _validate(self, state):
        if not isinstance(state, dict) or type(state.get('schema')) is not int or \
                state['schema'] != 1:
            raise BudgetStateError('invalid forward budget schema')
        if state.get('label') != self.label or not isinstance(state.get('ledger_id'), str):
            raise BudgetStateError('forward budget identity mismatch')
        if not state['ledger_id']:
            raise BudgetStateError('missing forward budget identity')
        for key in ('limit', 'used', 'bootstrap'):
            if type(state.get(key)) is not int or state[key] < 0:
                raise BudgetStateError(f'invalid forward budget {key}')
        if state['used'] < state['bootstrap']:
            raise BudgetStateError('forward budget consumption regressed below bootstrap')
        if not isinstance(state.get('legacy'), dict):
            raise BudgetStateError('missing forward budget bootstrap provenance')
        if not (state.get('last_reservation_id') is None
                or isinstance(state['last_reservation_id'], str)):
            raise BudgetStateError('invalid forward reservation identity')
        if state['used'] > state['bootstrap'] and not state.get('last_reservation_id'):
            raise BudgetStateError('consumed forward budget is missing its reservation ID')
        try:
            json.dumps(state, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise BudgetStateError('nonfinite or invalid forward budget state') from exc
        return state

    def _read(self, path):
        try:
            return self._validate(json.loads(path.read_text(encoding='utf8')))
        except (OSError, ValueError, TypeError) as exc:
            raise BudgetStateError(f'Cannot resume forward budget {path}; preserve history') from exc

    def _bootstrap(self):
        log = self.path.with_name(f'{self.label}_evals.jsonl')
        recorded, partial, digest = 0, 0, None
        if log.exists():
            rows = io.read_rows(log)
            for row in rows:
                fom = row.get('fom')
                if not isinstance(row.get('params'), list) or not row['params'] or \
                        isinstance(fom, bool) or not isinstance(fom, (float, int)) or \
                        not math.isfinite(fom):
                    raise BudgetStateError('legacy callback log contains a non-evaluation row')
            raw = log.read_bytes()
            recorded = len(rows)
            if raw.strip():
                try:
                    json.loads(next(line for line in reversed(raw.splitlines()) if line.strip()))
                except (ValueError, UnicodeError):
                    partial = 1  # Forward may have completed before callback publication.
            digest = hashlib.sha256(raw).hexdigest()
            warnings.warn('Forward budget bootstrapped from callback rows; old failed '
                          'unlogged forwards cannot be reconstructed', RuntimeWarning,
                          stacklevel=3)
        return dict(schema=1, label=self.label, ledger_id=io.new_id(),
                    limit=self.max_feval, used=recorded+partial,
                    bootstrap=recorded+partial, last_reservation_id=None,
                    legacy=dict(log=log.name, complete_rows=recorded,
                                interrupted_tail_count=partial, sha256=digest))

    def _load(self):
        states = [self._read(self.path)] if self.path.exists() else []
        history = self.path.parent/'state_history'
        states.extend(self._read(p) for p in history.glob(self.path.stem+'_*.json'))
        if states:
            base = states[0]
            for state in states:
                if any(state[k] != base[k] for k in ('ledger_id', 'bootstrap', 'legacy')):
                    raise BudgetStateError('conflicting forward budget history identities')
            highest = max(states, key=lambda s: s['used'])
            if any(s['last_reservation_id'] != highest['last_reservation_id']
                   for s in states if s['used'] == highest['used']):
                raise BudgetStateError('conflicting reservations; budget requires one writer')
            state = dict(highest, limit=max(s['limit'] for s in states))
            if self.max_feval < state['limit']:
                raise BudgetStateError(f'stale max_feval={self.max_feval}; ledger limit is '
                                       f"{state['limit']}; no implicit reset or reduction")
            state['limit'] = self.max_feval  # A changed, larger configured total is explicit.
            if not self.path.exists() or state != states[0]:
                io.save_state(self.path, self._validate(state))
        else:
            state = self._bootstrap()
            io.save_state(self.path, self._validate(state))
        self.state = state
        return state

    @property
    def used(self):
        return self._load()['used']

    def reserve(self):
        """Durably spend one forward attempt before starting it; return its ID."""
        state = self._load()
        if state['used'] >= state['limit']:
            raise BudgetExhausted(f"{self.label}: forward budget exhausted "
                                  f"({state['used']}/{state['limit']}); extend max_feval "
                                  'explicitly to permit additional attempts')
        reservation_id = io.new_id()
        updated = dict(state, used=state['used']+1,
                       last_reservation_id=reservation_id)
        io.save_state(self.path, self._validate(updated))
        self.state = updated
        return reservation_id
