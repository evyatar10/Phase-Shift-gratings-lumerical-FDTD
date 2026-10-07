"""Offline forward-budget gates; preserved temp artifacts, no CAD or solver."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from runners.lumopt2_design import persistence as io
from runners.lumopt2_design.evaluation_budget import (
    BudgetExhausted, BudgetStateError, ForwardEvaluationBudget,
    find_budget_exhausted)


class BudgetChecks(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='te_forward_budget_gate_'))

    def ledger(self, limit=2):
        return ForwardEvaluationBudget(self.root, 'te_gate', limit)

    def test_recenter_and_cold_resume_share_total_budget(self):
        first = self.ledger()
        first.reserve()
        recentered = self.ledger()
        recentered.reserve()
        resumed = self.ledger()
        self.assertEqual(resumed.used, 2)
        with self.assertRaises(BudgetExhausted):
            resumed.reserve()
        self.assertEqual(resumed.used, 2)

    def test_interrupted_forward_reservation_is_not_refunded(self):
        token = self.ledger(1).reserve()
        self.assertTrue(token)
        # The process dies before its solver or callback; no completion API.
        with self.assertRaises(BudgetExhausted):
            self.ledger(1).reserve()

    def test_interrupted_atomic_publication_counts_flushed_reservation(self):
        ledger = self.ledger(1)
        with patch.object(io.os, 'replace', side_effect=OSError('interruption')):
            with self.assertRaises(OSError):
                ledger.reserve()
        current = json.loads(ledger.path.read_text())
        self.assertEqual(current['used'], 0)
        self.assertEqual(self.ledger(1).used, 1)
        with self.assertRaises(BudgetExhausted):
            ledger.reserve()

    def test_missing_active_after_initial_publication_uses_own_history(self):
        with patch.object(io.os, 'replace', side_effect=OSError('interruption')):
            with self.assertRaises(OSError):
                self.ledger()
        self.assertFalse((self.root/'te_gate_forward_budget.json').exists())
        self.assertEqual(self.ledger().used, 0)

    def test_corrupt_active_is_fatal_even_with_valid_history(self):
        ledger = self.ledger()
        for raw in ('{bad', '[]', '{"schema":1,"used":-1}'):
            ledger.path.write_text(raw)
            with self.assertRaises(BudgetStateError):
                self.ledger()
            self.assertEqual(ledger.path.read_text(), raw)

    def test_invalid_or_conflicting_history_fails_closed(self):
        ledger = self.ledger()
        history = self.root/'state_history'
        path = history/(ledger.path.stem+'_bad.json')
        path.write_text('{}')
        with self.assertRaises(BudgetStateError):
            self.ledger()
        other = dict(ledger.state, ledger_id='other')
        path.write_text(json.dumps(other))
        with self.assertRaises(BudgetStateError):
            self.ledger()

    def test_invalid_numeric_state_is_never_overwritten(self):
        ledger = self.ledger()
        for changes in (dict(used=True), dict(limit=-1), dict(schema=1.0),
                        dict(used=1, last_reservation_id=None), dict(bootstrap=1)):
            raw = json.dumps(dict(ledger.state, **changes))
            ledger.path.write_text(raw)
            with self.assertRaises(BudgetStateError):
                self.ledger()
            self.assertEqual(ledger.path.read_text(), raw)
    def test_extension_is_explicit_total_and_stale_config_rejected(self):
        self.ledger(1).reserve()
        extended = self.ledger(3)
        self.assertEqual(extended.used, 1)
        with self.assertRaises(BudgetStateError):
            self.ledger(1)
        extended.reserve()
        extended.reserve()
        with self.assertRaises(BudgetExhausted):
            self.ledger(3).reserve()

    def test_legacy_bootstrap_counts_duplicate_ids_and_partial_attempt(self):
        log = self.root/'te_gate_evals.jsonl'
        row = dict(eval=0, params=[1., 2.], fom=.9)
        raw = ((json.dumps(row)+'\n')*2+'{"eval": 0, "params":').encode()
        log.write_bytes(raw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ledger = self.ledger(4)
        self.assertEqual(ledger.used, 3)
        self.assertEqual(log.read_bytes(), raw)
        ledger.reserve()
        with self.assertRaises(BudgetExhausted):
            self.ledger(4).reserve()

    def test_legacy_over_budget_cannot_start_another_forward(self):
        row = dict(params=[1.], fom=.9)
        (self.root/'te_gate_evals.jsonl').write_text((json.dumps(row)+'\n')*3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ledger = self.ledger(2)
        with self.assertRaises(BudgetExhausted):
            ledger.reserve()
        self.assertEqual(self.ledger(4).used, 3)

    def test_legacy_complete_last_record_and_blank_lines_are_not_partial(self):
        row = json.dumps(dict(eval=0, params=[1.], fom=.8))
        (self.root/'te_gate_evals.jsonl').write_text(row+'\n \n')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            ledger = self.ledger(2)
        self.assertEqual(ledger.used, 1)
        self.assertEqual(ledger.state['legacy']['interrupted_tail_count'], 0)

    def test_forward_queue_hook_counts_forwards_and_excludes_adjoints(self):
        ledger = self.ledger(2)
        calls = []
        def forward_queue():
            ledger.reserve()  # Hook also covers uncached compute_gradient.
            calls.append('forward')
        forward_queue()
        calls.append('adjoint')  # Cached gradient has no forward queue call.
        forward_queue()
        with self.assertRaises(BudgetExhausted):
            forward_queue()
        self.assertEqual(calls, ['forward', 'adjoint', 'forward'])

    def test_terminal_outcome_survives_vendor_exception_wrapping(self):
        exhausted = BudgetExhausted('spent')
        wrapped = RuntimeError('vendor')
        wrapped.__context__ = exhausted
        outer = RuntimeError('optimization')
        outer.__cause__ = wrapped
        self.assertIs(find_budget_exhausted(outer), exhausted)
        self.assertIsNone(find_budget_exhausted(RuntimeError('unrelated')))

    def test_bad_configuration_rejected(self):
        for limit in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.ledger(limit)
        with self.assertRaises(ValueError):
            ForwardEvaluationBudget(self.root, '../outside', 1)
        with self.assertRaises(BudgetExhausted):
            self.ledger(0).reserve()


if __name__ == '__main__':
    unittest.main(verbosity=2)
