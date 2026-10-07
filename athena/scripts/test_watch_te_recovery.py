"""Offline checks: monitoring must not turn incomplete evidence into success."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import watch_te_recovery as w


class WatcherChecks(unittest.TestCase):
    def data(self):
        return dict(accounting=dict(code=0, err='', out='\n'.join(
            j+'|COMPLETED|0:0' for j in sorted(w.EXPECTED_TASKS))),
            queue=dict(code=0, out='', err=''))

    def test_all_tasks_required(self):
        data = self.data()
        self.assertEqual(w.verdict(data), 'complete')
        data['accounting']['out'] = '170746_1|COMPLETED|0:0'
        self.assertEqual(w.verdict(data), 'active')
        data['accounting']['out'] = ''
        self.assertEqual(w.verdict(data), 'active')

    def test_query_errors_are_not_completion(self):
        for key in ('accounting', 'queue'):
            data = self.data()
            data[key].update(code=1, err='scheduler unavailable')
            with self.assertRaises(RuntimeError):
                w.verdict(data)

    def test_running_pending_and_failed(self):
        data = self.data()
        for state in ('RUNNING', 'PENDING'):
            other = copy.deepcopy(data)
            other['accounting']['out'] = other['accounting']['out'].replace(
                'COMPLETED', state, 1)
            self.assertEqual(w.verdict(other), 'active')
        for state in ('FAILED', 'CANCELLED by 42', 'TIMEOUT', 'OUT_OF_MEMORY'):
            other = copy.deepcopy(data)
            other['accounting']['out'] = other['accounting']['out'].replace(
                'COMPLETED', state, 1)
            self.assertEqual(w.verdict(other), 'failed')

    def test_address_override_keeps_host_verification(self):
        result = SimpleNamespace(returncode=0, stdout=json.dumps(self.data()).encode())
        with patch.object(w.subprocess, 'run', return_value=result) as run:
            w.collect('192.0.2.1')  # offline documentation address only
        command = run.call_args.args[0]
        self.assertIn('StrictHostKeyChecking=yes', command)
        self.assertIn('HostKeyAlias=athena.technion.ac.il', command)
        self.assertIn('HostName=192.0.2.1', command)
        self.assertIsInstance(run.call_args.kwargs['input'], bytes)


if __name__ == '__main__':
    unittest.main()
