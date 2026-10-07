"""Bounded S2 v3 persistence/plumbing smoke after peak3 calibration.

Derived from validate_te task 38: TE N=70/free=60, pitch=491.06 nm,
h=350 nm, bare, 1560.464 nm center with 2 nm/501-point window.
Two evaluations, no automatic recenter retries; N=70 is a pipeline surrogate,
not a width/physics certificate for the retained N=98 device. A new solve is
needed to exercise the corrected callback/driver/state on actual field data;
stored calibration fields cannot execute that optimization trajectory.

Do not dispatch until CALIBRATION_SHA256 pins a reviewed, passing peak3
checkpoint in calibrations/. No fallback to S1 or the legacy softmax fit.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import config
from runners.lumopt2_design import campaign_te_s2 as s2
from runners.lumopt2_design import lumopt2_design as eng
from runners.lumopt2_design import persistence
from runners.lumopt2_design import evaluation_budget, width_contract
from runners.lumopt2_design import recover_te_s2 as recovery
from runners.lumopt2_design import validate_te as val

N_TASKS = 1
SPEC = dataclasses.replace(s2.SPEC_V3, label='lumopt2_te_s2_v3_restart_smoke_20261007')
CALIBRATION_PATH = Path(__file__).parent/'calibrations/s2_peak_20261007.json'
CALIBRATION_SHA256 = None  # Fill only after reviewing the actual fit checkpoint.


def load_calibration(path=CALIBRATION_PATH, expected_sha256=None):
    if not expected_sha256:
        raise RuntimeError('S2 peak3 calibration has not passed review; no CAD/solve allowed')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError('Calibration evidence hash mismatch')
    payload = json.loads(raw)
    if payload['identity']['objective'] != 'peak3_abs_T_out':
        raise RuntimeError('Calibration uses a different objective')
    r = payload['result']
    A = np.array([r['re'], r['im']], dtype=float).T
    fd = np.asarray(r['fd'], dtype=float)
    supplied = np.asarray(r['coefficients'], dtype=float)
    if (A.shape != (6, 2) or fd.shape != (6,) or supplied.shape != (2,)
            or not np.isfinite(A).all() or not np.isfinite(fd).all()
            or not np.isfinite(supplied).all() or np.any(fd == 0)
            or np.linalg.matrix_rank(A) != 2):
        raise RuntimeError('Invalid calibration vectors')
    coeff = np.linalg.lstsq(A, fd, rcond=None)[0]
    model = A@coeff
    held_out = []
    for i in range(6):
        keep = np.arange(6) != i
        if np.linalg.matrix_rank(A[keep]) != 2:
            raise RuntimeError('Rank-deficient held-out calibration')
        c = np.linalg.lstsq(A[keep], fd[keep], rcond=None)[0]
        held_out.append((A[i]@c-fd[i])/fd[i])
    if (r['passed'] is not True
            or not np.allclose(coeff, supplied, rtol=1e-11, atol=1e-13)
            or np.max(np.abs(model/fd-1)) > .1
            or np.max(np.abs(held_out)) > .1
            or np.any(np.sign(model) != np.sign(fd))):
        raise RuntimeError('Peak3 calibration fails recomputed 10% gates')
    return tuple(float(x) for x in supplied)


def smoke_spec(coefficients):
    measured = SimpleNamespace(**vars(s2))
    measured.ADJ_FIX_PORT = coefficients
    return val._campaign_spec(measured, '_restart_smoke_20261007', v3=True,
                              n_periods_side=70, two_kl_floor=0., fwhm0_um=None,
                              max_iter=2, max_feval=4)


def audit_budget(out, spec, evaluations):
    path = out/(spec.label+'_forward_budget.json')
    if not path.is_file():
        raise RuntimeError('Smoke has no persistent forward budget')
    # The helper validates its own history/identity, including interrupted
    # reservations that must not be refunded on a cold restart.
    budget = evaluation_budget.ForwardEvaluationBudget(out, spec.label, spec.max_feval)
    used = budget.used
    if not evaluations <= used <= spec.max_feval:
        raise RuntimeError('Smoke forward budget does not cover its evaluations')
    return used


def audit(out, spec):
    ev = persistence.read_rows(out/(spec.label+'_evals.jsonl'))
    steps = persistence.read_rows(out/(spec.label+'_proj.jsonl'))
    state = eng._load_opt_state(str(out), spec.label)
    if len(ev) < 2 or len(steps) < 2:
        raise RuntimeError('Smoke did not complete two evaluations/decisions')
    ids = [r['eval_id'] for r in ev]
    profiles = [persistence.profile_path(out, spec.label, r, r['eval']) for r in ev]
    if len(set(ids)) != len(ids) or len(set(profiles)) != len(profiles):
        raise RuntimeError('Duplicate evaluation identity or profile path')
    if any(p is None or not Path(p).is_file() for p in profiles):
        raise RuntimeError('A smoke profile is missing')
    if any(r.get('port_objective_definition') != 'abs_T_out'
           or r.get('tout_peak3') is None
           or not np.isfinite(r['tout_peak3'])
           or not np.isclose(r['fom'], r['tout_peak3'], rtol=1e-8, atol=1e-10)
           for r in ev):
        raise RuntimeError('Logged FOM does not match the measured peak3 T_out objective')
    if not state.get('accepted_eval_ids') or not (out/'state_history').is_dir():
        raise RuntimeError('No persistent acceptance history')
    if not any(r.get('v3_mode') and r.get('rho_T') is not None for r in steps):
        raise RuntimeError('V3 constrained step was not exercised')
    if not any(r.get('mac') is not None for r in steps):
        raise RuntimeError('MAC was not exercised')
    used = audit_budget(out, spec, len(ev))
    val._upgrade_markers(str(out), spec.label, require_reuse=False)
    return dict(evaluations=len(ev), decisions=len(steps),
                accepted_ids=state['accepted_eval_ids'], profiles=profiles,
                forward_batches=used, max_feval=spec.max_feval,
                pipeline_only=True, physical_N98_validation=False)


def main(task_idx=0):
    if int(task_idx) != 0:
        raise ValueError(task_idx)
    coeff = load_calibration(expected_sha256=CALIBRATION_SHA256)
    spec = smoke_spec(coeff)
    out = Path(config.RESULTS_DIR)/spec.label
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(calibration_sha256=CALIBRATION_SHA256,
                    code={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                          for m in (eng, persistence, evaluation_budget, width_contract)},
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    spec=json.loads(json.dumps(dataclasses.asdict(spec))))
    done = out/'smoke_complete.json'
    existing = recovery.read_checkpoint(done, identity)
    if existing is not None:
        audit(out, spec)
        print('Verified completed smoke; no new solve', existing, flush=True)
        return existing
    print('S2 bounded pipeline smoke; C_port', coeff, 'evidence', CALIBRATION_SHA256,
          'N=70, TE bare, 1560.464 +/-1 nm, two evaluations', flush=True)
    old_max = eng.MAX_RESTARTS
    try:
        eng.MAX_RESTARTS = 0
        eng.run_campaign(spec, str(out))
    finally:
        eng.MAX_RESTARTS = old_max
    result = audit(out, spec)
    recovery.checkpoint(done, identity, result)
    print('S2 RESTART SMOKE PASSED', result, flush=True)
    return result
