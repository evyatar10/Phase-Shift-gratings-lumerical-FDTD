"""Campaign persistence without silently discarding recovery evidence.

Single writer per campaign directory. A reader can ignore an unterminated
JSONL tail left by interruption; corruption in any completed line is fatal.
The next append archives the original before replacing that partial tail.
State corruption is fatal, while an absent state means a fresh campaign.
"""
import json
import math
import os
from pathlib import Path
import uuid
import warnings

import numpy as np


def new_id():
    return uuid.uuid4().hex


def _jsonl(path):
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return b'', [], None
    rows, offset = [], 0
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            if index == len(lines)-1 and not line.endswith((b'\n', b'\r')):
                return raw, rows, offset
            raise RuntimeError(f'Corrupt completed JSONL record {path}:{index+1}') from exc
        if not isinstance(row, dict):
            raise RuntimeError(f'Expected JSON object at {path}:{index+1}')
        rows.append(row)
        offset += len(line)
    return raw, rows, None


def read_rows(path):
    _, rows, partial = _jsonl(path)
    if partial is not None:
        warnings.warn(f'Ignoring interrupted final record in {path}; original bytes preserved',
                      RuntimeWarning, stacklevel=2)
    return rows


def _write_new(path, data):
    with Path(path).open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _nan_to_none(v):
    """Non-finite floats -> None (claude-te 2026-10-07): a NaN DIAGNOSTIC in a
    log row must not kill a campaign; numbers that matter are checked where used."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: _nan_to_none(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_nan_to_none(x) for x in v]
    return v


def append_row(path, row):
    path = Path(path)
    encoded = (json.dumps(_nan_to_none(row), allow_nan=False)+'\n').encode('utf8')
    raw, _, partial = _jsonl(path)
    if partial is not None:
        backup = path.with_name(path.name+'.interrupted_'+new_id())
        _write_new(backup, raw)
        # A distinct, flushed file keeps the original intact if interrupted
        # during repair. No reader ever sees an in-place truncated journal.
        tmp = path.with_name(path.name+'.resume_'+new_id()+'.tmp')
        _write_new(tmp, raw[:partial]+encoded)
        os.replace(tmp, path)
        warnings.warn(f'Interrupted journal archived as {backup}', RuntimeWarning, stacklevel=2)
    else:
        with path.open('ab') as f:
            if raw and not raw.endswith(b'\n'):
                f.write(b'\n')
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())


def _validate_state(state):
    if not isinstance(state, dict):
        raise ValueError('optimizer state must be a JSON object')
    # Existing versions do not have a schema number. Retain their optional
    # keys, but validate the numerical fields that control restart decisions.
    json.dumps(state, allow_nan=False)
    for key in ('cap_nm', 'wgain', 'dTp0', 'dwdlam', 'lam_tgt_nm', 'reuse_travel',
                'reuse_W0', 'retry_shrink'):
        value = state.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                  or not math.isfinite(value)):
            raise ValueError(f'invalid optimizer state {key}')
    for key in ('n_acc', 'n_rej', 'n_noise_rej', 'reuse_age'):
        if key in state and (type(state[key]) is not int or state[key] < 0):
            raise ValueError(f'invalid optimizer state {key}')
    for key in ('cap_nm', 'wgain', 'retry_shrink'):
        if state.get(key) is not None and state[key] <= 0:
            raise ValueError(f'optimizer state {key} must be positive')
    for key in ('ineligible', 'accepted_eval_ids'):
        if key in state and (not isinstance(state[key], list)
                             or not all(isinstance(v, str) for v in state[key])):
            raise ValueError(f'invalid optimizer evaluation history: {key}')
    return state


def load_state(path):
    try:
        with Path(path).open(encoding='utf8') as f:
            return _validate_state(json.load(f))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f'Cannot safely resume optimizer state {path}; '
                           'preserve it and inspect state_history before recovery') from exc


def save_state(path, state):
    path = Path(path)
    data = (json.dumps(_validate_state(state), allow_nan=False)+'\n').encode('utf8')
    # Refuse to erase an existing corrupt state by silently starting afresh.
    if path.exists():
        load_state(path)
    history = path.parent/'state_history'
    history.mkdir(exist_ok=True)
    if path.exists():
        _write_new(history/(path.stem+'_previous_'+new_id()+'.json'), path.read_bytes())
    snapshot = history/(path.stem+'_'+new_id()+'.json')
    _write_new(snapshot, data)
    tmp = path.with_name(path.name+'.'+new_id()+'.tmp')
    os.link(snapshot, tmp)
    os.replace(tmp, path)


def save_profile(out_dir, label, row, **arrays):
    folder = Path(out_dir)/'profiles'
    folder.mkdir(exist_ok=True)
    target = folder/f"{label}_ev{int(row['eval']):04d}_{row['eval_id']}.npz"
    with target.open('xb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    return target.relative_to(out_dir).as_posix()


def profile_path(out_dir, label, row, fallback_eval):
    """New rows use their recorded immutable file; old callbacks remain readable."""
    if 'eval_id' not in row and 'profile_file' not in row:
        return str(Path(out_dir)/'profiles'/f'{label}_ev{int(fallback_eval):04d}.npz')
    name = row.get('profile_file')
    if not name:
        return None
    root = Path(out_dir).resolve()
    path = (root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Profile reference is outside the campaign directory')
    return str(path)
