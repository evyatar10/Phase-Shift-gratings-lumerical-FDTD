"""Non-solving source inventory and TE seed contract audit.

Run from any directory: python -X utf8 python_tools/audit_readiness.py
Optional --output writes a NEW JSON evidence file (never overwrites).
This compiles source in memory; it does not execute arbitrary study modules,
open CAD, contact servers, or certify FDTD physics. Seed modules are imported.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inventory():
    rows = []
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if (not p.is_file() or p.suffix not in ('.py', '.sh', '.m', '.lsf')
                or any(x.startswith('.') or x == '__pycache__'
                       or x.startswith('results_from_') for x in rel.parts)):
            continue
        raw = p.read_bytes()
        row = dict(path=rel.as_posix(), bytes=len(raw), lines=len(raw.splitlines()),
                   sha256=hashlib.sha256(raw).hexdigest(), language=p.suffix)
        if p.suffix == '.py':
            try:
                tree = ast.parse(raw, filename=str(rel))
                compile(tree, str(rel), 'exec')
                row['syntax'] = 'pass'
                row['top_level_symbols'] = [n.name for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            except (SyntaxError, UnicodeError, ValueError) as e:
                row['syntax'] = 'fail'
                row['error'] = str(e)
            try:
                raw.decode('utf-8-sig')
            except UnicodeError as e:
                row['utf8_warning'] = str(e)
        rows.append(row)
    return rows


def seeds():
    import numpy as np
    from runners.lumopt2_design import lumopt2_design as eng
    from runners.lumopt2_design import campaign_te_s1 as s1, campaign_te_s2 as s2
    out = {}
    for name, module in [('S1_uniform', s1), ('S2_overshoot', s2)]:
        spec = module.SPEC_V3
        p, b, L = eng.seed_params(spec), np.asarray(eng.param_bounds(spec)), eng.layout(spec.n_free)
        assert len(p) == 296 and np.isfinite(p).all()
        assert np.all(p >= b[:, 0]) and np.all(p <= b[:, 1])
        assert spec.polarization == 'TE' and spec.bare and not spec.free_comb
        assert (spec.n_periods_side, spec.n_free) == (98, 60)
        props = eng.make_func(spec)(p)
        assert all(np.isfinite(v) for v in props.values())
        assert all(v > 0 for k, v in props.items() if k.endswith(' span'))
        out[name] = dict(label=spec.label, parameters=len(p),
            active_parameters=int(np.sum(np.ptp(b, axis=1) >= .01)),
            pitch_nm=spec.pitch_nm, n_side=spec.n_periods_side, n_free=spec.n_free,
            corr_min_nm=float(p[L.SL_CORR].min()), corr_max_nm=float(p[L.SL_CORR].max()),
            avg_min_nm=float(p[L.SL_AVG].min()), avg_max_nm=float(p[L.SL_AVG].max()),
            bulk_corr_nm=spec.corr0_nm, cavity_width_nm=float(p[L.I_CAV]),
            physical_width_band_um=[.98*spec.fwhm0_um, 1.02*spec.fwhm0_um],
            softw_twin_anchor_um=module.SOFTW0_UM,
            scan_center_nm=spec.scan_center_nm, scan_width_nm=spec.scan_width_nm,
            n_wl_points=spec.n_wl_points, lambda_step_bound_nm=spec.wgp_v3_dlam_nm,
            port_calibration=module.ADJ_FIX_PORT, field_calibration=module.ADJ_FIX_FIELD,
            width_row_reuse=spec.wgp_reuse_k,
            corr_nm=p[L.SL_CORR].tolist(), avg_nm=p[L.SL_AVG].tolist())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = inventory()
    counts = Counter(r['language'] for r in rows)
    errors = [r for r in rows if r.get('syntax') == 'fail']
    report = dict(scope='Static syntax and seed contracts only; not physics validation',
                  counts=dict(counts), lines=sum(r['lines'] for r in rows),
                  syntax_errors=errors, seeds=seeds(), sources=rows)
    if args.output:
        with args.output.open('x', encoding='utf8') as f:
            json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ('seeds', 'sources')}, indent=2))
    for name, data in report['seeds'].items():
        print(name, data['parameters'], 'slots;', data['active_parameters'], 'active;',
              'C_port =', data['port_calibration'])
    for row in rows:
        if 'utf8_warning' in row:
            print('UTF8_WARNING', row['path'], row['utf8_warning'])
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
