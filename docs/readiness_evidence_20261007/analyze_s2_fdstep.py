"""Offline diagnostic arithmetic; never dispatches or promotes calibration.

Run with --self-test, or --evidence-dir DIR --old-spectra ZIP --output NEW.json.
DIR must contain s2_fdstep_task00/01/02_original.json and matching _ports.npz.
Output is exclusive-create so prior evidence cannot be silently overwritten.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


def peak3(t):
    t = np.asarray(t, dtype=float).ravel()
    if not np.all(np.isfinite(t)):
        raise ValueError('Nonfinite spectrum')
    k = int(np.argmax(t))
    if not 0 < k < len(t) - 1:
        raise ValueError('Peak at scan edge')
    b = t[k+1] - 2*t[k] + t[k-1]
    d = t[k+1] - t[k-1]
    if b >= 0:
        raise ValueError('Nonconcave peak')
    offset = -d/(2*b)
    if abs(offset) > .5 + 1e-9:
        raise ValueError('Peak vertex outside central cell')
    weights = np.array([offset*(offset-1)/2, 1-offset**2,
                        offset*(offset+1)/2])
    return k, float(t[k] - d*d/(8*b)), weights


def terms(plus, minus, center, h):
    if h <= 0:
        raise ValueError('Positive step required')
    odd = (plus-minus)/2
    even = (plus+minus-2*center)/2
    return dict(odd=float(odd), even=float(even),
                derivative=float(odd/h), second_difference=float(2*even/h**2))


def load_npz(source):
    with np.load(source, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def analyze(evidence, archive):
    rows, arrays, digests = [], [], []
    for i in range(3):
        path = evidence/f's2_fdstep_task{i:02d}_original.json'
        raw = path.read_bytes()
        digests.append(hashlib.sha256(raw).hexdigest())
        rows.append(json.loads(raw))
        arrays.append(load_npz(evidence/f's2_fdstep_task{i:02d}_ports.npz'))
    for i, sign in ((1, 1), (2, -1)):
        assert rows[i]['result']['reviewed_center_sha256'] == digests[0]
        assert rows[i]['identity']['reviewed_center_sha256'] == digests[0]
        assert rows[i]['identity']['code'] == rows[0]['identity']['code']
        assert rows[i]['result']['sign'] == sign
        assert rows[i]['result']['delta_nm'] == sign*.5
        delta = np.array(rows[i]['identity']['point']) - rows[0]['identity']['point']
        expected = np.zeros_like(delta)
        expected[29] = sign*.5
        np.testing.assert_array_equal(delta, expected)
    with zipfile.ZipFile(archive) as z:
        old = [load_npz(io.BytesIO(z.read(name))) for name in
               ('base_ports.npz', 'task_03_ports.npz', 'task_04_ports.npz')]
    for a in arrays + old:
        np.testing.assert_array_equal(a['p2_lambda'], old[0]['p2_lambda'])
        np.testing.assert_array_equal(a['p2_f'], old[0]['p2_f'])
    result = dict(checkpoint_sha256=digests, steps_nm=[.5, 2.], observables={})
    for metric, transform in [('peak3_tout', lambda a: abs(a['p2_T_out'].ravel())),
                              ('peak3_modal', lambda a: abs(a['p2_S'].ravel())**2)]:
        spectra = [transform(a) for a in arrays]
        prior = [transform(a) for a in old]
        peaks = [peak3(t) for t in spectra]
        old_peaks = [peak3(t) for t in prior]
        for row, (_, value, _) in zip(rows, peaks):
            np.testing.assert_allclose(value, row['result'][metric], rtol=0, atol=1e-12)
        small = terms(peaks[1][1], peaks[2][1], peaks[0][1], .5)
        large = terms(old_peaks[1][1], old_peaks[2][1], old_peaks[0][1], 2.)
        k, _, weights = peaks[0]
        spectral_d = (spectra[1]-spectra[2]) / 1.0
        result['observables'][metric] = dict(
            h_half=small, h_two=large,
            h_half_with_old_center=terms(peaks[1][1], peaks[2][1], old_peaks[0][1], .5),
            h_two_with_new_center=terms(old_peaks[1][1], old_peaks[2][1], peaks[0][1], 2.),
            derivative_difference=small['derivative']-large['derivative'],
            conditional_richardson=(16*small['derivative']-large['derivative'])/15,
            repeat_peak_delta=peaks[0][1]-old_peaks[0][1],
            peak_indices_center_plus_minus=[p[0] for p in peaks],
            fixed_center_sample_fd=float(spectral_d[k]),
            center_peak3_jacobian_applied_to_spectral_fd=float(weights @ spectral_d[k-1:k+2]),
            spectral_fd_max_abs=float(np.max(abs(spectral_d))))
    result['verdict'] = 'DIAGNOSTIC_ONLY_OVERALL_CALIBRATION_REMAINS_FAILED'
    result['limitations'] = ('Two steps cannot establish convergence order. Richardson is conditional; '
        'one repeat is not a noise bound. Fixed-center finite spectral differences need not equal '
        'moving peak derivatives at finite step. No spatial width or full gradient validation here.')
    return result


def self_test():
    # A quartic has an exact h^2 error in both central derivative and curvature.
    f = lambda x: 7 + 2*x + 3*x*x + 5*x**3 + 11*x**4
    small, large = [terms(f(h), f(-h), f(0), h) for h in (.5, 2)]
    assert abs((16*small['derivative']-large['derivative'])/15 - 2) < 1e-12
    assert abs((16*small['second_difference']-large['second_difference'])/15 - 6) < 1e-12
    # Off-grid concave parabola: vertex value and envelope Jacobian are exact.
    x = np.arange(-3, 4.)
    k, value, weights = peak3(10-(x-.2)**2)
    assert k == 3 and abs(value-10) < 1e-12
    assert abs(weights.sum()-1) < 1e-12
    for bad in ([1, 2, 3], [1, float('nan'), 0]):
        try:
            peak3(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid spectrum accepted')
    print('PASS: analytic quartic derivative/curvature, off-grid peak, invalid spectra')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--evidence-dir', type=Path)
    parser.add_argument('--old-spectra', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if any(v is None for v in (args.evidence_dir, args.old_spectra, args.output)):
            parser.error('Provide evidence directory, old spectra archive and new output path')
        data = analyze(args.evidence_dir, args.old_spectra)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2)
        print(json.dumps(data, indent=2))
