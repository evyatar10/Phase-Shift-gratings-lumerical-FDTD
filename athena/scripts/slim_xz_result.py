"""
Server-side reducer for XZ-side field results (CLAUDE.md §6: reduce field
data server-side before downloading).

    python3 athena/scripts/slim_xz_result.py <result_full.mat> <result_slim.mat>

Keeps every top-level variable except the 2D field structs, and keeps
field_xz_side with E_res / P_res ONLY at the lambda_3d bin closest to the
stored resonance_wavelength_nm (lambda_3d collapses to that one bin, so
matlab_plotting/plot_field_poynting_zoom.m still reshapes it correctly).
field_xy and field_yz_cross are dropped (they stay in the full file).
Study: xz_apod_vs_nonapod_n80, Athena job 175019, 2026-10-10.
"""
import sys
import numpy as np
import scipy.io as sio

src, dst = sys.argv[1], sys.argv[2]
m = sio.loadmat(src, squeeze_me=False, struct_as_record=False)
out = {k: v for k, v in m.items()
       if not k.startswith('__') and k not in ('field_xy', 'field_yz_cross', 'field_xz_side')}

d = m['field_xz_side'][0, 0]
x, y, z = np.ravel(d.x), np.ravel(d.y), np.ravel(d.z)
lam = np.ravel(d.lambda_3d)
nx, ny, nz, nl = x.size, y.size, z.size, lam.size
res_m = float(np.ravel(m['resonance_wavelength_nm'])[0]) * 1e-9
il = int(np.argmin(np.abs(lam - res_m)))
print(f"{src}: resonance {res_m*1e9:.4f} nm -> bin {il} = {lam[il]*1e9:.4f} nm "
      f"(|d|={abs(lam[il]-res_m)*1e9:.4f} nm); grid {nx}x{ny}x{nz}x{nl}")

slim = {'x': x, 'y': y, 'z': z, 'lambda_3d': np.array([lam[il]]),
        'lambda_3d_full': lam, 'lambda_index_kept': il + 1}   # 1-based for MATLAB
for key in ('E_res', 'P_res'):
    if not hasattr(d, key):
        continue
    a = np.asarray(getattr(d, key))
    if a.ndim != 5:
        a = a.reshape((nx, ny, nz, nl, 3), order='F')
    slim[key] = np.ascontiguousarray(a[:, :, :, il:il + 1, :])
out['field_xz_side'] = slim
sio.savemat(dst, out, do_compression=True)
print(f"wrote {dst}")
