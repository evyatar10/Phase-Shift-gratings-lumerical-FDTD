# scratch 2026-10-21 harm_matrix: cache normalised far fields (solve_v4 --short normalisation) + multipoles l<=40 per row.
import sys, io, os, glob, shutil, contextlib, numpy as np, scipy.io as sio
HERE = os.path.dirname(os.path.abspath(__file__)); SCR = HERE   # solve_v4.py sits next to this file (was the scratch parent dir)
RES = r"C:/Users/evyat/Lumerical/phase_shift_grating_FTDT_codes/results_from_athena/"
LM = 40
src = open(SCR + "/solve_v4.py", encoding="utf-8").read().split("rows = [load(p)")[0]
ns = {}; sys.argv = ["solve_v4.py", "--short"]; exec(src, ns)          # defines load(), sphere_field, multipoles, LMAX, short=True
def one(d):
    f = glob.glob(RES + d + "/results/*T340ps_ff.mat") or glob.glob(RES + d + "/results/*_ff.mat"); assert len(f) == 1, (d, f); return f[0]
FILES = {"ctrl": one("farfield_sph_15um_c_nt80")}
for d in sorted(os.listdir(RES + "farfield_sph_15um_bumps")):
    if os.path.isdir(RES + "farfield_sph_15um_bumps/" + d): FILES["bump:" + d] = one("farfield_sph_15um_bumps/" + d)
for d in sorted(os.listdir(RES + "farfield_sph_15um_contrast")):
    if os.path.isdir(RES + "farfield_sph_15um_contrast/" + d): FILES["contrast:" + d] = one("farfield_sph_15um_contrast/" + d)
for f in sorted(glob.glob(RES + "farfield_sph_15um_subteeth/results/*_ff.mat")): FILES["subteeth:" + os.path.basename(f)[9:-7]] = f
for f in sorted(glob.glob(RES + "farfield_sph_15um_phase/results/*_ff.mat")): FILES["tab:" + os.path.basename(f)[6:-7]] = f
grid = None
for k, p in FILES.items():
    out = HERE + "/row_" + k.replace(":", "_") + ".npz"
    if os.path.exists(out): continue
    short = HERE + "/tmp_in.mat"; shutil.copyfile(p, short)                     # MAX_PATH guard
    r = ns["load"](short)
    if grid is None:
        grid = ns["sphere_field"](sio.loadmat(short, squeeze_me=True, struct_as_record=False), ns["LMAX"])[:3]
        np.savez(HERE + "/grid.npz", x=grid[0], w=grid[1], phi=grid[2])
    aE, aM = ns["multipoles"](*grid, r["E"][0], r["E"][1], LM)
    np.savez(out, E=r["E"], dW=r["dW"][0], a=np.stack([aE, aM]), lam=r["lam"], T=r["T"], Q=r["Q"], W=r["W"], det=r["det"], src=p)
    print(k, f"lam {r['lam']:.4f} T {r['T']:.4f} Q {r['Q']:.0f} W {r['W']:.3f} det {r['det']:+.2f}", flush=True)
os.remove(HERE + "/tmp_in.mat") if os.path.exists(HERE + "/tmp_in.mat") else None
