"""
Post-strip probe: TRUE radiated response of ONE full-height SiN post beside a SiN strip guide.

Study dir: results/post_strip_probe/ (RUN_NAME = module name on Athena). Job ID: not yet dispatched.
Date: 2026-10-25.
Purpose: calibrate the Born dipole model of python_tools/radcancel/postarray_born.py
(p = k^2/(4 pi) * pi r^2 h * GAM * E_drive, free-oxide dipole far field) against FDTD with
the guide present: radiated power, far-field pattern and phase vs the drive field.

Scene: straight SiN strip (n 1.97, h 350 nm, width W) along x in SiO2 (1.444), object-defined
indices, conformal variant 0. TE fundamental mode source (+x) at 1619.4 nm. One full-height SiN
post (n 1.97, h 350, radius r) at (0, Y, 0), Y from the strip centre. Closed box of six DFT
monitors around the post (the strip pierces the two x faces), a 3D DFT "drive" monitor over the
post band (y 0.55-1.85 um: every Y below), and a point time monitor (decay check).
Each case is run WITH and WITHOUT the post at the identical mesh; the analysis subtracts E, H on
the faces (scattered field), removes the guided TE0 part on the x faces by mode projection
(mode profile = the no-post field on that face), and projects the rest to the far field.
FIXED run length, early shutoff OFF: the subtraction needs both runs to stop at the same step.

Mesh "prod" = the production post-array numerics at the post: dx 51.4 nm (pitch 514 / 10,
'optimization'), dy = dz = 50 nm max step outside the core override, core override
dy 750/13 = 57.7 nm over |y| <= 1.08 um (1.2 x the Nt80x's widest tooth, 1.8 um) and dz 350/7 = 50 nm.
Mesh "m25" = prod + a 25 nm (x, y, z) override box over the whole post band, present in the no-post
reference too. Rectilinear mesh: its z lines also refine the strip near z = 0 (same in both runs).

GPU OK: mode source + DFT monitors are GPU-supported (unlike the TFSF of post_mesh_probe).
Array: task t of a SLURM array runs the cases i with i % SLURM_ARRAY_TASK_COUNT == t (no array ->
all); a case whose npz exists is skipped (REQUEUE-safe).
Dispatch (2 tasks x 7 cases):  bash athena/deploy_athena.sh --option2 --run=post_strip_probe --pol-array
Analysis: scratch stripprobe/analyze.py (subtraction, mode projection, far field, Born).
Local:  python runners/single/post_strip_probe.py build <case idx> <out.fsp>    (no solve)
        python runners/single/post_strip_probe.py smoke <case idx> <out dir>    (CPU, 60 fs, plumbing only)
"""
import os
import sys
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config

try:
    import lumapi
except ImportError:
    _spec = importlib.util.spec_from_file_location("lumapi", config.LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(lumapi)

# ── knobs ─────────────────────────────────────────────────────────────────────
LAM = 1619.4e-9
BAND = 40e-9                        # source band, centred in FREQUENCY on c/LAM; monitors at c/LAM only
N_CORE, N_CLAD = 1.97, 1.444
H = 350e-9                          # strip and post height
MESHES = ["prod", "m25"]
# (strip width um, post Y um, post radius nm, mesh); r = 0 = the no-post reference of that (W, mesh)
CASES = ([(1.0, 0.0, 0, m) for m in MESHES]
         + [(1.0, Y, r, m) for m in MESHES for Y in (0.85, 1.2, 1.5) for r in (50, 100)])
SIM_X = (-4.0e-6, 4.0e-6)           # FDTD region (PML outside)
SIM_Y = (-2.0e-6, 3.1e-6)
SIM_Z = (-2.0e-6, 2.0e-6)
X_SRC = -3.2e-6
BOX_X = (-2.5e-6, 2.5e-6)           # closed DFT box around the post
BOX_Y = (-1.5e-6, 2.6e-6)
BOX_Z = (-1.5e-6, 1.5e-6)
DRIVE = ((-0.3e-6, 0.3e-6), (0.55e-6, 1.85e-6), (-0.25e-6, 0.25e-6))   # 3D drive monitor = m25 box
DX_PROD = 514e-9 / 10               # production x step at pitch 514 ('optimization')
DYZ_BG = 50e-9                      # bragg_device background dy, dz
CORE_DY, CORE_DZ = 750e-9 / 13, H / 7
CORE_Y_SPAN = 1.2 * 1.8e-6          # bragg_device: 1.2 x widest tooth of the Nt80x
FINE = 25e-9
SIM_TIME = 1000e-15                 # fixed; the pulse is gone by ~600 fs (tail asserted in analysis)
C0 = 299792458.0
F0 = C0 / LAM
FACES = ["box_x1", "box_x2", "box_y1", "box_y2", "box_z1", "box_z2"]


def case_name(W, Y, r, mesh):
    return f"W{W:.2f}_ref_{mesh}" if r == 0 else f"W{W:.2f}_Y{Y:.2f}_r{r}_{mesh}"


def _span(a, lo_hi):
    return {a: 0.5 * (lo_hi[0] + lo_hi[1]), a + " span": lo_hi[1] - lo_hi[0]}


def _setall(fdtd, props):
    for k, v in props.items():
        fdtd.set(k, v)


def build(fdtd, W, Y, r, mesh):
    fdtd.switchtolayout()
    fdtd.deleteall()
    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    for a, lo_hi in zip("xyz", (SIM_X, SIM_Y, SIM_Z)):
        _setall(fdtd, _span(a, lo_hi))
        fdtd.set("define " + a + " mesh by", "maximum mesh step")
        fdtd.set("allow grading in " + a, 1)
    fdtd.set("dx", DX_PROD)
    fdtd.set("dy", DYZ_BG)
    fdtd.set("dz", DYZ_BG)
    fdtd.set("mesh type", "custom non-uniform")
    fdtd.set("grading factor", 1.41421)
    fdtd.set("mesh refinement", "conformal variant 0")
    fdtd.set("dt stability factor", 0.7)
    fdtd.set("background material", "<Object defined dielectric>")
    fdtd.set("background index", N_CLAD)
    fdtd.set("simulation time", SIM_TIME)
    fdtd.set("use early shutoff", 0)
    fdtd.setdevice("GPU" if config.USE_GPU else "CPU")

    fdtd.addmesh()                                   # core override, as bragg_device (y, z only: x is global)
    _setall(fdtd, {"name": "core_mesh", "x": 0.0, "x span": SIM_X[1] - SIM_X[0] + 2e-6,
                   "y": 0.0, "y span": CORE_Y_SPAN, "z": 0.0, "z span": H,
                   "override x mesh": 0, "override y mesh": 1, "override z mesh": 1,
                   "dy": CORE_DY, "dz": CORE_DZ})
    if mesh == "m25":                                # kept in the no-post reference too
        fdtd.addmesh()
        fdtd.set("name", "post_mesh")
        for a, lo_hi in zip("xyz", DRIVE):
            _setall(fdtd, _span(a, lo_hi))
            fdtd.set("override " + a + " mesh", 1)
            fdtd.set("d" + a, FINE)

    fdtd.addrect()
    _setall(fdtd, {"name": "strip", "material": "<Object defined dielectric>", "index": N_CORE,
                   "x": 0.0, "x span": 20e-6, "y": 0.0, "y span": W * 1e-6, "z": 0.0, "z span": H})
    if r > 0:
        fdtd.addcircle()
        _setall(fdtd, {"name": "post", "material": "<Object defined dielectric>", "index": N_CORE,
                       "radius": r * 1e-9, "x": 0.0, "y": Y * 1e-6, "z": 0.0, "z span": H})

    fdtd.addmode()
    _setall(fdtd, {"name": "src", "injection axis": "x-axis", "direction": "Forward",
                   "x": X_SRC, "y": 0.0, "y span": 3.0e-6, "z": 0.0, "z span": 3.0e-6,
                   "mode selection": "fundamental TE mode", "override global source settings": 0})
    df = C0 * BAND / LAM**2
    fdtd.setglobalsource("set frequency", 1)
    fdtd.setglobalsource("frequency start", F0 - df / 2)
    fdtd.setglobalsource("frequency stop", F0 + df / 2)
    fdtd.setglobalmonitor("use source limits", 1)
    fdtd.setglobalmonitor("frequency points", 1)     # = c/LAM (asserted after the run)

    box = dict(x=BOX_X, y=BOX_Y, z=BOX_Z)
    for name in FACES:
        ax, side = name[4], int(name[5]) - 1
        fdtd.addpower()
        fdtd.set("name", name)
        fdtd.set("monitor type", "2D " + ax.upper() + "-normal")
        for a in "xyz":
            if a == ax:
                fdtd.set(a, box[a][side])
            else:
                _setall(fdtd, _span(a, box[a]))
    fdtd.addpower()
    fdtd.set("name", "drive")
    fdtd.set("monitor type", "3D")
    for a, lo_hi in zip("xyz", DRIVE):
        _setall(fdtd, _span(a, lo_hi))
    fdtd.addtime()
    _setall(fdtd, {"name": "t_scat", "monitor type": "Point", "x": 1.5e-6, "y": 2.2e-6, "z": 0.5e-6})


def run_case(fdtd, case, out_dir, fsp):
    W, Y, r, mesh = case
    build(fdtd, W, Y, r, mesh)
    fdtd.save(fsp)
    fdtd.run()
    f = float(np.ravel(fdtd.getresult("box_x1", "f"))[0])
    assert abs(C0 / f - LAM) < 0.05e-9, f"monitor at {C0/f*1e9:.3f} nm, not {LAM*1e9} nm"
    out = dict(W=W, Y=Y, r=r, mesh=mesh, f=f, source_power=float(np.ravel(fdtd.sourcepower(f))[0]),
               grid_x=np.ravel(fdtd.getresult("FDTD", "x")), grid_y=np.ravel(fdtd.getresult("FDTD", "y")),
               grid_z=np.ravel(fdtd.getresult("FDTD", "z")))
    for name in FACES + ["drive"]:
        Ed, Hd = fdtd.getresult(name, "E"), fdtd.getresult(name, "H")
        for a in "xyz":
            out[f"{name}_{a}"] = np.ravel(Ed[a])
        out[f"{name}_E"] = Ed["E"][:, :, :, 0, :]   # (nx, ny, nz, 3), single frequency
        out[f"{name}_H"] = Hd["H"][:, :, :, 0, :]
    T = fdtd.getresult("t_scat", "E")
    out["t_t"] = np.ravel(T["t"])
    out["t_E2"] = np.sum(np.abs(np.squeeze(T["E"])) ** 2, axis=-1)
    path = os.path.join(out_dir, f"post_strip_probe_{case_name(*case)}.npz")
    np.savez(path, **out)
    n_cells = out["grid_x"].size * out["grid_y"].size * out["grid_z"].size
    print(f"  {case_name(*case):24s} cells {n_cells/1e6:.2f} M  P_src {out['source_power']:.3e} W  "
          f"tail/peak |E|^2 {out['t_E2'][-20:].max() / out['t_E2'].max():.1e}  -> {path}", flush=True)
    return path


def run(cfg=None):
    names = [case_name(*c) for c in CASES]
    assert len(set(names)) == len(names), "case names (= output files) must be unique"
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    mine = [c for i, c in enumerate(CASES) if task == "" or i % n_tasks == int(task)]
    print(f"[post_strip_probe] task {task or '-'} of {n_tasks}: {[case_name(*c) for c in mine]}", flush=True)
    fdtd = lumapi.FDTD(hide=True)
    try:
        for c in mine:
            if os.path.exists(os.path.join(config.RESULTS_DIR, f"post_strip_probe_{case_name(*c)}.npz")):
                print(f"  {case_name(*c)}: npz exists, skipped")
                continue
            run_case(fdtd, c, config.RESULTS_DIR, os.path.join(config.LAYOUTS_DIR, f"post_strip_probe_{case_name(*c)}.fsp"))
    finally:
        fdtd.close()


if __name__ == "__main__":
    mode, idx, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    config.USE_GPU = False                       # local = CPU
    case = CASES[idx]
    fdtd = lumapi.FDTD(hide=True)
    if mode == "build":
        build(fdtd, *case)
        fdtd.save(out)
        fdtd.selectall()
        for i in range(1, int(fdtd.getnumber()) + 1):
            print(f"  {fdtd.get('name', i):10s} {fdtd.get('type', i)}")
        for obj, keys in (("strip", ["y span", "z span", "index"]), ("src", ["x", "mode selection", "direction"]),
                          ("core_mesh", ["y span", "dy", "dz"]), ("FDTD", ["dx", "dy", "dz", "simulation time", "use early shutoff"])):
            print(f"  {obj}: " + ", ".join(f"{k}={fdtd.getnamed(obj, k)}" for k in keys))
        if case[2] > 0:
            print("  post: " + ", ".join(f"{k}={fdtd.getnamed('post', k)}" for k in ("radius", "x", "y", "z span", "index")))
        if case[3] == "m25":
            print("  post_mesh: " + ", ".join(f"{k}={fdtd.getnamed('post_mesh', k)}" for k in ("x span", "y", "y span", "z span", "dx", "dy", "dz")))
        print(f"  {case_name(*case)}: saved {out}")
    else:
        SIM_TIME = 60e-15                         # plumbing only, numbers meaningless
        run_case(fdtd, case, out, os.path.join(out, f"post_strip_probe_{case_name(*case)}.fsp"))
    fdtd.close()
