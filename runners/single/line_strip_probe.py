"""
Line-strip probe: SiN LINES / dashes / ribbon / bar beside a SiN strip guide vs equal-area post, vs distributed Born.

Study dir: results/line_strip_probe/ (RUN_NAME = module name on Athena). Job ID: not yet dispatched.
Date: 2026-10-25.
Purpose: does a connected SiN side element (12-um line, arc, dashes, ribbon, +23 deg bar) scatter like a
distributed Born element of its own mask, and how much power does it carry along itself (eta_line)?
Plan (reviewed): scratchpad gptLINES/PLAN_lines_test.md, 16 cases; analysis: scratchpad stripprobe/lines/analyze_lines.py.

Scene = post_strip_probe.py enlarged (its numerics, constants and helpers are imported): strip W 1.0 um, h 350 nm,
n 1.97 in 1.444, 1619.4 nm, TE0 mode source at x = -8 (fwd) or +8 (bwd), FDTD x[-9,9] y[-3,5] z[-3,3] um,
strip x[-12,12] through the PML. Closed DFT box x +-7, y[-2.5,4.5], z +-2.5. Common recorders in EVERY case
(references included, they are the subtraction partners): 3D E/H + index over x[-6.3,6.3] y[0.55,1.85] z +-0.25,
11 transverse sections (x-normal, box cross-section), 6 end-cap planes, 6 point time traces.
Mode solves (mode-expansion monitors, profiles saved): refs at the two box x faces (core modes); line/arc/ribbon
at their interior sections. Fixed 1 ps, early shutoff off. Mesh "prod" = post_strip_probe prod; "m25" adds ONE
common 25-nm xyz override over the 3D recorder, in references too (no shape-dependent mesh boxes).
NOT implemented: two DFT accumulation windows in one solve (the GPU solver drops monitor apodization, CLAUDE.md sec. 2).

Array: task t runs the cases i with i % SLURM_ARRAY_TASK_COUNT == t (no array -> all); an existing npz is skipped.
Dispatch (2 tasks x 8 cases):  bash athena/deploy_athena.sh --option2 --run=line_strip_probe --pol-array
Local:  python runners/single/line_strip_probe.py build <case idx> <out.fsp>    (no solve)
        python runners/single/line_strip_probe.py smoke <case idx> <out dir>    (CPU, 40 fs, plumbing only)
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.single.post_strip_probe import (lumapi, LAM, BAND, N_CORE, N_CLAD, H, C0, F0, DX_PROD, DYZ_BG,
                                             CORE_DY, CORE_DZ, CORE_Y_SPAN, FINE, _span, _setall)

# ── knobs (um unless noted) ───────────────────────────────────────────────────
VERSION = "v1"
W_STRIP = 1.0
Y_LINE = 1.2
SIM = dict(x=(-9.0, 9.0), y=(-3.0, 5.0), z=(-3.0, 3.0))
BOX = dict(x=(-7.0, 7.0), y=(-2.5, 4.5), z=(-2.5, 2.5))
REC = dict(x=(-6.3, 6.3), y=(0.55, 1.85), z=(-0.25, 0.25))     # 3D drive/material recorder = m25 override
X_SRC = 8.0                                                     # fwd source at -X_SRC, bwd at +X_SRC
SECTIONS = [-5.8, -5.5, -5.3, -0.4, -0.1, 0.0, 0.1, 0.4, 5.3, 5.5, 5.8]   # transverse planes (line/arc/dash gap/ribbon)
CAPS = [-6.2, -6.05, -0.8, 0.8, 6.05, 6.2]                     # end-cap planes: line+arc 6.2, dash 6.05, ribbon 0.8
CAP_Y, CAP_Z = (0.2, 2.7), (-1.0, 1.0)                          # cap window (covers line y 1.2 and arc end y 1.6)
TRACES = {"core": (3.0, 0.0, 0.0), "line_c": (0.0, 1.2, 0.0), "line_m": (-6.0, 1.2, 0.0), "line_p": (6.0, 1.2, 0.0),
          "arc_p": (6.0, 1.6, 0.0), "remote": (1.5, 3.5, 0.5)}
N_MODES = 8                                                     # modes solved per section (sorted by neff)
SIM_TIME = 1000e-15
R_ARC = 45.2
DASHES = [(0.15, 0.35), (0.65, 1.01), (1.31, 1.58), (1.88, 2.21), (2.51, 2.75), (3.05, 3.43), (3.73, 4.02),
          (4.32, 4.63), (4.93, 5.15), (5.45, 5.85)]
# shape -> (rectangles [(xc, yc, L, w, angle_deg)], arcs [(R, w, half-chord)], circles [(xc, yc, r)], mode-solve sections)
SHAPES = {
    "bare":   ([], [], [], []),
    "line":   ([(0.0, Y_LINE, 12.0, 0.1, 0.0)], [], [], [-5.8, -5.5, 0.0, 5.5, 5.8]),
    "arc":    ([], [(R_ARC, 0.1, 6.0)], [], [-5.8, -5.5, 0.0, 5.5, 5.8]),
    "dash":   ([(s * 0.5 * (a + b), Y_LINE, b - a, 0.1, 0.0) for a, b in DASHES for s in (-1, 1)], [], [], []),
    "ribbon": ([(0.0, Y_LINE, 1.2, 0.2, 0.0)], [], [], [-0.4, -0.1, 0.0, 0.1, 0.4]),
    "bar":    ([(0.0, Y_LINE, 0.6, 0.1, 23.0)], [], [], []),
    "circle": ([], [], [(0.0, Y_LINE, float(np.sqrt(0.06 / np.pi)))], []),
}
ORDER = ["bare", "line", "arc", "dash", "ribbon", "bar", "circle"]
CASES = ([(s, "prod", "fwd") for s in ORDER] + [(s, "m25", "fwd") for s in ORDER]
         + [("bare", "m25", "bwd"), ("bar", "m25", "bwd")])
FACES = ["box_x1", "box_x2", "box_y1", "box_y2", "box_z1", "box_z2"]
U = 1e-6


def case_name(shape, mesh, direction):
    key = json.dumps([VERSION, SHAPES[shape][:3], SIM, BOX, REC, SIM_TIME, X_SRC], sort_keys=True)
    return f"{shape}_{mesh}_{direction}_{hashlib.sha1(key.encode()).hexdigest()[:6]}"


def add_shape(fdtd, shape):
    rects, arcs, circles, _ = SHAPES[shape]
    for i, (xc, yc, L, w, ang) in enumerate(rects):
        fdtd.addrect()
        _setall(fdtd, {"name": f"{shape}_{i}", "material": "<Object defined dielectric>", "index": N_CORE,
                       "x": xc * U, "y": yc * U, "x span": L * U, "y span": w * U, "z": 0.0, "z span": H})
        if ang:
            _setall(fdtd, {"first axis": "z", "rotation 1": ang})
    for i, (R, w, half) in enumerate(arcs):     # centreline y = Y_LINE + R - sqrt(R^2 - x^2), |x| <= half
        th = np.degrees(np.arcsin(half / R))
        fdtd.addring()
        _setall(fdtd, {"name": f"{shape}_{i}", "material": "<Object defined dielectric>", "index": N_CORE,
                       "x": 0.0, "y": (Y_LINE + R) * U, "z": 0.0, "z span": H,
                       "inner radius": (R - w / 2) * U, "outer radius": (R + w / 2) * U,
                       "theta start": 270.0 - th, "theta stop": 270.0 + th})
    for i, (xc, yc, r) in enumerate(circles):
        fdtd.addcircle()
        _setall(fdtd, {"name": f"{shape}_{i}", "material": "<Object defined dielectric>", "index": N_CORE,
                       "radius": r * U, "x": xc * U, "y": yc * U, "z": 0.0, "z span": H})


def _plane(fdtd, name, x, y, z):
    fdtd.addpower()
    fdtd.set("name", name)
    fdtd.set("monitor type", "2D X-normal")
    fdtd.set("x", x * U)
    _setall(fdtd, _span("y", (y[0] * U, y[1] * U)))
    _setall(fdtd, _span("z", (z[0] * U, z[1] * U)))


def _mode_solve(fdtd, name, dft, x):
    fdtd.addmodeexpansion()
    _setall(fdtd, {"name": name, "monitor type": "2D X-normal", "x": x * U, "mode selection": "user select",
                   "selected mode numbers": np.arange(1, N_MODES + 1)})
    _setall(fdtd, _span("y", (BOX["y"][0] * U, BOX["y"][1] * U)))
    _setall(fdtd, _span("z", (BOX["z"][0] * U, BOX["z"][1] * U)))
    fdtd.setexpansion(name, dft)


def build(fdtd, shape, mesh, direction):
    fdtd.switchtolayout()
    fdtd.deleteall()
    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    for a in "xyz":
        _setall(fdtd, _span(a, (SIM[a][0] * U, SIM[a][1] * U)))
        fdtd.set("define " + a + " mesh by", "maximum mesh step")
        fdtd.set("allow grading in " + a, 1)
    _setall(fdtd, {"dx": DX_PROD, "dy": DYZ_BG, "dz": DYZ_BG, "mesh type": "custom non-uniform",
                   "grading factor": 1.41421, "mesh refinement": "conformal variant 0", "dt stability factor": 0.7,
                   "background material": "<Object defined dielectric>", "background index": N_CLAD,
                   "simulation time": SIM_TIME, "use early shutoff": 0})
    fdtd.setdevice("GPU" if config.USE_GPU else "CPU")

    fdtd.addmesh()                                   # core override, identical to post_strip_probe
    _setall(fdtd, {"name": "core_mesh", "x": 0.0, "x span": (SIM["x"][1] - SIM["x"][0] + 2) * U,
                   "y": 0.0, "y span": CORE_Y_SPAN, "z": 0.0, "z span": H,
                   "override x mesh": 0, "override y mesh": 1, "override z mesh": 1, "dy": CORE_DY, "dz": CORE_DZ})
    if mesh == "m25":                                # ONE common box, references included
        fdtd.addmesh()
        fdtd.set("name", "rec_mesh")
        for a in "xyz":
            _setall(fdtd, _span(a, (REC[a][0] * U, REC[a][1] * U)))
            fdtd.set("override " + a + " mesh", 1)
            fdtd.set("d" + a, FINE)

    fdtd.addrect()
    _setall(fdtd, {"name": "strip", "material": "<Object defined dielectric>", "index": N_CORE,
                   "x": 0.0, "x span": 24 * U, "y": 0.0, "y span": W_STRIP * U, "z": 0.0, "z span": H})
    add_shape(fdtd, shape)

    fdtd.addmode()
    _setall(fdtd, {"name": "src", "injection axis": "x-axis",
                   "direction": "Forward" if direction == "fwd" else "Backward",
                   "x": (-X_SRC if direction == "fwd" else X_SRC) * U, "y": 0.0, "y span": 3.0 * U, "z": 0.0, "z span": 3.0 * U,
                   "mode selection": "fundamental TE mode", "override global source settings": 0})
    df = C0 * BAND / LAM**2
    fdtd.setglobalsource("set frequency", 1)
    fdtd.setglobalsource("frequency start", F0 - df / 2)
    fdtd.setglobalsource("frequency stop", F0 + df / 2)
    fdtd.setglobalmonitor("use source limits", 1)
    fdtd.setglobalmonitor("frequency points", 1)     # = c/LAM (asserted after the run)

    for name in FACES:
        ax, side = name[4], int(name[5]) - 1
        fdtd.addpower()
        fdtd.set("name", name)
        fdtd.set("monitor type", "2D " + ax.upper() + "-normal")
        for a in "xyz":
            if a == ax:
                fdtd.set(a, BOX[a][side] * U)
            else:
                _setall(fdtd, _span(a, (BOX[a][0] * U, BOX[a][1] * U)))
    for i, x in enumerate(SECTIONS):
        _plane(fdtd, f"sec_{i}", x, BOX["y"], BOX["z"])
    for i, x in enumerate(CAPS):
        _plane(fdtd, f"cap_{i}", x, CAP_Y, CAP_Z)
    fdtd.addpower()
    fdtd.set("name", "rec")
    fdtd.set("monitor type", "3D")
    fdtd.addindex()
    fdtd.set("name", "rec_index")
    fdtd.set("monitor type", "3D")
    for name in ("rec", "rec_index"):
        for a in "xyz":
            fdtd.setnamed(name, a, 0.5 * (REC[a][0] + REC[a][1]) * U)
            fdtd.setnamed(name, a + " span", (REC[a][1] - REC[a][0]) * U)
    for name, (x, y, z) in TRACES.items():
        fdtd.addtime()
        _setall(fdtd, {"name": "t_" + name, "monitor type": "Point", "x": x * U, "y": y * U, "z": z * U})

    if shape == "bare":                              # core modes at the box x faces (the same for every shape)
        _mode_solve(fdtd, "mx_box_x1", "box_x1", BOX["x"][0])
        _mode_solve(fdtd, "mx_box_x2", "box_x2", BOX["x"][1])
    for x in SHAPES[shape][3]:
        _mode_solve(fdtd, f"mx_sec_{SECTIONS.index(x)}", f"sec_{SECTIONS.index(x)}", x)


def _fields(fdtd, name, out):
    Ed, Hd = fdtd.getresult(name, "E"), fdtd.getresult(name, "H")
    for a in "xyz":
        out[f"{name}_{a}"] = np.ravel(Ed[a])
    out[f"{name}_E"] = Ed["E"][:, :, :, 0, :].astype(np.complex64)   # GPU fields are single precision
    out[f"{name}_H"] = Hd["H"][:, :, :, 0, :].astype(np.complex64)


def _modes(fdtd, name, out):
    """Mode profiles of a mode-expansion monitor (E, H on its own grid) and their neff."""
    mp = fdtd.getresult(name, "mode profiles")
    out[f"{name}_neff"] = np.ravel(fdtd.getresult(name, "neff")["neff"])
    for a in "yz":
        out[f"{name}_{a}"] = np.ravel(mp[a])
    out[f"{name}_E"] = np.stack([np.squeeze(mp[f"E{m}"]) for m in range(1, N_MODES + 1)]).astype(np.complex64)
    out[f"{name}_H"] = np.stack([np.squeeze(mp[f"H{m}"]) for m in range(1, N_MODES + 1)]).astype(np.complex64)


def run_case(fdtd, case, out_dir, fsp):
    shape, mesh, direction = case
    build(fdtd, *case)
    fdtd.save(fsp)
    fdtd.run()
    f = float(np.ravel(fdtd.getresult("box_x1", "f"))[0])
    assert abs(C0 / f - LAM) < 0.05e-9, f"monitor at {C0/f*1e9:.3f} nm, not {LAM*1e9} nm"
    out = dict(shape=shape, mesh=mesh, direction=direction, version=VERSION, geometry=json.dumps(SHAPES[shape][:3]),
               f=f, source_power=float(np.ravel(fdtd.sourcepower(f))[0]),
               sections=np.array(SECTIONS), caps=np.array(CAPS),
               grid_x=np.ravel(fdtd.getresult("FDTD", "x")), grid_y=np.ravel(fdtd.getresult("FDTD", "y")),
               grid_z=np.ravel(fdtd.getresult("FDTD", "z")))
    for name in FACES + [f"sec_{i}" for i in range(len(SECTIONS))] + [f"cap_{i}" for i in range(len(CAPS))] + ["rec"]:
        _fields(fdtd, name, out)
    ix = fdtd.getresult("rec_index", "index")             # realized (conformal-averaged) material on its own grid
    out["rec_index"] = np.stack([np.squeeze(ix["index_" + a]) for a in "xyz"], -1).astype(np.complex64)
    for a in "xyz":
        out[f"rec_index_{a}"] = np.ravel(ix[a])
    for name in (["mx_box_x1", "mx_box_x2"] if shape == "bare" else []) + [f"mx_sec_{SECTIONS.index(x)}" for x in SHAPES[shape][3]]:
        _modes(fdtd, name, out)
    for name in TRACES:
        T = fdtd.getresult("t_" + name, "E")
        out[f"t_{name}_t"] = np.ravel(T["t"])
        out[f"t_{name}_E"] = np.squeeze(T["E"]).astype(np.float32)
    path = os.path.join(out_dir, f"line_strip_probe_{case_name(*case)}.npz")
    np.savez(path, **out)
    n_cells = out["grid_x"].size * out["grid_y"].size * out["grid_z"].size
    tails = {k: np.sum(out[f"t_{k}_E"]**2, -1) for k in TRACES}
    worst = max(t[int(0.9 * t.size):].max() / max(t.max(), 1e-300) for t in tails.values())
    print(f"  {case_name(*case):28s} cells {n_cells/1e6:.2f} M  P_src {out['source_power']:.3e} W  "
          f"worst late/peak |E|^2 {worst:.1e}  -> {path}", flush=True)
    return path


def run(cfg=None):
    names = [case_name(*c) for c in CASES]
    assert len(set(names)) == len(names) == 16, "16 unique cases (= output files)"
    task = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    mine = [c for i, c in enumerate(CASES) if task == "" or i % n_tasks == int(task)]
    print(f"[line_strip_probe] task {task or '-'} of {n_tasks}: {[case_name(*c) for c in mine]}", flush=True)
    fdtd = lumapi.FDTD(hide=True)
    try:
        for c in mine:
            if os.path.exists(os.path.join(config.RESULTS_DIR, f"line_strip_probe_{case_name(*c)}.npz")):
                print(f"  {case_name(*c)}: npz exists, skipped")
                continue
            run_case(fdtd, c, config.RESULTS_DIR, os.path.join(config.LAYOUTS_DIR, f"line_strip_probe_{case_name(*c)}.fsp"))
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
        names = [fdtd.get("name", i) for i in range(1, int(fdtd.getnumber()) + 1)]
        print(f"  {case_name(*case)}: {len(names)} objects: " + ", ".join(n for n in names if not n.startswith(("sec_", "cap_", "t_"))))
        print(f"  + {sum(n.startswith('sec_') for n in names)} sections, {sum(n.startswith('cap_') for n in names)} caps, "
              f"{sum(n.startswith('t_') for n in names)} time traces")
        for n in names:
            if n.startswith(case[0] + "_"):
                t = fdtd.getnamed(n, "type")
                keys = {"Rectangle": ["x", "y", "x span", "y span", "z span", "rotation 1"], "Circle": ["x", "y", "radius", "z span"],
                        "Ring": ["y", "inner radius", "outer radius", "theta start", "theta stop", "z span"]}[t]
                print(f"  {n} {t}: " + ", ".join(f"{k}={fdtd.getnamed(n, k):.5g}" for k in keys))
        print("  src: " + ", ".join(f"{k}={fdtd.getnamed('src', k)}" for k in ("x", "direction", "mode selection")))
        for m in ("core_mesh", "rec_mesh"):
            if m in names:
                print(f"  {m}: " + ", ".join(f"{k}={fdtd.getnamed(m, k):.4g}" for k in ("x span", "y", "y span", "z span", "dy", "dz")))
        print("  modes: " + ", ".join(n for n in names if n.startswith("mx_")))
        print(f"  saved {out}")
    else:
        SIM_TIME = 40e-15                         # plumbing only, numbers meaningless
        run_case(fdtd, case, out, os.path.join(out, f"line_strip_probe_{case_name(*case)}.fsp"))
    fdtd.close()
