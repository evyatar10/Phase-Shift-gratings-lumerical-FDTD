"""
Post mesh probe: scattering cross-section of ONE small SiN post vs FDTD mesh.

Study dir: results/post_mesh_probe/ (RUN_NAME = module name on Athena). Job ID: not yet dispatched.
Date: 2026-10-24.
Purpose: decide what mesh is needed to simulate SMALL full-height SiN posts (r 25-100 nm)
correctly, by measuring sigma_scat(r, mesh) of an isolated post and comparing to the 12.5 nm mesh.

Scene: full-height SiN cylinder (n 1.97, h 350 nm, axis z, at the origin) in SiO2 (1.444),
object-defined indices (bragg_device's constant-material "object" mode), conformal variant 0,
uniform mesh-override cube around post + TFSF + monitor box, 50 nm max step outside.
TFSF plane wave propagating +x, two polarisations:
  "Ey" = E perpendicular to the post axis (in-plane, GAM_perp = 2 de/(ep+eo))
  "Ez" = E along the post axis              (GAM_par  = de/eo)
E_x is impossible for +x propagation; by the cylinder's rotational symmetry about z an E_x
case (propagating along y) is the same physics as "Ey", so these two are the distinct cases.
The TFSF angle convention is asserted at runtime from an incident-field time monitor.

sigma = scattered power through a closed box of 6 DFT power monitors in the scattered-field
region / source intensity. TFSF leakage is linear in the field, so the r = 0 run at the same
mesh+pol is subtracted at the FIELD level on the 6 faces (E_s = E(r) - E(0), H_s likewise);
`sigma` is that subtracted value, `sigma_raw` the plain transmission sum (leak included).

CPU ONLY: TFSF is on Ansys' GPU-unsupported list (with BFAST; see
runners/lumopt2_design/HANDOFF.md), so config.USE_GPU is deliberately ignored and the
device is forced to CPU even inside the GPU job (athena_run's patch sets GPU first).

Local build-only check (no solve):  python runners/single/post_mesh_probe.py <out.fsp>
"""
import os
import sys
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import j1

import config

try:
    import lumapi
except ImportError:
    _spec = importlib.util.spec_from_file_location("lumapi", config.LUMAPI_PATH)
    lumapi = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(lumapi)

# ── knobs ─────────────────────────────────────────────────────────────────────
LAM = 1619.5e-9
BAND = 40e-9                       # source band (+-20 nm), centred in FREQUENCY on c/LAM
N_POST, N_CLAD = 1.97, 1.444
H_POST = 350e-9
RADII_NM = [0, 25, 35, 50, 75, 100]   # r = 0 is the TFSF-leakage reference, subtracted
MESH_NM = [50, 25, 12.5]              # cheapest first: a setup error dies in seconds
POLS = {"Ey": 0.0, "Ez": 90.0}        # TFSF polarization angle for injection axis x
SIM_SPAN = 2.2e-6                     # cube, PML outside
TFSF_SPAN = 0.6e-6                    # cube; post is 0.2 x 0.2 x 0.35 um at most
BOX_SPAN = 0.9e-6                     # closed monitor box, scattered-field region
OVR_SPAN = 1.1e-6                     # uniform mesh override encloses TFSF + box
BG_STEP = 50e-9                       # outside the override, production max step
SIM_TIME = 1000e-15                   # cap; auto shutoff ends it after the pulse leaves
AUTO_SHUTOFF = 1e-7                   # production value
C0 = 299792458.0
F0 = C0 / LAM
K = 2 * np.pi * N_CLAD / LAM
FACES = [("box_x1", "x", -1), ("box_x2", "x", +1), ("box_y1", "y", -1),
         ("box_y2", "y", +1), ("box_z1", "z", -1), ("box_z2", "z", +1)]


def build(fdtd, r, mesh, pol):
    fdtd.switchtolayout()
    fdtd.deleteall()
    fdtd.addfdtd()
    fdtd.set("dimension", "3D")
    for a in "xyz":
        fdtd.set(a, 0.0)
        fdtd.set(a + " span", SIM_SPAN)
        fdtd.set("define " + a + " mesh by", "maximum mesh step")
        fdtd.set("d" + a, BG_STEP)
        fdtd.set("allow grading in " + a, 1)
        fdtd.set("force symmetric " + a + " mesh", 1)
    fdtd.set("mesh type", "custom non-uniform")
    fdtd.set("grading factor", 1.41421)
    fdtd.set("mesh refinement", "conformal variant 0")
    fdtd.set("dt stability factor", 0.7)
    fdtd.set("background material", "<Object defined dielectric>")
    fdtd.set("background index", N_CLAD)
    fdtd.set("simulation time", SIM_TIME)
    fdtd.set("auto shutoff min", AUTO_SHUTOFF)
    fdtd.setdevice("CPU")   # TFSF is GPU-unsupported (see header)

    fdtd.addmesh()
    fdtd.set("name", "post_mesh")
    for a in "xyz":
        fdtd.set(a, 0.0)
        fdtd.set(a + " span", OVR_SPAN)
        fdtd.set("override " + a + " mesh", 1)
        fdtd.set("d" + a, mesh * 1e-9)

    if r > 0:
        fdtd.addcircle()
        fdtd.set("name", "post")
        fdtd.set("material", "<Object defined dielectric>")
        fdtd.set("index", N_POST)
        fdtd.set("radius", r * 1e-9)
        fdtd.set("x", 0.0)
        fdtd.set("y", 0.0)
        fdtd.set("z", 0.0)
        fdtd.set("z span", H_POST)

    fdtd.addtfsf()
    fdtd.set("name", "tfsf")
    fdtd.set("injection axis", "x")
    fdtd.set("direction", "forward")
    fdtd.set("polarization angle", POLS[pol])
    fdtd.set("override global source settings", 0)   # addtfsf defaults to 1 (own 400-700 nm band)
    for a in "xyz":
        fdtd.set(a, 0.0)
        fdtd.set(a + " span", TFSF_SPAN)
    df = C0 * BAND / LAM**2
    fdtd.setglobalsource("set frequency", 1)
    fdtd.setglobalsource("frequency start", F0 - df / 2)
    fdtd.setglobalsource("frequency stop", F0 + df / 2)
    fdtd.setglobalmonitor("use source limits", 1)
    fdtd.setglobalmonitor("frequency points", 1)   # = centre frequency = c/LAM (asserted after run)

    h = BOX_SPAN / 2
    for name, ax, sign in FACES:
        fdtd.addpower()
        fdtd.set("name", name)
        fdtd.set("monitor type", "2D " + ax.upper() + "-normal")
        for a in "xyz":
            if a == ax:
                fdtd.set(a, sign * h)
            else:
                fdtd.set(a, 0.0)
                fdtd.set(a + " span", BOX_SPAN)

    # incident-field probe inside TFSF (polarisation assert) and scattered-field probe (decay)
    for name, pos in (("t_inc", (-0.2e-6, 0.2e-6, 0.25e-6)), ("t_scat", (0.4e-6, 0.4e-6, 0.3e-6))):
        fdtd.addtime()
        fdtd.set("name", name)
        fdtd.set("monitor type", "Point")
        fdtd.set("x", pos[0])
        fdtd.set("y", pos[1])
        fdtd.set("z", pos[2])


def face_power(E, H, x, y, z, ax, sign):
    """Outward flux 0.5 Re(E x H*) . n through one face (single frequency, uniform-mesh trapz)."""
    E = np.squeeze(E)
    H = np.squeeze(H)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[ax]   # S_ax = E_i H_j* - E_j H_i*
    S = 0.5 * np.real(E[..., i] * np.conj(H[..., j]) - E[..., j] * np.conj(H[..., i]))
    u, v = {"x": (y, z), "y": (x, z), "z": (x, y)}[ax]
    return sign * trapezoid(trapezoid(S, v, axis=1), u, axis=0)


def solve_case(fdtd, r, mesh, pol, path, ref):
    build(fdtd, r, mesh, pol)
    fdtd.save(path)
    fdtd.run()

    f = float(np.ravel(fdtd.getresult("box_x1", "f"))[0])
    assert abs(C0 / f - LAM) < 0.05e-9, f"monitor at {C0/f*1e9:.3f} nm, not {LAM*1e9} nm"
    p_src = float(np.ravel(fdtd.sourcepower(f))[0])
    I_src = float(np.ravel(fdtd.sourceintensity(f))[0])

    fields, P_raw_T, P_raw_F, P_sub = {}, 0.0, 0.0, 0.0
    for name, ax, sign in FACES:
        Ed = fdtd.getresult(name, "E")
        Hd = fdtd.getresult(name, "H")
        x, y, z = (np.ravel(Ed[a]) for a in "xyz")
        E, H = Ed["E"], Hd["H"]
        fields[name] = (E, H, y)
        P_raw_T += sign * float(np.ravel(fdtd.transmission(name))[0]) * p_src
        P_raw_F += face_power(E, H, x, y, z, ax, sign)
        if ref is not None:
            E0, H0, y0 = ref[name]
            assert np.allclose(y, y0), f"{name}: grid differs from the r = 0 reference"
            P_sub += face_power(E - E0, H - H0, x, y, z, ax, sign)
        if name == "box_x1":   # achieved mesh on the face must be the override step
            dy = np.median(np.diff(y))
            assert abs(dy - mesh * 1e-9) < 0.02 * mesh * 1e-9, f"face dy {dy*1e9:.2f} nm != {mesh} nm"

    Ei = np.squeeze(fdtd.getresult("t_inc", "E")["E"])           # (nt, 3)
    comp = np.max(np.abs(Ei), axis=0)
    want = {"Ey": 1, "Ez": 2}[pol]
    if r == 0:
        assert comp[want] > 10 * max(np.delete(comp, want)), f"{pol}: incident |E| per component {comp}"
    Es = np.sum(np.abs(np.squeeze(fdtd.getresult("t_scat", "E")["E"]))**2, axis=-1)
    tail = Es[int(0.9 * len(Es)):].max() / Es.max()

    print(f"  r {r:5.1f} nm  mesh {mesh:5.1f} nm  {pol}: sigma_raw {P_raw_T/I_src:.4e} m^2  "
          f"flux check field/transmission {P_raw_F/P_raw_T:.4f}  "
          f"sigma_sub {P_sub/I_src:.4e} m^2  tail/peak |E|^2 {tail:.1e}", flush=True)
    return P_sub / I_src, P_raw_T / I_src, tail, fields


def born_sigma(r_nm, pol):
    """First-order (Born, quasi-static GAM of postarray_born.py) sigma of one post, with the
    finite-size form factor sinc(k h n_z/2) * 2 J1(q r)/(q r); also the point-dipole k^4 |alpha|^2/6pi."""
    ep, eo = N_POST**2, N_CLAD**2
    gam = {"Ey": 2 * (ep - eo) / (ep + eo), "Ez": (ep - eo) / eo}[pol]
    r = r_nm * 1e-9
    alpha = np.pi * r**2 * H_POST * gam
    point = K**4 * alpha**2 / (6 * np.pi)
    th = np.linspace(0, np.pi, 361)[:, None]
    ph = np.linspace(0, 2 * np.pi, 721)[None, :]
    n = np.stack(np.broadcast_arrays(np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th) + 0 * ph))
    e = {"Ey": 1, "Ez": 2}[pol]
    q = K * np.hypot(1 - n[0], n[1])                       # in-plane momentum transfer, incidence +x
    disk = np.where(q * r > 1e-9, 2 * j1(q * r) / np.maximum(q * r, 1e-30), 1.0)
    rod = np.sinc(K * H_POST * n[2] / (2 * np.pi))         # np.sinc(x) = sin(pi x)/(pi x)
    integrand = (1 - n[e]**2) * (disk * rod)**2 * np.sin(th)
    full = K**4 * alpha**2 / (16 * np.pi**2) * trapezoid(trapezoid(integrand, ph[0], axis=1), th[:, 0])
    return point, full


def run(cfg=None):
    out_dir = config.RESULTS_DIR
    fsp = os.path.join(config.LAYOUTS_DIR, "post_mesh_probe.fsp")
    fdtd = lumapi.FDTD(hide=True)
    fdtd.setresource("FDTD", 1, "device type", "CPU")   # undo athena_run's GPU patch: TFSF is CPU-only
    fdtd.setresource("FDTD", 1, "processes", 1)
    fdtd.setresource("FDTD", 1, "threads", int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    rows = []
    try:
        for mesh in MESH_NM:
            for pol in POLS:
                ref = None
                for r in RADII_NM:
                    s, s_raw, tail, fields = solve_case(fdtd, r, mesh, pol, fsp, ref)
                    if r == 0:
                        ref = fields
                    rows.append((r, mesh, pol, s, s_raw, tail))
                    np.savez(os.path.join(out_dir, "post_mesh_probe.npz"),
                             r=np.array([w[0] for w in rows], float), mesh=np.array([w[1] for w in rows], float),
                             pol=np.array([w[2] for w in rows]), sigma=np.array([w[3] for w in rows]),
                             sigma_raw=np.array([w[4] for w in rows]), tail=np.array([w[5] for w in rows]))
    finally:
        fdtd.close()

    sig = {(w[0], w[1], w[2]): w[3] for w in rows}
    for pol in POLS:
        print(f"\n{pol}: sigma(r, mesh) / sigma(r, 12.5 nm)   [sigma in um^2; Born = point dipole / with form factor]")
        print("  r nm  " + "  ".join(f"{m:>7.1f}" for m in MESH_NM) + "   sigma(12.5)   Born_pt    Born_ff")
        for r in RADII_NM[1:]:
            base = sig[(r, 12.5, pol)]
            pt, ff = born_sigma(r, pol)
            print(f"  {r:4d}  " + "  ".join(f"{sig[(r, m, pol)] / base:7.3f}" for m in MESH_NM)
                  + f"   {base*1e12:.3e}   {pt*1e12:.3e}  {ff*1e12:.3e}")
        print("  (r = 0 rows are the reference; raw leak sigma per mesh: "
              + ", ".join(f"{m}: {w[4]*1e12:.2e}" for m in MESH_NM for w in rows if w[0] == 0 and w[1] == m and w[2] == pol)
              + " um^2)")
    print(f"\nsaved {os.path.join(out_dir, 'post_mesh_probe.npz')}")


if __name__ == "__main__":   # local build-only check: r = 50 nm, mesh 25 nm, Ey; no solve
    out = sys.argv[1]
    fdtd = lumapi.FDTD(hide=True)
    build(fdtd, 50, 25, "Ey")
    fdtd.save(out)
    fdtd.selectall()
    for i in range(1, int(fdtd.getnumber()) + 1):
        print(f"  {fdtd.get('name', i):10s} {fdtd.get('type', i)}")
    print("post radius", fdtd.getnamed("post", "radius"), "z span", fdtd.getnamed("post", "z span"),
          "index", fdtd.getnamed("post", "index"))
    print("tfsf axis", fdtd.getnamed("tfsf", "injection axis"), "pol angle", fdtd.getnamed("tfsf", "polarization angle"),
          "f start/stop nm", C0 / fdtd.getnamed("tfsf", "frequency stop") * 1e9, C0 / fdtd.getnamed("tfsf", "frequency start") * 1e9)
    print("override dx", fdtd.getnamed("post_mesh", "dx"), "bg index", fdtd.getnamed("FDTD", "index"))
    print("born r50", born_sigma(50, "Ey"), born_sigma(50, "Ez"))
    fdtd.close()
    print("saved", out)
