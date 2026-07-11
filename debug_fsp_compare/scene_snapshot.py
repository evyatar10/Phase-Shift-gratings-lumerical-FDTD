"""Scene-regression harness: prove a bragg_device.py edit did not change the physics.

Builds a fixed set of reference configurations (spanning the builder's code
paths: TE baseline, apodization, inner shift, TM + scatterers, shaped cavity,
two-device) and dumps a CANONICAL TEXT INVENTORY of every object in the scene
(name, type-relevant geometry properties, material/index, plus the FDTD region,
boundary conditions, mesh, ports, and global source/monitor settings).

Usage (build-only; needs a local Lumerical design license, no solve):
    python debug_fsp_compare/scene_snapshot.py --out debug_fsp_compare/snapshots
    # ... edit bragg_device.py ...
    python debug_fsp_compare/scene_snapshot.py --out /tmp/after
    diff -r debug_fsp_compare/snapshots /tmp/after   # empty diff = same scenes

The committed snapshots/ directory is the reference: regenerate + diff before
committing any geometry change. Floats are rounded to 1e-15 m so the dump is
stable across runs.
"""
import argparse
import copy
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_DIR)

import numpy as np

from bragg_device import PiShiftBraggFDTD
from sim_helpers import apply_monitor_overrides
from simulation_config import SimulationConfig


# ── Reference configurations ──────────────────────────────────────────────────
def _te_baseline():
    cfg = SimulationConfig()
    cfg.mesh.simulation_mode = "optimization"
    cfg.grating.cavity_neg_detuning_nm = 5.76
    return cfg, {}


def _te_apod_tanh():
    cfg, _ = _te_baseline()
    cfg.apodization.enabled = True
    cfg.apodization.method = "tanh"
    cfg.apodization.n_apod_periods_each_side = 10
    cfg.apodization.center_mod_depth_nm = 20.0
    return cfg, {}


def _te_shift_innersize():
    cfg, _ = _te_baseline()
    return cfg, dict(innermost_tooth_shift_m=100e-9,
                     use_apodization=True,
                     n_apod_periods_each_side=1,
                     center_mod_depth_nm=125.0)


def _tm_anchored_scatterer():
    # The anchored TM device with the mirrored scatterer pair (exercises
    # _add_scatterers and the TM symmetry parities).
    from runners.sweeps._tm_base import build_base
    cfg = build_base()
    cfg.scatterer.radius_m = 150e-9
    cfg.scatterer.x_m = 810e-9
    return cfg, {}


def _tm_shaped_cavity():
    cfg, _ = _te_baseline()
    cfg.source.polarization = "TM"
    cfg.grating.pitch_m = 516.83e-9
    cfg.geometry.corrugation_depth_m = 400e-9
    return cfg, dict(cavity_shape="hann", cavity_shape_depth_m=150e-9,
                     cavity_width_m=1050e-9)


def _two_device():
    cfg, _ = _te_baseline()
    cfg.grating.n_periods_each_side = 20     # keep the pair scene small
    cfg.symmetry.use_y_symmetry = False
    return cfg, dict(n_devices=2, device_gap_m=3.0e-6, device_stagger_m=0.0)


CONFIGS = {
    "te_baseline": _te_baseline,
    "te_apod_tanh": _te_apod_tanh,
    "te_shift_innersize": _te_shift_innersize,
    "tm_anchored_scatterer": _tm_anchored_scatterer,
    "tm_shaped_cavity": _tm_shaped_cavity,
    "two_device": _two_device,
}

# Properties probed per top-level object (missing ones are skipped silently).
OBJ_PROPS = [
    "x", "y", "z", "x span", "y span", "z span", "x min", "x max",
    "z min", "z max", "radius", "index", "material", "mesh order",
    "override mesh order from material database",
    "monitor type", "override global monitor settings", "use source limits",
    "frequency points", "down sample x", "down sample y", "down sample z",
    "override x mesh", "override y mesh", "override z mesh", "dx", "dy", "dz",
]
FDTD_PROPS = [
    "dimension", "x", "x span", "y", "y span", "z", "z span",
    "x min bc", "x max bc", "y min bc", "y max bc", "z min bc", "z max bc",
    "force symmetric y mesh", "force symmetric z mesh",
    "background material", "background index",
    "simulation time", "auto shutoff min", "mesh type",
    "define x mesh by", "define y mesh by", "define z mesh by",
    "dx", "dy", "dz", "allow grading in x", "allow grading in y",
    "allow grading in z", "grading factor", "mesh refinement",
    "dt stability factor",
]
PORT_PROPS = [
    "injection axis", "direction", "mode selection",
    "frequency dependent profile", "x", "y", "y span", "z", "z span",
]


def _fmt(v):
    if isinstance(v, np.ndarray):
        return np.array2string(np.round(v.astype(float), 15), threshold=10**6,
                               max_line_width=120)
    if isinstance(v, float):
        return f"{round(v, 15):.15g}"
    return str(v)


def dump_scene(sim, out_path):
    fdtd = sim.fdtd
    lines = []

    def probe(getter, props, label):
        lines.append(f"### {label}")
        for p in props:
            try:
                lines.append(f"  {p} = {_fmt(getter(p))}")
            except Exception:
                pass

    probe(lambda p: fdtd.getglobalsource(p),
          ["wavelength start", "wavelength stop"], "global source")
    probe(lambda p: fdtd.getglobalmonitor(p),
          ["frequency points", "use source limits"], "global monitor")

    # Enumerate every top-level object by name.
    fdtd.eval('selectall; tmpn = getnumber; tmpnames = "";'
              'for(i=1:tmpn){ tmpnames = tmpnames + get("name", i) + endl; }')
    names = [n for n in str(fdtd.getv("tmpnames")).split("\n") if n]
    lines.append(f"### object count = {len(names)}")
    for name in sorted(names):
        props = FDTD_PROPS if name == "FDTD" else OBJ_PROPS
        probe(lambda p, _n=name: fdtd.getnamed(_n, p), props, f"object {name}")
        if name not in ("FDTD", "mesh_override"):
            try:
                lines.append(f"  vertices = {_fmt(fdtd.getnamed(name, 'vertices'))}")
            except Exception:
                pass

    for port in ("Port_1", "Port_2", "Port_3", "Port_4"):
        try:
            fdtd.getnamed(f"FDTD::ports::{port}", "x")
        except Exception:
            continue
        probe(lambda p, _n=port: fdtd.getnamed(f"FDTD::ports::{_n}", p),
              PORT_PROPS, f"port {port}")
    try:
        lines.append(f"### source port = {fdtd.getnamed('FDTD::ports', 'source port')}")
    except Exception:
        pass

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="snapshot output directory")
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of: " + ",".join(CONFIGS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    todo = args.configs.split(",") if args.configs else list(CONFIGS)

    for key in todo:
        cfg, extra = CONFIGS[key]()
        kwargs = copy.deepcopy(cfg).to_device_kwargs()
        kwargs.update(extra)
        print(f"=== building {key} ===")
        sim = PiShiftBraggFDTD(**kwargs)
        try:
            sim.build()
            sim.update_scan(center_lambda_m=cfg.spectral.center_wavelength_m,
                            width_nm=cfg.spectral.scan_width_nm,
                            n_points=cfg.spectral.n_wl_points)
            apply_monitor_overrides(sim, cfg)
            out_path = os.path.join(args.out, f"{key}.txt")
            dump_scene(sim, out_path)
            print(f"SNAPSHOT_SAVED:{out_path}")
        finally:
            sim.close()


if __name__ == "__main__":
    main()
