"""TINY GPU probe: at what x-size does a FieldRegion source stop launching?

Study dir: runners/lumopt2_design/  |  Created 2026-08-24  |  Job(s): TBD
Purpose: answer the FieldRegion-on-GPU launch question in MINUTES instead of
hours. The failure is a CUDA kernel-LAUNCH error and has nothing to do with our
device, so it does not need our device: this builds an empty box with one field
region and a dipole, and asks only "does the source-mode run launch?".

WHY THIS FILE EXISTS (user rule 2026-08-24): the same question was first asked
with FULL-DEVICE rungs at ~45-70 min each (jobs 136799/136826/136869/136907),
which made every bisection step cost an hour. CLAUDE.md §5 already says to
gate a new gradient method on a TINY problem first; this is that gate.

Each size costs two short solves (fill the monitor, then re-run in source mode)
on a box that is thin in y/z and short in time, so the whole ladder is minutes.

Dispatch:
    SBATCH_MEM=64G LUMOPT2_TIME=01:00:00 bash athena/deploy_athena.sh \
        --lumopt2-design=runners.lumopt2_design.gpu_probe
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

import config
from runners.lumopt2_design import lumopt2_design as eng

# The dispatch harness requires a top-level SPEC + main() for kind
# lumopt2_design (athena/scripts/build_sweep_list.py:171-178). Nothing in this
# probe builds the device, so the spec only carries the run label.
SPEC = eng.CampaignSpec(label="lumopt2_gpu_probe")
N_TASKS = 2      # 0 = x-threshold ladder | 1 = tiling superposition identity

DX_NM = 50.0                     # same cell size as the real optimization region
X_CELLS = [256, 512, 768, 896, 960, 1000, 1024, 1040, 1080, 1152, 1536, 2112]
Y_SPAN_UM = 1.5                  # a few cells; y is measured NOT to matter
Z_SPAN_UM = 1.5
SIM_TIME_FS = 10.0               # just long enough to launch and step
DIPOLE = "probe_dipole"          # explicit: adddipole's default name is not "dipole"
PAD_UM = 4.0                     # FDTD box margin around the widest region


def _fresh_scene(fdtd, x_span_m):
    """Empty box + dipole, sized for the widest region we will test.

    lumapi's add* helpers do NOT take geometry keywords in this build
    (measured, job 136914: "unexpected keyword argument 'x'") — use
    add() then set(), the same pattern as bragg_device/build_base_fsp.
    """
    fdtd.switchtolayout()
    fdtd.deleteall()
    fdtd.addfdtd()
    for k, v in (("dimension", "3D"),
                 ("x", 0.0), ("x span", x_span_m + PAD_UM * 1e-6),
                 ("y", 0.0), ("y span", (Y_SPAN_UM + 1.0) * 1e-6),
                 ("z", 0.0), ("z span", (Z_SPAN_UM + 0.5) * 1e-6),
                 ("mesh type", "uniform"),
                 ("dx", DX_NM * 1e-9), ("dy", DX_NM * 1e-9),
                 ("dz", DX_NM * 1e-9),
                 ("simulation time", SIM_TIME_FS * 1e-15)):
        fdtd.set(k, v)
    fdtd.adddipole()
    # ★name it explicitly: adddipole's default name is NOT "dipole"
    # (measured, job 136918: setnamed found no such object, died in seconds)
    fdtd.set("name", DIPOLE)
    for k, v in (("x", 0.0), ("y", 0.0), ("z", 0.0),
                 ("override global source settings", 1),
                 ("wavelength start", 1.55e-6), ("wavelength stop", 1.55e-6)):
        fdtd.set(k, v)


def probe(fdtd, n_cells):
    """(launched, note) for a field region n_cells wide in x, in source mode."""
    x_span = n_cells * DX_NM * 1e-9
    _fresh_scene(fdtd, x_span)
    fdtd.addfieldregion()
    fdtd.set("name", "probe_region")
    fdtd.set("monitor type", "2D Z-normal")
    fdtd.set("x", 0.0)
    fdtd.set("x span", x_span)
    fdtd.set("y", 0.0)
    fdtd.set("y span", Y_SPAN_UM * 1e-6)
    fdtd.set("z", 0.0)
    fdtd.set("override global monitor settings", 1)
    fdtd.set("use source limits", 0)
    fdtd.set("frequency points", 1)
    fdtd.set("wavelength center", 1.55e-6)
    fdtd.set("wavelength span", 0.0)
    fdtd.set("source mode", 0)

    # pass 1 — plain monitor, to get a shape-correct dataset to inject back
    fdtd.run("FDTD", "GPU")
    prof = fdtd.getresult("probe_region", "E")

    # pass 2 — THE TEST: same object as a volumetric current source
    fdtd.switchtolayout()
    fdtd.setnamed(DIPOLE, "enabled", False)
    fdtd.setnamed("probe_region", "source mode", True)
    fdtd.select("probe_region")
    fdtd.importdataset(prof)
    try:
        fdtd.run("FDTD", "GPU")
    except Exception as e:                 # loud on purpose: the string IS the datum
        return False, str(e).strip().replace("\n", " | ")[:220]
    E = fdtd.getresult("probe_region", "E")["E"]
    peak = float(np.abs(np.asarray(E)).max())
    return True, f"launched, adjoint-side max|E| = {peak:.4g}"


TILE_TEST_CELLS = 512        # a width that is MEASURED to work single-source
TILE_TEST_N = 4              # 512 / 4 = 128 cells per tile


def tiling_identity(fdtd, resource="GPU"):
    """Does one wide source == the same source split across N tiles?

    This is the correctness gate for CampaignSpec.wg_src_tiles, and it needs
    NO device: source superposition is generic physics. Inject a synthetic
    weighted profile (a) through one region, (b) through N x-partitioned
    tiles carrying disjoint slices of the SAME data, and compare the field
    recorded by an independent monitor. Exact partition + linear superposition
    ⇒ the two fields agree to roundoff.
    """
    x_span = TILE_TEST_CELLS * DX_NM * 1e-9
    _fresh_scene(fdtd, x_span)
    fdtd.setnamed(DIPOLE, "enabled", True)

    fdtd.addfieldregion()                      # the wide source, pass 1
    fdtd.set("name", "src_full")
    for k, v in (("monitor type", "2D Z-normal"), ("x", 0.0),
                 ("x span", x_span), ("y", 0.0), ("y span", Y_SPAN_UM * 1e-6),
                 ("z", 0.0), ("override global monitor settings", 1),
                 ("use source limits", 0), ("frequency points", 1),
                 ("wavelength center", 1.55e-6), ("wavelength span", 0.0),
                 ("source mode", 0)):
        fdtd.set(k, v)

    fdtd.addpower()                            # independent witness monitor
    fdtd.set("name", "witness")
    for k, v in (("monitor type", "2D Z-normal"), ("x", 0.0),
                 ("x span", x_span * 0.5), ("y", 0.0),
                 ("y span", Y_SPAN_UM * 1e-6), ("z", 0.4e-6),
                 ("override global monitor settings", 1),
                 ("use source limits", 0), ("frequency points", 1),
                 ("wavelength center", 1.55e-6), ("wavelength span", 0.0)):
        fdtd.set(k, v)

    fdtd.run("FDTD", resource)                 # fill src_full as a monitor
    prof = fdtd.getresult("src_full", "E")

    # synthetic WEIGHT with structure in x, so a mis-sliced tile cannot cancel
    x = np.squeeze(np.asarray(prof["x"])).astype(float)
    w = (1.0 + np.sin(2 * np.pi * x / (x[-1] - x[0]) * 3.0)).astype(float)
    E = np.asarray(prof["E"])
    shaped = dict(prof)
    shaped["E"] = np.conj(E) * w.reshape((-1,) + (1,) * (E.ndim - 1))

    def _field_from(setup):
        fdtd.switchtolayout()
        fdtd.setnamed(DIPOLE, "enabled", False)
        setup()
        fdtd.run("FDTD", resource)
        return np.asarray(fdtd.getresult("witness", "E")["E"])

    def _one():
        fdtd.setnamed("src_full", "source mode", True)
        fdtd.select("src_full")
        fdtd.importdataset(shaped)

    def _tiled():
        fdtd.setnamed("src_full", "source mode", False)
        edges, bounds = eng.tile_x_edges(x, TILE_TEST_N)
        parts = eng.split_dataset_x(shaped, bounds)
        for t, part in enumerate(parts):
            fdtd.addfieldregion()
            fdtd.set("name", f"src_t{t}")
            for k, v in (("monitor type", "2D Z-normal"),
                         ("x", 0.5 * (edges[t] + edges[t + 1])),
                         ("x span", edges[t + 1] - edges[t]),
                         ("y", 0.0), ("y span", Y_SPAN_UM * 1e-6), ("z", 0.0),
                         ("override global monitor settings", 1),
                         ("use source limits", 0), ("frequency points", 1),
                         ("wavelength center", 1.55e-6),
                         ("wavelength span", 0.0), ("source mode", 1)):
                fdtd.set(k, v)
            fdtd.select(f"src_t{t}")
            fdtd.importdataset(part)

    a = _field_from(_one)
    b = _field_from(_tiled)
    scale = float(np.abs(a).max())
    err = float(np.abs(a - b).max()) / scale if scale else float("inf")
    print(f"[tiling] witness max|E| single {scale:.6e} | tiled "
          f"{float(np.abs(b).max()):.6e} | max REL diff {err:.3e}", flush=True)
    print(f"[tiling] {'PASS' if err < 1e-6 else 'FAIL'} "
          f"(pass = rel diff < 1e-6; exact partition + linear superposition)",
          flush=True)
    return err


def main(task_idx):
    sys.path.insert(0, os.path.dirname(config.LUMAPI_PATH))
    import lumapi
    if task_idx == 1:
        print(f"[tiling] identity test: {TILE_TEST_CELLS} cells, one source "
              f"vs {TILE_TEST_N} tiles, on a DUMMY scene (no device)",
              flush=True)
        with lumapi.FDTD(hide=True) as fdtd:
            tiling_identity(fdtd)
        return
    print(f"[gpu_probe] dx {DX_NM} nm, y span {Y_SPAN_UM} um, "
          f"sim time {SIM_TIME_FS} fs, sizes {X_CELLS}", flush=True)
    rows = []
    with lumapi.FDTD(hide=True) as fdtd:
        for n in X_CELLS:
            try:
                ok, note = probe(fdtd, n)
            except Exception as e:         # a build/import failure is NOT a verdict
                ok, note = None, f"INCONCLUSIVE (before the run): {e}"
            rows.append((n, ok, note))
            tag = {True: "LAUNCHED", False: "REJECTED", None: "INCONCL."}[ok]
            print(f"[gpu_probe] x={n:5d} cells ({n * DX_NM / 1000:7.3f} um)  "
                  f"{tag:9s}  {note}", flush=True)

    good = [n for n, ok, _ in rows if ok is True]
    bad = [n for n, ok, _ in rows if ok is False]
    print(f"\n[gpu_probe] LAUNCHED: {good}\n[gpu_probe] REJECTED: {bad}")
    if good and bad:
        print(f"[gpu_probe] ★THRESHOLD between {max(good)} and {min(bad)} cells "
              f"(CUDA max threads/block = 1024 predicts 1024 / 1040)")
    print("[gpu_probe] NOTE: 'launched' here means the KERNEL launched on a "
          "dummy scene. It does NOT validate the physics — the real gradient "
          "still needs the FD gate on the device.")
