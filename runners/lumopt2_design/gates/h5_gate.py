"""h5 NON-ZERO gate for the GPU width-adjoint (HANDOFF: mandatory).

Runs on the Athena LOGIN node. A plausible runtime is NOT evidence an adjoint
injected: the 52-minute "success" of 2026-08-23 had EVERY monitor field
exactly 0.0 (dead source at z=0). This reports max|E| per component per file
so a dead source is visible immediately.

Usage (one ssh):
  python3 h5_gate.py gfr_full gfr_yhalf gfr_quart
"""
import glob
import os
import sys

import h5py
import numpy as np

BASE = os.path.expanduser(
    "~/bragg_sim_athena/results/validate_c325/results/lumopt2_val_c325")


def scan(path):
    out = []
    with h5py.File(path, "r") as h:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.size == 0:
                return
            if not np.issubdtype(obj.dtype, np.number) and obj.dtype.names is None:
                return
            try:
                a = obj[()]
            except Exception:
                return
            if a.dtype.names:                      # compound (re, im)
                a = np.stack([a[n] for n in a.dtype.names], -1)
            m = float(np.abs(np.asarray(a, dtype=float)).max()) if a.size else 0.0
            if "E" in name or "field" in name.lower():
                out.append((name, a.shape, m))
        h.visititems(visit)
    return out


for label in sys.argv[1:]:
    d = os.path.join(BASE, f"lumopt2_val_c325_{label}_files")
    files = sorted(glob.glob(os.path.join(d, "**", "*.h5"), recursive=True))
    print(f"\n=== {label} — {len(files)} h5 under {d}")
    for f in files:
        rows = scan(f)
        peak = max([r[2] for r in rows], default=0.0)
        verdict = "DEAD (all-zero)" if peak == 0.0 else f"alive max|E|={peak:.4g}"
        print(f"  {os.path.basename(f)}: {verdict}")
        for name, shape, m in rows[:6]:
            print(f"      {name} {shape} max {m:.4g}")
