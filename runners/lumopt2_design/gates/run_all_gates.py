"""Run the four mandatory pre-dispatch gates (CLAUDE.md section 5), fail fast.

Study dir: runners/lumopt2_design/gates/ | Created 2026-09-01 | zero GPU.
Usage:  python run_all_gates.py     (exit 0 = all green, safe to dispatch)
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GATES = ["gate_projection_local.py", "gate_lam_chain.py",
         "gate_lam_chain_plumbing.py", "predispatch_check.py"]

env = dict(os.environ, PYTHONIOENCODING="utf-8")   # the lam-chain gate prints λ
for g in GATES:
    print(f"=== {g} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, g)], env=env)
    if r.returncode != 0:
        print(f"*** GATE FAILED: {g} — do NOT dispatch ***")
        sys.exit(1)
print("ALL FOUR GATES GREEN — safe to dispatch")
