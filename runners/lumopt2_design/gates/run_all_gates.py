"""Run the thirteen offline pre-dispatch gates, fail fast.

Study dir: runners/lumopt2_design/gates/ | Created 2026-09-01 | zero GPU.
Usage: python run_all_gates.py (exit 0 = offline checks passed).
TE bounds are included in predispatch_check. Separately run gate_te_local.py
--generate for licensed build checks. Calibration, hardware smoke, resources
and authorization must also be satisfied before dispatch.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GATES = ["gate_projection_local.py", "gate_lam_chain.py",
         "gate_lam_chain_plumbing.py", "predispatch_check.py",
         "gate_tm_identity.py",
         "gate_v3_local.py", "gate_recovery.py",
         "gate_persistence.py", "gate_s2_smoke.py", "gate_width_contract.py",
         "gate_evaluation_budget.py", "gate_campaign_budget.py",
         "gate_modal_guard.py"]

env = dict(os.environ, PYTHONIOENCODING="utf-8")   # the lam-chain gate prints λ
for g in GATES:
    print(f"=== {g} ===", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, g)], env=env)
    if r.returncode != 0:
        print(f"*** GATE FAILED: {g} — do NOT dispatch ***")
        sys.exit(1)
print(f"ALL {len(GATES)} OFFLINE GATES GREEN — build, calibration and dispatch checks still apply")
