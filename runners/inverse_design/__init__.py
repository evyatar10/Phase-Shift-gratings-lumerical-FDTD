"""
Inverse design for the pi-shift Bragg grating.

Three files:
    inverse_design.py          — all machinery (dataclass + geometry + lumopt driver)
    optimize_transmission.py   — the study (what to optimize, bounds, starting point)
    test_geometry.py           — local self-test (runs without Lumerical)

Run on the cluster:
    bash athena/deploy_athena.sh --inverse-design=runners.inverse_design.optimize_transmission
    bash dgx/deploy_dgx.sh       --inverse-design=runners.inverse_design.optimize_transmission

Run one driver locally (useful for inspecting param vectors):
    python -m runners.inverse_design.inverse_design --spec runners.inverse_design.optimize_transmission
"""
