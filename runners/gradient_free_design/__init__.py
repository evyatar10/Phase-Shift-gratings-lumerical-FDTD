"""
Gradient-free inverse design for the pi-shift Bragg grating.

Lumerical-native — uses `addsweep("Optimization")` (Particle Swarm or
Generic Algorithm) running inside an lumapi.FDTD session. No scipy.optimize,
no external Python optimizers.

Three files:
    gradient_free_design.py    — main machinery (spec + Lumerical PSO driver)
    optimize_transmission.py   — the study (what to optimize, bounds, init point)
    test_geometry.py           — local self-test (no Lumerical needed)

Run on the cluster:
    bash athena/deploy_athena.sh --gradient-free-design=runners.gradient_free_design.optimize_transmission
    bash dgx/deploy_dgx.sh       --gradient-free-design=runners.gradient_free_design.optimize_transmission

Run one driver locally (useful for sanity-checking the .fsp build):
    python -m runners.gradient_free_design.gradient_free_design --spec runners.gradient_free_design.optimize_transmission
"""
