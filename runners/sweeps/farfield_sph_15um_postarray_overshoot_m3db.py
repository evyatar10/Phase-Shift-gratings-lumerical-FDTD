"""farfield_sph_15um_postarray_overshoot, variant "m3db": DIRECT -3 dB device of the best design (2026-10-24, memory project_postarray_night_2026-10-23).
nw_r2pack (measured x1.89 at N 96, Q_i 1.07e6, Q_c 1.256e5) lengthened to N 117 (Q_c(N) = 1.256e5 exp(0.059 (N-96)) -> 4.3e5 = 0.414 Q_i -> T ~0.5),
run 1100 ps = ~2 amplitude decay times at the expected Q_L ~3.1e5 (skill highq-short-run). Reading: Q_L (= Q at -3 dB) from the trace decay, T x gamma^2."""
import os
os.environ["FF15_PA_SET"] = "m3db"
os.environ["FF15_WIN"] = "nw_r2pack"
os.environ["FF15_N"] = "117"
os.environ["FF15_PS"] = "1100"
from runners.sweeps.farfield_sph_15um_postarray_overshoot import SPEC, BASE  # noqa: E402,F401
