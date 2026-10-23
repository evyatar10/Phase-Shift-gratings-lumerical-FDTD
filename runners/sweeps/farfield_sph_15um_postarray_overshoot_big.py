"""farfield_sph_15um_postarray_overshoot, variant "big" (see FF15_PA_SET in the main runner). Created 2026-10-23; Athena (memory project_radcancel_overnight_2026-10-23)."""
import os
os.environ["FF15_PA_SET"] = "big"
from runners.sweeps.farfield_sph_15um_postarray_overshoot import SPEC, BASE  # noqa: E402,F401
