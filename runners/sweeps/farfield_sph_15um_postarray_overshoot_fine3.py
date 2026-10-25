"""farfield_sph_15um_postarray_overshoot, variant "fine3" (see FF15_PA_SET in the main runner). Created 2026-10-25; Athena (memory project_postarray_night_2026-10-23)."""
import os
os.environ["FF15_PA_SET"] = "fine3"
from runners.sweeps.farfield_sph_15um_postarray_overshoot import SPEC, BASE  # noqa: E402,F401
