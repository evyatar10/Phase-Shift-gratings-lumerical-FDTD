"""farfield_sph_15um_postarray_overshoot, variant "v21f" (see FF15_PA_SET in the main runner). Created 2026-10-26; Athena (memory project_postarray_night_2026-10-23)."""
import os
os.environ["FF15_PA_SET"] = "v21f"
from runners.sweeps.farfield_sph_15um_postarray_overshoot import SPEC, BASE  # noqa: E402,F401
