"""
Recovery runner: the 4 TM-apodization points that FAILED in array job 97304.

Job 97304 (runners/sweeps/tm_apod_pitch518.py, 8-task --option3 array) lost tasks
4-7 to a shared-sweep_list.txt collision: a concurrent spec deploy overwrote
/work/data/sweep_list.txt with a shorter list, so the QOS-delayed tasks read an
out-of-range index and died in ~5 s. The 4 early tasks (A2, A5 × linear/tanh)
completed fine.

This runner re-runs ONLY the 4 missing points as ONE sequential --option2 GPU job.
A single job uses no shared sweep_list (it loops run_single_sim in-process), so it
is immune to the collision; and as one normally-queued job it does not preempt or
starve the other jobs in the queue.

  apodized teeth/side : {10, 20}          (the 2 missing per method)
  apod_method         : {linear, tanh}    (tanh_steepness = 2.0)
  polarization        : {TM}
  => 2 × 2 = 4 sims, run sequentially in this one job.

STUDY_DIR_NAME pins the output folder to the original study (tm_apod_pitch518), so
these 4 result_*.mat land alongside the A2/A5 files already there. Everything else
(pitch 518.3, N=80, constant materials, 150 nm/6001-pt window, record_2d_fields,
far-field off) is identical to runners/sweeps/tm_apod_pitch518.py.

Dispatch (one sequential GPU job, immune to the sweep_list collision):
    bash athena/deploy_athena.sh --option2 --run=tm_apod_pitch518_rest

Outputs: <BASE_SAVE_DIR>/tm_apod_pitch518/results/ on Athena, sync via
`--results-no-fsp`.
"""

from runners.sweeps.sweep_spec import SweepSpec, run_sweep_spec
from runners.tm._tm_vs_te_common import build_base_cfg
from simulation_config import SimulationConfig

# Write into the SAME results folder as the original 8-task study so the 4
# recovered files sit next to the A2/A5 ones (athena_run.py honors this).
STUDY_DIR_NAME = "tm_apod_pitch518"


def _base() -> SimulationConfig:
    """Identical baseline to runners/sweeps/tm_apod_pitch518.py."""
    base = build_base_cfg(SimulationConfig())
    base.grating.pitch_m = 518.3e-9               # TM-corrected pitch (~1571 nm)
    base.spectral.center_wavelength_m = 1.550e-6
    base.spectral.scan_width_nm       = 150.0
    base.spectral.n_wl_points         = 6001
    base.monitors.record_2d_fields    = True
    base.farfield.enabled             = False
    return base


def run(cfg=None):
    """Run the 4 missing points sequentially. `cfg` from athena_run.py is ignored
    (it carries the wrong detuning); the baseline is rebuilt via build_base_cfg."""
    spec = SweepSpec(
        n_apod_periods_each_side = [10, 20],
        polarization             = ["TM"],
        apod_method              = ["linear", "tanh"],
        tanh_steepness           = [2.0],
        center_mod_depth_nm      = [4.0],
        label = "tm_apod_pitch518_rest",
    )
    return run_sweep_spec(spec, target="local", base=_base())


if __name__ == "__main__":
    run()
