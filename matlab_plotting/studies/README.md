# matlab_plotting/studies/ — one-off study plot scripts (archived, runnable)

Job/study-specific plot scripts moved out of `matlab_plotting/` so the root
holds only the general engine scripts (plot_transmission, plot_farfield, ...).
Nothing here was edited — each script still targets its original
`results_from_athena/<study>/` directory, and `startup.m` adds this folder to
the MATLAB path, so every script runs exactly as before (by name from MATLAB,
or `matlab -batch "cd matlab_plotting/studies; <script>"`).

Convention: each script's header states the study directory and job ID it
plots. When a new study closes, its plot script moves here.
