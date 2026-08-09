%% plot_trench_flush_top — T(lambda) of the flush-top trench run vs stored anchors
% Study: runners/metal_mirror/trench_flush_top.py | Job: Athena 128918 | 2026-08-06
% One spectrum (TM N=80 corr-400 + air trench w800/d1800/L84, z -3.975..+0.175 um,
% top flush with SiN top, z-sym OFF). Anchor T values (identical numerics, stored):
% ctrl 0.8862 / full-z trench 0.9037 / floor-only 0.9035 — drawn as reference lines.
res_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'results_from_athena', 'trench_flush_top', 'results');
f = dir(fullfile(res_dir, 'result_*Zminm3975.mat'));
d = load(fullfile(res_dir, f(1).name));

Q = d.resonance_wavelength_nm / abs(d.spectral_fwhm_nm);

fig = figure('Position', [100 100 900 520], 'Visible', 'off');
plot(d.wl_nm, d.T, 'LineWidth', 1.4); hold on;
plot(d.resonance_wavelength_nm, d.resonance_transmission, 'v', ...
    'MarkerSize', 8, 'MarkerFaceColor', [0.85 0.33 0.10]);
yline(0.8862, '--', 'ctrl 0.8862', 'LabelHorizontalAlignment', 'left');
yline(0.9037, ':',  'full-z trench 0.9037', 'LabelHorizontalAlignment', 'left');
grid on; xlabel('\lambda (nm)'); ylabel('Transmission');
xlim([1548.5 1568.5]);
title(sprintf(['TM N=80 corr 400, trench w800/d1800 flush with SiN top (z -3.975..+0.175 \\mum)' ...
    newline '\\lambda_{res} = %.3f nm,  peak T = %.4f,  Q = %.0f'], ...
    d.resonance_wavelength_nm, d.resonance_transmission, Q), 'FontSize', 12);
legend({'T(\lambda)', sprintf('resonance %.3f nm', d.resonance_wavelength_nm)}, ...
    'Location', 'southwest', 'FontSize', 10);

savefig(fig, fullfile(res_dir, 'trench_flush_top_T.fig'));
exportgraphics(fig, fullfile(res_dir, 'trench_flush_top_T.png'), 'Resolution', 200);
fprintf('lambda %.3f nm, T %.4f, Q %.0f -> %s\n', d.resonance_wavelength_nm, ...
    d.resonance_transmission, Q, fullfile(res_dir, 'trench_flush_top_T.png'));
