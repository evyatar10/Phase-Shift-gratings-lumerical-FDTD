% plot_barrel_field_overlay.m — proof that the barrel does NOT change the mode.
% Overlays the longitudinal energy-density envelope of the control vs barrel-300
% device (accurate mesh). They lie on top of each other: same mode, 27% less loss.

clear; close all;
d = load('c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\barrel_followup\barrel_field_overlay.mat');

fig = figure('Visible', 'off', 'Position', [80 80 1000 460]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile; hold on; grid on;
plot(d.xc*1e6, d.edc, '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 2.4, 'DisplayName', 'control (rect cavity)');
plot(d.xb*1e6, d.edb, '--', 'Color', [0.13 0.55 0.33], 'LineWidth', 1.6, 'DisplayName', 'barrel cavity (+300 nm)');
xlabel('x along guide (\mum)'); ylabel('normalized energy density');
title('Full mode envelope — identical');
legend('Location', 'northeast'); xlim([-25 25]);

nexttile; hold on; grid on;
plot(d.xc*1e6, d.edc, '-', 'Color', [0.35 0.35 0.35], 'LineWidth', 2.4, 'DisplayName', 'control');
plot(d.xb*1e6, d.edb, '--', 'Color', [0.13 0.55 0.33], 'LineWidth', 1.6, 'DisplayName', 'barrel');
xlabel('x along guide (\mum)'); ylabel('normalized energy density');
title('Zoom on the defect (\pm2 \mum) — still identical');
legend('Location', 'northeast'); xlim([-2 2]);

sgtitle(sprintf(['Why the barrel does NOT widen the mode: the mode is UNCHANGED\n' ...
    'same envelope, same fwhm (15.3\\rightarrow15.4 \\mum) \\cdot yet loss 0.117\\rightarrow0.086 (\\bf-27%%\\rm)']), ...
    'FontSize', 12);

out = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\barrel_followup';
exportgraphics(fig, fullfile(out, 'barrel_field_overlay.png'), 'Resolution', 200);
savefig(fig, fullfile(out, 'barrel_field_overlay.fig'));
fprintf('saved: %s\n', fullfile(out, 'barrel_field_overlay.png'));
