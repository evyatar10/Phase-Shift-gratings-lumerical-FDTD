% plot_metal_mirror_dscan.m — PEC metal-mirror d-scan verdict figure.
% Study dir: results_from_athena/metal_mirror_dscan | Job 124253 | 2026-07-18
% Purpose: dT vs mirror standoff d for the core-height (350 nm) PEC film pair
% beside the TM corr-400 W800 device; overlay = cos fit at the predicted
% lambda_y/2 = 2.68 um image-interference period; band = 0.0018 jitter floor.

d_um = [3.000 3.675 4.350 5.025 5.700];
dT   = [0.0019 -0.0011 -0.0008 0.0006 0.0007];       % MEASURED job 124253
T_ctrl = 0.8851; lam_res = 1558.612; cyc = 2.68;

% cos fit at fixed period (amplitude/phase from the analysis script)
dd = linspace(2.8, 5.9, 300);
fit = 0.0014 * cosd(360 * dd / cyc + 342);

fig = figure('Visible', 'on', 'Position', [80 80 900 520]);
hold on;
fill([2.8 5.9 5.9 2.8], 1e3 * [-0.0018 -0.0018 0.0018 0.0018], ...
     [0.92 0.92 0.92], 'EdgeColor', 'none');
yline(0, 'k-', 'LineWidth', 0.5);
plot(dd, 1e3 * fit, '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2);
plot(d_um, 1e3 * dT, 'o', 'MarkerSize', 7, 'MarkerFaceColor', [0 0.45 0.74], ...
     'Color', [0 0.45 0.74]);
hold off;
grid on; box on;
xlabel('mirror standoff d (\mum)');
ylabel('\DeltaT vs control (\times10^{-3})');
legend({'jitter floor \pm0.0018', '', ...
        sprintf('cos fit, period %.2f \\mum', cyc), 'MEASURED'}, ...
       'Location', 'southeast');
title(sprintf(['PEC film pair (350 nm layer, L=82.6 \\mum, t=200 nm) — TM corr-400 W800, ' ...
               '\\lambda_{res}=%.3f nm, T_{ctrl}=%.4f'], lam_res, T_ctrl));

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
                   'results_from_athena', 'metal_mirror_dscan');
exportgraphics(fig, fullfile(out_dir, 'metal_mirror_dscan.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'metal_mirror_dscan.fig'));
fprintf('saved: %s\nsaved: %s\n', fullfile(out_dir, 'metal_mirror_dscan.png'), ...
        fullfile(out_dir, 'metal_mirror_dscan.fig'));
