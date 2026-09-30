% plot_te_apod_tradeoff.m — TE apodization: transmission gain against mode widening.
% Study dirs: results_from_athena/tm_te_apod (job 96506) and tm_te_apod_tanh; baseline
% reused from run_tm_vs_te | plot 2026-09-14 for the research overview deck.
% Purpose: one figure for the slide "TE: apodization reduces loss and widens the mode".
% Peak T vs spatial mode width (fwhm_m, post_processing convention) for the plain grating
% and for linear and tanh apodization of 2, 5, 10, 20 periods per side. MEASURED.
% Device: TE, W 800, corr 300, h 350, pitch 500, N 80/side, centre modulation 4 nm.

root = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results_from_athena');
base = load(fullfile(root, 'run_tm_vs_te', 'results', 'result_N80_avg_te.mat'));
napod = [2 5 10 20];
lin = struct('T', [], 'w', []); th = lin;
for n = napod
    m = load(fullfile(root, 'tm_te_apod', 'results', sprintf('result_N80_A%d_M4_avg.mat', n)));
    lin.T(end + 1) = m.resonance_transmission; lin.w(end + 1) = m.fwhm_m * 1e6;
    m = load(fullfile(root, 'tm_te_apod_tanh', 'results', sprintf('result_N80_A%d_th_M4_avg.mat', n)));
    th.T(end + 1) = m.resonance_transmission; th.w(end + 1) = m.fwhm_m * 1e6;
end
T0 = base.resonance_transmission; w0 = base.fwhm_m * 1e6;

blue = [0 0.45 0.74]; red = [0.85 0.33 0.10]; green = [0.47 0.67 0.19];
fig = figure('Visible', 'off', 'Position', [80 80 820 520]);
ax = axes(fig); hold(ax, 'on');
plot(ax, [w0 lin.w], [T0 lin.T], '-o', 'Color', red, 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', red);
plot(ax, [w0 th.w], [T0 th.T], '-s', 'Color', green, 'LineWidth', 1.5, 'MarkerSize', 7, 'MarkerFaceColor', green);
plot(ax, w0, T0, 'o', 'Color', blue, 'MarkerSize', 9, 'MarkerFaceColor', blue);
for k = 1:numel(napod)
    text(ax, lin.w(k) + 0.25, lin.T(k) - 0.006, sprintf('%d', napod(k)), 'Color', red, 'FontSize', 10);
    text(ax, th.w(k) + 0.25, th.T(k) - 0.006, sprintf('%d', napod(k)), 'Color', green, 'FontSize', 10);
end
text(ax, w0 + 0.25, T0 - 0.006, 'no apodization', 'Color', blue, 'FontSize', 10);
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on');
xlim(ax, [14.5 27]); ylim(ax, [0.84 1.0]);
xlabel(ax, 'spatial mode width, FWHM (\mum)'); ylabel(ax, 'peak transmission');
legend(ax, {'linear apodization', 'tanh apodization', 'plain grating'}, 'Location', 'southeast', 'FontSize', 10);
title(ax, sprintf('TE \\pi shift grating, W 800 nm, corr 300 nm, pitch 500 nm, N 80/side, \\lambda_{res} %.1f nm', base.resonance_wavelength_nm), 'FontSize', 11.5);
subtitle(ax, 'numbers = apodized periods per side', 'FontSize', 10);

out = fullfile(root, 'tm_te_apod', 'te_apod_tradeoff');
exportgraphics(fig, [out '.png'], 'Resolution', 200);
savefig(fig, [out '.fig']);
fprintf('saved: %s.png\nsaved: %s.fig\n', out, out);
