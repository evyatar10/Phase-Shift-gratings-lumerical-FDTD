% plot_hole_lattice_spectra.m — why SiO2 holes inside the core do not work.
% Study dir: results_from_athena/tm_hole_lattice | Job 123303 | 2026-07-18 (plot 2026-09-14)
% Purpose: T(lambda) and loss(lambda) of the anchored TM device (W800, corr 400,
% h 350, pitch 516.83, N 80/side) with and without an r = 100 nm SiO2 hole at the
% centre of every narrow segment. The stopband survives (it widens: the holes add
% grating strength); the defect resonance survives; but the loss at the resonance
% triples and the peak collapses from T 0.83 to 0.03. MEASURED, identical numerics.

base = fullfile(fileparts(mfilename('fullpath')), '..', '..', ...
                'results_from_athena', 'tm_hole_lattice');
cases = { ...
  'result_N80_TM_avg.mat',                                                 'no holes (control)',           [1554 1563]; ...
  'result_N80_TM_avg_scR100_arr160_X-41346to41088_Y0_C400_hole.mat',      'hole per period, corr 400 nm', [1544 1552]; ...
  'result_N80_TM_avg_scR100_arr160_X-41346to41088_Y0_C300_hole.mat',      'hole per period, corr 300 nm', [1544 1552]};
col = [0 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19];
win = [1520 1600];

fig = figure('Visible', 'off', 'Position', [80 80 1000 640]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
axT = nexttile(tl); hold(axT, 'on');
axL = nexttile(tl); hold(axL, 'on');
leg = cell(1, size(cases, 1));
for k = 1:size(cases, 1)
    m = load(fullfile(base, cases{k, 1}));
    wl = m.wl_nm(:); T = m.T(:); L = m.loss(:);
    % resonance = the sharp peak inside the stopband, never the passband max; the stored
    % resonance_wavelength_nm is a passband mis-pick for the corr 400 case, so the peak is
    % taken as max T inside a per-case window that brackets the defect peak only
    inb = wl > cases{k, 3}(1) & wl < cases{k, 3}(2);
    [Tpk, i] = max(T .* inb);
    lam = wl(i);
    plot(axT, wl, T, '-', 'Color', col(k, :), 'LineWidth', 1.4);
    plot(axL, wl, L, '-', 'Color', col(k, :), 'LineWidth', 1.4);
    plot(axT, lam, Tpk, 'o', 'MarkerSize', 6, 'Color', col(k, :), 'MarkerFaceColor', col(k, :), 'HandleVisibility', 'off');
    plot(axL, lam, L(i), 'o', 'MarkerSize', 6, 'Color', col(k, :), 'MarkerFaceColor', col(k, :), 'HandleVisibility', 'off');
    leg{k} = sprintf('%s: \\lambda_{res} %.1f nm, T %.2f, loss %.2f', cases{k, 2}, lam, Tpk, L(i));
end
hold(axT, 'off'); hold(axL, 'off');
set([axT axL], 'XLim', win, 'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', 'FontSize', 11);
ylabel(axT, 'transmission');
ylabel(axL, 'loss (1 - T - R)');
xlabel(axL, 'wavelength (nm)');
set(axT, 'XTickLabel', []);
ylim(axT, [0 1]); ylim(axL, [0 0.7]);
legend(axT, leg, 'Location', 'northwest', 'FontSize', 10);
title(tl, sprintf(['TM \\pi shift grating, W 800 nm, h 350 nm, pitch 516.83 nm, N 80/side: ' ...
                   'SiO_2 holes (r 100 nm) in every narrow segment']), 'FontSize', 12);
subtitle(tl, 'the stopband and the defect peak survive; the loss at the resonance triples and the peak collapses', ...
         'FontSize', 10.5);

out = fullfile(base, 'hole_lattice_spectra');
exportgraphics(fig, [out '.png'], 'Resolution', 200);
savefig(fig, [out '.fig']);
fprintf('saved: %s.png\nsaved: %s.fig\n', out, out);
