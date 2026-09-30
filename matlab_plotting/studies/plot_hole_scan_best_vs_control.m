% plot_hole_scan_best_vs_control.m — a single SiO2 hole in the core never beats the control.
% Study dir: results_from_athena/tm_hole_scan | Jobs 116152/116272 | 2026-07 (plot 2026-09-14)
% Purpose: same device as the control (TM, W 800, corr 400, h 350, pitch 516.83, N 80/side)
% with ONE r = 100 nm SiO2 hole on the axis, stepped along +x at pitch/8. Left: T(lambda)
% of the control, the best hole position and the worst. Right: peak T vs hole position,
% the standing-wave oscillation whose maxima approach the control but never exceed it.
% MEASURED, identical numerics; peak = max T inside the stopband window 1550-1566 nm.

base = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results_from_athena', 'tm_hole_scan', 'results');
win = [1550 1566];
files = dir(fullfile(base, 'result_N80_TM_avg_scR100_X*_Y0_hole.mat'));
x_nm = zeros(numel(files), 1); Tpk = x_nm; lam = x_nm;
for k = 1:numel(files)
    tok = regexp(files(k).name, '_X(-?\d+)_', 'tokens', 'once');
    x_nm(k) = str2double(tok{1});
    m = load(fullfile(base, files(k).name));
    wl = m.wl_nm(:); T = m.T(:);
    [Tpk(k), i] = max(T .* (wl > win(1) & wl < win(2)));
    lam(k) = wl(i);
end
[x_nm, o] = sort(x_nm); Tpk = Tpk(o); lam = lam(o); files = files(o);
c = load(fullfile(base, 'result_N80_TM_avg.mat'));
[Tc, ic] = max(c.T(:) .* (c.wl_nm(:) > win(1) & c.wl_nm(:) < win(2)));
lam_c = c.wl_nm(ic);
[~, ib] = max(Tpk); [~, iw] = min(Tpk);
b = load(fullfile(base, files(ib).name)); w = load(fullfile(base, files(iw).name));

col = [0 0.45 0.74; 0.47 0.67 0.19; 0.85 0.33 0.10; 0.49 0.18 0.56];
fig = figure('Visible', 'off', 'Position', [80 80 1250 460]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl); hold(ax1, 'on');
plot(ax1, c.wl_nm, c.T, '-', 'Color', col(1, :), 'LineWidth', 1.6);
plot(ax1, b.wl_nm, b.T, '--', 'Color', col(2, :), 'LineWidth', 1.6);
plot(ax1, w.wl_nm, w.T, '-', 'Color', col(3, :), 'LineWidth', 1.2);
hold(ax1, 'off'); grid(ax1, 'on'); box(ax1, 'on');
xlim(ax1, [lam_c - 6, lam_c + 6]); ylim(ax1, [0 0.9]);
xlabel(ax1, 'wavelength (nm)'); ylabel(ax1, 'transmission');
legend(ax1, {sprintf('no hole (control): T %.3f', Tc), ...
             sprintf('hole at x = %.2f \\mum (best): T %.3f', x_nm(ib) / 1e3, Tpk(ib)), ...
             sprintf('hole at x = %.2f \\mum (worst): T %.3f', x_nm(iw) / 1e3, Tpk(iw))}, ...
       'Location', 'northwest', 'FontSize', 10);
title(ax1, 'transmission at the resonance', 'FontWeight', 'normal');

ax2 = nexttile(tl); hold(ax2, 'on');
yline(ax2, Tc, '--', 'Color', col(1, :), 'LineWidth', 1.4);
plot(ax2, x_nm / 1e3, Tpk, '-o', 'Color', col(4, :), 'MarkerSize', 4, 'MarkerFaceColor', col(4, :), 'LineWidth', 1.1);
plot(ax2, x_nm(ib) / 1e3, Tpk(ib), 'o', 'MarkerSize', 9, 'Color', col(2, :), 'LineWidth', 1.8);
plot(ax2, x_nm(iw) / 1e3, Tpk(iw), 'o', 'MarkerSize', 9, 'Color', col(3, :), 'LineWidth', 1.8);
hold(ax2, 'off'); grid(ax2, 'on'); box(ax2, 'on');
xlabel(ax2, 'hole position x from the cavity centre (\mum)'); ylabel(ax2, 'peak transmission');
ylim(ax2, [0.70 0.85]);
legend(ax2, {'no hole (control)', 'one SiO_2 hole, r 100 nm', 'best', 'worst'}, 'Location', 'southeast', 'FontSize', 10);
title(ax2, 'peak transmission vs hole position (step = pitch/8)', 'FontWeight', 'normal');

title(tl, sprintf(['TM \\pi shift grating, W 800 nm, h 350 nm, pitch 516.83 nm, corr 400 nm, N 80/side, ' ...
                   'one SiO_2 hole in the core: \\lambda_{res} %.2f nm, T_{control} %.3f'], lam_c, Tc), 'FontSize', 12);
subtitle(tl, 'peak T oscillates with the standing wave; its maxima approach the control and never exceed it', 'FontSize', 10.5);

out = fullfile(base, '..', 'hole_scan_best_vs_control');
exportgraphics(fig, [out '.png'], 'Resolution', 200);
savefig(fig, [out '.fig']);
fprintf('saved: %s.png\nsaved: %s.fig\n', out, out);
