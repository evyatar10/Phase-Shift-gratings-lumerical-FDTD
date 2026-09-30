% plot_incore_oxide_periodic.m — periodic SiO2 holes INSIDE the core never reach the control.
% Study dirs: results_from_igum/scat_x_incore (IGUM, 2026-08-10) and
%             results_from_athena/tm_hole_lattice (job 123303, 2026-07-18) | plot 2026-09-14
% Purpose: every periodic in-core oxide structure measured, vs the control of the same
% device (TM, W 800, corr 400, h 350, pitch 516.83, N 80/side) at identical numerics.
% Left: peak T vs comb phase for the in-core oxide comb (Lambda 531, r 80, y +/-250 nm,
% 9 holes); only two phases were run, so points, no line. Right: every variant vs its
% own control (the lattice rows sit at a different box and window than the comb rows).

root = fullfile(fileparts(mfilename('fullpath')), '..', '..');
ig = fullfile(root, 'results_from_igum', 'scat_x_incore', 'results');
lat_dir = fullfile(root, 'results_from_athena', 'tm_hole_lattice');
ctrl_ig = fullfile(root, 'results_from_igum', 'scat_q_r80phase', 'results', 'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat');
comb = { ...            % file, phase (deg), label
  'result_N80_TM_avg_Ybox16p0_Zbox8p8_scR80_arr9_X-1992to2256_Y250to250_C400_pair_hole_ff.mat',   90, '9 holes, 90\circ'; ...
  'result_N80_TM_avg_Ybox16p0_Zbox8p8_scR80_arr9_X-1726to2522_Y250to250_C400_pair_hole_ff.mat',  270, '9 holes, 270\circ'; ...
  'result_N80_TM_avg_Ybox16p0_Zbox8p8_scR80_arr31_X-7567to8363_Y250to250_C400_pair_hole_ff.mat', 270, '31 holes, 270\circ'};
lat = { ...
  'result_N80_TM_avg_scR100_arr160_X-41346to41088_Y0_C400_hole.mat', 'hole per period, \Lambda 516.83'; ...
  'result_N80_TM_avg_scR100_arr151_X-40875to41148_Y0_C400_hole.mat', 'hole per period, \Lambda 545'; ...
  'result_N80_TM_avg_scR100_arr160_X-41217to41217_Y0_C400_hole.mat', 'hole per period, shifted \Lambda/4'};
peak = @(m, lo, hi) max(m.T(:) .* (m.wl_nm(:) > lo & m.wl_nm(:) < hi));   % defect peak only, never the passband

c = load(ctrl_ig); Tc = peak(c, 1550, 1566);
ca = load(fullfile(lat_dir, 'result_N80_TM_avg.mat')); Tca = peak(ca, 1554, 1563);
n = size(comb, 1); ph = zeros(n, 1); Tk = ph;
for k = 1:n
    m = load(fullfile(ig, comb{k, 1})); ph(k) = comb{k, 2}; Tk(k) = peak(m, 1550, 1566);
end
Tl = zeros(size(lat, 1), 1);
for k = 1:size(lat, 1)
    m = load(fullfile(lat_dir, lat{k, 1})); Tl(k) = peak(m, 1544, 1552);   % all three defect peaks at 1548 +/- 1 nm
end

blue = [0 0.45 0.74]; red = [0.85 0.33 0.10];
fig = figure('Visible', 'off', 'Position', [80 80 1200 440]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile(tl); hold(ax1, 'on');
yline(ax1, Tc, '--', 'Color', blue, 'LineWidth', 1.4);
plot(ax1, ph(1:2), Tk(1:2), 'o', 'MarkerSize', 9, 'Color', red, 'MarkerFaceColor', red);
plot(ax1, ph(3), Tk(3), 's', 'MarkerSize', 9, 'Color', red, 'MarkerFaceColor', 'w', 'LineWidth', 1.5);
for k = 1:n
    text(ax1, ph(k) + 8, Tk(k), sprintf('%.3f', Tk(k)), 'FontSize', 10, 'VerticalAlignment', 'middle');
end
hold(ax1, 'off'); grid(ax1, 'on'); box(ax1, 'on');
xlim(ax1, [0 360]); ylim(ax1, [0.5 0.92]); xticks(ax1, 0:90:360);
xlabel(ax1, 'comb phase (\circ)'); ylabel(ax1, 'peak transmission');
legend(ax1, {'no holes', '9 holes', '31 holes'}, 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 10);
title(ax1, 'oxide comb in the core, \Lambda 531 nm, r 80 nm', 'FontWeight', 'normal');

dT = [Tk - Tc; Tl - Tca];
labels = [comb(:, 3); lat(:, 2)];
ax2 = nexttile(tl); hold(ax2, 'on');
barh(ax2, 1:numel(dT), dT, 0.6, 'FaceColor', red, 'EdgeColor', 'none');
xline(ax2, 0, 'k-', 'LineWidth', 1.0);
hold(ax2, 'off'); grid(ax2, 'on'); box(ax2, 'on');
set(ax2, 'YTick', 1:numel(dT), 'YTickLabel', labels, 'YDir', 'reverse', 'FontSize', 10);
xlim(ax2, [-0.85 0.05]);
xlabel(ax2, 'peak transmission minus its control');
title(ax2, 'every in-core variant, vs its own control', 'FontWeight', 'normal');

title(tl, sprintf('SiO_2 holes inside the core, TM, N 80/side, corr 400 nm: control T %.3f', Tc), 'FontSize', 12);

out = fullfile(root, 'results_from_igum', 'scat_x_incore', 'incore_oxide_periodic');
exportgraphics(fig, [out '.png'], 'Resolution', 200);
savefig(fig, [out '.fig']);
fprintf('saved: %s.png\nsaved: %s.fig\n', out, out);
