% plot_scat_s_refine.m
% Study: results_from_athena/scat_s_refine (night refine, jobs 130117 + 130135)
% Date: 2026-08-10
% Purpose: refine-wave verdict — T vs comb period (the soft-cutoff plateau),
%   T vs radius, T vs phase. All at 270 deg phase, d=1.8 um, vs ctrl 0.8851.

ROOT = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results_from_athena');
RES = fullfile(ROOT, 'scat_s_refine', 'results');
T0 = 0.8851;   % stored ctrl (job 123563)

% (Lambda, label-x, T) — measured this night, incl. stage-R anchors at 536/540
lamT = [530 0.8967; 531 0.8966; 532 0.8962; 534 0.8949; 536 0.8928; 540 0.8872];
rT_536 = [85 0.8932; 92 0.8936; 100 0.8936; 110 0.8928];
rT_532 = [92 0.8951; 110 0.8962];
phT = [250 0.8910; 270 0.8928; 290 0.8912];

fig = figure('Visible', 'off', 'Position', [70 70 940 640]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl, [1 2]);
hold(ax, 'on');
plot(ax, lamT(:,1), lamT(:,2), 'o-', 'LineWidth', 1.5, 'Color', [0 0.45 0.74]);
yline(ax, T0, 'k:', 'control 0.8851', 'FontSize', 8);
xline(ax, 530.7, ':', 'nominal cutoff \Lambda_c', 'FontSize', 8, 'LabelVerticalAlignment', 'bottom');
grid(ax, 'on');
xlabel(ax, 'comb period \Lambda  [nm]  (all at 270\circ phase, r=110)');
ylabel(ax, 'peak T');
title(ax, 'A — T vs period: rises INTO the soft cutoff; plateau 530-532 (ties at the floor); B* = 531');

ax = nexttile(tl);
hold(ax, 'on');
plot(ax, rT_536(:,1), rT_536(:,2), 's-', 'LineWidth', 1.3, 'DisplayName', '\Lambda=536');
plot(ax, rT_532(:,1), rT_532(:,2), 'o-', 'LineWidth', 1.3, 'DisplayName', '\Lambda=532');
yline(ax, T0, 'k:', 'HandleVisibility', 'off');
grid(ax, 'on');
xlabel(ax, 'post radius r  [nm]');
ylabel(ax, 'peak T');
legend(ax, 'Location', 'southeast', 'FontSize', 8);
title(ax, 'B — radius: flat 85-100 at 536; optimum moves up near cutoff');

ax = nexttile(tl);
hold(ax, 'on');
plot(ax, phT(:,1), phT(:,2), 'd-', 'LineWidth', 1.3, 'Color', [0.49 0.18 0.56]);
yline(ax, T0, 'k:', 'HandleVisibility', 'off');
grid(ax, 'on');
xlabel(ax, 'comb phase  [deg]  (\Lambda=536, r=110)');
ylabel(ax, 'peak T');
title(ax, 'C — phase: 270\circ confirmed as the peak');

title(tl, {'Night refine waves (jobs 130117 + 130135) — anti-needle comb around the winner'; ...
    'TM corr 400, W800, N=80, d=1.8 \mum, h350 — best: \Lambda=531, 270\circ, r=110: T 0.8966 (+0.0115, 6.4\times floor)'});

set(fig, 'Visible', 'on');
savefig(fig, fullfile(ROOT, 'scat_s_refine', 'scat_s_refine.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(ROOT, 'scat_s_refine', 'scat_s_refine.png'), 'Resolution', 150);
disp(fullfile(ROOT, 'scat_s_refine', 'scat_s_refine.png'));
