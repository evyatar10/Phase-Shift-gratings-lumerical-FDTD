% plot_scat_p_antineedle.m
% Study: results_from_athena/scat_p_antineedle (stage P, job 129989, 2026-08-09)
% Purpose: anti-needle comb verdict — needle-bin power and T vs comb x-shift
%   (the interference phase circle, Lambda=545) and vs period (dx=0).
%   All vs stored identical-numerics ctrl T=0.8851 (job 123563).

ROOT = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results_from_athena');
RES = fullfile(ROOT, 'scat_p_antineedle', 'results');
ctrl = load(fullfile(ROOT, 'scat_h_retrocomb', 'results', ...
    'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'));
mask = abs(ctrl.farfield_side.ux) > 0.94;
N0 = sum(ctrl.farfield_side.E2(mask, :), 'all');
T0 = ctrl.resonance_transmission;

% rows: {x-range tag, radius tag, Lambda_nm, dx_nm}
rows = {'X-8085to8085', 'R110', 539, 0;  'X-8130to8130', 'R110', 542, 0; ...
        'X-8175to8175', 'R110', 545, 0;  'X-8220to8220', 'R110', 548, 0; ...
        'X-8265to8265', 'R110', 551, 0;  'X-8039to8311', 'R110', 545, 136; ...
        'X-7902to8448', 'R110', 545, 273; 'X-7766to8584', 'R110', 545, 409};
n = size(rows, 1);
[needle, T, lam, dx] = deal(zeros(1, n));
for k = 1:n
    fl = dir(fullfile(RES, sprintf('*sc%s_*%s_*_ff.mat', rows{k, 2}, rows{k, 1})));
    d = load(fullfile(fl(1).folder, fl(1).name));
    needle(k) = sum(d.farfield_side.E2(mask, :), 'all') / N0;
    T(k) = d.resonance_transmission;
    lam(k) = rows{k, 3};
    dx(k) = rows{k, 4};
end

fig = figure('Visible', 'off', 'Position', [80 80 900 640]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% A — phase circle at Lambda=545 (dx = 0/136/273/409)
ax = nexttile(tl);
m = lam == 545 & ismember(dx, [0 136 273 409]);
yyaxis(ax, 'left');
plot(ax, dx(m), needle(m), 'o-', 'LineWidth', 1.4);
ylabel(ax, 'grazing-lobe power  (\times ctrl)');
yline(ax, 1, ':', 'no comb', 'FontSize', 8);
yyaxis(ax, 'right');
plot(ax, dx(m), T(m), 's--', 'LineWidth', 1.1);
yline(ax, T0, ':', 'ctrl T', 'FontSize', 8);
ylabel(ax, 'peak T');
grid(ax, 'on');
xlabel(ax, 'comb x-shift \deltax  [nm]   (\Lambda = 545 nm: 136 = 90\circ, 273 = 180\circ, 409 = 270\circ)');
title(ax, 'A — interference phase circle: lobes \times2.2 at 90\circ, \times0.45 at 270\circ (mechanism confirmed)');

% B — period scan at dx=0
ax = nexttile(tl);
m = dx == 0;
yyaxis(ax, 'left');
plot(ax, lam(m), needle(m), 'o-', 'LineWidth', 1.4);
ylabel(ax, 'grazing-lobe power  (\times ctrl)');
yline(ax, 1, ':', 'no comb', 'FontSize', 8);
yyaxis(ax, 'right');
plot(ax, lam(m), T(m), 's--', 'LineWidth', 1.1);
yline(ax, T0, ':', 'ctrl T', 'FontSize', 8);
ylabel(ax, 'peak T');
grid(ax, 'on');
xlabel(ax, 'comb period \Lambda  [nm]   (\deltax = 0)');
title(ax, 'B — period scan at fixed phase (\deltax=0 sits near the constructive side)');
title(tl, {'Anti-needle comb, measured (job 129989) — TM corr 400, W800, N=80'; ...
    '31 SiN posts r=110 nm, h=350 nm, d=1.8 \mum, \lambda_{res} 1558.64 nm, ctrl T=0.885'});

set(fig, 'Visible', 'on');
savefig(fig, fullfile(ROOT, 'scat_p_antineedle', 'scat_p_antineedle.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(ROOT, 'scat_p_antineedle', 'scat_p_antineedle.png'), 'Resolution', 150);
disp(fullfile(ROOT, 'scat_p_antineedle', 'scat_p_antineedle.png'));
