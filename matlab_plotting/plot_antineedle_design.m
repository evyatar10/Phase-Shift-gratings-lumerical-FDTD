% plot_antineedle_design.m
% Study: anti-needle comb design (zero-GPU, 2026-08-09) — renders
%   docs/antineedle_comb_design.mat produced by python_tools/antineedle_comb_design.py.
% Panels: (A) measured needle + measured green comb beam + designed beam;
%         (B) predicted cancellation vs comb period; (C) the phase-flip vs comb x-shift.

ROOT = fileparts(fileparts(mfilename('fullpath')));
D = load(fullfile(ROOT, 'docs', 'antineedle_comb_design.mat'));
OUT = fullfile(ROOT, 'docs');

fig = figure('Visible', 'off', 'Position', [60 60 900 860]);
tl = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% A — field profiles near the (negative) needle
ax = nexttile(tl);
hold(ax, 'on');
nrm = @(p) p / max(p);
zoom_m = D.ux <= -0.80 & D.ux >= -1.0;
plot(ax, D.ux(zoom_m), nrm(D.prof_needle(zoom_m)), 'k', 'LineWidth', 1.6, ...
    'DisplayName', 'device needle (measured)');
plot(ax, D.ux(zoom_m), nrm(D.prof_green(zoom_m)), 'Color', [0.47 0.67 0.19], ...
    'LineWidth', 1.2, 'DisplayName', 'comb beam, \Lambda=551 nm (measured, wrong angle)');
plot(ax, D.ux_win, nrm(D.prof_design), '--', 'Color', [0 0.45 0.74], 'LineWidth', 1.4, ...
    'DisplayName', sprintf('designed beam, \\Lambda=%.0f nm, L=17 \\mum', D.best_lambda_nm));
grid(ax, 'on');
xlabel(ax, 'u_x');
ylabel(ax, '|E| (norm.)');
title(ax, 'A — aim the carrier-out-coupled beam AT the needle');
legend(ax, 'Location', 'northwest', 'FontSize', 8);

% B — cancellation vs period
ax = nexttile(tl);
hold(ax, 'on');
plot(ax, D.lam_scan, 100*D.cancel_L17, 'o-', 'Color', [0 0.45 0.74], ...
    'LineWidth', 1.3, 'DisplayName', 'L = 17 \mum (width-matched)');
plot(ax, D.lam_scan, 100*D.cancel_L83, 's-', 'Color', [0.85 0.33 0.10], ...
    'LineWidth', 1.1, 'DisplayName', 'L = 83 \mum (full-length)');
xline(ax, 551, ':', '\Lambda=551 (green run)', 'FontSize', 8, 'HandleVisibility', 'off');
grid(ax, 'on');
xlabel(ax, 'comb period \Lambda  [nm]');
ylabel(ax, 'needle power removable  [%]');
title(ax, 'B — cancellation ceiling vs period (optimal amplitude+phase)');
legend(ax, 'Location', 'northwest', 'FontSize', 8);

% C — the flip: needle power vs comb x-shift
ax = nexttile(tl);
plot(ax, D.dx_nm, 10*log10(D.P_dx), 'LineWidth', 1.4, 'Color', [0.49 0.18 0.56]);
yline(ax, 0, 'k-', 'no comb', 'FontSize', 8);
grid(ax, 'on');
xlabel(ax, sprintf('comb x-shift \\deltax  [nm]  (one period = %.0f nm = 360\\circ)', ...
    D.best_lambda_nm));
ylabel(ax, 'needle power  [dB vs no comb]');
title(ax, 'C — the flip knob: \deltax sets constructive \leftrightarrow destructive');
title(tl, {'Anti-needle comb design (zero-GPU, calibrated on measured runs)'; ...
    'TM corr 400, W800, N=80 — SiN comb in cladding, d=1.8 \mum'});

set(fig, 'Visible', 'on');
savefig(fig, fullfile(OUT, 'antineedle_comb_design.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(OUT, 'antineedle_comb_design.png'), 'Resolution', 150);
disp(fullfile(OUT, 'antineedle_comb_design.png'));
