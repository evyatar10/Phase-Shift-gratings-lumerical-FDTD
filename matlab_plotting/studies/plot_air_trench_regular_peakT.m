% plot_air_trench_regular_peakT.m
% Peak transmission (at resonance) vs lateral trench offset d, for each trench
% width, from the tm_air_trench_regular sweep (job 119488) — the REGULAR
% pi-shift device (NO stack). Companion to plot_air_trench_peakT.m (stack).
% Peak T = resonance_transmission (built-in finder, NOT argmax(T)).
%
% Output: PNG + editable FIG in results_from_athena/tm_air_trench_regular/.

clear; close all;

RES = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'tm_air_trench_regular', 'results');
OUT = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'tm_air_trench_regular');

pk = @(f) getfield(load(fullfile(RES, f)), 'resonance_transmission');
rl = @(f) getfield(load(fullfile(RES, f)), 'resonance_wavelength_nm');

pfx = 'result_N80_TM_avg_Ybox8p0_Zbox8p8';

% --- control (no trench) ---
ctrl_f = [pfx '.mat'];
T_ctrl = pk(ctrl_f);
lam_ctrl = rl(ctrl_f);

d = [1.2 1.5 1.8 2.1 2.4];

f800 = arrayfun(@(y) sprintf('%s_scRECT_L84000xW800_X0_Y%d_pair_hole.mat', pfx, round(y*1000)), d, 'uni', 0);
f400 = arrayfun(@(y) sprintf('%s_scRECT_L84000xW400_X0_Y%d_pair_hole.mat', pfx, round(y*1000)), d, 'uni', 0);
T800 = cellfun(pk, f800);
T400 = cellfun(pk, f400);

% --- sanity: resonances stay in the scan window (1558.5 +/- 20 nm) ---
lam_all = [lam_ctrl, cellfun(rl, [f800 f400])];
assert(all(lam_all > 1538 & lam_all < 1579), 'a resonance fell off-window');

% ----------------------------- plot -----------------------------
fig = figure('Color','w','Position',[100 100 780 540]);
hold on; grid on; box on;

c800 = [0.10 0.45 0.80];
c400 = [0.85 0.33 0.10];

yline(T_ctrl, '--', sprintf('no trench  %.4f', T_ctrl), ...
    'Color',[0.35 0.35 0.35], 'LineWidth',1.4, ...
    'LabelHorizontalAlignment','left','FontSize',10);

p1 = plot(d, T800, '-o', 'Color',c800, 'MarkerFaceColor',c800, ...
    'LineWidth',1.8, 'MarkerSize',7);
p2 = plot(d, T400, '-s', 'Color',c400, 'MarkerFaceColor',c400, ...
    'LineWidth',1.8, 'MarkerSize',8);

xlabel('trench offset d from guide axis (\mum)');
ylabel('peak transmission at resonance');
title(sprintf('Air-trench recycling, REGULAR device: peak T vs offset  (\\lambda_{res} \\approx %.1f nm)', lam_ctrl));

legend([p1 p2], {'air trench w = 800 nm','air trench w = 400 nm'}, ...
    'Location','best');

xlim([1.05 2.55]);
set(gca,'FontSize',11);

% ----------------------------- save -----------------------------
png = fullfile(OUT, 'air_trench_regular_peakT_vs_offset.png');
figf = fullfile(OUT, 'air_trench_regular_peakT_vs_offset.fig');
exportgraphics(fig, png, 'Resolution', 200);
savefig(fig, figf);

fprintf('REGULAR device control (no trench) peak T = %.5f  (lam=%.2f)\n', T_ctrl, lam_ctrl);
fprintf('w800  d=%.2f  T=%.5f\n', [d; T800]);
fprintf('w400  d=%.2f  T=%.5f\n', [d; T400]);
fprintf('best air-trench delta vs control = %+.5f\n', max([T800 T400]) - T_ctrl);
fprintf('SAVED:\n  %s\n  %s\n', png, figf);
