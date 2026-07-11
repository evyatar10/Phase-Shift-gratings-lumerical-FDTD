% plot_air_trench_peakT.m
% Peak transmission (at resonance) vs lateral trench offset d, for each
% trench width, from the tm_air_trench sweep. Peak T = resonance_transmission
% (built-in resonance finder, NOT argmax(T)).
%
% Device = the stack: W1050 + [+20,+20] + see-saw 1040/980, accurate mesh.
% Control = same stack, NO trench (horizontal reference).
% Output: PNG + editable FIG in results_from_athena/tm_air_trench/.

clear; close all;

RES = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'tm_air_trench', 'results');
OUT = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'tm_air_trench');

pk = @(f) getfield(load(fullfile(RES, f)), 'resonance_transmission');
rl = @(f) getfield(load(fullfile(RES, f)), 'resonance_wavelength_nm');

pfx = 'result_N80_TM_W1050_dsh2S40s20_ptw2W1040to980_Ybox8p0_Zbox8p8';

% --- control (no trench) ---
ctrl_f = [pfx '.mat'];
T_ctrl = pk(ctrl_f);
lam_ctrl = rl(ctrl_f);

% --- width 800 nm air trench: d = 1.2 .. 2.4 um ---
d800  = [1.2 1.5 1.8 2.1 2.4];
f800  = { ...
    [pfx '_scRECT_L84000xW800_X0_Y1200_pair_hole.mat'], ...
    [pfx '_scRECT_L84000xW800_X0_Y1500_pair_hole.mat'], ...
    [pfx '_scRECT_L84000xW800_X0_Y1800_pair_hole.mat'], ...
    [pfx '_scRECT_L84000xW800_X0_Y2100_pair_hole.mat'], ...
    [pfx '_scRECT_L84000xW800_X0_Y2400_pair_hole.mat']};
T800 = cellfun(pk, f800);

% --- width 400 nm air trench: d = 1.2 .. 2.4 um (endpoints added, job 119464) ---
d400  = [1.2 1.5 1.8 2.1 2.4];
f400  = { ...
    [pfx '_scRECT_L84000xW400_X0_Y1200_pair_hole.mat'], ...
    [pfx '_scRECT_L84000xW400_X0_Y1500_pair_hole.mat'], ...
    [pfx '_scRECT_L84000xW400_X0_Y1800_pair_hole.mat'], ...
    [pfx '_scRECT_L84000xW400_X0_Y2100_pair_hole.mat'], ...
    [pfx '_scRECT_L84000xW400_X0_Y2400_pair_hole.mat']};
T400 = cellfun(pk, f400);

% --- SiN high-index strip control at d=1.8, w=800 (P2 comparison) ---
T_sin = pk([pfx '_scRECT_L84000xW800_X0_Y1800_pair.mat']);

% --- sanity: resonances stay in the scan window (1556.5 +/- 20 nm) ---
lam_all = [lam_ctrl, cellfun(rl, [f800 f400])];
assert(all(lam_all > 1536 & lam_all < 1577), 'a resonance fell off-window');

% ----------------------------- plot -----------------------------
fig = figure('Color','w','Position',[100 100 780 540]);
hold on; grid on; box on;

c800 = [0.10 0.45 0.80];
c400 = [0.85 0.33 0.10];

% control reference line
yl_ctrl = yline(T_ctrl, '--', sprintf('no trench  %.4f', T_ctrl), ...
    'Color',[0.35 0.35 0.35], 'LineWidth',1.4, ...
    'LabelHorizontalAlignment','left','FontSize',10);

p1 = plot(d800, T800, '-o', 'Color',c800, 'MarkerFaceColor',c800, ...
    'LineWidth',1.8, 'MarkerSize',7);
p2 = plot(d400, T400, '-s', 'Color',c400, 'MarkerFaceColor',c400, ...
    'LineWidth',1.8, 'MarkerSize',8);

xlabel('trench offset d from guide axis (\mum)');
ylabel('peak transmission at resonance');
title(sprintf('Air-trench recycling: peak T vs offset  (stack, \\lambda_{res} \\approx %.1f nm)', lam_ctrl));

legend([p1 p2], {'air trench w = 800 nm','air trench w = 400 nm'}, ...
    'Location','southwest');

xlim([1.05 2.55]);
ylim([0.920 0.965]);
% SiN high-index strip control lands far below (lossy) -> note, not on-scale
text(2.52, 0.9225, sprintf('SiN strip control (d=1.8): T=%.3f', T_sin), ...
    'HorizontalAlignment','right','FontSize',9,'Color',[0.20 0.55 0.20]);
set(gca,'FontSize',11);

% ----------------------------- save -----------------------------
png = fullfile(OUT, 'air_trench_peakT_vs_offset.png');
figf = fullfile(OUT, 'air_trench_peakT_vs_offset.fig');
exportgraphics(fig, png, 'Resolution', 200);
savefig(fig, figf);

fprintf('control (no trench) peak T = %.5f\n', T_ctrl);
fprintf('w800  d=%.2f  T=%.5f\n', [d800; T800]);
fprintf('w400  d=%.2f  T=%.5f\n', [d400; T400]);
fprintf('SiN strip d=1.8 T=%.5f\n', T_sin);
fprintf('best air-trench delta vs control = %+.5f\n', max([T800 T400]) - T_ctrl);
fprintf('SAVED:\n  %s\n  %s\n', png, figf);
