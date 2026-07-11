% plot_detune_T_gap34.m
% TM side-by-side: device-1 peak transmission (at resonance) vs device-2
% corrugation, for gap Dy = 3 and 4 um only, over the full corr-2 scan
% (360..650 nm). Peak T = resonance_transmission (built-in resonance finder,
% NOT argmax(T)). Device-1 corrugation fixed at 400 nm, pitch 516.83, N=80, Dx=0.
%
% Benchmark = the plain single pi-shift Bragg grating (same corr 400, N=80,
% pitch 516.83), NO neighbour, at the SAME numerics as the side-by-side runs:
% optimization mesh dx=50 nm and the default span_mult=1.8 z-box (z ~ 3.2 um,
% under-converged in z -> T depressed by ~0.09 vs a tall converged box). Using
% the converged-in-z single device (~0.886) would be apples-to-oranges here
% because these coupled runs are themselves at the short-z box. Proxy file:
% tm_span_convergence result at Ybox 6.8 / Zbox 3.2 um (T=0.7926, loss 19.5%).
% Output: PNG + editable FIG in results_from_athena/side_by_side_tm_detune_400nm/.

clear; close all;

RES = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'side_by_side_tm_detune_400nm', 'results');
OUT = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'side_by_side_tm_detune_400nm');
BENCH = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'tm_span_convergence', 'results', ...
    'result_N80_TM_avg_Ybox6p8_Zbox3p2.mat');   % same numerics as side-by-side (opt mesh, z~3.2um)

pk = @(f) getfield(load(fullfile(RES, f)), 'resonance_transmission');
rl = @(f) getfield(load(fullfile(RES, f)), 'resonance_wavelength_nm');

pfx = 'result_N80_TM_avg_2pishift_p516p8';   % ..._Ygap{gap}nm_Xstag0nm_corr1400nm_corr2{c}nm

corr2 = [360 380 400 420 440 450 500 550 600 650];

fname = @(gap, c) sprintf('%s_Ygap%dnm_Xstag0nm_corr1400nm_corr2%dnm.mat', pfx, gap, c);

T3 = arrayfun(@(c) pk(fname(3000, c)), corr2);
T4 = arrayfun(@(c) pk(fname(4000, c)), corr2);

% resonance for the title (use the matched-reference corr2=400 point, gap 3 um)
lam0 = rl(fname(3000, 400));

% --- benchmark: single device, no scatterers / no neighbour ---
b = load(BENCH);
T_bench = b.resonance_transmission;

% --- sanity: resonances stay in the scan window ---
lam_all = [arrayfun(@(c) rl(fname(3000,c)), corr2), ...
           arrayfun(@(c) rl(fname(4000,c)), corr2)];
assert(all(lam_all > 1548 & lam_all < 1570), 'a resonance fell off-window');

% ----------------------------- plot -----------------------------
fig = figure('Color','w','Position',[100 100 820 560]);
hold on; grid on; box on;

c3 = [0.10 0.45 0.80];
c4 = [0.85 0.33 0.10];

% benchmark reference line (single device, SAME box/mesh as these runs)
yline(T_bench, '--', sprintf('baseline  %.4f', T_bench), ...
    'Color',[0.35 0.35 0.35], 'LineWidth',1.4, ...
    'LabelHorizontalAlignment','left', 'LabelVerticalAlignment','bottom', ...
    'FontSize',10, 'Interpreter','none');

% matched-reference marker (device-1 = 400 nm)
xline(400, ':', 'Color',[0.6 0.6 0.6], 'LineWidth',1.0, 'HandleVisibility','off');

p3 = plot(corr2, T3, '-o', 'Color',c3, 'MarkerFaceColor',c3, ...
    'LineWidth',1.8, 'MarkerSize',7);
p4 = plot(corr2, T4, '-s', 'Color',c4, 'MarkerFaceColor',c4, ...
    'LineWidth',1.8, 'MarkerSize',8);

xlabel('device-2 corrugation (nm)');
ylabel('peak transmission at resonance  (|S_{21}|^2)');
title(sprintf(['TM side-by-side: device-1 peak T vs device-2 corrugation\n' ...
    'device-1 corr 400 nm, pitch 516.83 nm, N=80, \\Deltax=0, \\lambda_{res} \\approx %.1f nm'], lam0));

legend([p3 p4], {'gap 3 \mum','gap 4 \mum'}, 'Location','east');

xlim([350 660]);
set(gca,'FontSize',11);

% ----------------------------- save -----------------------------
png  = fullfile(OUT, 'detune_transmission_gap34_TM.png');
figf = fullfile(OUT, 'detune_transmission_gap34_TM.fig');
exportgraphics(fig, png, 'Resolution', 200);
savefig(fig, figf);

fprintf('benchmark (single device, same box: opt mesh, z~3.2um) peak T = %.5f  (lam = %.2f nm)\n', ...
    T_bench, b.resonance_wavelength_nm);
fprintf('gap 3um  corr2=%d  T=%.5f\n', [corr2; T3]);
fprintf('gap 4um  corr2=%d  T=%.5f\n', [corr2; T4]);
fprintf('SAVED:\n  %s\n  %s\n', png, figf);
