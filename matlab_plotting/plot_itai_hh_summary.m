% Itai's HH apodization vs our Q3dB devices -- the comparison figure.
% Study: runners/sweeps/itai_hh_apod.py + itai_hh_tm_cross.py | IGUM 63438/63451/63454/63491
% Every point plotted has T+R < 1 (rows that violated energy conservation are excluded
% upstream by hh_synth.py). Q(-3dB) = 0.293*Q_i, a relation MEASURED on our own
% N=166..215 ladder and re-confirmed on his geometry across N=98..140.
clear; close all;
d = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_igum', 'itai_hh_summary.csv');
T = readtable(d);

his = strcmp(T.who, 'his');  ours = ~his;
te  = strcmp(T.pol, 'TE');   tm   = strcmp(T.pol, 'TM');

fig = figure('Position', [80 80 1150 500], 'Color', 'w');

% ---- panel 1: intrinsic Q vs mode width -------------------------------------
subplot(1,2,1);
BL = [0 0.35 0.85]; RD = [0.85 0.25 0];
semilogy(T.fwhm(his&te), T.Qi(his&te), 'o', 'MarkerSize', 10, 'LineWidth', 1.5, ...
         'Color', BL, 'MarkerFaceColor', BL); hold on;
semilogy(T.fwhm(his&tm), T.Qi(his&tm), 's', 'MarkerSize', 10, 'LineWidth', 1.5, ...
         'Color', RD, 'MarkerFaceColor', RD);
semilogy(T.fwhm(ours&te), T.Qi(ours&te), 'p', 'MarkerSize', 17, 'LineWidth', 2, ...
         'Color', 'k', 'MarkerFaceColor', [0.75 0.85 1]);
semilogy(T.fwhm(ours&tm), T.Qi(ours&tm), 'h', 'MarkerSize', 17, 'LineWidth', 2, ...
         'Color', 'k', 'MarkerFaceColor', [1 0.82 0.72]);
xline(20, 'k:', 'LineWidth', 1.2);
% gain annotations
text(20.72, 378044*1.35, 'x9.5', 'Color', BL, 'FontWeight', 'bold', 'FontSize', 12);
text(20.30, 121007*1.45, 'x2.5', 'Color', RD, 'FontWeight', 'bold', 'FontSize', 12);
grid on; xlabel('spatial mode FWHM (\mum)'); ylabel('intrinsic Q_i');
ylim([2e4 1.2e6]); xlim([19.2 21.3]);
legend({'HH apod TE','HH apod TM','our Q3dB TE','our Q3dB TM','20 \mum spec'}, ...
       'Location','southwest','FontSize',9);
title({'Intrinsic Q at matched mode width', ...
       'circles/squares = Itai HH apodization,  stars = our devices'});

% ---- panel 2: TM crossing ladder --------------------------------------------
subplot(1,2,2);
c = strcmp(T.who,'his') & tm & ~isnan(T.N);
[Ns, k] = sort(T.N(c)); Ts = T.Tres(c); Ts = Ts(k);
plot(Ns, Ts, 'o-', 'LineWidth', 1.8, 'MarkerSize', 8, ...
     'Color', [0.8 0.25 0], 'MarkerFaceColor', [1 0.8 0.7]); hold on;
yline(0.5, 'k--', 'LineWidth', 1.2);
xline(189, 'k:', 'LineWidth', 1.2);
grid on; xlabel('periods per side, N'); ylabel('peak transmission T');
xlim([90 205]); ylim([0.4 1.0]);
plot(189, 0.52747, 'p', 'MarkerSize', 18, 'LineWidth', 2, 'Color', 'k', 'MarkerFaceColor', [1 0.85 0.2]);
legend({'HH apod TM (scale 0.72, 20.2-20.4 \mum)','-3 dB','crossing N = 190','MEASURED at N=189: T=0.527, Q=34006'}, ...
       'Location','southwest','FontSize',9);
title({'TM crossing ladder (same method as our own anchors)', ...
       'Q_c grows 3.19% per period;  Q_i N-independent to 9%'});

out = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_igum');
savefig(fig, fullfile(out, 'itai_hh_summary.fig'));
exportgraphics(fig, fullfile(out, 'itai_hh_summary.png'), 'Resolution', 150);
fprintf('summary figure written\n');
