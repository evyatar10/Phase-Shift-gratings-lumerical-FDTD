% Transmission spectrum (dB) of the N=1300/side pi-shift Bragg grating.
% Study: results_from_athena/tm_h200_w1800_p504_n1300   Job: 127443 (2026-08-03)
% Purpose: single-panel T(lambda) in dB with resonance/Q/T annotation.
% Q = 7.0e5 is the energy-decay fit (tau_E = 549-567 ps) from the run's solver
% log — the 16 pm spectral line is truncation-limited (500 ps record), so the
% spectrum alone cannot show it; T_res = 0.0031 is the measured (same-limit) peak.

RESULTS_DIR = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'tm_h200_w1800_p504_n1300', 'results');
MAT = fullfile(RESULTS_DIR, 'result_N1300_TM_avg_Wavg1800_C400_Ybox9p5_Zbox7p7.mat');

LAM_RES_NM = 1492.124;   % defect peak, read manually (stored scalar is a band-edge mis-pick)
Q_DECAY    = 7.0e5;      % from tau_E fit of the solver decay trace
T_RES      = 0.0031;     % measured peak T (truncation-limited)
DB_FLOOR   = -80;        % clip: stopband floor ~4e-7, exact zeros -> -Inf

d = load(MAT);
wl = d.wl_nm(:);
TdB = 10 * log10(max(d.T(:), 10^(DB_FLOOR / 10)));

fig = figure('Position', [100 100 900 560], 'Color', 'w');
plot(wl, TdB, 'b-', 'LineWidth', 0.9);
hold on;
plot(LAM_RES_NM, 10 * log10(T_RES), 'rv', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
text(LAM_RES_NM + 0.15, 10 * log10(T_RES), ...
    sprintf('\\lambda_{res} = %.3f nm', LAM_RES_NM), 'Color', 'r');
hold off;
grid on;
xlim([min(wl) max(wl)]);
ylim([DB_FLOOR 5]);
xlabel('Wavelength [nm]');
ylabel('Transmission [dB]');
title(sprintf(['\\pi-shift Bragg, TM, h=200 nm, w_{avg}=1800 nm, corr=400 nm, ' ...
    '\\Lambda=504 nm, 1300 periods']));
subtitle(sprintf('\\lambda_{res} = %.3f nm,  Q = 7.0\\times10^5,  T_{res} = %.1f dB', ...
    LAM_RES_NM, 10 * log10(T_RES)));

savefig(fig, fullfile(RESULTS_DIR, 'transmission_N1300_dB.fig'));
exportgraphics(fig, fullfile(RESULTS_DIR, 'transmission_N1300_dB.png'), 'Resolution', 200);
fprintf('saved: %s\n', fullfile(RESULTS_DIR, 'transmission_N1300_dB.png'));

% ── Linear-scale version (same data, T_res quoted in linear units) ───────────
fig2 = figure('Position', [100 100 900 560], 'Color', 'w');
plot(wl, d.T(:), 'b-', 'LineWidth', 0.9);
hold on;
plot(LAM_RES_NM, T_RES, 'rv', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
text(LAM_RES_NM + 0.15, T_RES + 0.03, ...
    sprintf('\\lambda_{res} = %.3f nm', LAM_RES_NM), 'Color', 'r');
hold off;
grid on;
xlim([min(wl) max(wl)]);
ylim([0 1.05]);
xlabel('Wavelength [nm]');
ylabel('Transmission');
title(sprintf(['\\pi-shift Bragg, TM, h=200 nm, w_{avg}=1800 nm, corr=400 nm, ' ...
    '\\Lambda=504 nm, 1300 periods']));
subtitle(sprintf('\\lambda_{res} = %.3f nm,  Q = 7.0\\times10^5,  T_{res} = %.4f', ...
    LAM_RES_NM, T_RES));

savefig(fig2, fullfile(RESULTS_DIR, 'transmission_N1300_linear.fig'));
exportgraphics(fig2, fullfile(RESULTS_DIR, 'transmission_N1300_linear.png'), 'Resolution', 200);
fprintf('saved: %s\n', fullfile(RESULTS_DIR, 'transmission_N1300_linear.png'));
