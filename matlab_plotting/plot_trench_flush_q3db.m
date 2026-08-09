%% plot_trench_flush_q3db — flush-top trench N ladder: T(dB) and Q vs N
% Study: runners/metal_mirror/trench_flush_q3db.py | Jobs: Athena 128925+129103,
% ctrl IGUM 50733 | 2026-08-08
% z-sym-OFF family (flush requires it): 4 flush points + 1 ctrl point.
% Stored z-sym-ON curves (trench_q3db_20um, IGUM) dashed for context — they
% carry a measured numerics offset at corr 325 (ctrl N165: dT -0.21 dB, dQ +7%).
res_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'results_from_athena', 'trench_flush_q3db', 'results');

N_flush  = [167 168 169 170];
dB_flush = [-3.161 -3.259 -3.362 -3.467];
Q_flush  = [17526 17982 18439 18897];
ctrl_off = struct('N', 165, 'dB', -3.304, 'Q', 14904);   % IGUM 50733, z-sym OFF
% stored z-sym-ON anchors (context only)
N_ctrl_on  = [150 165 180 195];   dB_ctrl_on  = [-1.966 -3.093 -4.749 -7.106];
N_tr_on    = [165 169 170 185];   dB_tr_on    = [-2.563 -2.903 -2.992 -4.672];

p = polyfit(N_flush, dB_flush, 1);
N_cross = (-3.0 - p(2)) / p(1);
Q_cross = interp1(N_flush, Q_flush, N_cross, 'linear', 'extrap');

fig = figure('Position', [80 80 980 540], 'Visible', 'off');
yyaxis left
plot(N_flush, dB_flush, 'o-', 'LineWidth', 1.6, 'MarkerFaceColor', 'auto'); hold on;
plot(ctrl_off.N, ctrl_off.dB, 's', 'MarkerSize', 10, 'LineWidth', 1.6);
plot(N_ctrl_on, dB_ctrl_on, '--', 'Color', [0.5 0.5 0.5]);
plot(N_tr_on, dB_tr_on, ':', 'Color', [0.5 0.5 0.5]);
yline(-3, '-', '-3 dB', 'LineWidth', 1.0);
xline(N_cross, ':', sprintf('N* = %.1f', N_cross));
ylabel('peak T (dB)'); ylim([-8 -1.5]);
yyaxis right
plot(N_flush, Q_flush/1e3, '^-', 'LineWidth', 1.6, 'MarkerFaceColor', 'auto');
plot(ctrl_off.N, ctrl_off.Q/1e3, 'v', 'MarkerSize', 10, 'LineWidth', 1.6);
ylabel('Q_L (x10^3)');
xlabel('N periods each side'); grid on; xlim([148 200]);
title(sprintf(['Flush-top trench (w800/d1800, z -3.975..+0.175 \\mum), corr 325, 20 \\mum mode' ...
    newline 'T = -3 dB at N* = %.1f, Q = %.1fk, fwhm 19.4 \\mum (z-sym-OFF family)'], ...
    N_cross, Q_cross/1e3), 'FontSize', 12);
legend({'flush T(dB)', 'ctrl z-sym-OFF T(dB)', 'stored ctrl (z-sym ON)', ...
    'stored full-z trench (z-sym ON)', '', '', 'flush Q', 'ctrl z-sym-OFF Q'}, ...
    'Location', 'southwest', 'FontSize', 9);

savefig(fig, fullfile(res_dir, 'trench_flush_q3db_T_Q.fig'));
exportgraphics(fig, fullfile(res_dir, 'trench_flush_q3db_T_Q.png'), 'Resolution', 200);
fprintf('N* = %.2f, Q at crossing = %.0f -> %s\n', N_cross, Q_cross, ...
    fullfile(res_dir, 'trench_flush_q3db_T_Q.png'));
