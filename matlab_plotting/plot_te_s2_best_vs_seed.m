% plot_te_s2_best_vs_seed.m — S2 overshoot pi-shift Bragg grating, N=98: Itai's seed vs the
% inverse-design best TE_S2_N98_T9739 (conformal 150 ps: T 0.98995 vs 0.98939).
% Study dir: results_from_athena/te_s2_best_vs_seed/ | 2026-10-13 | jobs 172603_4 (design), 179082_5/6 (conformal)
d = readtable(fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_athena', 'te_s2_best_vs_seed', 's2_seed_vs_t9739.csv'));
t = d.tooth;
f = figure('Position', [100 100 1150 720], 'Color', 'w');
tiledlayout(f, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; plot(t, d.seed_corr_nm, 'k--', t, d.best_corr_nm, 'r-', 'LineWidth', 1.4); grid on
ylabel('corrugation (nm)'); legend({'seed (Itai)', 'best T9739'}, 'Location', 'southeast');
title('S2 overshoot \pi-shift grating, N=98, pitch 491.06 nm, \lambda 1559.67 nm: T 0.98939 \rightarrow 0.98995 (conformal)');
nexttile; stem(t, d.dcorr_nm, 'filled', 'MarkerSize', 4); grid on; ylabel('\Delta corrugation (nm)');
nexttile; plot(t, d.seed_shift_nm, 'k--', t, d.best_shift_nm, 'r-', 'LineWidth', 1.4); grid on
xlabel('tooth from cavity'); ylabel('tooth shift (nm)'); legend({'seed', 'best T9739'}, 'Location', 'northeast');
out = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_athena', 'te_s2_best_vs_seed', 's2_seed_vs_t9739');
savefig(f, [out '.fig']); exportgraphics(f, [out '.png'], 'Resolution', 150);
