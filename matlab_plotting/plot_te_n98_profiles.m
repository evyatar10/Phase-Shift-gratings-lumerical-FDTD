% plot_te_n98_profiles.m — per-tooth corrugation and tooth-shift profiles of the
% N=98 TE inverse-designed pi-shift Bragg gratings, both seeds, seed vs best.
% Study dir: results_from_athena/te_n98_profiles/ | 2026-10-10 | jobs 172603 (bests), 174284_9 (S2 shift-fix)
% Input CSV from the campaign vectors (runners/lumopt2_design/best_designs.py); tooth 1 = innermost.
d = readtable(fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_athena', 'te_n98_profiles', 'te_n98_profiles.csv'));
t = d.tooth;
f = figure('Position', [100 100 1100 700], 'Color', 'w');
tl = tiledlayout(f, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; plot(t, d.s1_seed_corr, 'k--', t, d.s1_best_corr, 'b-', 'LineWidth', 1.4);
ylabel('corrugation (nm)'); title('S1 plain W800, N=98: λ 1561.04 nm, T 0.974 (150 ps)'); legend({'seed 250 nm', 'best'}, 'Location', 'south'); grid on
nexttile; plot(t, d.s2_seed_corr, 'k--', t, d.s2_best_corr, 'r-', t, d.s2_shiftfix_corr, 'm-', 'LineWidth', 1.4);
title('S2 overshoot Nt60, N=98: λ 1560.42 nm, T 0.986 (150 ps)'); legend({'seed (Itai)', 'best', 'shift-fix lane'}, 'Location', 'north'); grid on
nexttile; plot(t, d.s1_seed_shift, 'k--', t, d.s1_best_shift, 'b-', 'LineWidth', 1.4);
xlabel('tooth from cavity'); ylabel('tooth shift (nm)'); title('S1 tooth shifts'); grid on
nexttile; plot(t, d.s2_seed_shift, 'k--', t, d.s2_best_shift, 'r-', t, d.s2_shiftfix_shift, 'm-', 'LineWidth', 1.4);
xlabel('tooth from cavity'); title('S2 tooth shifts'); legend({'seed', 'best', 'shift-fix lane'}, 'Location', 'north'); grid on
out = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_athena', 'te_n98_profiles', 'te_n98_profiles');
savefig(f, [out '.fig']); exportgraphics(f, [out '.png'], 'Resolution', 150);
