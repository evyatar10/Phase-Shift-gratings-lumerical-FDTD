% plot_nt80x_contrast_search.m -- Nt80x inner-pair contrast line search: radiation per stored energy, Q_i, and the harmonic table.
% Study dir: runners/sweeps/ (farfield_sph_15um_bumps, _contrast) | Jobs: Athena 265046 (control), 265111, 265257 | 2026-10-20
% Inputs: the *_multipoles.csv next to each *_ff.mat (python_tools/farfield_multipole.py --lmax 150 --csv) and the MEASURED
% scalars below (ringdown_budget / solve_v4). Output: results_from_athena/farfield_sph_15um_contrast/nt80x_contrast_search.{fig,png}
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
out = fullfile(root, 'results_from_athena', 'farfield_sph_15um_contrast');
d_nm  = [0 5 10 15 20];                                   % contrast step on the two innermost periods (wide +d/2, narrow -d/2)
PU    = [1.000 0.862 0.767 0.720 0.684];                     % MEASURED radiation per stored energy / control
Qi    = [5.35 6.13 6.79 7.23 7.49];                         % MEASURED Q_i (1e5)
dirs  = {fullfile(root,'results_from_athena','farfield_sph_15um_c_nt80','results'), ...
         fullfile(root,'results_from_athena','farfield_sph_15um_bumps','diff_g0001_p5nm','results'), ...
         fullfile(root,'results_from_athena','farfield_sph_15um_contrast','diff_g0001_p10nm','results'), ...
         fullfile(root,'results_from_athena','farfield_sph_15um_contrast','diff_g0001_p15nm','results'), ...
         fullfile(root,'results_from_athena','farfield_sph_15um_contrast','diff_g0001_p20nm','results')};
f = figure('Color','w','Position',[80 80 1500 560]);
subplot(1,3,1); yyaxis left; plot(d_nm, PU, 'o-', 'LineWidth', 1.6, 'MarkerFaceColor', 'auto'); ylabel('radiation per stored energy / control'); ylim([0.6 1.05]);
yyaxis right; plot(d_nm, Qi, 's--', 'LineWidth', 1.4); ylabel('Q_i (\times10^5)'); ylim([5 8]);
xlabel('extra contrast on periods {0,1} (nm)'); grid on; title('Nt80x: inner-pair contrast, \lambda and W held', 'FontSize', 11);
% harmonic tables: share of the CONTROL's leak (scale each device by its P/U so bars are comparable in absolute power)
keys = {}; T = containers.Map();
for k = 1:numel(dirs)
    c = dir(fullfile(dirs{k}, '*_multipoles.csv')); if isempty(c), continue; end
    M = readmatrix(fullfile(dirs{k}, c(1).name));       % l, m, frac_E, frac_M
    for r = 1:size(M,1)
        key = sprintf('%s(%d,%+d)', char('E'*(M(r,3)>=M(r,4)) + 'M'*(M(r,3)<M(r,4))), M(r,1), M(r,2));
        v = (M(r,3)+M(r,4)) * PU(k);
        if ~isKey(T, key), T(key) = zeros(1, numel(dirs)); keys{end+1} = key; end %#ok<AGROW>
        tmp = T(key); tmp(k) = v; T(key) = tmp;
    end
end
vals = cell2mat(cellfun(@(k) T(k)', keys, 'UniformOutput', false))';   % n_keys x n_dev
[~, order] = sort(vals(:,1), 'descend'); top = order(1:12);
subplot(1,3,[2 3]); bar(100*vals(top,:)); set(gca, 'XTick', 1:12, 'XTickLabel', keys(top), 'XTickLabelRotation', 35, 'FontSize', 9);
ylabel('% of the control''s radiated power'); legend({'control', '+5 nm', '+10 nm', '+15 nm', '+20 nm'}, 'Location', 'northeast');
title('Largest harmonics of the control and what the contrast knob leaves of them (per stored energy)', 'FontSize', 11); grid on;
lsum = @(k, L) sum(vals(cellfun(@(s) sscanf(s(3:end), '%d') <= L, keys), k));
fprintf('l<=5 share of control leak: ctrl %.1f%%  +5 %.1f%%  +10 %.1f%%  +15 %.1f%%\n', 100*lsum(1,5), 100*lsum(2,5), 100*lsum(3,5), 100*lsum(4,5));
sgtitle('Nt80x (N=96, pitch 514 nm, \lambda 1619.47 nm): the re-shaping baseline, MEASURED (Athena 265111 / 265257)', 'FontSize', 12);
exportgraphics(f, fullfile(out, 'nt80x_contrast_search.png'), 'Resolution', 160); savefig(f, fullfile(out, 'nt80x_contrast_search.fig'));
fprintf('saved %s\n', out);
