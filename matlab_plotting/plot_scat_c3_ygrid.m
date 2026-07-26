% plot_scat_c3_ygrid.m — 2D y-grid single-scatterer response summary (stage C3)
% Study: runners/scatterers/scat_c3_ygrid.py | Jobs: 121525 + 121614 | 2026-07-15
% Purpose: per-row far-field landscapes (rows y = 700..1510 nm), measured y-decay
% law + cross-row phase rotation, from c3_summary.mat (exported by the solver).
% Output: editable .fig + PNG in results_from_athena/scat_c3_ygrid/.

root = fileparts(fileparts(mfilename('fullpath')));
ddir = fullfile(root, 'results_from_athena', 'scat_c3_ygrid');
m = load(fullfile(ddir, 'c3_summary.mat'));

rows = m.rows_y_nm(:);
fig = figure('Visible', 'on', 'Position', [80 80 1150 780]);

subplot(2, 2, 1); hold on; grid on;
plot(m.land_x_nm{1}/1000, m.land_gain_pct{1}, 'o-', 'MarkerSize', 3);
xlabel('x (\mum)'); ylabel('FF gain (%)');
title(sprintf('Row y = %.0f nm (reference row)', rows(1)));

subplot(2, 2, 2); hold on; grid on;
co = lines(numel(rows) - 1);
for k = 2:numel(rows)
    plot(m.land_x_nm{k}/1000, m.land_gain_pct{k}, '.-', 'Color', co(k-1, :), ...
        'MarkerSize', 8, 'DisplayName', sprintf('y = %.0f', rows(k)));
end
xlabel('x (\mum)'); ylabel('FF gain (%)');
legend('Location', 'southoutside', 'FontSize', 7, 'NumColumns', 3);
title('Rows y = 900..1510 nm');

subplot(2, 2, 3);
semilogy(m.decay_y_nm - 700, m.decay_ratio, 'o-'); grid on;
xlabel('\Deltay from row 700 (nm)'); ylabel('median |r_y / r_{700}|');
title('Amplitude decay — smooth, no oscillation');

subplot(2, 2, 4);
ph = rad2deg(unwrap(deg2rad(m.decay_phase_deg)));   % continuous rotation, no ±180 wrap
plot(m.decay_y_nm - 700, ph, 's-'); grid on;
xlabel('\Deltay from row 700 (nm)'); ylabel('median arg \rho (deg, unwrapped)');
title('Cross-row phase rotation');

sgtitle(['\pi-shift Bragg TM, corr 400 nm, pitch 516.83 nm, \lambda_{res} ' ...
    '1558.6 nm — single-scatterer response grid'], 'FontSize', 11);

savefig(fig, fullfile(ddir, 'c3_ygrid_summary.fig'));
exportgraphics(fig, fullfile(ddir, 'c3_ygrid_summary.png'), 'Resolution', 170);
disp(['saved: ' fullfile(ddir, 'c3_ygrid_summary.png')]);
