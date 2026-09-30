% plot_incore_axis_vs_pair.m — in-core SiO2 cylinders, ONE on the axis (y 0) vs the PAIR at y +/-250: radius series.
% Study dirs: results_from_athena/{scat_x10_incore_r50_axis, scat_x11_incore_axis_r30_40_60} + results_from_igum/
% scat_x12_incore_axis_r80_110 (r 80; r 110 re-run on Athena into the same study dir) — jobs 151320/151333/90593/151353;
% pair rows from scat_x4_incore_below530 / x5 / x6 / x7 (149982/150391/150429/150458). Plot 2026-09-16.
% Three panels vs hole radius at Lambda 524 / 270 deg / 31 sites: peak T, Q (= lambda/|spectral FWHM|), spatial width.
% Device: TM, W 800, corr 400, h 350, pitch 516.83, N 80/side, box y 16, 20 nm / 1501 pts. pchip through the points.
root = fullfile(fileparts(mfilename('fullpath')), '..', '..');
ctrl = load(fullfile(root, 'results_from_athena', 'scat_h_retrocomb', 'results', 'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'), ...
            'resonance_transmission', 'resonance_wavelength_nm', 'spectral_fwhm_nm', 'fwhm_m');
c0 = [ctrl.resonance_transmission, ctrl.resonance_wavelength_nm / abs(ctrl.spectral_fwhm_nm), ctrl.fwhm_m * 1e6];

axis_rows = rows([dir(fullfile(root, 'results_from_athena', 'scat_x1*_incore*', 'results', '*Y0to0*.mat')); ...
                  dir(fullfile(root, 'results_from_igum',   'scat_x1*_incore*', 'results', '*Y0to0*.mat'))]);
pair_rows = rows([dir(fullfile(root, 'results_from_athena', 'scat_x4_incore_below530', 'results', '*arr31_X-7467to8253_Y250to250_C400*.mat')); ...
                  dir(fullfile(root, 'results_from_athena', 'scat_x5_incore_r110',      'results', '*arr31_X-7467to8253_Y250to250_C400*.mat')); ...
                  dir(fullfile(root, 'results_from_athena', 'scat_x6_incore_r50',       'results', '*arr31_X-7467to8253_Y250to250_C400*.mat')); ...
                  dir(fullfile(root, 'results_from_athena', 'scat_x7_incore_r40_r30',   'results', '*arr31_X-7467to8253_Y250to250_C400*.mat'))]);

cA = [0.85 0.33 0.10]; cP = [0.49 0.18 0.56]; grey = [0.45 0.45 0.45];
ylab = {'peak transmission', 'Q', 'spatial mode width FWHM (\mum)'};
ctrl_name = {sprintf('no holes, T %.3f', c0(1)), sprintf('no holes, Q %.0f', c0(2)), ['no holes, ' sprintf('%.1f', c0(3)) ' \mum']};
fig = figure('Visible', 'off', 'Position', [60 60 1500 480]);
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:3
    ax = nexttile(tl); hold(ax, 'on');
    yline(ax, c0(p), '--', 'Color', grey, 'LineWidth', 1.2, 'DisplayName', ctrl_name{p});
    curve(ax, axis_rows(:, 1), axis_rows(:, p + 1), 'o', cA, 'one cylinder on the axis (y 0)');
    curve(ax, pair_rows(:, 1), pair_rows(:, p + 1), 's', cP, 'pair at y \pm250 nm');
    hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); xlim(ax, [20 120]);
    xlabel(ax, 'hole radius r (nm)'); ylabel(ax, ylab{p});
    legend(ax, 'Location', 'southoutside', 'FontSize', 9);
end
title(tl, 'SiO_2 cylinders in the core, \Lambda 524, 270\circ, 31 sites - TM, corr 400, N 80/side');
out = fullfile(root, 'results_from_athena', 'scat_x11_incore_axis_r30_40_60', 'incore_axis_vs_pair_radius');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150); disp(['wrote ' out '.png']);
disp('axis rows [r T Q w]:'); disp(axis_rows); disp('pair rows [r T Q w]:'); disp(pair_rows);

function curve(ax, x, y, mk, c, name)
    xx = linspace(min(x), max(x), 300);
    plot(ax, xx, pchip(x, y, xx), '-', 'Color', c, 'LineWidth', 1.6, 'HandleVisibility', 'off');
    plot(ax, x, y, mk, 'Color', c, 'MarkerFaceColor', c, 'MarkerSize', 7, 'HandleVisibility', 'off');
    plot(ax, NaN, NaN, ['-' mk], 'Color', c, 'MarkerFaceColor', c, 'MarkerSize', 7, 'LineWidth', 1.6, 'DisplayName', name);
end

function M = rows(L)
    % [r  T  Q  width_um] per file, sorted by radius; peak T = built-in finder, Q = lambda/|spectral FWHM|
    M = zeros(0, 4);
    for k = 1:numel(L)
        tok = regexp(L(k).name, 'scR(\d+)_', 'tokens', 'once'); if isempty(tok), continue, end
        m = load(fullfile(L(k).folder, L(k).name), 'resonance_transmission', 'resonance_wavelength_nm', 'spectral_fwhm_nm', 'fwhm_m');
        M(end + 1, :) = [str2double(tok{1}), m.resonance_transmission, m.resonance_wavelength_nm / abs(m.spectral_fwhm_nm), m.fwhm_m * 1e6]; %#ok<AGROW>
    end
    M = sortrows(M, 1);
end
