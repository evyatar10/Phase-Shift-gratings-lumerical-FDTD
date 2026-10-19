% plot_farfield_sph_15um.m — spherical-harmonic triangle tables of the far field of the
% 15.5-um-mode pi-shift gratings at N = 96 (study runners/sweeps/farfield_sph_15um.py).
% Phase A = Athena 264973 (2026-10-19): plain TE corr 316 / pitch 500 and plain TM corr 400 /
% pitch 516.83, W800, steady state 150 ps, tube projection. Rows l, columns |m| (+-m summed),
% cell = % of radiated power; electric (top) and magnetic (bottom) multipoles.
% Input: *_multipoles.csv from python_tools/farfield_multipole.py.
clear; close all;
D = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
             'results_from_athena', 'farfield_sph_15um', 'results');
LMAX = 8;
dev = { ...
  'result_N96_avg_C316_Ybox6p8_Zbox6p8_AS1em12_ff', {'TE plain, corr 316 nm, \Lambda 500 nm', '\lambda 1558.51 nm, T 0.707, mode 15.29 \mum'}; ...
  'result_N96_TM_avg_Ybox8p0_Zbox8p8_AS1em12_ff',   {'TM plain, corr 400 nm, \Lambda 516.83 nm', '\lambda 1558.61 nm, T 0.792, mode 15.90 \mum'}};

fig = figure('Color', 'w', 'Position', [50 50 1300 1000], 'Visible', 'off');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
cmax = 15;
for d = 1:size(dev, 1)
    T = readtable(fullfile(D, [dev{d, 1} '_multipoles.csv']));
    PE = nan(LMAX + 1); PM = nan(LMAX + 1);
    for l = 0:LMAX, PE(l + 1, 1:l + 1) = 0; PM(l + 1, 1:l + 1) = 0; end
    for k = 1:height(T)
        l = T.l(k); m = abs(T.m(k));
        if l <= LMAX
            PE(l + 1, m + 1) = PE(l + 1, m + 1) + 100 * T.frac_E(k);
            PM(l + 1, m + 1) = PM(l + 1, m + 1) + 100 * T.frac_M(k);
        end
    end
    for typ = 1:2
        if typ == 1, P = PE; nm = 'electric'; else, P = PM; nm = 'magnetic'; end
        ax = nexttile((typ - 1) * 2 + d);
        im = imagesc(0:LMAX, 0:LMAX, P); set(im, 'AlphaData', ~isnan(P));
        set(ax, 'YDir', 'normal', 'XTick', 0:LMAX, 'YTick', 0:LMAX, 'FontSize', 10, ...
                'Color', 'w', 'TickLength', [0 0]); clim([0 cmax]); box on;
        xlabel('|m|'); yl = ylabel('l', 'Rotation', 0); yl.Position(1) = -1.1;
        for l = 0:LMAX
            for m = 0:l
                v = P(l + 1, m + 1);
                if v >= 0.05
                    text(m, l, sprintf('%.1f', v), 'HorizontalAlignment', 'center', ...
                         'FontSize', 9, 'Color', [0 0 0] + (v > 0.55 * cmax) * [1 1 1]);
                end
            end
        end
        title({sprintf('(%s) %s, %s', char('a' + (typ - 1) * 2 + d - 1), dev{d, 2}{1}, nm), dev{d, 2}{2}}, ...
              'FontSize', 10, 'FontWeight', 'normal');
    end
end
colormap(flipud(hot)); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '% of radiated power';
out = fullfile(D, '..', 'farfield_sph_15um_plain_triangles');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150);
fprintf('saved %s.fig/.png\n', out);

%% ── Figure 2 (2026-10-20): the two corr-500 overshoots, pitch 514, N=96 (Athena 265046 / 265047).
% Short runs (~2 decay times, gate B2 passed); T DERIVED = gamma^2-corrected port T. Taller triangle:
% the overshoot radiates to high l.
LMAX = 24; cmax = 5;
dev = { ...
  fullfile('farfield_sph_15um_c_nt80', 'results', 'result_N96_W950_Wavg1000_C500_ptw96W950to1250_ptn96W950to750_Ybox6p8_Zbox6p8_AS1em12_RD0ps_T340ps_ff'), ...
      {'Nt80 overshoot (new), corr 500 nm, \Lambda 514 nm', '\lambda 1619.47 nm, Q_L 9.9e4, T 0.66, mode 16.15 \mum'}; ...
  fullfile('farfield_sph_15um_c_hh60', 'results', 'result_N96_W950_Wavg1000_C500_ptw96W950to1250_ptn96W950to750_Ybox6p8_Zbox6p8_AS1em12_RD0ps_T265ps_ff'), ...
      {'HH60 overshoot (old, unscaled), corr 500 nm, \Lambda 514 nm', '\lambda 1619.42 nm, Q_L 7.6e4, T 0.73, mode 15.45 \mum'}};
DA = fileparts(D);  % results_from_athena
DA = fileparts(DA);
fig = figure('Color', 'w', 'Position', [50 50 1500 1300], 'Visible', 'off');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for d = 1:size(dev, 1)
    T = readtable(fullfile(DA, [dev{d, 1} '_multipoles.csv']));
    PE = nan(LMAX + 1); PM = nan(LMAX + 1);
    for l = 0:LMAX, PE(l + 1, 1:l + 1) = 0; PM(l + 1, 1:l + 1) = 0; end
    for k = 1:height(T)
        l = T.l(k); m = abs(T.m(k));
        if l <= LMAX
            PE(l + 1, m + 1) = PE(l + 1, m + 1) + 100 * T.frac_E(k);
            PM(l + 1, m + 1) = PM(l + 1, m + 1) + 100 * T.frac_M(k);
        end
    end
    for typ = 1:2
        if typ == 1, P = PE; nm = 'electric'; else, P = PM; nm = 'magnetic'; end
        ax = nexttile((typ - 1) * 2 + d);
        im = imagesc(0:LMAX, 0:LMAX, P); set(im, 'AlphaData', ~isnan(P));
        set(ax, 'YDir', 'normal', 'XTick', 0:4:LMAX, 'YTick', 0:4:LMAX, 'FontSize', 10, ...
                'Color', 'w', 'TickLength', [0 0]); clim([0 cmax]); box on;
        xlabel('|m|'); yl = ylabel('l', 'Rotation', 0); yl.Position(1) = -2.5;
        for l = 0:LMAX
            for m = 0:l
                v = P(l + 1, m + 1);
                if v >= 0.5
                    text(m, l, sprintf('%.1f', v), 'HorizontalAlignment', 'center', ...
                         'FontSize', 6.5, 'Color', [0 0 0] + (v > 0.55 * cmax) * [1 1 1]);
                end
            end
        end
        title({sprintf('(%s) %s, %s', char('a' + (typ - 1) * 2 + d - 1), dev{d, 2}{1}, nm), dev{d, 2}{2}}, ...
              'FontSize', 10, 'FontWeight', 'normal');
    end
end
colormap(flipud(hot)); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '% of radiated power';
out = fullfile(D, '..', 'farfield_sph_15um_overshoot_triangles');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150);
fprintf('saved %s.fig/.png\n', out);
