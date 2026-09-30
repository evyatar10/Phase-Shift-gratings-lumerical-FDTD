% plot_farfield_sph_20um.m — spherical-harmonic (multipole) triangle tables of the
% far field of three 20-um-mode pi-shift gratings at N = 98 (Athena job 164893,
% 2026-09-29). Rows l, columns |m| (+-m summed, identical by y-mirror symmetry),
% cell = % of radiated power; electric (top row) and magnetic (bottom row) multipoles.
% Input: the *_multipoles.csv files written by python_tools/farfield_multipole.py.
% Study: runners/sweeps/farfield_sph_20um.py
clear; close all;
D = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
             'results_from_athena', 'farfield_sph_20um', 'results');
LMAX = 12;
dev = { ...
  'result_N98_avg_C250_Ybox6p8_Zbox6p8_ff', ...
    sprintf('TE plain, corr 250 nm, N=98\n\\lambda 1559.99 nm, T 0.912, mode 19.1 \\mum'); ...
  'result_N98_W951_Wavg1000_C494_ptw98W951to1247_ptn98W951to753_Ybox6p8_Zbox6p8_ff', ...
    sprintf('TE overshoot apodization (Nt60), N=98\n\\lambda 1559.87 nm, T 0.973, mode 19.6 \\mum'); ...
  'result_N98_TM_avg_C325_Ybox8p0_Zbox8p8_ff', ...
    sprintf('TM plain, corr 325 nm, N=98\n\\lambda 1559.07 nm, T 0.915, mode 19.2 \\mum')};

fig = figure('Color', 'w', 'Position', [50 50 1750 1000]);
tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
cmax = 20;
for d = 1:size(dev, 1)
    T = readtable(fullfile(D, [dev{d, 1} '_multipoles.csv']));
    PE = nan(LMAX + 1); PM = nan(LMAX + 1);          % (l+1, |m|+1), NaN above the diagonal
    for l = 0:LMAX, PE(l + 1, 1:l + 1) = 0; PM(l + 1, 1:l + 1) = 0; end
    for k = 1:height(T)
        l = T.l(k); m = abs(T.m(k));
        if l <= LMAX
            PE(l + 1, m + 1) = PE(l + 1, m + 1) + 100 * T.frac_E(k);
            PM(l + 1, m + 1) = PM(l + 1, m + 1) + 100 * T.frac_M(k);
        end
    end
    restE = 100 * sum(T.frac_E(T.l > LMAX)); restM = 100 * sum(T.frac_M(T.l > LMAX));
    for typ = 1:2
        if typ == 1, P = PE; nm = 'Electric'; rest = restE; else, P = PM; nm = 'Magnetic'; rest = restM; end
        ax = nexttile((typ - 1) * 3 + d);
        im = imagesc(0:LMAX, 0:LMAX, P); set(im, 'AlphaData', ~isnan(P));
        set(ax, 'YDir', 'normal', 'XTick', 0:LMAX, 'YTick', 0:LMAX, 'FontSize', 9, ...
                'Color', 'w', 'TickLength', [0 0]); clim([0 cmax]); box on;
        xlabel('|m|'); ylabel('l');
        for l = 0:LMAX
            for m = 0:l
                v = P(l + 1, m + 1);
                if v >= 0.05
                    text(m, l, sprintf('%.1f', v), 'HorizontalAlignment', 'center', ...
                         'FontSize', 7.5, 'Color', [0 0 0] + (v > 0.55 * cmax) * [1 1 1]);
                end
            end
        end
        title(sprintf('%s\n%s multipoles: %.1f%% of power (l > %d: %.1f%%)', ...
              dev{d, 2}, nm, 100 * sum(T.(sprintf('frac_%s', nm(1)))), LMAX, rest), 'FontSize', 9.5);
    end
end
colormap(flipud(hot)); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '% of radiated power';
title(tl, 'Far-field vector spherical-harmonic content at resonance (% of radiated power per (l, |m|), \pm m summed)', 'FontSize', 11);

out = fullfile(D, '..', 'farfield_sph_20um_triangles');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150);
fprintf('saved %s.fig/.png\n', out);

% ── Figure 2: the overshoot device needs a taller triangle (49% of its power above l = 12)
LMAX2 = 24; d = 2;
T = readtable(fullfile(D, [dev{d, 1} '_multipoles.csv']));
fig2 = figure('Color', 'w', 'Position', [50 50 1700 800]);
tl2 = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for typ = 1:2
    if typ == 1, col = T.frac_E; nm = 'Electric'; else, col = T.frac_M; nm = 'Magnetic'; end
    P = nan(LMAX2 + 1); for l = 0:LMAX2, P(l + 1, 1:l + 1) = 0; end
    for k = 1:height(T)
        if T.l(k) <= LMAX2, P(T.l(k) + 1, abs(T.m(k)) + 1) = P(T.l(k) + 1, abs(T.m(k)) + 1) + 100 * col(k); end
    end
    ax = nexttile; im = imagesc(0:LMAX2, 0:LMAX2, P); set(im, 'AlphaData', ~isnan(P));
    set(ax, 'YDir', 'normal', 'XTick', 0:2:LMAX2, 'YTick', 0:2:LMAX2, 'FontSize', 9, 'TickLength', [0 0]);
    clim([0 6]); box on; xlabel('|m|'); ylabel('l');
    for l = 0:LMAX2, for m = 0:l
        v = P(l + 1, m + 1);
        if v >= 0.3, text(m, l, sprintf('%.1f', v), 'HorizontalAlignment', 'center', 'FontSize', 6.5, 'Color', [0 0 0] + (v > 3.5) * [1 1 1]); end
    end, end
    title(sprintf('%s\n%s multipoles: %.1f%% of power (l > %d: %.1f%%)', dev{d, 2}, nm, 100 * sum(col), LMAX2, 100 * sum(col(T.l > LMAX2))), 'FontSize', 9.5);
end
colormap(flipud(hot)); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '% of radiated power';
title(tl2, 'TE overshoot device: far-field spherical-harmonic content to l = 24 (\pm m summed)', 'FontSize', 11);
out2 = fullfile(D, '..', 'farfield_sph_20um_overshoot_l24');
savefig(fig2, [out2 '.fig']); exportgraphics(fig2, [out2 '.png'], 'Resolution', 150);

% ── Figure 3: T(lambda) of each device with the wavelength the far field was projected at
fig3 = figure('Color', 'w', 'Position', [50 50 1500 420]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for d = 1:3
    S = load(fullfile(D, [dev{d, 1} '.mat']));
    nexttile; plot(S.wl_nm, S.T, 'b', 'LineWidth', 1.2); hold on;
    lam_ff = 1e9 * S.farfield_top.lam;
    xline(S.resonance_wavelength_nm, 'k--', sprintf('res %.3f', S.resonance_wavelength_nm), 'LabelOrientation', 'horizontal');
    xline(lam_ff, 'r-', sprintf('far field %.3f', lam_ff), 'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom');
    fw = abs(S.spectral_fwhm_nm); xlim(S.resonance_wavelength_nm + 3 * fw * [-1 1]); ylim([0 1]);
    xlabel('\lambda (nm)'); ylabel('T'); grid on;
    title(sprintf('%s\nlinewidth %.3f nm, offset %.0f%% of it', dev{d, 2}, fw, 100 * abs(lam_ff - S.resonance_wavelength_nm) / fw), 'FontSize', 9.5);
end
out3 = fullfile(D, '..', 'farfield_sph_20um_on_resonance');
savefig(fig3, [out3 '.fig']); exportgraphics(fig3, [out3 '.png'], 'Resolution', 150);
fprintf('saved %s and %s\n', out2, out3);

% ── Figures 4-6: ONE Wikipedia-style triangle per device: apex l = 0 at the top,
% columns m = -l..l (both sides shown), electric + magnetic summed, log colour scale.
LMAX3 = 24; short = {'te_plain', 'te_overshoot', 'tm_plain'};
vmin = 0.05; vmax = 20;                                   % colour range, % (log10)
cmap = parula(256);
for d = 1:3
    T = readtable(fullfile(D, [dev{d, 1} '_multipoles.csv']));
    V = zeros(LMAX3 + 1, 2 * LMAX3 + 1);                 % (l+1, m+LMAX3+1)
    for k = 1:height(T)
        if T.l(k) <= LMAX3
            V(T.l(k) + 1, T.m(k) + LMAX3 + 1) = V(T.l(k) + 1, T.m(k) + LMAX3 + 1) + 100 * (T.frac_E(k) + T.frac_M(k));
        end
    end
    rest = 100 * sum(T.frac_E(T.l > LMAX3) + T.frac_M(T.l > LMAX3));
    f4 = figure('Color', 'w', 'Position', [50 50 2200 1150]);
    ax = axes(f4); hold(ax, 'on');
    for l = 0:LMAX3
        for m = -l:l
            v = V(l + 1, m + LMAX3 + 1);
            if v < vmin
                c = [1 1 1];
            else
                c = cmap(1 + round(255 * (log10(v) - log10(vmin)) / (log10(vmax) - log10(vmin))), :);
            end
            rectangle(ax, 'Position', [m - 0.5, -l - 0.5, 1, 1], 'FaceColor', c, 'EdgeColor', [0.85 0.85 0.85]);
            if v >= 0.3
                text(ax, m, -l, sprintf('%.1f', v), 'HorizontalAlignment', 'center', 'FontSize', 6, ...
                     'Color', [0 0 0] + (v > 6) * [1 1 1]);
            end
        end
    end
    axis(ax, 'equal'); xlim(ax, [-LMAX3 - 0.6, LMAX3 + 0.6]); ylim(ax, [-LMAX3 - 0.6, 0.6]);
    set(ax, 'XTick', -LMAX3:2:LMAX3, 'YTick', -LMAX3:0, 'YTickLabel', LMAX3:-1:0, 'FontSize', 9, 'TickLength', [0 0]);
    xlabel(ax, 'm'); ylabel(ax, 'l'); box(ax, 'on');
    colormap(ax, cmap); cb = colorbar(ax); clim(ax, [log10(vmin) log10(vmax)]);
    cb.Ticks = log10([0.05 0.1 0.2 0.5 1 2 5 10 20]); cb.TickLabels = {'0.05', '0.1', '0.2', '0.5', '1', '2', '5', '10', '20'};
    cb.Label.String = '% of radiated power (log scale; white < 0.05%)';
    title(ax, {dev{d, 2}; sprintf('Far-field spherical-harmonic content, E + M summed, l > %d: %.1f%%', LMAX3, rest)}, 'FontSize', 10);
    o = fullfile(D, '..', ['farfield_sph_20um_triangle_' short{d}]);
    savefig(f4, [o '.fig']); exportgraphics(f4, [o '.png'], 'Resolution', 150); fprintf('saved %s\n', o);
end
