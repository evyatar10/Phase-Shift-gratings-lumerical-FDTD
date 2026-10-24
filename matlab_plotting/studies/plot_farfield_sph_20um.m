% plot_farfield_sph_20um.m — spherical-harmonic (multipole) triangle tables of the
% far field of three 20-um-mode pi-shift gratings at N = 98 (Athena job 164893,
% 2026-09-29). Rows l, columns |m| (+-m summed, identical by y-mirror symmetry),
% cell = % of radiated power; electric (top row) and magnetic (bottom row) multipoles.
% Input: the *_multipoles.csv files written by python_tools/farfield_multipole.py.
% Study: runners/sweeps/farfield_sph_20um.py
clear; close all;
% 2026-10-10: input = the CORRECTED tube-projection far fields (IGUM 100034); figures land in
% results_from_igum/farfield_sph_20um/. Use 'results_from_athena' for the original 164893 planar version.
D = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
             'results_from_igum', 'farfield_sph_20um', 'results');
LMAX = 12;
dev = { ...
  'result_N98_avg_C250_Ybox6p8_Zbox6p8_ff', 'TE, plain grating'; ...
  'result_N98_W951_Wavg1000_C494_ptw98W951to1247_ptn98W951to753_Ybox6p8_Zbox6p8_ff', 'TE, overshoot apodization'; ...
  'result_N98_TM_avg_C325_Ybox8p0_Zbox8p8_ff', 'TM, plain grating'};
% 2026-10-10: article style titles (panel letter + device); lambda, T and mode width live in the captions.

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
        xlabel('|m|'); ylabel('l', 'Rotation', 0);
        for l = 0:LMAX
            for m = 0:l
                v = P(l + 1, m + 1);
                if v >= 0.05
                    text(m, l, sprintf('%.1f', v), 'HorizontalAlignment', 'center', ...
                         'FontSize', 7.5, 'Color', [0 0 0] + (v > 0.55 * cmax) * [1 1 1]);
                end
            end
        end
        title(sprintf('(%s) %s, %s', char('a' + (typ - 1) * 3 + d - 1), dev{d, 2}, lower(nm)), ...
              'FontSize', 11, 'FontWeight', 'normal');
    end
end
colormap(flipud(hot)); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '% of radiated power';

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
    clim([0 6]); box on; xlabel('|m|'); ylabel('l', 'Rotation', 0);
    for l = 0:LMAX2, for m = 0:l
        v = P(l + 1, m + 1);
        if v >= 0.3, text(m, l, sprintf('%.1f', v), 'HorizontalAlignment', 'center', 'FontSize', 6.5, 'Color', [0 0 0] + (v > 3.5) * [1 1 1]); end
    end, end
    title(sprintf('(%s) %s, %s', char('a' + typ - 1), dev{d, 2}, lower(nm)), 'FontSize', 11, 'FontWeight', 'normal');
end
colormap(flipud(hot)); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '% of radiated power';
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
    title(sprintf('(%s) %s', char('a' + d - 1), dev{d, 2}), 'FontSize', 11, 'FontWeight', 'normal');
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
    title(ax, sprintf('%s, electric + magnetic', dev{d, 2}), 'FontSize', 12, 'FontWeight', 'normal');
    o = fullfile(D, '..', ['farfield_sph_20um_triangle_' short{d}]);
    savefig(f4, [o '.fig']); exportgraphics(f4, [o '.png'], 'Resolution', 150); fprintf('saved %s\n', o);
end

%% ── Report figures (v7 model note, 2026-10-05): CORRECTED far field (tube projection, IGUM 100034)
% Input: results_from_igum/farfield_sph_20um/report_data.mat (prepared from the *_ff.mat files with
% python_tools/farfield_surface.py + farfield_multipole.py). Output: farfield_v7_*.fig/.png next to it.
DR = fullfile(fileparts(fileparts(fileparts(D))), 'results_from_igum', 'farfield_sph_20um');
RD = load(fullfile(DR, 'report_data.mat'));
tags = {'A', 'B', 'C'};
nm3 = {'TE plain (corr 250 nm)', 'TE overshoot apodization', 'TM plain (corr 325 nm)'};
col3 = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.47 0.67 0.19];
lab = @(S, d) {nm3{d}, sprintf('N = 98, \\lambda = %.2f nm, T = %.3f, mode width %.1f \\mum', S.lam_nm, S.T, S.fwhm_um)};

% R1: where the light goes -- far-field intensity maps, top (+z) and side (+y) views, dB
f = figure('Color', 'w', 'Position', [40 40 1500 900]);
tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for v = 1:2
    for d = 1:3
        S = RD.(tags{d}); u = S.u_map;
        if v == 1, M = S.map_top; yl = 'u_y'; vn = 'Top view (looking down, +z)'; else, M = S.map_side; yl = 'u_z'; vn = 'Side view (+y)'; end
        ax = nexttile; imagesc(u, u, 10 * log10(M.')); set(ax, 'YDir', 'normal'); axis image; clim([-30 0]);
        xlabel('u_x (along the waveguide)'); ylabel(yl);
        title([lab(S, d), {vn}], 'FontSize', 9);
    end
end
colormap(turbo); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = 'far-field intensity (dB, peak = 0)';
title(tl, 'Where the radiated light goes (corrected far field at resonance)', 'FontSize', 12);
save2(f, DR, 'farfield_v7_maps');

% R2: how much power in each spherical-harmonic order l, electric + magnetic stacked, with cumulative share
f = figure('Color', 'w', 'Position', [40 40 1500 460]);
tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for d = 1:3
    S = RD.(tags{d}); l = 0:numel(S.PE_l) - 1; L = 25;
    ax = nexttile; b = bar(l(1:L + 1), [S.PE_l(1:L + 1); S.PM_l(1:L + 1)].', 'stacked', 'BarWidth', 0.85);
    b(1).FaceColor = [0.20 0.40 0.80]; b(2).FaceColor = [0.90 0.55 0.15];
    ylabel('% of radiated power'); xlabel('order l'); xlim([-0.6 L + 0.6]); grid on;
    yyaxis right; plot(l(1:L + 1), cumsum(S.PE_l(1:L + 1) + S.PM_l(1:L + 1)), 'k-o', 'MarkerSize', 3, 'LineWidth', 1.1);
    ylim([0 100]); ylabel('cumulative %'); ax.YAxis(2).Color = 'k';
    legend({'electric', 'magnetic', 'cumulative'}, 'Location', 'east', 'FontSize', 8);
    title([lab(S, d), {sprintf('l \\leq 5: %.0f%%,  mean l = %.1f', sum(S.PE_l(1:6) + S.PM_l(1:6)), sum(l .* (S.PE_l + S.PM_l)) / 100)}], 'FontSize', 9);
end
title(tl, 'Spherical-harmonic content of the radiation, by order l (origin at the cavity centre)', 'FontSize', 12);
save2(f, DR, 'farfield_v7_by_order');

% R3: the full (l, |m|) tables, E + M summed, l <= 12
f = figure('Color', 'w', 'Position', [40 40 1600 560]);
tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for d = 1:3
    S = RD.(tags{d}); P = S.triE + S.triM; P(triu(true(13), 1)) = NaN;
    ax = nexttile; im = imagesc(0:12, 0:12, P); set(im, 'AlphaData', ~isnan(P));
    set(ax, 'YDir', 'normal', 'XTick', 0:12, 'YTick', 0:12, 'FontSize', 8, 'TickLength', [0 0]); clim([0 20]); box on;
    for l = 0:12, for m = 0:l
        if P(l + 1, m + 1) >= 0.5
            text(m, l, sprintf('%.1f', P(l + 1, m + 1)), 'HorizontalAlignment', 'center', 'FontSize', 7, ...
                 'Color', [0 0 0] + (P(l + 1, m + 1) > 11) * [1 1 1]);
        end
    end, end
    xlabel('|m| (\pm m summed)'); ylabel('order l');
    title([lab(S, d), {sprintf('shown: %.0f%% of the power (l \\leq 12)', sum(P(:), 'omitnan'))}], 'FontSize', 9);
end
colormap(flipud(hot)); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '% of radiated power';
title(tl, 'Power in each spherical harmonic (l, |m|), electric + magnetic', 'FontSize', 12);
save2(f, DR, 'farfield_v7_lm_tables');

% R4: two facts behind "cancel one harmonic does not work": the table moves with the origin, and
% the leak comes from a compact spot (TE plain) vs a spread region (overshoot)
f = figure('Color', 'w', 'Position', [40 40 1500 470]);
tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
S = RD.A; sh = {'shift_m190', 'shift_p000', 'shift_p190'}; shl = {'origin -0.19 \mum', 'origin 0', 'origin +0.19 \mum'};
Y = zeros(8, 3); for k = 1:3, Y(:, k) = S.(sh{k})(2:9); end
nexttile; bar(1:8, Y, 'BarWidth', 0.85); grid on; xlabel('order l'); ylabel('% of radiated power');
legend(shl, 'Location', 'northeast'); title([lab(S, 1), {'Same field, origin moved by less than half a pitch'}], 'FontSize', 9);
nexttile; hold on;
for d = 1:3
    S = RD.(tags{d}); plot(S.reach_L, 100 * S.reach, '-o', 'Color', col3(d, :), 'LineWidth', 1.5, 'MarkerFaceColor', col3(d, :));
end
set(gca, 'XScale', 'log'); grid on; ylim([0 100]); xlim([0.2 12]);
xlabel('half-width of the source region around the \pi shift (\mum)'); ylabel('% of the leak pattern reproduced');
legend(nm3, 'Location', 'southeast'); title({'Where the leak comes from (frozen-field fit, a ceiling)'; 'TE plain: 83% from \pm0.5 \mum, i.e. the junction'}, 'FontSize', 9);
title(tl, 'Why one-harmonic cancellation fails, and what the leak is instead', 'FontSize', 12);
save2(f, DR, 'farfield_v7_origin_and_source');

% R5: the measurement error and the energy budget
f = figure('Color', 'w', 'Position', [40 40 1500 450]);
tl = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
A = RD.aperture; nexttile; hold on;
plot(A.angle_deg, A.box68, '-o', 'LineWidth', 1.6); plot(A.angle_deg, A.box12, '-s', 'LineWidth', 1.6);
yline(1, 'k--'); xline(45, ':', 'seam between the two monitors', 'LabelOrientation', 'horizontal');
grid on; ylim([0 1.3]); xlabel('angle from the monitor normal (deg)'); ylabel('projected / exact power');
legend({'monitor 2.15 \mum from axis, \pm3.4 \mum wide (used)', 'monitor 5.15 \mum, \pm6 \mum (bigger box)'}, 'Location', 'southwest', 'FontSize', 8);
title({'Old method: one flat monitor, exact dipole source'; 'right only within ~40\circ of its normal'}, 'FontSize', 9);
nexttile; Bv = zeros(3, 2); for d = 1:3, Bv(d, :) = [RD.(tags{d}).loss_tube, RD.(tags{d}).loss_ports - RD.(tags{d}).loss_tube]; end
b = bar(categorical(nm3, nm3), Bv, 'stacked'); b(1).FaceColor = [0.30 0.60 0.85]; b(2).FaceColor = [0.80 0.80 0.80];
ylabel('% of input power'); grid on; ylim([0 11]);
legend({'radiated by the cavity (through the surface)', 'rest of 1-T-R (likely the grating ends)'}, 'Location', 'northeast', 'FontSize', 8);
title({'Loss budget at resonance (corrected method)'; 'part of the port loss is not cavity radiation'}, 'FontSize', 9);
title(tl, 'The far-field measurement: what was wrong, and what the corrected one shows', 'FontSize', 12);
save2(f, DR, 'farfield_v7_method_and_budget');
fprintf('saved report figures in %s\n', DR);

%% ── Cancellation deck figures (2026-10-10), article style: panel letter + device, details in captions.
% Input: *_multipoles.csv (tube projection, IGUM 100034) and the TE plain files written with
% farfield_multipole.py --origin-x (expansion centre moved along the grating). Output: cancel_*.fig/.png in DR.
% A channel = (type, l, |m|): the y mirror ties +m and -m, so each channel is one complex number.
FS = 11; cE = [0.00 0.45 0.74]; cM = [0.85 0.33 0.10]; cG = [0.47 0.67 0.19];
devc = {dev{1, 1}, dev{2, 1}, dev{3, 1}}; dname = {dev{1, 2}, dev{2, 2}, dev{3, 2}};
xo = [-0.19 -0.125 0 0.125 0.19];                        % TE plain expansion centres (um); 0 = pi shift
csvA = @(x0) fullfile(D, [devc{1} iif(x0 == 0, '_multipoles.csv', sprintf('_multipoles_x%+.3fum.csv', x0))]);

% C1: power per order l, electric and magnetic stacked, cumulative share on the right axis
f = figure('Color', 'w', 'Position', [40 40 1500 420]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact'); LM = [10 30 20];
for d = 1:3
    T = readtable(fullfile(D, [devc{d} '_multipoles.csv'])); L = LM(d);
    PE = accumarray(T.l + 1, T.frac_E); PM = accumarray(T.l + 1, T.frac_M);
    ax = nexttile; b = bar(0:L, 100 * [PE(1:L + 1) PM(1:L + 1)], 'stacked', 'BarWidth', 0.8, 'EdgeColor', 'none');
    b(1).FaceColor = cE; b(2).FaceColor = cM; xlim([0.3 L + 0.7]);
    xlabel('multipole order l'); ylabel('radiated power (%)');
    cum = 100 * cumsum(PE(1:L + 1) + PM(1:L + 1));
    yyaxis right; hc = plot(1:L, cum(2:end), 'k-o', 'MarkerSize', 3, 'LineWidth', 1.1);
    ylim([0 100]); ylabel('cumulative (%)'); ax.YAxis(2).Color = 'k'; yyaxis left; ax.YAxis(1).Color = 'k';
    set(ax, 'FontSize', FS, 'TickDir', 'out', 'Box', 'off');
    title(sprintf('(%s) %s', char('a' + d - 1), dname{d}), 'FontWeight', 'normal');
    if d == 1, legend([b(1) b(2) hc], {'electric', 'magnetic', 'cumulative'}, 'Location', 'east', 'Box', 'off'); end
end
save2(f, DR, 'cancel_order_spectra');

% C2: power per channel (l, |m|), electric (beta_lm) and magnetic (alpha_lm) side by side, one figure per device
short = {'te_plain', 'te_overshoot', 'tm_plain'}; LC = [7 12 10]; CM = [20 20 20];    % one colour scale for all three devices (user, 2026-10-11)
for d = 1:3
    f = figure('Color', 'w', 'Position', [40 40 900 400]);    % small canvas = large text on a slide
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    T = readtable(fullfile(D, [devc{d} '_multipoles.csv'])); L = LC(d); fsz = 12 - 3 * (L > 7);
    for typ = 1:2
        col = T.frac_E; nm = 'electric (\beta_{lm})'; if typ == 2, col = T.frac_M; nm = 'magnetic (\alpha_{lm})'; end
        P = nan(L + 1); for l = 1:L, P(l + 1, 1:l + 1) = 0; end
        for k = 1:height(T)
            if T.l(k) <= L, P(T.l(k) + 1, abs(T.m(k)) + 1) = P(T.l(k) + 1, abs(T.m(k)) + 1) + 100 * col(k); end
        end
        ax = nexttile; im = imagesc(0:L, 0:L, P); set(im, 'AlphaData', ~isnan(P)); clim([0 CM(d)]);
        set(ax, 'YDir', 'normal', 'XTick', 0:L, 'YTick', 1:L, 'FontSize', 13, 'TickLength', [0 0], 'XTickLabelRotation', 0); ylim([0.5 L + 0.5]);
        for l = 1:L, for m = 0:l
            v = P(l + 1, m + 1);
            if v >= 0.3 + 0.2 * (L > 7), text(m, l, sprintf('%.1f', v), 'HorizontalAlignment', 'center', 'FontSize', fsz, 'Color', [0 0 0] + (v > 0.55 * CM(d)) * [1 1 1]); end
        end, end
        xlabel('|m|'); ylabel('l', 'Rotation', 0); title(sprintf('(%s) %s', char('a' + typ - 1), nm), 'FontWeight', 'normal');
    end
    colormap(flipud(hot)); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = 'radiated power (%)';
    save2(f, DR, ['cancel_channels_' short{d}]);
    fprintf('%s: shown l <= %d holds %.1f%% (E %.1f, M %.1f)\n', short{d}, L, 100 * sum(T.frac_E(T.l <= L) + T.frac_M(T.l <= L)), 100 * sum(T.frac_E), 100 * sum(T.frac_M));
end

% C3: cancellation ceiling: loss left after the K largest channels are removed exactly
f = figure('Color', 'w', 'Position', [40 40 760 520]); ax = axes(f); hold(ax, 'on'); K = 1:30;
R = zeros(numel(xo), numel(K));
for j = 1:numel(xo), r = 1 - cumsum(chshare(readtable(csvA(xo(j))))); R(j, :) = 100 * r(K); end
fill([K fliplr(K)], [min(R) fliplr(max(R))], cE, 'FaceAlpha', 0.15, 'EdgeColor', 'none');
h(1) = plot(K, R(xo == 0, :), '-o', 'Color', cE, 'LineWidth', 1.6, 'MarkerFaceColor', cE, 'MarkerSize', 4);
r = 1 - cumsum(chshare(readtable(fullfile(D, [devc{2} '_multipoles.csv'])))); h(2) = plot(K, 100 * r(K), '-s', 'Color', cM, 'LineWidth', 1.6, 'MarkerSize', 4);
r = 1 - cumsum(chshare(readtable(fullfile(D, [devc{3} '_multipoles.csv'])))); h(3) = plot(K, 100 * r(K), '-^', 'Color', cG, 'LineWidth', 1.6, 'MarkerSize', 4);
set(ax, 'YScale', 'log', 'FontSize', FS, 'TickDir', 'out', 'Box', 'off', 'XTick', [1 5:5:30]); ylim([1 100]); xlim([0.5 30.5]);
set(ax, 'YTick', [1 2 5 10 20 50 100]); grid(ax, 'on'); ax.GridAlpha = 0.12; ax.MinorGridAlpha = 0.05;
xlabel('channels cancelled, K (real knobs needed: 2K + 2)'); ylabel('radiated power left (%)');
legend(h, {'TE, plain grating (band: expansion centre within \pm0.19 \mum)', dname{2}, dname{3}}, 'Location', 'southwest', 'Box', 'off');
save2(f, DR, 'cancel_ceiling');

% C4: TE plain, share per order l for five expansion centres
f = figure('Color', 'w', 'Position', [40 40 820 480]); Y = zeros(7, numel(xo));
for j = 1:numel(xo)
    T = readtable(csvA(xo(j))); Pl = accumarray(T.l + 1, T.frac_E + T.frac_M); Y(:, j) = 100 * Pl(2:8);
end
b = bar(1:7, Y, 'BarWidth', 0.85, 'EdgeColor', 'none'); cm = flipud(parula(numel(xo) + 1));
for j = 1:numel(xo), b(j).FaceColor = cm(j, :); end
set(gca, 'FontSize', FS, 'TickDir', 'out', 'Box', 'off'); xlabel('multipole order l'); ylabel('radiated power (%)');
legend(arrayfun(@(v) strrep(sprintf('x_0 = %+.3g \\mum', v), '+0 ', '0 '), xo, 'UniformOutput', false), 'Location', 'northeast', 'Box', 'off');
save2(f, DR, 'cancel_origin_dependence');

% C5: one knob of strength q against the leak: P/P_c = 1 - eta^2 + eta^2 (q/q_* - 1)^2
f = figure('Color', 'w', 'Position', [40 40 760 480]); ax = axes(f); hold(ax, 'on'); q = linspace(-0.5, 2.5, 301);
patch([2 2.5 2.5 2], [1 1 2.6 2.6], [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
patch([-0.5 0 0 -0.5], [1 1 2.6 2.6], [0.85 0.85 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
eta2 = [0.25 0.5 0.9]; cc = [cG; cM; cE]; hh = gobjects(1, 3);
for j = 1:3, hh(j) = plot(q, 1 - eta2(j) + eta2(j) * (q - 1).^2, 'Color', cc(j, :), 'LineWidth', 1.8); end
yline(1, 'k:'); xline([0 1 2], 'k:');
set(ax, 'FontSize', FS, 'TickDir', 'out', 'Box', 'off', 'XTick', [0 1 2], 'XTickLabel', {'0', 'q_*', '2q_*'}); ylim([0 2.6]); xlim([-0.5 2.5]);
xlabel('knob strength q'); ylabel('radiated power P / P_c');
legend(hh, arrayfun(@(e) sprintf('\\eta^2 = %.2g', e), eta2, 'UniformOutput', false), 'Location', 'north', 'Box', 'off');
text(2.25, 2.4, 'worse', 'HorizontalAlignment', 'center', 'FontSize', FS); text(-0.25, 2.4, 'worse', 'HorizontalAlignment', 'center', 'FontSize', FS);
save2(f, DR, 'cancel_single_knob');

% C6: radiated power per solid angle (far-field Poynting flux), seen from above (+z) and from the side (+y),
% dB re peak; tube projection (report_data.mat). White outside the unit disk, tick labels on the outer panels only.
f = figure('Color', 'w', 'Position', [40 40 1000 650]);    % small canvas = readable labels on a slide
tiledlayout(2, 3, 'TileSpacing', 'tight', 'Padding', 'compact');
sn = {'TE plain', 'TE overshoot', 'TM plain'}; vw = {'from above', 'from the side'}; yl = {'u_y', 'u_z'};
th = linspace(0, 2 * pi, 361);
for v = 1:2
    for d = 1:3
        S = RD.(tags{d}); u = S.u_map; M = S.map_top; if v == 2, M = S.map_side; end
        [UX, UY] = meshgrid(u, u); M = max(10 * log10(M.'), -30); M(UX.^2 + UY.^2 > 1) = NaN;
        ax = nexttile; hold(ax, 'on');
        im = imagesc(ax, u, u, M); set(im, 'AlphaData', ~isnan(M)); plot(ax, cos(th), sin(th), 'k-', 'LineWidth', 0.6);
        axis(ax, 'image'); xlim(ax, [-1.02 1.02]); ylim(ax, [-1.02 1.02]); clim(ax, [-30 0]);
        set(ax, 'FontSize', 12, 'TickDir', 'out', 'XTick', -1:0.5:1, 'YTick', -1:0.5:1, 'Box', 'off', 'Color', 'w');
        if v == 2, xlabel(ax, 'u_x'); else, set(ax, 'XTickLabel', []); end
        if d == 1, ylabel(ax, yl{v}); else, set(ax, 'YTickLabel', []); end
        title(ax, sprintf('(%s) %s, %s', char('a' + (v - 1) * 3 + d - 1), sn{d}, vw{v}), 'FontWeight', 'normal', 'FontSize', 12);
    end
end
colormap(turbo); cb = colorbar; cb.Layout.Tile = 'east'; cb.Ticks = -30:10:0; cb.FontSize = 12;
cb.Label.String = 'radiated power per solid angle (dB)';
save2(f, DR, 'cancel_farfield_maps');
fprintf('saved cancellation figures in %s\n', DR);

function s = chshare(T)
% channel shares (type, l, |m|) of one multipole csv, sorted large to small
key = [T.l * 1000 + abs(T.m); 1e6 + T.l * 1000 + abs(T.m)];
[~, ~, g] = unique(key); s = sort(accumarray(g, [T.frac_E; T.frac_M]), 'descend'); s = s / sum(s);
end

function out = iif(c, a, b)
if c, out = a; else, out = b; end
end

function save2(f, folder, name)
savefig(f, fullfile(folder, [name '.fig']));
exportgraphics(f, fullfile(folder, [name '.png']), 'Resolution', 170);
end
