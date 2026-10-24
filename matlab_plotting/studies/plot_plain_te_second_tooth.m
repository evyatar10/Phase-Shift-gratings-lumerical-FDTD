% plot_plain_te_second_tooth.m -- second tooth (150 nm x +79 nm/side, every period) on the plain TE pi-shift grating,
% position sweep DELTA (study runners/sweeps/plain_te_second_tooth.py, Athena 267575, 2026-10-22).
% Input: summary .mat written by the session's reduction (Q_L from the cavity trace, T and tube radiation x gamma^2,
% harmonics at lmax 60, far-field intensity re-binned about the grating axis x). Row 1 = stored plain control (150 ps).
% Output (results_from_athena/plain_te_second_tooth/): _summary, _farfield_top_view, _farfield_side_view, _dPdkx, _where_radiation_crosses, _on_resonance, _harmonics_electric, _harmonics_magnetic (.png + .fig); harmonics table csv from the reduction.
clear; close all;
D = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results_from_athena', 'plain_te_second_tooth');
S = load(fullfile(D, 'plain_te_second_tooth_summary.mat'));
names = cellstr(S.names); n = numel(names);
dlt = nan(1, n); for k = 2:n, dlt(k) = sscanf(names{k}, 'Delta %d'); end
sub = 'Plain TE \pi-shift grating (corr 316 / \Lambda 500 / W800 / N96) + 2nd tooth 150 nm \times 79 nm/side in every period';

% ---- 1. scalar observables vs DELTA (control = dashed line) ----
f1 = figure('Color', 'w', 'Position', [50 50 1300 800], 'Visible', 'off');
tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
Y = {S.lam_nm, '\lambda_{res} (nm)'; S.T, 'peak T (port)'; S.Q_L, 'Q_L (\lambda / FWHM)';
     S.Q_rad, 'Q_{rad} (tube)'; S.W_um, 'mode FWHM (\mum)'; 100 * S.rad ./ S.loss, 'radiated / total loss (%)'};
for k = 1:6
    ax = nexttile; y = Y{k, 1};
    plot(ax, dlt(2:end), y(2:end), 'o-', 'LineWidth', 1.5, 'MarkerFaceColor', 'auto'); hold(ax, 'on');
    yline(ax, y(1), '--k', 'plain', 'LabelHorizontalAlignment', 'left');
    xline(ax, [0 125 250], ':', {'N centre', 'N\rightarrowW step', 'W centre'}, 'LabelVerticalAlignment', 'bottom');
    xlabel(ax, '\Delta (nm) from the narrow-half centre'); ylabel(ax, Y{k, 2}); grid(ax, 'on'); xlim(ax, [-225 275]);
end
title(tl, {sub, '100-ps runs, >= 8 decay times (steady state): Q_L = \lambda/FWHM, port T, tube radiation'});
exportgraphics(f1, fullfile(D, 'plain_te_second_tooth_summary.png'), 'Resolution', 150); savefig(f1, fullfile(D, 'plain_te_second_tooth_summary.fig'));

% ---- 2. far-field maps, same style as plot_farfield_sph_20um R1 (v7): top view (+z) and side view (+y), dB, peak = 0 per device ----
views = {'map_top', 'u_y', 'Top view (looking down, +z)', 'farfield_top_view'; 'map_side', 'u_z', 'Side view (+y)', 'farfield_side_view'};
for v = 1:2
    f2 = figure('Color', 'w', 'Position', [40 40 1700 1300], 'Visible', 'off');
    tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:n
        ax = nexttile; M = squeeze(S.(views{v, 1})(k, :, :));
        imagesc(ax, S.u_map, S.u_map, 10 * log10(M.')); set(ax, 'YDir', 'normal'); axis(ax, 'image'); clim(ax, [-30 0]);
        xlabel(ax, 'u_x (along the waveguide)'); ylabel(ax, views{v, 2});
        if k == 1, nm = 'TE plain (corr 316 nm), no 2nd tooth'; else, nm = sprintf('2nd tooth at \\Delta = %+d nm', dlt(k)); end
        title(ax, {nm, sprintf('N = 96, \\lambda = %.2f nm, T = %.3f, mode width %.1f \\mum', S.lam_nm(k), S.T(k), S.W_um(k)), views{v, 3}}, 'FontSize', 9);
    end
    colormap(f2, turbo); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = 'far-field intensity (dB, peak = 0)';
    title(tl, {'Where the radiated light goes (far field at resonance, closed-tube projection)', sub}, 'FontSize', 12);
    o = fullfile(D, ['plain_te_second_tooth_' views{v, 4}]);
    exportgraphics(f2, [o '.png'], 'Resolution', 150); savefig(f2, [o '.fig']);
end

% ---- 3. 1D far field, proposal-figure style (proposal_fig_combined_dPdk panel b): lateral (+y half space) dP/dk_x per
%      stored energy, Gaussian-averaged over k_x (sigma 0.03 k_clad), 0 dB = the plain device's averaged peak ----
f3 = figure('Color', 'w', 'Position', [50 50 1400 700], 'Visible', 'off'); ax = axes(f3); hold(ax, 'on');
ref = max(S.dPdk_lat(1, :)); cm = turbo(n + 1);
patch(ax, [-1.05 -1 -1 -1.05], [-60 -60 10 10], [0.9 0.9 0.9], 'EdgeColor', 'none', 'HandleVisibility', 'off');
patch(ax, [1 1.05 1.05 1], [-60 -60 10 10], [0.9 0.9 0.9], 'EdgeColor', 'none', 'HandleVisibility', 'off');
plot(ax, S.ux, 10 * log10(S.dPdk_lat(1, :) / ref), 'k-', 'LineWidth', 2.5, 'DisplayName', 'plain (no 2nd tooth)');
for k = 2:n
    plot(ax, S.ux, 10 * log10(S.dPdk_lat(k, :) / ref), '-', 'Color', cm(k, :), 'LineWidth', 1.6, ...
         'DisplayName', sprintf('\\Delta %+d nm (T %.2f, Q_L %.0f)', dlt(k), S.T(k), S.Q_L(k)));
end
all_db = 10 * log10(S.dPdk_lat / ref);
xlim(ax, [-1.05 1.05]); ylim(ax, [5 * floor(min(all_db(:)) / 5) 5 * ceil(max(all_db(:)) / 5)]); grid(ax, 'on'); box(ax, 'on');
xlabel(ax, 'k_x / k_{clad}'); ylabel(ax, 'dP/dk_x per stored energy (dB)'); legend(ax, 'Location', 'eastoutside');
title(ax, {'Lateral far-field radiation (side, +y), 0 dB = plain peak', sub});
exportgraphics(f3, fullfile(D, 'plain_te_second_tooth_dPdkx.png'), 'Resolution', 150); savefig(f3, fullfile(D, 'plain_te_second_tooth_dPdkx.fig'));

% ---- 3b. 1D results: where the radiation crosses the tube along x (top +z and side +y faces) vs the mode |E|^2 envelope ----
f5 = figure('Color', 'w', 'Position', [40 40 1700 1100], 'Visible', 'off');
tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:n
    ax = nexttile; hold(ax, 'on');
    plot(ax, S.x_um, S.flux_top(k, :), '-', 'LineWidth', 1.2, 'DisplayName', 'top face (+z)');
    plot(ax, S.x_um, S.flux_side(k, :), '-', 'LineWidth', 1.2, 'DisplayName', 'side face (+y)');
    plot(ax, S.x_um, S.env(k, :), 'r--', 'LineWidth', 1, 'DisplayName', 'mode energy envelope');
    xline(ax, [-40 40], ':k', 'HandleVisibility', 'off'); xlim(ax, [-50 50]); ylim(ax, [-0.05 1.05]); grid(ax, 'on');
    if k == 1, nm = 'TE plain (corr 316 nm)'; else, nm = sprintf('2nd tooth at \\Delta = %+d nm', dlt(k)); end
    title(ax, {nm, sprintf('\\lambda %.2f nm, T %.3f, mode %.1f \\mum', S.lam_nm(k), S.T(k), S.W_um(k))}, 'FontSize', 9);
    xlabel(ax, 'x along the tube (\mum)'); ylabel(ax, 'flux density (norm)');
    if k == 1, legend(ax, 'Location', 'northeast', 'FontSize', 7); end
end
title(tl, {'Where the radiation crosses the far-field tube along x (Poynting flux, dotted = tube ends)', sub}, 'FontSize', 12);
exportgraphics(f5, fullfile(D, 'plain_te_second_tooth_where_radiation_crosses.png'), 'Resolution', 150);
savefig(f5, fullfile(D, 'plain_te_second_tooth_where_radiation_crosses.fig'));

% ---- 3c. 1D results: T(lambda) around each resonance, far-field projection wavelength marked ----
f6 = figure('Color', 'w', 'Position', [40 40 1700 1100], 'Visible', 'off');
tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:n
    ax = nexttile; hold(ax, 'on'); lw = S.lam_nm(k) / S.Q_L(k);
    plot(ax, S.wl_nm(k, :), S.T_spec(k, :), 'b-', 'LineWidth', 1.5);
    xline(ax, S.lam_nm(k), 'k--', sprintf('res %.3f', S.lam_nm(k)), 'LabelVerticalAlignment', 'top');
    xline(ax, S.lam_ff_nm(k), 'r-', sprintf('far field %.3f', S.lam_ff_nm(k)), 'LabelVerticalAlignment', 'bottom', 'Color', 'r');
    xlim(ax, S.lam_nm(k) + [-6 6] * lw); ylim(ax, [0 1]); grid(ax, 'on'); xlabel(ax, '\lambda (nm)'); ylabel(ax, 'T (port, uncorrected)');
    if k == 1, nm = 'TE plain (corr 316 nm)'; else, nm = sprintf('2nd tooth at \\Delta = %+d nm', dlt(k)); end
    title(ax, {nm, sprintf('\\lambda %.2f nm, T %.3f, mode %.1f \\mum', S.lam_nm(k), S.T(k), S.W_um(k)), ...
               sprintf('linewidth %.3f nm, offset %.0f%% of it', lw, 100 * abs(S.lam_ff_nm(k) - S.lam_nm(k)) / lw)}, 'FontSize', 9);
end
title(tl, {'Transmission at resonance (port T, steady state)', sub}, 'FontSize', 12);
exportgraphics(f6, fullfile(D, 'plain_te_second_tooth_on_resonance.png'), 'Resolution', 150); savefig(f6, fullfile(D, 'plain_te_second_tooth_on_resonance.fig'));

% ---- 4. harmonic tables, same style as plot_farfield_sph_15um.m: rows l, columns |m| (+-m summed), cell = % of radiated
%      power, electric and magnetic separately, every cell >= 0.05 % printed; one figure per type ----
LMAX = 8; cmax = 15;
for typ = 1:2
    if typ == 1, F = S.tri_E; nmT = 'electric'; else, F = S.tri_M; nmT = 'magnetic'; end
    f4 = figure('Color', 'w', 'Position', [40 40 1800 1350], 'Visible', 'off');
    tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    for k = 1:n
        ax = nexttile; P = 100 * squeeze(F(k, :, :));
        for l = 0:LMAX, P(l + 1, l + 2:end) = nan; end
        im = imagesc(ax, 0:LMAX, 0:LMAX, P); set(im, 'AlphaData', ~isnan(P)); clim(ax, [0 cmax]);
        set(ax, 'YDir', 'normal', 'XTick', 0:LMAX, 'YTick', 0:LMAX, 'FontSize', 8, 'Color', 'w', 'TickLength', [0 0]); box(ax, 'on');
        for l = 0:LMAX
            for m = 0:l
                if P(l + 1, m + 1) >= 0.05
                    text(ax, m, l, sprintf('%.1f', P(l + 1, m + 1)), 'HorizontalAlignment', 'center', 'FontSize', 6.5);
                end
            end
        end
        if k == 1, nm = 'TE plain (corr 316 nm)'; else, nm = sprintf('2nd tooth at \\Delta = %+d nm', dlt(k)); end
        title(ax, {nm, sprintf('E %.0f%% / M %.0f%%, <l> %.1f, l<=%d holds %.0f%%', 100 * S.frac_E(k), 100 - 100 * S.frac_E(k), ...
                   S.l_mean(k), LMAX, 100 * sum(S.P_l(k, 1:LMAX + 1)))}, 'FontSize', 9);
        xlabel(ax, '|m|'); ylabel(ax, 'l');
    end
    colormap(f4, parula); cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '% of radiated power';
    title(tl, {sprintf('Spherical-harmonic content of the far field at resonance: %s multipoles', nmT), sub}, 'FontSize', 12);
    o = fullfile(D, ['plain_te_second_tooth_harmonics_' nmT]);
    exportgraphics(f4, [o '.png'], 'Resolution', 150); savefig(f4, [o '.fig']);
end
