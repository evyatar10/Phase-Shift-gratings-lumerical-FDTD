% plot_trench_n150_maps.m — N=150 full-device |E|^2 field maps, trench vs control.
% Study dir: results_from_athena/trench_n150_full | Jobs 124551 (controls) +
% 124590 (full-z trenches) | 2026-07-20; revised 2026-07-21 (naming/scale),
% split per-device + geometry overlays (this version).
% Purpose: per-device comparison figures (control vs full-z air trench) for the
% three full-size TM devices, with the TRUE device geometry drawn on top:
% white step outline of every tooth + cavity highlight (extracted from the
% BUILT scenes -> outlines_n150.mat, scratchpad export_n150_outlines.py) and
% the air trenches in bold pink with pointer arrows (top view only — the
% trenches at y=+/-1.8 um do not intersect the side view's y=0 plane).
% Views (standard convention, user-set 2026-07-21): XY = "Top view" (y
% vertical), XZ = "Side view" (z vertical). Scale: RAW |E|^2 in dB, ONE
% shared color scale per view across all six runs so the three images stay
% directly comparable. Outputs (6): trench_n150_tm_{topview,sideview}_
% outline_{w800,w1050,apod10}.{fig,png}. Older combined 6-panel versions
% remain on disk (trench_n150_tm_*.png, figs_v1/).

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
                   'results_from_athena', 'trench_n150_full', 'results');
out_dir = fullfile(res_dir, '..', 'figures');

tr_tag = '_scRECT_L156000xW800_X0_Y1800_pair_hole';

% per device: {key, display name, ctrl file, trench file, ctrl stats, trench stats}
% trench rows carry the dB value AND the gain vs the regular device
% (10*log10 of the measured T ratio: +1.01 / +1.31 / +0.69 dB)
DEV = { ...
 'w800',   'average cavity width 800 nm', ...
   'result_N150_TM_avg_Ybox8p0_Zbox8p8', ...
   'regular:  T 0.184 (-7.35 dB),  Q 14256,  FWHM 16.2 \mum', ...
   'with air trenches:  T 0.232 (-6.34 dB, +1.01 dB),  Q 17902,  FWHM 16.0 \mum'; ...
 'w1050',  'optimized cavity width 1050 nm', ...
   'result_N150_TM_W1050_Ybox8p0_Zbox8p8', ...
   'regular:  T 0.292 (-5.34 dB),  Q 18001,  FWHM 16.3 \mum', ...
   'with air trenches:  T 0.395 (-4.03 dB, +1.31 dB),  Q 23079,  FWHM 16.1 \mum'; ...
 'apod10', '10-period apodization', ...
   'result_N150_A10_TM_avg_Ybox8p0_Zbox8p8', ...
   'regular:  T 0.762 (-1.18 dB),  Q 27584,  FWHM 20.3 \mum', ...
   'with air trenches:  T 0.894 (-0.49 dB, +0.69 dB),  Q 33861,  FWHM 19.9 \mum'};

DB_RANGE  = 70;                                % dynamic range below the global peak
TRENCH    = struct('x', 78.0, 'y_in', 1.4, 'y_out', 2.2);   % um
TR_COLOR  = [1.0, 0.1, 0.9];                   % bold pink (absent from turbo)
TR_LINE_W = 2.2;
ol = load(fullfile(res_dir, '..', 'outlines_n150.mat'));

% One shared absolute dB scale per view across ALL runs (images comparable).
pk_xy = -inf; pk_xz = -inf;
for k = 1:3
    for f = {[DEV{k,3} '_planes.mat'], [DEV{k,3} tr_tag '_planes.mat']}
        d = load(fullfile(res_dir, f{1}));
        pk_xy = max(pk_xy, max(double(d.field_xy_E2(:))));
        pk_xz = max(pk_xz, max(double(d.field_xz_E2(:))));
    end
end
top_xy = 10*log10(pk_xy);
top_xz = 10*log10(pk_xz);

for k = 1:3
    key = DEV{k,1};
    files  = {[DEV{k,3} '_planes.mat'], [DEV{k,3} tr_tag '_planes.mat']};
    labels = DEV(k, 4:5);
    dx = ol.([key '_x'])  * 1e6;
    dh = ol.([key '_hw']) * 1e6;
    cav = ol.([key '_cavity']) * 1e6;

    % ── Top view (XY) ────────────────────────────────────────────────────────
    fig = figure('Visible', 'on', 'Position', [40 40 1500 640]);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for p = 1:2
        d = load(fullfile(res_dir, files{p}));
        nexttile(tl);
        imagesc(d.field_xy_x*1e6, d.field_xy_y*1e6, ...
                10*log10(max(double(d.field_xy_E2'), 1e-30)));
        axis xy; clim([top_xy - DB_RANGE, top_xy]); colormap(turbo);
        ylabel('y (\mum)');
        if p == 2, xlabel('x (\mum)'); end
        hold on;
        plot(dx,  dh, 'w-', 'LineWidth', 0.4);
        plot(dx, -dh, 'w-', 'LineWidth', 0.4);
        rectangle('Position', [cav(1), -cav(3), cav(2)-cav(1), 2*cav(3)], ...
                  'EdgeColor', 'w', 'LineWidth', 1.5);
        if p == 2      % the air trenches (bold rectangles, labeled "Air")
            for s = [1, -1]
                y_lo = min(s*TRENCH.y_in, s*TRENCH.y_out);
                rectangle('Position', [-TRENCH.x, y_lo, 2*TRENCH.x, ...
                          TRENCH.y_out-TRENCH.y_in], ...
                          'EdgeColor', TR_COLOR, 'LineWidth', TR_LINE_W);
                text(-70, s*1.8, 'Air', 'Color', TR_COLOR, 'FontSize', 13, ...
                     'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle');
            end
        end
        hold off;
        title(labels{p}, 'FontSize', 13);
        set(gca, 'FontSize', 12);
    end
    title(tl, ['TM, N=150, ' DEV{k,2} ' — Top view (XY), |E|^2 in dB, ' ...
               'resonance \lambda \approx 1558 nm'], 'FontSize', 16);
    cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '|E|^2 (dB)';
    cb.Label.FontSize = 13; cb.FontSize = 12;
    savefig(fig, fullfile(out_dir, ['trench_n150_tm_topview_outline_' key '.fig']));
    exportgraphics(fig, fullfile(out_dir, ['trench_n150_tm_topview_outline_' key '.png']), ...
                   'Resolution', 160);

    % ── Side view (XZ) ───────────────────────────────────────────────────────
    fig = figure('Visible', 'on', 'Position', [60 60 1500 640]);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for p = 1:2
        d = load(fullfile(res_dir, files{p}));
        nexttile(tl);
        imagesc(d.field_xz_x*1e6, d.field_xz_z*1e6, ...
                10*log10(max(double(d.field_xz_E2'), 1e-30)));
        axis xy; clim([top_xz - DB_RANGE, top_xz]); colormap(turbo);
        ylabel('z (\mum)');
        if p == 2, xlabel('x (\mum)'); end
        hold on;   % core slab + cavity (trenches are out of this y=0 plane)
        rectangle('Position', [min(dx), -0.175, max(dx)-min(dx), 0.35], ...
                  'EdgeColor', 'w', 'LineWidth', 0.4);
        rectangle('Position', [cav(1), -0.175, cav(2)-cav(1), 0.35], ...
                  'EdgeColor', 'w', 'LineWidth', 1.5);
        hold off;
        title(labels{p}, 'FontSize', 13);
        set(gca, 'FontSize', 12);
    end
    title(tl, ['TM, N=150, ' DEV{k,2} ' — Side view (XZ), |E|^2 in dB, ' ...
               'resonance \lambda \approx 1558 nm'], 'FontSize', 16);
    cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '|E|^2 (dB)';
    cb.Label.FontSize = 13; cb.FontSize = 12;
    savefig(fig, fullfile(out_dir, ['trench_n150_tm_sideview_outline_' key '.fig']));
    exportgraphics(fig, fullfile(out_dir, ['trench_n150_tm_sideview_outline_' key '.png']), ...
                   'Resolution', 160);

    % ── Cross-section view (YZ plane at the cavity edge, x ~ +0.13 um) ───────
    % y horizontal, z vertical (looking down the propagation axis). The full-z
    % trenches DO intersect this plane: two pink wall cross-sections.
    yzfiles = {[DEV{k,3} '_yz_planes.mat'], [DEV{k,3} tr_tag '_yz_planes.mat']};
    fig = figure('Visible', 'on', 'Position', [80 80 1100 900]);
    tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for p = 1:2
        d = load(fullfile(res_dir, yzfiles{p}));
        nexttile(tl);
        imagesc(d.yz_y*1e6, d.yz_z*1e6, ...
                10*log10(max(double(d.yz_E2'), 1e-30)));
        axis xy; clim([top_xy - DB_RANGE, top_xy]); colormap(turbo);
        ylabel('z (\mum)');
        if p == 2, xlabel('y (\mum)'); end
        hold on;   % core cross-section at the cavity + the trench walls
        rectangle('Position', [-cav(3), -0.175, 2*cav(3), 0.35], ...
                  'EdgeColor', 'w', 'LineWidth', 1.5);
        if p == 2
            zl = ylim;
            for s = [1, -1]
                y_lo = min(s*TRENCH.y_in, s*TRENCH.y_out);
                rectangle('Position', [y_lo, zl(1), TRENCH.y_out-TRENCH.y_in, ...
                          zl(2)-zl(1)], 'EdgeColor', TR_COLOR, 'LineWidth', TR_LINE_W);
                text(s*1.8, 3.4, 'Air', 'Color', TR_COLOR, 'FontSize', 13, ...
                     'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
                     'VerticalAlignment', 'middle');
            end
        end
        hold off;
        title(labels{p}, 'FontSize', 13);
        set(gca, 'FontSize', 12);
    end
    title(tl, ['TM, N=150, ' DEV{k,2} ' — Cross-section (YZ) at the cavity, ' ...
               '|E|^2 in dB'], 'FontSize', 14);
    cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = '|E|^2 (dB)';
    cb.Label.FontSize = 13; cb.FontSize = 12;
    savefig(fig, fullfile(out_dir, ['trench_n150_tm_crossview_outline_' key '.fig']));
    exportgraphics(fig, fullfile(out_dir, ['trench_n150_tm_crossview_outline_' key '.png']), ...
                   'Resolution', 160);
end
fprintf('Saved 9 per-device figures to %s\n', out_dir);

% ═══ Benchmark comparison (user 2026-07-21): the trench raises BOTH the peak
% transmission and the Q factor on every device (mode width ~unchanged — shown
% via the envelope panel, not benchmarked). Data: bars = MEASURED resonance
% values; curves = spectra + envelopes from benchmark_data.mat (server-extracted).
bd = load(fullfile(res_dir, 'benchmark_data.mat'));
dev_names = {'average\newlinewidth 800 nm', 'optimized\newlinewidth 1050 nm', ...
             '10-period\newlineapodization'};
T_vals = [0.1841 0.2322; 0.2923 0.3953; 0.7624 0.8944];
Q_vals = [14256 17902; 18001 23079; 27584 33861];
FW_um  = [16.2 16.0; 16.3 16.1; 20.3 19.9];
keys   = {'w800', 'w1050', 'apod'};
dcol   = [0.000 0.447 0.741; 0.850 0.325 0.098; 0.466 0.674 0.188];
c_reg  = [0.55 0.55 0.60];

fig = figure('Visible', 'on', 'Position', [60 60 1500 950]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile(tl);                                  % (a) peak transmission
b = bar(T_vals, 'grouped');
b(1).FaceColor = c_reg; b(2).FaceColor = TR_COLOR;
set(gca, 'XTickLabel', dev_names, 'FontSize', 12); grid on;
ylabel('peak transmission T'); ylim([0 1.05]);
for i = 1:3
    text(b(1).XEndPoints(i), T_vals(i,1)+0.03, sprintf('%.3f', T_vals(i,1)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
    text(b(2).XEndPoints(i), T_vals(i,2)+0.03, sprintf('%.3f\n+%.2f dB', ...
         T_vals(i,2), 10*log10(T_vals(i,2)/T_vals(i,1))), ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end
legend({'regular', 'with air trenches'}, 'Location', 'northwest', 'FontSize', 11);
title('(a) peak resonance transmission', 'FontSize', 13);

nexttile(tl);                                  % (b) quality factor
b = bar(Q_vals, 'grouped');
b(1).FaceColor = c_reg; b(2).FaceColor = TR_COLOR;
set(gca, 'XTickLabel', dev_names, 'FontSize', 12); grid on;
ylabel('Q factor'); ylim([0 max(Q_vals(:))*1.18]);
for i = 1:3
    text(b(1).XEndPoints(i), Q_vals(i,1)+300, sprintf('%d', Q_vals(i,1)), ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 10);
    text(b(2).XEndPoints(i), Q_vals(i,2)+300, sprintf('%d  (+%.0f%%)', Q_vals(i,2), ...
         100*(Q_vals(i,2)/Q_vals(i,1)-1)), 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', 'FontSize', 10);
end
title('(b) quality factor', 'FontSize', 13);

nexttile(tl);                                  % (c) linear T(lambda)
hold on;
for i = 1:3
    plot(bd.([keys{i} '_ctrl_wl']), bd.([keys{i} '_ctrl_T']), '--', ...
         'Color', dcol(i,:), 'LineWidth', 1.1);
    plot(bd.([keys{i} '_tr_wl']),   bd.([keys{i} '_tr_T']),   '-', ...
         'Color', dcol(i,:), 'LineWidth', 1.4);
end
hold off; grid on; xlim([1556.5 1560.5]); set(gca, 'FontSize', 12);
xlabel('wavelength (nm)'); ylabel('T');
legend({'800 regular', '800 +trench', '1050 regular', '1050 +trench', ...
        'apod regular', 'apod +trench'}, 'FontSize', 9, 'NumColumns', 2);
title('(c) transmission spectra (linear)', 'FontSize', 13);

nexttile(tl);                                  % (d) longitudinal mode envelopes
hold on;
leg = cell(1, 6); n = 0;
for i = 1:3
    for v = {'ctrl', 'tr'}
        n = n + 1;
        e = bd.([keys{i} '_' v{1} '_env']); e = e / max(e);
        st = '--'; lw = 1.1; if strcmp(v{1}, 'tr'), st = '-'; lw = 1.4; end
        plot(bd.([keys{i} '_' v{1} '_fx'])*1e6, e, st, 'Color', dcol(i,:), 'LineWidth', lw);
        fw = FW_um(i, 1 + strcmp(v{1}, 'tr'));
        tag = 'regular'; if strcmp(v{1}, 'tr'), tag = '+trench'; end
        short = {'800 nm', '1050 nm', 'apod-10'};
        leg{n} = sprintf('%s %s, FWHM %.1f \\mum', short{i}, tag, fw);
    end
end
hold off; grid on; xlim([-60 60]); set(gca, 'FontSize', 12);
xlabel('x (\mum)'); ylabel('normalized envelope');
legend(leg, 'FontSize', 9, 'Location', 'northeast');
title('(d) longitudinal mode envelope (integrated |E|^2)', 'FontSize', 13);

title(tl, ['TM, N=150 full devices — benchmark: air trenches raise peak T ' ...
           'and Q on every device'], 'FontSize', 16);
savefig(fig, fullfile(out_dir, 'trench_n150_tm_benchmark.fig'));
exportgraphics(fig, fullfile(out_dir, 'trench_n150_tm_benchmark.png'), 'Resolution', 160);
fprintf('Saved: %s\n', fullfile(out_dir, 'trench_n150_tm_benchmark.png'));

% ═══ Standalone spectra figure in dB (user 2026-07-21) ═══
fig = figure('Visible', 'on', 'Position', [80 80 1100 700]);
hold on;
for i = 1:3
    plot(bd.([keys{i} '_ctrl_wl']), 10*log10(max(bd.([keys{i} '_ctrl_T']), 1e-10)), ...
         '--', 'Color', dcol(i,:), 'LineWidth', 1.2);
    plot(bd.([keys{i} '_tr_wl']),   10*log10(max(bd.([keys{i} '_tr_T']), 1e-10)), ...
         '-',  'Color', dcol(i,:), 'LineWidth', 1.5);
end
% y: 0 to -30 dB; x: cropped where the leftmost curve (800 +trench) and the
% rightmost curve (apod regular) touch -30 dB.
dbL = 10*log10(max(bd.w800_tr_T, 1e-10));
dbR = 10*log10(max(bd.apod_ctrl_T, 1e-10));
xL = min(bd.w800_tr_wl(dbL >= -30));
xR = max(bd.apod_ctrl_wl(dbR >= -30));
hold off; grid on; xlim([xL xR]); ylim([-30 0]);
set(gca, 'FontSize', 12);
xlabel('wavelength (nm)', 'FontSize', 13);
ylabel('transmission (dB)', 'FontSize', 13);
legend({'800 regular', '800 +trench', '1050 regular', '1050 +trench', ...
        'apod regular', 'apod +trench'}, 'FontSize', 11, 'NumColumns', 2, ...
        'Location', 'south');
title('TM, N=150 full devices — transmission spectra (dB)', 'FontSize', 15);
savefig(fig, fullfile(out_dir, 'trench_n150_tm_spectra_db.fig'));
exportgraphics(fig, fullfile(out_dir, 'trench_n150_tm_spectra_db.png'), 'Resolution', 160);
fprintf('Saved: %s\n', fullfile(out_dir, 'trench_n150_tm_spectra_db.png'));

% ═══ Presentation figure (user 2026-07-21): left column = benchmark bars
% (T in dB with linear in parentheses; Q with +%), right = cropped dB spectra.
% Sized 16:9 for a PowerPoint slide, large fonts.
ER_dB = -10*log10(T_vals);                     % extinction ratio (positive dB, lower = better)
fig = figure('Visible', 'on', 'Position', [40 40 1600 900]);
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

c_reg_txt = [0.25 0.25 0.30];                  % label colors match the bars
c_tr_txt  = [0.75 0.00 0.65];

nexttile(tl, 1);                               % (a) extinction ratio (positive dB)
b = bar(ER_dB, 'grouped');
b(1).FaceColor = c_reg; b(2).FaceColor = TR_COLOR;
set(gca, 'XTickLabel', dev_names, 'FontSize', 15); grid on;
ylabel('extinction ratio (dB)', 'FontSize', 16); ylim([0 11]);
% one stacked, color-coded label block per group (no collisions by design)
for i = 1:3
    ytop = max(ER_dB(i,:));
    xg = max(i, 1.18);           % group 1: keep the wide labels clear of the y-axis
    text(xg, ytop + 1.75, sprintf('%.2f dB (T %.3f)', ER_dB(i,1), T_vals(i,1)), ...
         'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', c_reg_txt);
    text(xg, ytop + 0.75, sprintf('%.2f dB (T %.3f),  +%.2f dB', ER_dB(i,2), ...
         T_vals(i,2), ER_dB(i,1)-ER_dB(i,2)), 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'FontWeight', 'bold', 'Color', c_tr_txt);
end
legend({'regular', 'with air trenches'}, 'Location', 'northeast', 'FontSize', 15);
title('(a) peak transmission — extinction ratio', 'FontSize', 17);

nexttile(tl, 3);                               % (b) quality factor (ratio labels)
b = bar(Q_vals, 'grouped');
b(1).FaceColor = c_reg; b(2).FaceColor = TR_COLOR;
set(gca, 'XTickLabel', dev_names, 'FontSize', 15); grid on;
ylabel('Q factor', 'FontSize', 16); ylim([0 max(Q_vals(:))*1.42]);
for i = 1:3
    ytop = max(Q_vals(i,:));
    text(i, ytop + 6300, sprintf('%d', Q_vals(i,1)), ...
         'HorizontalAlignment', 'center', 'FontSize', 14, 'Color', c_reg_txt);
    text(i, ytop + 2600, sprintf('%d  (\\times%.2f)', Q_vals(i,2), ...
         Q_vals(i,2)/Q_vals(i,1)), 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'FontWeight', 'bold', 'Color', c_tr_txt);
end
title('(b) quality factor', 'FontSize', 17);

nexttile(tl, 2, [2 1]);                        % (c) dB spectra, cropped
hold on;
for i = 1:3
    plot(bd.([keys{i} '_ctrl_wl']), 10*log10(max(bd.([keys{i} '_ctrl_T']), 1e-10)), ...
         '--', 'Color', dcol(i,:), 'LineWidth', 1.8);
    plot(bd.([keys{i} '_tr_wl']),   10*log10(max(bd.([keys{i} '_tr_T']), 1e-10)), ...
         '-',  'Color', dcol(i,:), 'LineWidth', 2.2);
end
hold off; grid on; xlim([xL xR]); ylim([-30 0]); set(gca, 'FontSize', 16);
xlabel('wavelength (nm)', 'FontSize', 17);
ylabel('transmission (dB)', 'FontSize', 17);
legend({'800 regular', '800 +trench', '1050 regular', '1050 +trench', ...
        'apod regular', 'apod +trench'}, 'FontSize', 16, 'NumColumns', 2, ...
        'Location', 'south');
title('(c) transmission spectra', 'FontSize', 17);

title(tl, 'TM, N=150 full devices — air trenches raise peak transmission and Q', ...
      'FontSize', 19);
savefig(fig, fullfile(out_dir, 'trench_n150_tm_presentation.fig'));
exportgraphics(fig, fullfile(out_dir, 'trench_n150_tm_presentation.png'), 'Resolution', 160);
fprintf('Saved: %s\n', fullfile(out_dir, 'trench_n150_tm_presentation.png'));

% ═══ Standalone 6-curve spectra, presentation styling (user 2026-07-22):
% solid = regular, dashed = + trench; strong per-device colors; big legend.
c6 = [0.20 0.20 0.20;                          % 800  — near-black
      0.15 0.45 0.85;                          % 1050 — blue
      0.80 0.15 0.55];                         % apod — magenta
fig = figure('Visible', 'on', 'Position', [60 60 1250 750]);
hold on;
for i = 1:3
    plot(bd.([keys{i} '_ctrl_wl']), 10*log10(max(bd.([keys{i} '_ctrl_T']), 1e-10)), ...
         '-',  'Color', c6(i,:), 'LineWidth', 2.4);
    plot(bd.([keys{i} '_tr_wl']),   10*log10(max(bd.([keys{i} '_tr_T']), 1e-10)), ...
         '--', 'Color', c6(i,:), 'LineWidth', 2.4);
end
hold off; grid on; xlim([xL xR]); ylim([-30 0]); set(gca, 'FontSize', 16);
xlabel('wavelength (nm)', 'FontSize', 17);
ylabel('transmission (dB)', 'FontSize', 17);
legend({'800 nm regular', '800 nm + trench', '1050 nm regular', '1050 nm + trench', ...
        'apod-10 regular', 'apod-10 + trench'}, 'FontSize', 17, 'NumColumns', 1, ...
        'Location', 'southeast');
title('TM, N=150 full devices — transmission spectra', 'FontSize', 18);
savefig(fig, fullfile(out_dir, 'trench_n150_tm_spectra6.fig'));
exportgraphics(fig, fullfile(out_dir, 'trench_n150_tm_spectra6.png'), 'Resolution', 160);
fprintf('Saved: %s\n', fullfile(out_dir, 'trench_n150_tm_spectra6.png'));
