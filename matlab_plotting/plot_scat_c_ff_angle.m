% plot_scat_c_ff_angle.m
% Study: results_from_athena/scat_c_response + scat_c2_row2 + scat_c3_ygrid
%        (jobs 120976 / 121392 / 121525+121614)
% Date: 2026-08-09
% Purpose: does ANY single r=80 scatterer-pair position touch the grazing lobes?
%   All 293 measured sites: needle-bin (|ux|>0.95, side monitor) and total FF
%   power change vs site angle theta = atan(y/|x|) from the cavity, with the
%   measured needle emission angle (10.9-12.5 deg) marked.

ROOT = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results_from_athena');
DIRS = {fullfile(ROOT, 'scat_c_response', 'results'), ...
        fullfile(ROOT, 'scat_c2_row2', 'results'), ...
        fullfile(ROOT, 'scat_c3_ygrid', 'results')};
OUT_DIR = fullfile(ROOT, 'scat_c_response');
NEEDLE_DEG = [10.9 12.5];   % measured needle angular range (sub-pixel fit, stage E)

ctrl = load(fullfile(DIRS{1}, 'result_N80_TM_avg_Ybox6p8_Zbox8p8_ff.mat'), ...
    'farfield_side', 'farfield_top', 'resonance_wavelength_nm', 'resonance_transmission');
mask = abs(ctrl.farfield_side.ux) > 0.95;      % E2 rows = ux (verified)
N0 = sum(ctrl.farfield_side.E2(mask, :), 'all');
P0 = sum(ctrl.farfield_side.E2(:)) + sum(ctrl.farfield_top.E2(:));

x_nm = []; y_nm = []; dN = []; dP = [];
seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
for i = 1:numel(DIRS)
    fl = dir(fullfile(DIRS{i}, 'result_*_scR80_arr1_X*_Y*_pair_ff.mat'));
    for j = 1:numel(fl)
        tok = regexp(fl(j).name, 'X(-?\d+)to-?\d+_Y(\d+)_pair', 'tokens', 'once');
        if isempty(tok) || isKey(seen, [tok{1} '_' tok{2}]); continue; end
        seen([tok{1} '_' tok{2}]) = true;
        d = load(fullfile(fl(j).folder, fl(j).name), 'farfield_side', 'farfield_top');
        x_nm(end+1) = str2double(tok{1});                                  %#ok<SAGROW>
        y_nm(end+1) = str2double(tok{2});                                  %#ok<SAGROW>
        dN(end+1) = 100*(sum(d.farfield_side.E2(mask, :), 'all')/N0 - 1);  %#ok<SAGROW>
        dP(end+1) = 100*((sum(d.farfield_side.E2(:)) + sum(d.farfield_top.E2(:)))/P0 - 1); %#ok<SAGROW>
    end
end
theta = atan2d(y_nm, abs(x_nm));

fig = figure('Visible', 'off', 'Position', [80 80 950 700]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
rows_y = unique(y_nm);
cols = lines(numel(rows_y));
panels = {dN, {'grazing-lobe power'; 'change vs bare device  [%]'}, ...
          'Grazing side-lobes only (|u_x|>0.95): change vs device with NO scatterer'; ...
          dP, {'total radiated power'; 'change vs bare device  [%]'}, ...
          'Total radiated power (all angles, both monitors): change vs device with NO scatterer'};
for p = 1:2
    ax = nexttile(tl);
    hold(ax, 'on');
    xr = xregion(ax, NEEDLE_DEG(1), NEEDLE_DEG(2), 'FaceColor', [0.85 0.3 0.1], 'FaceAlpha', 0.15);
    for k = 1:numel(rows_y)
        idx = y_nm == rows_y(k);
        scatter(ax, theta(idx), panels{p, 1}(idx), 14, cols(k, :), 'filled', ...
            'DisplayName', sprintf('y=%.2f\\mum', rows_y(k)/1000));
    end
    yline(ax, 0, 'k-', '0 = radiates same as bare device', ...
        'LabelHorizontalAlignment', 'left', 'FontSize', 8, 'HandleVisibility', 'off');
    yl = [min(panels{p, 1}) max(panels{p, 1})] + [-1 1];
    text(ax, 1, 0.8*yl(2), {'\uparrow MORE radiation'; '(scatterer = extra antenna)'}, ...
        'FontSize', 8, 'Color', [0.7 0.2 0.1], 'HorizontalAlignment', 'left');
    text(ax, 1, 0.8*yl(1), {'\downarrow LESS radiation'; '(scatterer cancels leak)'}, ...
        'FontSize', 8, 'Color', [0.1 0.5 0.2], 'HorizontalAlignment', 'left');
    grid(ax, 'on');
    xlabel(ax, 'site angle from cavity  \theta = atan(y/|x|)  [deg]');
    ylabel(ax, panels{p, 2});
    title(ax, panels{p, 3});
    if p == 1
        set(xr, 'DisplayName', 'lobe emission angle 10.9-12.5\circ');
        legend(ax, 'Location', 'northeast', 'FontSize', 8, 'NumColumns', 2);
        text(ax, 30, min(dN) + 2, 'air trench (extended, d=1.8\mum): needle -94% — off scale', ...
            'FontSize', 8, 'Color', [0.4 0.4 0.4]);
    end
end
title(tl, {'All 293 measured single scatterer-pair sites (SiN r=80) — effect on the far field'; ...
    sprintf('TM corr 400, W800, N=80/side, \\lambda_{res} %.2f nm, ctrl T=%.3f', ...
    ctrl.resonance_wavelength_nm, ctrl.resonance_transmission)});

set(fig, 'Visible', 'on');
savefig(fig, fullfile(OUT_DIR, 'scat_c_ff_angle.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(OUT_DIR, 'scat_c_ff_angle.png'), 'Resolution', 150);
fprintf('%d sites plotted; saved to %s\n', numel(dN), OUT_DIR);
