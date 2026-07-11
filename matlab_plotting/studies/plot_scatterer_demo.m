%% Scatterer demo: top-view |E|^2 maps — baseline vs constructive vs destructive vs hole
% Study: runners/sweeps/tm_scatterer_demo.py -> results_from_athena/tm_scatterer_demo/
%
% Reads the SERVER-REDUCED slice_*.mat files (z=0 XY plane at the resonance
% wavelength, extracted on Athena by scratchpad/reduce_demo_fields.py — ~1 MB each
% instead of the ~650 MB full-field .mat). One panel per demo case, dB scale SHARED
% across panels so brightness is comparable:
%   baseline (no scatterer) | constructive pillar pair (r=100 @ x=0.81 um)
%   destructive pillar pair (r=200 @ x=1.62 um) | in-core hole (r=100 @ x=0)
% White dashed circles mark the scatterers. Headless-safe; deliverables .fig+PNG.

addpath(fileparts(fileparts(mfilename('fullpath'))));  % project root on path

FONT_SIZE = 11;
X_RANGE_UM = [-4, 6];      % crop: cavity + scatterer region
DB_SPAN = 45;              % dynamic range below the global max

if ~exist('data_dir', 'var') || isempty(data_dir)
    proj = fileparts(fileparts(mfilename('fullpath')));
    data_dir = fullfile(proj, 'results_from_athena', 'tm_scatterer_demo', 'slices');
end
files = dir(fullfile(data_dir, 'slice_*.mat'));
assert(~isempty(files), 'No slice_*.mat in %s', data_dir);

cases = struct('name', {}, 'I', {}, 'x', {}, 'y', {}, 'T', {}, ...
               'r_nm', {}, 'x_nm', {}, 'y_nm', {}, 'is_hole', {});
for k = 1:numel(files)
    d = load(fullfile(files(k).folder, files(k).name));
    r_nm = round(double(d.scatterer_r_m) * 1e9);
    x_nm = round(double(d.scatterer_x_m) * 1e9);
    y_nm = round(double(d.scatterer_y_m) * 1e9);
    is_hole = double(d.scatterer_n) > 0 && double(d.scatterer_n) < 1.9;
    if r_nm == 0
        nm = 'baseline (no scatterer)';
    elseif is_hole
        nm = sprintf('in-core SiO_2 hole r=%d nm @ x=%.2f \\mum', r_nm, x_nm/1000);
    else
        nm = sprintf('SiN pair r=%d nm @ (%.2f, \\pm%.1f) \\mum', r_nm, x_nm/1000, y_nm/1000);
    end
    cases(end+1) = struct('name', nm, 'I', double(d.I_xy), ...
        'x', double(d.x(:))', 'y', double(d.y(:))', ...
        'T', double(d.resonance_transmission), ...
        'r_nm', r_nm, 'x_nm', x_nm, 'y_nm', y_nm, 'is_hole', is_hole); %#ok<SAGROW>
end

% Order: baseline, constructive (r=100 off-axis), destructive (r=200), hole
idx = [find([cases.r_nm] == 0, 1), ...
       find([cases.r_nm] == 100 & ~[cases.is_hole], 1), ...
       find([cases.r_nm] == 200, 1), ...
       find([cases.is_hole], 1)];
cases = cases(idx(idx > 0));

gmax = -inf;
for c = cases, gmax = max(gmax, 10*log10(max(c.I(:)))); end

fig = figure('Position', [60 60 1150 850], 'Color', 'w');
tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = gobjects(1, numel(cases));
for k = 1:numel(cases)
    c = cases(k);
    ax(k) = nexttile;  hold(ax(k), 'on');
    IdB = 10 * log10(c.I);
    sel = c.x*1e6 >= X_RANGE_UM(1) & c.x*1e6 <= X_RANGE_UM(2);
    imagesc(ax(k), c.x(sel)*1e6, c.y*1e6, IdB(sel, :)');
    set(ax(k), 'YDir', 'normal');
    colormap(ax(k), 'hot');
    clim(ax(k), [gmax - DB_SPAN, gmax]);
    axis(ax(k), 'tight');
    if c.r_nm > 0
        ys = c.y_nm / 1000;  yset = ys;  if ys ~= 0, yset = [ys, -ys]; end
        for yy = yset
            rectangle(ax(k), 'Position', ...
                [c.x_nm/1000 - c.r_nm/1000, yy - c.r_nm/1000, 2*c.r_nm/1000, 2*c.r_nm/1000], ...
                'Curvature', [1 1], 'EdgeColor', 'w', 'LineStyle', '--', 'LineWidth', 1.2);
        end
    end
    title(ax(k), sprintf('%s — T = %.4f', c.name, c.T), 'FontSize', FONT_SIZE);
    xlabel(ax(k), 'x [\mum]', 'FontSize', FONT_SIZE);
    ylabel(ax(k), 'y [\mum]', 'FontSize', FONT_SIZE);
    set(ax(k), 'FontSize', FONT_SIZE - 1);
end
cb = colorbar(ax(end));  cb.Layout.Tile = 'east';
ylabel(cb, '10\cdotlog_{10}|E|^2  [dB, shared scale]', 'FontSize', FONT_SIZE);
title(tl, {'\pi-shift Bragg TM — Top view |E|^2 at resonance', ...
    'baseline vs constructive vs destructive scatterers'}, 'FontSize', FONT_SIZE + 1);

out_dir = fileparts(data_dir);
png_path = fullfile(out_dir, 'scatterer_demo_fieldmaps.png');
fig_path = fullfile(out_dir, 'scatterer_demo_fieldmaps.fig');
exportgraphics(fig, png_path, 'Resolution', 200);
savefig(fig, fig_path);
fprintf('Saved: %s\nSaved: %s\n', png_path, fig_path);

%% Figure 2 — DIFFERENCE maps: where does each scatterer change the field?
% 10*log10(I_case / I_baseline), diverging blue-white-red, symmetric clim.
% Red = field enhanced vs baseline, blue = suppressed. The guided mode is
% identical in all cases, so it cancels and the scatterer's action stands out.
ib = find([cases.r_nm] == 0, 1);
assert(~isempty(ib), 'baseline case required for difference maps');
Ib = cases(ib).I;
DIFF_CLIM = 6;   % dB
nmap = 256;  half = nmap/2;
bwr = [linspace(0,1,half)', linspace(0,1,half)', ones(half,1); ...
       ones(half,1), linspace(1,0,half)', linspace(1,0,half)'];

others = cases([cases.r_nm] > 0);
fig2 = figure('Position', [80 80 1150 420*numel(others)/1.4], 'Color', 'w');
tl2 = tiledlayout(numel(others), 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax2 = gobjects(1, numel(others));
for k = 1:numel(others)
    c = others(k);
    ax2(k) = nexttile;  hold(ax2(k), 'on');
    D = 10 * log10(c.I ./ Ib);
    sel = c.x*1e6 >= X_RANGE_UM(1) & c.x*1e6 <= X_RANGE_UM(2);
    imagesc(ax2(k), c.x(sel)*1e6, c.y*1e6, D(sel, :)');
    set(ax2(k), 'YDir', 'normal');
    colormap(ax2(k), bwr);
    clim(ax2(k), [-DIFF_CLIM, DIFF_CLIM]);
    axis(ax2(k), 'tight');
    ys = c.y_nm / 1000;  yset = ys;  if ys ~= 0, yset = [ys, -ys]; end
    for yy = yset
        rectangle(ax2(k), 'Position', ...
            [c.x_nm/1000 - c.r_nm/1000, yy - c.r_nm/1000, 2*c.r_nm/1000, 2*c.r_nm/1000], ...
            'Curvature', [1 1], 'EdgeColor', 'k', 'LineStyle', '--', 'LineWidth', 1.2);
    end
    title(ax2(k), sprintf('%s — T = %.4f', c.name, c.T), 'FontSize', FONT_SIZE);
    ylabel(ax2(k), 'y [\mum]', 'FontSize', FONT_SIZE);
    set(ax2(k), 'FontSize', FONT_SIZE - 1);
    if k < numel(others), set(ax2(k), 'XTickLabel', []); end
end
xlabel(ax2(end), 'x [\mum]', 'FontSize', FONT_SIZE);
cb2 = colorbar(ax2(end));  cb2.Layout.Tile = 'east';
ylabel(cb2, '\Delta|E|^2 vs baseline  [dB]', 'FontSize', FONT_SIZE);
title(tl2, {'\pi-shift Bragg TM — field CHANGE vs no-scatterer baseline', ...
    'red = enhanced, blue = suppressed'}, 'FontSize', FONT_SIZE + 1);

png2 = fullfile(out_dir, 'scatterer_demo_diffmaps.png');
fig2p = fullfile(out_dir, 'scatterer_demo_diffmaps.fig');
exportgraphics(fig2, png2, 'Resolution', 200);
savefig(fig2, fig2p);
fprintf('Saved: %s\nSaved: %s\n', png2, fig2p);
