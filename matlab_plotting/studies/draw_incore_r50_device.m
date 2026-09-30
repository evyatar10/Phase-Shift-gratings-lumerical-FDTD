% draw_incore_r50_device.m — top-view drawing of the in-core r 50 cylinder comb device (stage X6, job 150429).
% Geometry reproduced from bragg_device.py's layout walk for the stage-X spec (2026-09-16):
%   pitch 516.83 nm, N 80/side, half-pitch narrow (600) / wide (1000) segments, corr 400, avg 800;
%   cavity = pitch/2 at avg width (cavity_width_option 'avg'), centred at x = 0; port width 1000.
%   Holes: SiO2 cylinders r 50, 31 per side at x = k*524 + 393 nm (k = -15..15), y = +/-250 nm, full core height.
% Not a Lumerical render (no local license at draw time) — a to-scale plan view. Output next to the X6 results.
pitch = 516.83; hp = pitch/2; N = 80; Wn = 600; Ww = 1000; Wc = 800; Wport = 1000;
lam = 524; dx = 393; r = 50; yh = 250; xh = (-15:15)*lam + dx;
cSiN = [0.55 0.65 0.80]; cHole = [1 1 1]; cEdge = [0.25 0.30 0.40];

segs = [];                                   % [x0 x1 width]
x = -(N*pitch + hp/2);
segs(end+1,:) = [x - 3000, x, Wport];
for d = 1:N, segs(end+1,:) = [x, x+hp, Wn]; x = x+hp; segs(end+1,:) = [x, x+hp, Ww]; x = x+hp; end %#ok<*SAGROW>
segs(end+1,:) = [x, x+hp, Wc]; x = x+hp;
for d = 1:N, segs(end+1,:) = [x, x+hp, Wn]; x = x+hp; segs(end+1,:) = [x, x+hp, Ww]; x = x+hp; end
segs(end+1,:) = [x, x + 3000, Wport];

fig = figure('Visible', 'off', 'Position', [60 60 1400 640], 'Color', 'w');
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'loose', 'Padding', 'compact');
lims = {[-9.5 9.5], [-1.6 1.6]};
ttl = {'Top view - \pi-shift Bragg grating, TM, corr 400, N 80/side, SiO_2 cylinders r 50 in the core (31 per side, \Lambda 524, 270\circ)', 'centre: \pi-shift cavity and the innermost holes'};
for p = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    for k = 1:size(segs,1)
        patch(ax, [segs(k,1) segs(k,2) segs(k,2) segs(k,1)]/1e3, [-1 -1 1 1]*segs(k,3)/2e3, cSiN, 'EdgeColor', cEdge, 'LineWidth', 0.4);
    end
    th = linspace(0, 2*pi, 60);
    for k = 1:numel(xh)
        for s = [-1 1]
            patch(ax, (xh(k) + r*cos(th))/1e3, (s*yh + r*sin(th))/1e3, cHole, 'EdgeColor', [0.7 0.1 0.1], 'LineWidth', 0.8);
        end
    end
    xline(ax, 0, ':', 'Color', [0.3 0.3 0.3]);
    hold(ax, 'off'); axis(ax, 'equal'); xlim(ax, lims{p}); ylim(ax, [-0.75 0.75]); box(ax, 'on');
    xlabel(ax, 'x (\mum)'); ylabel(ax, 'y (\mum)'); title(ax, ttl{p}, 'FontWeight', 'normal');
end
out = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results_from_athena', 'scat_x6_incore_r50', 'incore_r50_device_topview');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 200); disp(['wrote ' out '.png']);
