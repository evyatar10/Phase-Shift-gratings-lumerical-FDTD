% plot_postarray_overshoot.m -- top view of the TE overshoot (Nt80x) with the Born-designed SiN post arrays (pa3: 3 rows, pa2: 2 rows).
% Study dir: runners/sweeps/ (farfield_sph_15um_postarray_overshoot) | Jobs: Athena (see memory project_radcancel_overnight_2026-10-22) | 2026-10-23
% Input: results_from_athena/comb_overshoot/postarray_devices.json (written by the build check). Output: postarray_devices.{fig,png} there.
if ~exist('jsonfile', 'var'), jsonfile = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results_from_athena', 'comb_overshoot', 'postarray_devices.json'); end
out = fileparts(jsonfile); D = jsondecode(fileread(jsonfile));   % caller may preset jsonfile (e.g. the plain-TE device file)
P = D.pitch * 1e-3; nar = D.nar(:)' * 1e-3; wid = D.wid(:)' * 1e-3; cav = D.cav * 1e-3;
SiN = [0.75 0.82 0.95]; POST = [0.85 0.3 0.15];
f = figure('Color', 'w', 'Position', [40 40 1500 1100]);
if ~exist('names', 'var'), names = {'pa3_r100', 'pa2_r100'}; end   % caller may preset, e.g. {'oneline','pair2row'}
if ~exist('meas', 'var'), meas = repmat({''}, size(names)); end   % optional MEASURED text per design
tl = tiledlayout(f, numel(names) + 1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:numel(names)
    d = D.designs.(names{k}); ax = nexttile(tl); hold(ax, 'on');
    XL = max(26, ceil(max(abs(d.x_nm)) / 1e3) + 1); YL = max(2.1, max(d.y_nm) / 1e3 + 0.3);   % fit the whole array
    draw_device(ax, nar, wid, cav, P, [-XL XL], SiN); draw_posts(ax, d.x_nm(:)' * 1e-3, d.y_nm(:)' * 1e-3, d.r_nm(:)' * 1e-3, POST);
    xlim(ax, [-XL XL]); ylim(ax, [-YL YL]); set(ax, 'FontSize', 10); ylabel(ax, 'y (\mum)');
    rows = unique(d.y_nm);
    title(ax, sprintf('%s: %d posts per side on rows y = %s um, r %.0f-%.0f nm, model-predicted leak %.3f x control %s', names{k}, numel(d.x_nm), ...
        strjoin(arrayfun(@(v) sprintf('%.1f', v / 1e3), rows, 'UniformOutput', false), '/'), min(d.r_nm), max(d.r_nm), d.born_P_over_Pc, meas{k}), 'Interpreter', 'none', 'FontSize', 11);
end
xlabel(ax, 'x along the waveguide (\mum); \pi-shift cavity at x = 0');
d = D.designs.(names{1}); ax = nexttile(tl); hold(ax, 'on');
draw_device(ax, nar, wid, cav, P, [10 16], SiN); draw_posts(ax, d.x_nm(:)' * 1e-3, d.y_nm(:)' * 1e-3, d.r_nm(:)' * 1e-3, POST);
axis(ax, 'equal'); xlim(ax, [10 16]); ylim(ax, [-0.2 max(d.y_nm) / 1e3 + 0.25]); set(ax, 'FontSize', 10); xlabel(ax, 'x (\mum)'); ylabel(ax, 'y (\mum)');
title(ax, sprintf('%s, zoom x = 10-16 um (posts: full 350 nm SiN, n 1.97)', names{1}), 'Interpreter', 'none', 'FontSize', 11);
if isfield(D, 'title'), ttl = D.title; else, ttl = sprintf('TE overshoot Nt80x (pitch %.0f nm, %d periods/side, cavity %.1f nm, resonance %.1f nm) + designed side-post arrays', D.pitch, D.n_side, D.cav, D.lam); end
title(tl, ttl, 'FontSize', 12);
tag = strjoin(names, '_'); exportgraphics(f, fullfile(out, ['devices_' tag '.png']), 'Resolution', 150); savefig(f, fullfile(out, ['devices_' tag '.fig']));

function draw_device(ax, nar, wid, cav, P, XL, col)
xe = XL(1):0.002:XL(2); ye = zeros(size(xe));
for i = 1:numel(xe)
    s = abs(xe(i)) - P / 4;
    if s < 0, ye(i) = cav / 2; continue; end
    j = min(floor(s / P) + 1, numel(nar)); fr = (s - (j - 1) * P) / P;
    if xe(i) > 0, w = nar(j) * (fr < 0.5) + wid(j) * (fr >= 0.5); else, w = wid(j) * (fr < 0.5) + nar(j) * (fr >= 0.5); end
    ye(i) = w / 2;
end
fill(ax, [xe fliplr(xe)], [ye -fliplr(ye)], col, 'EdgeColor', 'none');
plot(ax, xe, ye, xe, -ye, 'Color', [0.2 0.35 0.7], 'LineWidth', 0.8);
end

function draw_posts(ax, xs, ys, rs, col)
t = linspace(0, 2 * pi, 32);
for i = 1:numel(xs)
    for s = [-1 1]
        fill(ax, xs(i) + rs(i) * cos(t), s * ys(i) + rs(i) * sin(t), col, 'EdgeColor', 'none');
    end
end
end
