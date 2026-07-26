% 2D side far-field maps: regular vs + air trench (W800, W1050).
% Study dirs: results_from_athena/air_trench_dscan + air_trench_w1050
% Jobs: 124379 (W800 d-scan) + 124400 (W1050)  |  2026-07-22
% Data: trench_ff_maps.mat (extracted by scratchpad extract_trench_ff.py from
% the *_ff.mat; h=2 um trenches at d=1.8 um, box y=16 um, N=80, opt mesh).
% NOTE: absolute needle values carry the box-16 clipping caveat; the
% regular-vs-trench comparison at identical numerics is the valid content.

res_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
    'results_from_athena', 'air_trench_dscan', 'figures');
d = load(fullfile(res_dir, 'trench_ff_maps.mat'));

P0 = max(d.w800_ctrl_E2(:));                   % one shared scale for all panels
keys = {'w800_ctrl', 'w800_tr'; 'w1050_ctrl', 'w1050_tr'};
titles = {'regular 800 nm', '800 nm + trench (d = 1.8 \mum)'; ...
          'wide cavity 1050 nm', '1050 nm + trench (d = 1.8 \mum)'};
FS = 14;

fig = figure('Position', [60 60 1150 850], 'Color', 'w');
tl = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for r = 1:2
    for c = 1:2
        nexttile
        k = keys{r, c};
        imagesc(d.([k '_ux']), d.([k '_uy']), ...
            10*log10(max(d.([k '_E2'])', P0*1e-5) / P0));
        axis xy; axis([-1 1 -1 1]);
        set(gca, 'FontSize', FS-1);
        title(titles{r, c}, 'FontSize', FS);
        if r == 2, xlabel('u_x', 'FontSize', FS); end
        if c == 1, ylabel('u_z', 'FontSize', FS); end
    end
end
colormap turbo
clim_all = [-40 0];
for ax = findall(fig, 'Type', 'axes')', set(ax, 'CLim', clim_all); end
cb = colorbar;
cb.Layout.Tile = 'east'; cb.FontSize = FS-1;
cb.Label.String = 'radiated power (dB, shared scale)'; cb.Label.FontSize = FS;
title(tl, ['TM N=80 side far field, \lambda\approx1558.5 nm — ' ...
    'air trench removes the grazing needles'], 'FontSize', FS+2);

exportgraphics(fig, fullfile(res_dir, 'trench_farfield_2d.png'), 'Resolution', 180);
set(fig, 'Visible', 'on'); savefig(fig, fullfile(res_dir, 'trench_farfield_2d.fig'));
disp('DONE: trench_farfield_2d');

%% ---- 1D version, same style as the scatterer far-field figure ----
% P(ux) = E2 integrated over uz; solid = regular, dashed = + trench.
P = struct();
for k = {'w800_ctrl', 'w800_tr', 'w1050_ctrl', 'w1050_tr'}
    key = k{1};
    P.(key) = trapz(d.([key '_uy']), d.([key '_E2']), 2);
end
ux = d.w800_ctrl_ux;
P1 = max(P.w800_ctrl);

C800  = [0.20 0.20 0.20];
C1050 = [0.15 0.45 0.85];
fig = figure('Position', [100 100 950 540], 'Color', 'w');
hold on
for u = [-0.980 0.980]   % needle directions
    xline(u, '--', 'Color', [0.85 0.55 0.10], 'LineWidth', 1.8);
end
for s = [-1 1]           % beyond |ux|=1: evanescent in the cladding (bound)
    fill(s*[1 1.05 1.05 1], [-40 -40 3 3], [0.90 0.90 0.90], 'EdgeColor', 'none');
end
h = gobjects(1, 2);
h(1) = plot(ux, 10*log10(P.w800_ctrl / P1), '-',  'Color', C800,  'LineWidth', 2.2);
h(2) = plot(ux, 10*log10(P.w800_tr / P1),   '--', 'Color', C1050, 'LineWidth', 2.2);
text(-0.955, 1.5, '\theta \approx -11.5\circ', 'HorizontalAlignment', 'left', ...
    'FontSize', FS, 'FontWeight', 'bold', 'Color', [0.75 0.45 0.05]);
text(0.955, 1.5, '\theta \approx +11.5\circ', 'HorizontalAlignment', 'right', ...
    'FontSize', FS, 'FontWeight', 'bold', 'Color', [0.75 0.45 0.05]);
set(gca, 'FontSize', FS);
xlabel('direction cosine u_x', 'FontSize', FS);
ylabel('radiated power (dB, norm.)', 'FontSize', FS);
title('TM N=80 side far field — air trench removes the grazing needles', ...
    'FontSize', FS+1);
xlim([-1.05 1.05]); ylim([-30 3]); grid on; box on
legend(h, {'regular 800 nm', '800 nm + trench'}, ...
    'Location', 'south', 'FontSize', FS+1);

exportgraphics(fig, fullfile(res_dir, 'trench_farfield_1d.png'), 'Resolution', 200);
set(fig, 'Visible', 'on'); savefig(fig, fullfile(res_dir, 'trench_farfield_1d.fig'));
disp('DONE: trench_farfield_1d');
