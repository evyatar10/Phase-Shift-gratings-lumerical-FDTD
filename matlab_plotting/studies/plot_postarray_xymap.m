% plot_postarray_xymap.m -- z = 0 in-plane |E|^2 maps ("Side view" = XY monitor, project naming) of the bare TE overshoot vs a
% post-array device. Study: runners/sweeps/farfield_sph_15um_postarray_overshoot | Jobs: Athena 269582 (control map), 269626_1 (nw_r2pack),
% 270593 (nw_r2pack_ext42) | 2026-10-25. Input: results_from_athena/comb_overshoot/xymap_ctrl_vs_<dev>.mat (python extraction: field at the
% bin nearest each resonance, |E|^2 / U with U = int |E|^2 dx of the core monitor -> both panels on the same per-stored-energy scale;
% label / npost text fields carry the device name and Q_i gain). Other device: matlab -batch "dev='nw_r2pack_ext42'; plot_postarray_xymap"
if ~exist('dev', 'var'), dev = 'nw_r2pack'; end
R = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results_from_athena', 'comb_overshoot');
d = load(fullfile(R, ['xymap_ctrl_vs_' dev '.mat']));
xl = max(36, ceil(max(abs(d.px) + d.pr)));   % widen past the map when the posts reach beyond |x| = 36 um
f = figure('Color', 'w', 'Position', [40 40 1500 900]); tl = tiledlayout(f, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
lo = log10(max(d.I_ctrl(:))) - 5; hi = log10(max(d.I_ctrl(:)));
tags = {'ctrl', 'posts'}; ttl = {sprintf('Bare overshoot (control): \\lambda %.2f nm, peak T %.3f', d.lam_ctrl, d.T_ctrl), ...
       sprintf('%s (%s): \\lambda %.2f nm, peak T %.3f', d.npost, d.label, d.lam_posts, d.T_posts)};
for k = 1:2
    ax = nexttile(tl); I = d.(['I_' tags{k}]);
    imagesc(ax, d.(['x_' tags{k}]), d.(['y_' tags{k}]), log10(I)); axis(ax, 'xy'); clim(ax, [lo hi]); colormap(ax, 'turbo');
    cm = turbo(256); ax.Color = cm(1, :);   % dark axes background so white post circles beyond the map (|x| > 36 um) stay visible
    hold(ax, 'on'); if k == 2, th = linspace(0, 2*pi, 24);
        for j = 1:numel(d.px), for s = [1 -1], plot(ax, d.px(j) + d.pr(j)*cos(th), s*d.py(j) + d.pr(j)*sin(th), 'w-', 'LineWidth', 0.4); end, end, end
    ylim(ax, [-2.3 2.3]); xlim(ax, [-xl xl]); ylabel(ax, 'y (\mum)'); title(ax, ttl{k}, 'FontSize', 11);
    cb = colorbar(ax); cb.Label.String = 'log_{10} |E|^2 / U';
end
ax = nexttile(tl); Ic = d.I_ctrl; Ip = d.I_posts;
imagesc(ax, d.x_posts, d.y_posts, log10(Ip ./ max(Ic, 1e-30))); axis(ax, 'xy'); clim(ax, [-1.5 1.5]); colormap(ax, [linspace(0,1,128)' linspace(0,1,128)' ones(128,1); ones(128,1) linspace(1,0,128)' linspace(1,0,128)']);
ylim(ax, [-2.3 2.3]); xlim(ax, [-xl xl]); xlabel(ax, 'x along the waveguide (\mum); \pi-shift at x = 0'); ylabel(ax, 'y (\mum)');
title(ax, 'Ratio posts / control (log_{10}): blue = weaker field with posts, red = stronger', 'FontSize', 11); cb = colorbar(ax); cb.Label.String = 'log_{10} ratio';
title(tl, 'Side view (z = 0): TE \pi-shift overshoot Nt80x, SiN 350 nm, pitch 514 nm, 96 periods/side', 'FontSize', 12);
exportgraphics(f, fullfile(R, ['xymap_ctrl_vs_' dev '.png']), 'Resolution', 150); savefig(f, fullfile(R, ['xymap_ctrl_vs_' dev '.fig']));
