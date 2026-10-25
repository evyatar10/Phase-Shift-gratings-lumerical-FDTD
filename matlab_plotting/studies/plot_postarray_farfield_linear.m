% plot_postarray_farfield_linear.m -- (linear-scale version of plot_postarray_farfield.m) far field of the bare TE overshoot vs a post device: 2-D intensity on the
% top (z>0, u_x-u_y) and side (y>0, u_x-u_z) hemispheres + 1-D dP/du_x over the whole sphere. Study: runners/sweeps/
% farfield_sph_15um_postarray_overshoot | Jobs: Athena 267605_0 (control), 269626_1 (nw_r2pack), 270593 (nw_r2pack_ext42) | 2026-10-25.
% Input: results_from_athena/comb_overshoot/farfield_ctrl_vs_<dev>.mat (python: solve_v4 --short --energy normalisation, both
% divided by the control's total -> same scale; raw P/Pc without the short-run g-correction; label / T / lam_nm fields for the text).
% Other device: matlab -batch "dev='nw_r2pack_ext42'; plot_postarray_farfield_linear"
if ~exist('dev', 'var'), dev = 'nw_r2pack'; end
R = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results_from_athena', 'comb_overshoot');
d = load(fullfile(R, ['farfield_ctrl_vs_' dev '.mat'])); u = d.grid;
f = figure('Color', 'w', 'Position', [40 40 1400 1350]); tl = tiledlayout(f, 4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
hi = 0.5 * max([d.top_ctrl(:); d.side_ctrl(:)], [], 'omitnan');   % LINEAR colour scale, same for both devices (saturates at half the control max)
lab = {'ctrl', 'posts'}; nm = {'Bare overshoot', ['+ posts (' d.label ')']};
for k = 1:2
    for v = {'top', 'side'}
        ax = nexttile(tl); M = d.([v{1} '_' lab{k}]); imagesc(ax, u, u, M); axis(ax, 'xy', 'equal'); xlim(ax, [-1 1]); ylim(ax, [-1 1]);
        clim(ax, [0 hi]); colormap(ax, 'turbo'); cb = colorbar(ax); cb.Label.String = 'P (per control total, linear)';
        if strcmp(v{1}, 'top'), ylabel(ax, 'u_y'); t = 'top hemisphere (z > 0)'; else, ylabel(ax, 'u_z'); t = 'side hemisphere (y > 0)'; end
        xlabel(ax, 'u_x (along the waveguide)'); title(ax, sprintf('%s: %s', nm{k}, t), 'FontSize', 11);
    end
end
ax = nexttile(tl, [1 2]); plot(ax, d.ux, d.dPdux_ctrl, 'k-', 'LineWidth', 1.6); hold(ax, 'on'); plot(ax, d.ux, d.dPdux_posts, '-', 'Color', [0.85 0.3 0.15], 'LineWidth', 1.6);
grid(ax, 'on'); xlabel(ax, 'u_x = k_x / k'); ylabel(ax, 'dP/du_x (per control total, linear)');
legend(ax, {sprintf('Bare overshoot: total 1.000'), sprintf('+ posts: total %.3f (raw, no g-correction)', d.Ptot_posts)}, 'Location', 'north', 'Interpreter', 'none');
title(ax, '1-D far field (LINEAR): power per unit u_x, integrated over all other directions', 'FontSize', 11);
ax = nexttile(tl); plot(ax, d.ux, d.dPdux_posts ./ d.dPdux_ctrl, '-', 'Color', [0.85 0.3 0.15], 'LineWidth', 1.6); hold(ax, 'on'); yline(ax, 1, 'k:');
yline(ax, d.Ptot_posts, 'b--', sprintf('total %.2f', d.Ptot_posts)); grid(ax, 'on'); ylim(ax, [0 1.6]); xlabel(ax, 'u_x'); ylabel(ax, 'posts / control');
title(ax, 'Ratio per u_x (linear): < 1 = radiation removed', 'FontSize', 11);
ax = nexttile(tl); du = mean(diff(d.ux)); plot(ax, d.ux, cumsum(d.dPdux_ctrl) * du, 'k-', 'LineWidth', 1.6); hold(ax, 'on');
plot(ax, d.ux, cumsum(d.dPdux_posts) * du, '-', 'Color', [0.85 0.3 0.15], 'LineWidth', 1.6); grid(ax, 'on'); xlabel(ax, 'u_x');
ylabel(ax, 'cumulative radiated power'); legend(ax, {'bare overshoot', '+ posts'}, 'Location', 'northwest');
title(ax, sprintf('Cumulative power from u_x = -1 (linear): ends at 1.00 vs %.2f', d.Ptot_posts), 'FontSize', 11);
title(tl, sprintf('Far field: TE \\pi-shift overshoot Nt80x, SiN 350 nm, 96 periods/side, %.1f nm (peak T %.3f \\rightarrow %.3f)', d.lam_nm, d.T_ctrl, d.T_posts), 'FontSize', 12);
exportgraphics(f, fullfile(R, ['farfield_ctrl_vs_' dev '_linear.png']), 'Resolution', 150); savefig(f, fullfile(R, ['farfield_ctrl_vs_' dev '_linear.fig']));
