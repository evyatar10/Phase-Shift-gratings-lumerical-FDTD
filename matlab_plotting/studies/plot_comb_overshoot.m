% plot_comb_overshoot.m -- top view of the TE overshoot (Nt80x) with the side comb(s) of the confirmation rows.
% Study dir: runners/sweeps/ (farfield_sph_15um_comb_overshoot) | Jobs: Athena 267523 (phase circle), 267628 (rows 3-4), 268986 (pairRL) | 2026-10-22/23
% Purpose (user): drawing of the single comb and the independently phased comb pair (MEASURED Q_i +7.5 % / +16.8 %) with all dimensions.
% Input: results_from_athena/comb_overshoot/comb_devices.json (written from the runner). Output: comb_devices.{fig,png} there.
out = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results_from_athena', 'comb_overshoot');
D = jsondecode(fileread(fullfile(out, 'comb_devices.json')));
P = D.pitch * 1e-3; nar = D.nar(:)' * 1e-3; wid = D.wid(:)' * 1e-3; cav = D.cav * 1e-3; d = D.d * 1e-3;
SiN = [0.75 0.82 0.95]; POST = [0.85 0.3 0.15];
f = figure('Color', 'w', 'Position', [40 40 1500 1100]);
tl = tiledlayout(f, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
rows = {D.single, 'Single comb at +12 um, phase 51 deg: 31 posts per side, r 63 nm. MEASURED Q_i 5.75e5 (+7.5 %), T 0.682 (Athena 267628)';
        D.pairRL, 'Pair: right comb +12 um at 51 deg + left comb -12 um at 147 deg, 2 x 31 posts per side, r 63 nm. MEASURED Q_i 6.25e5 (+16.8 %), T 0.701, width unchanged (Athena 268986)'};
for k = 1:2
    ax = nexttile(tl); hold(ax, 'on');
    draw_device(ax, nar, wid, cav, P, [-26 26], SiN);
    draw_posts(ax, rows{k, 1}.x(:)' * 1e-3, d, rows{k, 1}.r * 1e-3, POST);
    xlim(ax, [-26 26]); ylim(ax, [-2.2 2.2]); set(ax, 'FontSize', 10); ylabel(ax, 'y (\mum)');
    title(ax, rows{k, 2}, 'Interpreter', 'none', 'FontSize', 11);
    xline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
end
xlabel(ax, 'x along the waveguide (\mum); \pi-shift cavity at x = 0');
% zoom: first posts of the +12 comb with dimensions
ax = nexttile(tl); hold(ax, 'on');
xs = D.single.x(:)' * 1e-3; r = D.single.r * 1e-3;
draw_device(ax, nar, wid, cav, P, [3.4 6.6], SiN); draw_posts(ax, xs, d, r, POST);
axis(ax, 'equal'); xlim(ax, [3.4 6.6]); ylim(ax, [-0.3 2.0]); set(ax, 'FontSize', 10);
plot(ax, [xs(1) xs(2)], [1.85 1.85], 'k-', 'LineWidth', 1); text(ax, mean(xs(1:2)), 1.93, sprintf('\\Lambda_c = %.1f nm', D.lam_c), 'HorizontalAlignment', 'center', 'FontSize', 10);
plot(ax, [xs(3) xs(3)], [0 d], 'k--'); text(ax, xs(3) + 0.05, d / 2, sprintf('d = %.1f \\mum (axis to post centre)', d), 'FontSize', 10);
text(ax, xs(1), d - 0.22, sprintf('r = %.0f nm', D.single.r), 'HorizontalAlignment', 'center', 'FontSize', 10);
title(ax, sprintf(['Zoom on the start of the +12 um comb: SiN posts (n 1.97, full 350 nm height), period %.1f nm, radius %.0f nm, ' ...
    'first post at x = %.3f um (phase 51 deg = dx %.1f nm)'], D.lam_c, D.single.r, xs(1), 51 / 360 * D.lam_c), 'Interpreter', 'tex', 'FontSize', 11);
xlabel(ax, 'x (\mum)'); ylabel(ax, 'y (\mum)');
title(tl, sprintf('TE overshoot Nt80x (pitch %.0f nm, %d periods/side, cavity %.1f nm wide, \\lambda %.1f nm) + side comb', D.pitch, D.n_side, D.cav, D.lam), 'FontSize', 12);
exportgraphics(f, fullfile(out, 'comb_devices.png'), 'Resolution', 150); savefig(f, fullfile(out, 'comb_devices.fig'));

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

function draw_posts(ax, xs, d, r, col)
t = linspace(0, 2 * pi, 40);
for x = xs
    for s = [-1 1]
        fill(ax, x + r * cos(t), s * d + r * sin(t), col, 'EdgeColor', 'none');
    end
end
end
