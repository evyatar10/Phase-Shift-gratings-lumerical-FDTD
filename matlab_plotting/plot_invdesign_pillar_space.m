% plot_invdesign_pillar_space.m
% Study: inverse-design phase (planning) | Created 2026-08-10 | Job: none (schematic)
% Purpose: user-requested drawing of the pillar design space for inverse design.
%   The periodic comb (measured winner) is the seed; the general parametrization
%   frees every site's position and radius. The sparse 2-pillar pair was REMOVED
%   from consideration by user decision 2026-08-10 (not drawn).
% Output: docs/invdesign_pillar_space.{png,fig}

OUT = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'docs');
PITCH = 0.51683;                  % um
WIDE = 0.4 + 0.325/2;             % half-width of wide tooth (um, corr-325 device)
NARROW = 0.4 - 0.325/2;
RSHOW = 3;                        % post radii exaggerated x3 for visibility

fig = figure('Visible', 'off', 'Position', [60 60 980 560]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---- Panel 1: periodic aimed row (the comb) ------------------------------
ax2 = nexttile(tl);
draw_grating(ax2, 20, PITCH, WIDE, NARROW);
xc = -15:0.531:15;
draw_posts(ax2, xc, 1.9 * ones(size(xc)), 0.08 * RSHOW * ones(size(xc)));
plot(ax2, [0 -18], [0.45 2.3], 'k--', 'LineWidth', 1.0);
plot(ax2, [0 -18], [2.0 3.6], '--', 'Color', [0.85 0.2 0.1], 'LineWidth', 1.0);
text(ax2, -11.5, 0.75, 'needle (grazing, u_x\approx0.98)', 'FontSize', 8.5, 'VerticalAlignment', 'top');
text(ax2, -13.2, 3.55, 'anti-beam (antiphase, 270\circ)', 'FontSize', 8.5, 'Color', [0.85 0.2 0.1]);
text(ax2, 3.5, 2.9, '\Lambda = 531 nm, r = 80 nm, d = 1.9 \mum, 57 posts', 'FontSize', 9);
xlim(ax2, [-20 20]); ylim(ax2, [-4 4.4]);
title(ax2, 'Periodic aimed row — the comb, the SEED   (MEASURED: +0.045 T at q3db N=165)', 'FontSize', 10);
ylabel(ax2, 'y (\mum)');

% ---- Panel 2: inverse-design free sites ----------------------------------
ax3 = nexttile(tl);
draw_grating(ax3, 20, PITCH, WIDE, NARROW);
x3 = [-14.3 -11.9 -9.2 -6.1 -4.4 -2.6 -0.9 0.3 1.8 3.3 5.7 8.4 11.2 13.6];
r3 = [0.06 0.09 0.075 0.11 0.08 0.13 0.07 0.12 0.095 0.06 0.105 0.08 0.07 0.09];
draw_posts(ax3, x3, 1.9 * ones(size(x3)), r3 * RSHOW);
plot(ax3, [x3(6) x3(6)], [1.9 3.4], 'k:', 'LineWidth', 0.8);
text(ax3, x3(6) + 0.4, 3.35, 'x_i free, r_i free (\geq mesh floor), d free', 'FontSize', 9);
text(ax3, -19.3, -3.3, 'optimizer relaxes the comb: per-site position + radius, joint with per-tooth grating knobs', ...
    'FontSize', 8.5);
xlim(ax3, [-20 20]); ylim(ax3, [-4 4.4]);
title(ax3, 'Inverse design — the comb with every site freed (x_i, r_i)', 'FontSize', 10);
xlabel(ax3, 'x (\mum)'); ylabel(ax3, 'y (\mum)');

title(tl, {'Pillar row for inverse design: measured comb as seed \rightarrow per-site free relaxation'; ...
    '\pi-shift grating (TM, h = 350 nm), SiN posts in the cladding; post sizes \times3 for visibility'}, ...
    'FontSize', 11);

set(fig, 'Visible', 'on');
savefig(fig, fullfile(OUT, 'invdesign_pillar_space.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(OUT, 'invdesign_pillar_space.png'), 'Resolution', 160);
disp(fullfile(OUT, 'invdesign_pillar_space.png'));

% ---- local helpers -------------------------------------------------------
function draw_grating(ax, xmax, pitch, wide, narrow)
    hold(ax, 'on');
    for k = ceil(-xmax / (pitch/2)):floor(xmax / (pitch/2))
        x0 = k * pitch / 2;
        if abs(x0) < pitch / 2          % pi-shift cavity segment
            col = [0.93 0.62 0.32]; h = wide;
        elseif mod(k, 2) == 0
            col = [0.55 0.72 0.88]; h = wide;
        else
            col = [0.55 0.72 0.88]; h = narrow;
        end
        rectangle(ax, 'Position', [x0, -h, pitch/2, 2*h], 'FaceColor', col, 'EdgeColor', 'none');
    end
    text(ax, 0, 0, '\pi', 'FontSize', 9, 'HorizontalAlignment', 'center');
    axis(ax, 'equal');
end

function draw_posts(ax, x, y, r)
    for i = 1:numel(x)
        for s = [1 -1]
            rectangle(ax, 'Position', [x(i)-r(i), s*y(i)-r(i), 2*r(i), 2*r(i)], ...
                'Curvature', [1 1], 'FaceColor', [0.15 0.35 0.6], 'EdgeColor', 'none');
        end
    end
end
