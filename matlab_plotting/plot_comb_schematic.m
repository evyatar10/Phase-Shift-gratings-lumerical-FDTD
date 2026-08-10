% plot_comb_schematic.m
% Study: scat_r_aim536 (winner device illustration, job 130091_3)   |   2026-08-10
% Purpose: to-scale schematic of the pi-shift Bragg grating + anti-needle comb
%   (winner: Lambda=536 nm, r=110 nm, dx=402 nm = 270 deg, d=1.8 um, h=350 nm).
%   Panel A = top view (x-y); panel B = cross-section (y-z) at a post.

PITCH = 0.51683;  W_WIDE = 1.0;  W_NARROW = 0.6;   % um
LAM = 0.536;  DX = 0.402;  R = 0.110;  D = 1.8;  N_HALF = 15;
CORE_H = 0.35;
XLIM = 10.5;                                       % um, zoom window

col_sin  = [0.35 0.62 0.80];
col_sio2 = [0.93 0.93 0.90];

fig = figure('Visible', 'off', 'Position', [60 60 980 620]);
tl = tiledlayout(fig, 5, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% ── A: top view ────────────────────────────────────────────────────────────
ax = nexttile(tl, [4 1]);
hold(ax, 'on');
patch(ax, XLIM*[-1 1 1 -1], 3.2*[-1 -1 1 1], col_sio2, 'EdgeColor', 'none');
% grating teeth: half-period segments, wide/narrow alternating, pi-shift cavity
% at |x| < pitch/2 (extra half-period of avg width = the cavity segment)
cav = 0.5*PITCH;
patch(ax, cav*[-1 1 1 -1], 0.4*[-1 -1 1 1], col_sin, 'EdgeColor', 'k', 'LineWidth', 0.3);
for s = [-1 1]
    for k = 0:ceil((XLIM-cav)/(0.5*PITCH))
        x0 = cav + k*0.5*PITCH;
        if x0 > XLIM; break; end
        w = W_WIDE/2;  if mod(k, 2) == 1; w = W_NARROW/2; end
        patch(ax, s*[x0 x0+0.5*PITCH x0+0.5*PITCH x0], [-w -w w w], ...
            col_sin, 'EdgeColor', 'k', 'LineWidth', 0.3);
    end
end
% comb posts (same litho layer)
th = linspace(0, 2*pi, 32);
for k = -N_HALF:N_HALF
    xc = k*LAM + DX;
    for s = [-1 1]
        patch(ax, xc + R*cos(th), s*D + R*sin(th), col_sin, 'EdgeColor', 'k', 'LineWidth', 0.3);
    end
end
% annotations
plot(ax, [DX DX+LAM], [D+0.55 D+0.55], 'k-', 'LineWidth', 0.8);
text(ax, DX+LAM/2, D+0.85, '\Lambda = 536 nm', 'HorizontalAlignment', 'center', 'FontSize', 9);
plot(ax, [-9.8 -9.8], [0 D], 'k-', 'LineWidth', 0.8);
text(ax, -9.6, D/2, 'd = 1.8 \mum', 'FontSize', 9);
text(ax, 0, -0.85, {'\pi-shift'; 'cavity'}, 'HorizontalAlignment', 'center', 'FontSize', 9);
text(ax, -6, 0.9, 'corrugated waveguide (SiN, pitch 516.8 nm; continues to \pm43 \mum)', 'FontSize', 9);
text(ax, XLIM-0.3, -2.45, 'comb: 31 SiN posts, r=110 nm, \deltax=+402 nm (270\circ)', ...
    'FontSize', 9, 'HorizontalAlignment', 'right');
text(ax, -XLIM+0.3, -2.75, 'SiO_2 cladding', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);
axis(ax, 'equal');
xlim(ax, XLIM*[-1 1]); ylim(ax, [-3.0 3.0]);
xlabel(ax, 'x  [\mum]  (propagation)');
ylabel(ax, 'y  [\mum]');
text(ax, -XLIM+0.3, 2.7, 'Top view — single litho layer (grating + comb)', ...
    'FontSize', 9, 'FontWeight', 'bold');

% ── B: cross-section at a post ─────────────────────────────────────────────
ax = nexttile(tl);
hold(ax, 'on');
patch(ax, 3.2*[-1 1 1 -1], [-1 -1 1 1], col_sio2, 'EdgeColor', 'none');
patch(ax, 0.4*[-1 1 1 -1], CORE_H/2*[-1 -1 1 1], col_sin, 'EdgeColor', 'k');
for s = [-1 1]
    patch(ax, s*D + R*[-1 1 1 -1], CORE_H/2*[-1 -1 1 1], col_sin, 'EdgeColor', 'k');
end
text(ax, 0, 0.62, 'core 800\times350 nm', 'HorizontalAlignment', 'center', 'FontSize', 8);
text(ax, D, 0.62, 'post', 'HorizontalAlignment', 'center', 'FontSize', 8);
axis(ax, 'equal');
xlim(ax, 3.2*[-1 1]); ylim(ax, [-1 1]);
xlabel(ax, 'y  [\mum]');
ylabel(ax, 'z  [\mum]');
title(ax, 'Cross-section (y-z)');

title(tl, {'\pi-shift Bragg grating + anti-needle comb — measured winner'; ...
    'TM, pitch 516.83 nm, corr 400 nm, N=80/side, \lambda_{res} 1558.64 nm — T 0.893 (ctrl 0.885), side lobes \times0.54'});

set(fig, 'Visible', 'on');
OUT = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results_from_athena', 'scat_r_aim536');
savefig(fig, fullfile(OUT, 'comb_device_schematic.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(OUT, 'comb_device_schematic.png'), 'Resolution', 180);
disp(fullfile(OUT, 'comb_device_schematic.png'));
