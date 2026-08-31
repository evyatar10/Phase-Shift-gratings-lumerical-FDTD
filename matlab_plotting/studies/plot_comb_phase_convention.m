% plot_comb_phase_convention.m
% Study: comb handoff (runners/scatterers/COMB_HANDOFF.md)  |  Date: 2026-08-27
% Purpose: DEFINE the comb phase -- what "270 deg" is measured relative to.
%   Panel A: to-scale geometry near the cavity. The phase-0 reference lattice
%   (a post ON the pi-shift defect, dashed) vs the actual comb (filled), and the
%   offset dx between them. Also drawn: the identical lattice reached by shifting
%   a quarter period BACKWARD (+0.75*Lambda == -0.25*Lambda, mod Lambda).
%   Panel B: the phase dial with the MEASURED peak T at each quadrant
%   (stage P, job 129989: Lambda 545, r 110, 31 posts, d 1.8 um, corr-400 N=80).
% Output: results_from_athena/scat_rect_comb/comb_phase_convention.{png,fig}

ROOT = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % studies/ -> repo root
OUT  = fullfile(ROOT, 'results_from_athena', 'scat_rect_comb');
if ~isfolder(OUT); mkdir(OUT); end

PITCH = 0.51683; W_WIDE = 1.0; W_NARROW = 0.6;      % um (corr 400 at W800 avg)
LAM = 0.531; DX = 0.398; R = 0.080; D = 1.9;        % the q3db comb, um
XLIM = 2.20;                                        % um, window around the defect

col_sin = [0.35 0.62 0.80];  col_sio2 = [0.94 0.94 0.91];
col_ref = [0.75 0.35 0.15];  col_hi = [0.10 0.45 0.20];

fig = figure('Visible', 'off', 'Position', [60 60 1180 560]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% ── A: what the phase is measured FROM ────────────────────────────────────
ax = nexttile(tl);
hold(ax, 'on');
patch(ax, XLIM*[-1 1 1 -1], (D+0.5)*[-1 -1 1 1], col_sio2, 'EdgeColor', 'none');

cav = 0.5 * PITCH;                                  % the pi-shift cavity segment
patch(ax, cav*[-1 1 1 -1], 0.4*[-1 -1 1 1], col_sin, 'EdgeColor', 'k', 'LineWidth', 0.3);
for s = [-1 1]
    for k = 0:ceil((XLIM - cav) / (0.5*PITCH))
        x0 = cav + k*0.5*PITCH;
        if x0 > XLIM; break; end
        w = W_WIDE/2;  if mod(k, 2) == 1; w = W_NARROW/2; end
        patch(ax, s*[x0 x0+0.5*PITCH x0+0.5*PITCH x0], [-w -w w w], ...
              col_sin, 'EdgeColor', 'k', 'LineWidth', 0.3);
    end
end

th = linspace(0, 2*pi, 40);
for k = -4:4                                        % phase-0 reference lattice
    xc = k * LAM;
    if abs(xc) > XLIM - R; continue; end
    for s = [-1 1]
        plot(ax, xc + R*cos(th), s*D + R*sin(th), '--', 'Color', col_ref, 'LineWidth', 1.1);
    end
end
for k = -4:4                                        % the actual comb
    xc = k*LAM + DX;
    if abs(xc) > XLIM - R; continue; end
    for s = [-1 1]
        patch(ax, xc + R*cos(th), s*D + R*sin(th), col_sin, 'EdgeColor', 'k', 'LineWidth', 0.4);
    end
end

plot(ax, [0 0], [-(D+0.50) D+0.50], 'k--', 'LineWidth', 1.0);
text(ax, 0.08, -(D+0.30), 'x = 0: the \pi-shift defect', 'FontSize', 9);

yA = D + 0.26;                                      % forward arrow 0 -> +dx
quiver(ax, 0, yA, DX, 0, 0, 'Color', col_hi, 'LineWidth', 1.6, 'MaxHeadSize', 0.9);
text(ax, DX/2, yA + 0.16, '\deltax = +398 nm = 0.75\Lambda = 270\circ', ...
     'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', col_hi, 'FontWeight', 'bold');
yB = D - 0.70;                                      % equivalent backward arrow
quiver(ax, LAM, yB, -(LAM - DX), 0, 0, 'Color', col_hi, 'LineWidth', 1.2, ...
       'LineStyle', ':', 'MaxHeadSize', 0.9);
text(ax, LAM + 0.10, yB, '\equiv -133 nm = -0.25\Lambda (the same lattice)', ...
     'HorizontalAlignment', 'left', 'FontSize', 8.5, 'Color', col_hi);

plot(ax, [DX DX+LAM], [D - 0.30 D - 0.30], 'k-', 'LineWidth', 0.8);
plot(ax, [DX DX], (D-0.30) + [-0.05 0.05], 'k-', 'LineWidth', 0.8);
plot(ax, DX+LAM + [0 0], (D-0.30) + [-0.05 0.05], 'k-', 'LineWidth', 0.8);
text(ax, DX + LAM/2, D - 0.46, '\Lambda = 531 nm', 'HorizontalAlignment', 'center', 'FontSize', 9);
plot(ax, [-1.55 -1.55], [0 D], 'k-', 'LineWidth', 0.8);
text(ax, -1.50, D/2, 'd = 1.9 \mum', 'FontSize', 9);

text(ax, -XLIM+0.08, -(D+0.52), 'dashed = phase 0 (a post ON the defect)', ...
     'FontSize', 8.5, 'Color', col_ref);
axis(ax, 'equal');
xlim(ax, XLIM*[-1 1]); ylim(ax, [-(D+0.70) D+0.62]);
xlabel(ax, 'x  [\mum]  (propagation \rightarrow)'); ylabel(ax, 'y  [\mum]');
title(ax, 'Phase = rigid comb shift from the cavity centre', 'FontSize', 10);
box(ax, 'on');

% ── B: the phase dial, with MEASURED peak T per quadrant ──────────────────
ax2 = nexttile(tl);
hold(ax2, 'on');
tt = linspace(0, 2*pi, 400);
plot(ax2, cos(tt), sin(tt), '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 1.0);
plot(ax2, [-1.25 1.25], [0 0], 'k-', 'LineWidth', 0.4);
plot(ax2, [0 0], [-1.25 1.25], 'k-', 'LineWidth', 0.4);

phi = [0 90 180 270];                       % deg
Tm  = [0.8694 0.8586 0.8689 0.8797];        % MEASURED, scat_p_antineedle (Lam 545)
lbl = {'0\circ', '90\circ', '180\circ', '270\circ'};
for k = 1:4
    a = deg2rad(phi(k));
    isbest = (phi(k) == 270);
    c = [0.45 0.45 0.45];  if isbest; c = col_hi; end
    plot(ax2, cos(a), sin(a), 'o', 'MarkerSize', 9, 'MarkerFaceColor', c, 'MarkerEdgeColor', 'k');
    txt = sprintf('%s   T = %.4f', lbl{k}, Tm(k));
    ha = 'left';  off = 0.10;
    if cos(a) < -0.5; ha = 'right'; off = -0.10; end
    text(ax2, cos(a)*1.06 + off, sin(a)*1.14 + 0.10*(abs(sin(a)) < 0.5), txt, 'HorizontalAlignment', ha, ...
         'FontSize', 9.5, 'Color', c, 'FontWeight', ternary(isbest, 'bold', 'normal'));
end
quiver(ax2, 0, 0, cos(deg2rad(270))*0.88, sin(deg2rad(270))*0.88, 0, ...
       'Color', col_hi, 'LineWidth', 2.0, 'MaxHeadSize', 0.5);
text(ax2, 0, 0.42, {'one full turn ='; 'one comb period \Lambda'}, ...
     'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.35 0.35 0.35]);
text(ax2, 0, -0.30, sprintf('\\phi = 360\\circ \\times \\deltax / \\Lambda'), ...
     'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(ax2, 0, 1.45, 'no comb: T = 0.8851', 'HorizontalAlignment', 'center', ...
     'FontSize', 9, 'Color', [0.3 0.3 0.3]);
axis(ax2, 'equal'); axis(ax2, 'off');
xlim(ax2, [-1.75 1.75]); ylim(ax2, [-1.6 1.6]);
title(ax2, 'Measured phase circle (\Lambda 545, r 110, 31 posts, corr-400 N=80)', 'FontSize', 10);

sgtitle(fig, 'Comb phase convention — \deltax measured from the \pi-shift defect, in units of \Lambda', ...
        'FontSize', 12, 'FontWeight', 'bold');

exportgraphics(fig, fullfile(OUT, 'comb_phase_convention.png'), 'Resolution', 200);
savefig(fig, fullfile(OUT, 'comb_phase_convention.fig'));
close(fig);
fprintf('wrote %s\n', fullfile(OUT, 'comb_phase_convention.png'));

function out = ternary(c, a, b)
if c; out = a; else; out = b; end
end
