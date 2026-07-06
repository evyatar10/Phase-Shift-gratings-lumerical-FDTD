% plot_strip_geometry.m — schematic of the lateral STRIP REFLECTOR geometry
% (job 118360). Two SiN strips in the oxide cladding, parallel to the guide,
% mirrored at y = +-d, full core height, same litho layer as the teeth.
% Top view (x = propagation, horizontal; y = lateral, vertical) + a transverse
% cross-section. Schematic — teeth count compressed, y exaggerated for clarity.
% Headless-safe.

clear; close all;
proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
out_dir = fullfile(proj, 'results_from_athena', 'tm_strip_reflector');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

sin = [0.85 0.62 0.30]; sinE = [0.45 0.30 0.10];    % SiN core/strip
ox  = [0.90 0.94 0.99];                              % oxide cladding
hp = 0.51683/2; Wn = 0.60; Ww = 1.00; Wc = 1.05;
d = 1.20; ws = 0.198;                                % strip offset / width (nm-scale)

fig = figure('Visible', 'off', 'Position', [40 40 1180 720], 'Color', 'w');
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% ================= (a) TOP VIEW =================
nexttile; hold on;
xmax = 6.0;
rectangle('Position', [-xmax -2.2 2*xmax 4.4], 'FaceColor', ox, 'EdgeColor', 'none');

drawteeth = @(sgn) 0;   % placeholder
    function draw_rect(x0, x1, w, c, ec)
        fill([x0 x1 x1 x0], [-w -w w w]/2, c, 'EdgeColor', ec, 'LineWidth', 0.5);
    end

% cavity block
draw_rect(-hp, hp, Wc, sin, sinE);
% right arm: narrow, tooth1(wide 1040 + shifted), narrow, tooth2(narrow 980), then nominal
xr = hp;
for k = 1:9
    draw_rect(xr, xr+hp, Wn, sin, sinE); xr = xr+hp;
    if k == 1, w = 1.04; elseif k == 2, w = 0.98; else, w = Ww; end
    draw_rect(xr, xr+hp, w, sin, sinE); xr = xr+hp;
end
% left arm mirror
xl = -hp;
for k = 1:9
    draw_rect(xl-hp, xl, Wn, sin, sinE); xl = xl-hp;
    if k == 1, w = 1.04; elseif k == 2, w = 0.98; else, w = Ww; end
    draw_rect(xl-hp, xl, w, sin, sinE); xl = xl-hp;
end

% the two strips (mirrored pair), full length (drawn to plot edge + arrows)
for s = [1 -1]
    fill([-xmax xmax xmax -xmax], s*d + [-ws -ws ws ws]/2, sin, ...
        'EdgeColor', sinE, 'LineWidth', 0.5);
end
% length arrows / break markers
plot([-xmax xmax], [0 0]*0 + 2.05, 'k', 'HandleVisibility', 'off');
text(0, 2.15, 'strips run the FULL arm length  L \approx 84 \mum  (\pm42 \mum)', ...
    'HorizontalAlignment', 'center', 'FontSize', 9);
% dimension annotations
annotation('doublearrow', [0.5 0.5], [0.62 0.70]);   % rough; also textual below
text(1.3, d, sprintf('SiN strip: width w = %.0f nm', ws*1000), 'FontSize', 9, 'Color', sinE);
text(1.3, -d, 'mirrored strip (\pm y)', 'FontSize', 9, 'Color', sinE);
plot([2.6 2.6], [0 d], 'k-'); plot(2.6, 0, 'k.'); plot(2.6, d, 'k.');
text(2.72, d/2, sprintf('offset d = %.1f \\mum', d), 'FontSize', 9);
text(0, 0.0, '\pi', 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(-4.3, 1.55, 'guided TM mode radiates near-axially (in-plane) \rightarrow', ...
    'FontSize', 8, 'Color', [0.2 0.4 0.7]);
xlim([-xmax xmax]); ylim([-2.2 2.4]); axis equal; box on;
set(gca, 'YTick', [-d 0 d], 'YTickLabel', {'-d', '0', '+d'});
xlabel('x  (\mum) — propagation'); ylabel('y  (\mum)');
title('(a) Top view — SiN reflector strips in the oxide, parallel to the guide, mirrored \pm d');

% ================= (b) TRANSVERSE CROSS-SECTION =================
nexttile; hold on;
h = 0.35;                                            % core height (um)
rectangle('Position', [-3 -1.5 6 3], 'FaceColor', ox, 'EdgeColor', 'none');
% ridge (guide) core
fill([-Ww Ww Ww -Ww]/2, [-h -h h h]/2, sin, 'EdgeColor', sinE, 'LineWidth', 0.7);
% strips at +-d, same height
for s = [1 -1]
    fill(s*d + [-ws ws ws -ws]/2, [-h -h h h]/2, sin, 'EdgeColor', sinE, 'LineWidth', 0.7);
end
plot([-3 3], [-h -h]/2 - 0.02, 'k-', 'LineWidth', 0.5);   % substrate line hint
text(0, h/2 + 0.18, 'ridge (guide)', 'HorizontalAlignment', 'center', 'FontSize', 9);
text(d, h/2 + 0.18, 'SiN strip', 'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', sinE);
text(-d, h/2 + 0.18, 'SiN strip', 'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', sinE);
text(2.55, -0.55, 'all at h = 350 nm (same etch layer)', 'FontSize', 8);
% height arrow
plot([-Ww/2-0.15 -Ww/2-0.15], [-h h]/2, 'k-');
text(-Ww/2-0.55, 0, 'h', 'FontSize', 9);
xlim([-3 3]); ylim([-1.0 1.0]); axis equal; box on;
xlabel('y  (\mum) — lateral'); ylabel('z  (\mum) — height');
title('(b) Transverse cross-section — strips are just SiN blocks beside the ridge in oxide');

title(tl, sprintf(['Lateral strip reflector — recycles the near-axial in-plane TM radiation ' ...
    'back into the guide\n(job 118360; base = best device W1050 + gap-pair[+20,+20] + ' ...
    'see-saw(1040,980)). d = %.1f-2.2 \\mum, w = 198-400 nm scanned'], d), ...
    'FontSize', 11, 'Interpreter', 'tex');

exportgraphics(fig, fullfile(out_dir, 'strip_geometry_schematic.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'strip_geometry_schematic.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'strip_geometry_schematic.png'));
