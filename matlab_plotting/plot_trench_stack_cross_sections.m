% Experimental-stack cross sections (Si / 3.8um BOX / SiN / SiO2 cover) for the
% four trench cases of the N=150 W800 pi-shift device, annotated with measured
% peak T and Q from the SYMMETRIC all-oxide simulations (established valid:
% Si-substrate study measured the full fab stack equal within 0.0004 in T).
% Styled after the original fabrication slide (flat outlined blocks, no axes),
% drawn TO SCALE; engineering dimension lines (extension lines + filled
% arrowheads, arrows outside when the span is too small for them).
%
% Study dir: results_from_igum/trench_n150_hscan/   |   2026-07-28
% Jobs: IGUM 41767+41802 (N=150 height ladder, self-contained numerics);
%       "full" panel = plateau h = 5.4/6.9/12 um (T 0.2315/0.2322/0.2315,
%       spread < half floor) — no exact 3.8-um-depth N=150 sim exists; the
%       depth tie was measured at N=80 (jobs 43459/43519/126104).

CORE_H   = 0.35;      % um, SiN core height
BOX_T    = 3.8;       % um, oxide below core bottom -> Si top
TOP_CLAD = 2.2;       % um, SiO2 cover above core top (illustrative thickness)
CORE_W   = 0.8;       % um, regular waveguide width (W800)
TRENCH_Y = [1.4 2.2]; % um, |y| extent of the air trench (d=1800, w=800)
X_HALF   = 4.0;       % um, drawn half-width of the chip block
SI_T     = 1.3;       % um, drawn Si thickness

zc  = CORE_H/2;                  % 0.175
zsi = -(zc + BOX_T);             % -3.975 = Si top face
ztp = zc + TOP_CLAD;             % chip top surface
zbt = zsi - SI_T;                % bottom of drawn Si

% (label, trench z-extent [z1 z2] or [] = none, T, Q)
CASES = { ...
  'No trench',                  [],              0.1842, 14193; ...
  'Trench h = 350 nm',          [-zc  zc],       0.1999, 15405; ...
  'Trench h = 3.25 \mum',       [-1.625 1.625],  0.2280, 17539; ...
  'Full trench, down to Si',    [zsi  ztp],      0.2320, 17740};

C_SI  = [0.470 0.470 0.470];     % slide palette
C_OX  = [0.847 0.871 0.925];
C_SIN = [0.165 0.640 0.610];
INK   = [0.13 0.13 0.13];
EDGE  = {'EdgeColor', INK, 'LineWidth', 1.1};

fig = figure('Position', [40 40 900 950], 'Color', 'w');
tl = tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

for k = 1:size(CASES, 1)
    ax = nexttile(tl);
    hold(ax, 'on');

    rectangle(ax, 'Position', [-X_HALF zbt 2*X_HALF zsi-zbt], 'FaceColor', C_SI,  EDGE{:});
    rectangle(ax, 'Position', [-X_HALF zsi 2*X_HALF ztp-zsi], 'FaceColor', C_OX,  EDGE{:});
    rectangle(ax, 'Position', [-CORE_W/2 -zc CORE_W CORE_H],  'FaceColor', C_SIN, EDGE{:});

    trz = CASES{k, 2};
    if ~isempty(trz)
        for s = [-1 1]
            y0 = min(s*TRENCH_Y);
            rectangle(ax, 'Position', [y0 trz(1) diff(TRENCH_Y) diff(trz)], ...
                      'FaceColor', 'w', EDGE{:});
        end
        if diff(trz) > 1.2
            z_air = mean(trz);
            if z_air < -0.5, z_air = 0.9; end   % down-to-Si trench: keep the
            %                                     label up, clear of the gap dim
            text(ax, mean(TRENCH_Y), z_air, 'Air', 'Rotation', 90, ...
                 'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', INK);
        end
    end

    % material labels (slide style: large, plain ink)
    text(ax, 0, (zsi+zbt)/2, 'Si', 'HorizontalAlignment', 'center', ...
         'FontSize', 14, 'Color', 'w');
    text(ax, -3.0,  1.35, 'SiO_2', 'HorizontalAlignment', 'center', 'FontSize', 13, 'Color', INK);
    text(ax,  0,  -2.7,  'SiO_2', 'HorizontalAlignment', 'center', 'FontSize', 13, 'Color', INK);
    % SiN: to-scale core is too small for an inside label -> leader from above
    text(ax, 0, 1.45, 'SiN', 'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', INK);
    plot(ax, [0 0], [1.15 zc+0.06], '-', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.7);

    title(ax, CASES{k, 1}, 'FontSize', 14, 'FontWeight', 'normal', 'Color', INK);
    text(ax, 0, zbt - 0.95, sprintf('T = %.2f dB      Q = %.1fk', ...
         10*log10(CASES{k, 3}), CASES{k, 4}/1e3), 'HorizontalAlignment', 'center', ...
         'FontSize', 13.5, 'FontWeight', 'bold', 'Color', INK);

    axis(ax, 'equal');
    xlim(ax, [-X_HALF-1.5, X_HALF+1.5]);
    ylim(ax, [zbt-1.6, ztp+1.3]);
    axis(ax, 'off');
end

% ── dimensions: plain two-sided arrows beside each feature, ALL labels
%    horizontal (slide style — no extension lines, no rotated text) ───────────
arrow_v(gobj(tl,1), -3.0, zsi, -zc, 'BOX 3.8 \mum', 'right');
arrow_v(gobj(tl,1),  0.70, -zc,  zc, '350 nm', 'right');
arrow_h(gobj(tl,1), -CORE_W/2, CORE_W/2, -0.60, '800 nm', 'below');
arrow_v(gobj(tl,2), -2.45, -zc,   zc,    '350 nm', 'left');
arrow_v(gobj(tl,3),  2.45, -1.625, 1.625, '3.25 \mum', 'right');
arrow_h(gobj(tl,4), -TRENCH_Y(2), -TRENCH_Y(1), ztp+0.45, '800 nm', 'above');
arrow_h(gobj(tl,4),  CORE_W/2, TRENCH_Y(1), -1.2, '1.0 \mum', 'above');

title(tl, ['\pi-shift Bragg grating, 150 periods, TM,  ' ...
           '\lambda_{res} \approx 1558 nm'], 'FontSize', 16, 'Color', INK);

% invisible full-figure frame so exportgraphics keeps a white margin all around
annotation(fig, 'rectangle', [0.002 0.002 0.996 0.996], 'Color', 'w');

out_dir = fullfile(fileparts(mfilename('fullpath')), '..', ...
                   'results_from_igum', 'trench_n150_hscan');
exportgraphics(fig, fullfile(out_dir, 'trench_stack_cross_sections.png'), 'Resolution', 220);
savefig(fig, fullfile(out_dir, 'trench_stack_cross_sections.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'trench_stack_cross_sections.png'));

% ── helpers ──────────────────────────────────────────────────────────────────
function ax = gobj(tl, k)
ax = nexttile(tl, k);
end

function arrow_h(ax, x1, x2, y, label, labelpos)
% Simple horizontal double-headed arrow x1<->x2 at height y, horizontal label.
ink = [0.13 0.13 0.13];
AH = min(0.22, 0.42*(x2-x1)); AW = AH/2.6;
plot(ax, [x1 x2], [y y], '-', 'Color', ink, 'LineWidth', 1.0);
patch(ax, x1 + [0 AH AH], y + [0 AW -AW], ink, 'EdgeColor', 'none');
patch(ax, x2 - [0 AH AH], y + [0 AW -AW], ink, 'EdgeColor', 'none');
dy = 0.42; if strcmp(labelpos, 'above'), s = 1; else, s = -1; end
text(ax, (x1+x2)/2, y + s*dy, label, 'HorizontalAlignment', 'center', ...
     'FontSize', 11.5, 'Color', ink);
end

function arrow_v(ax, x, z1, z2, label, labelpos)
% Simple vertical double-headed arrow z1<->z2 at x, HORIZONTAL label beside it.
ink = [0.13 0.13 0.13];
AH = min(0.22, 0.42*(z2-z1)); AW = AH/2.6;
plot(ax, [x x], [z1 z2], '-', 'Color', ink, 'LineWidth', 1.0);
patch(ax, x + [0 AW -AW], z1 + [0 AH AH], ink, 'EdgeColor', 'none');
patch(ax, x + [0 AW -AW], z2 - [0 AH AH], ink, 'EdgeColor', 'none');
if strcmp(labelpos, 'right')
    text(ax, x + 0.22, (z1+z2)/2, label, 'HorizontalAlignment', 'left', ...
         'FontSize', 11.5, 'Color', ink);
else
    text(ax, x - 0.22, (z1+z2)/2, label, 'HorizontalAlignment', 'right', ...
         'FontSize', 11.5, 'Color', ink);
end
end
