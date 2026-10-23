% plot_radcancel_planar_devices.m -- top views + cross-sections of the planar feature classes proposed for radiation cancellation.
% Study dir: runners/sweeps/ (farfield_sph_15um_*plain*) | Jobs: none yet (devices for review before dispatch) | 2026-10-21
% Purpose (user): "image of the devices we want to check" before execution. Plain TE pi-shift grating (pitch 500, narrow 642 /
% wide 958 nm, core 350 nm, oxide clad). Base D* = best rectangular cavity: width 1050 nm is the best MEASURED so far (placeholder
% until Phase A). Feature sizes are PLACEHOLDERS until the model sets them. All features full-height in the SiN layer, y-mirrored.
% Output: results_from_athena/radcancel_planar/planar_devices.{fig,png}
out = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results_from_athena', 'radcancel_planar');
if ~exist(out, 'dir'), mkdir(out); end
P = 0.5; NAR = 0.642; WID = 0.958; H = 0.35;          % um
W0 = 0.800; WSTAR = 1.050;                             % cavity widths: D0 control, D* placeholder
XL = [-1.6 1.6]; YL = [-0.85 0.85];
SiN = [0.75 0.82 0.95]; FEAT = [0.85 0.3 0.15];
% feature placeholders (x0, y0 = centre of the upper copy; lx, ly = size), all in um
shoulder = [0.175 NaN 0.10 0.010];                     % on the narrow wall right of the +125 nm step (drawn x5 in height)
slot     = [0.125 0     0.10 0.060];                   % oxide slot straddling the +125 nm step, on the axis
pad      = [0.250 NAR/2 + 0.10 + 0.05 0.15 0.10];      % SiN pad, 100 nm gap from the narrow wall
rails    = [0.250 NAR/2 + 0.15 + 0.05 0.25 0.10;       % SiN rails beside the narrow segments of the first periods (gap 150 nm)
           -0.500 NAR/2 + 0.15 + 0.05 0.25 0.10;
            0.750 NAR/2 + 0.15 + 0.05 0.25 0.10];
panels = {
  {W0,    {},                     'D0  plain control (stored): cavity 800 x 250 nm, Q_i 2.02e4, radiation 1.000'}
  {WSTAR, {},                     'D*  best rectangular cavity = new base (1050 nm best measured so far, 0.899; Phase A finds the optimum)'}
  {WSTAR, {'shoulder', shoulder}, 'A  sidewall shoulder at the right cavity step (existing type; drawn x5 in height)'}
  {WSTAR, {'slot', slot},         'B  oxide slot through the core on the axis, at the right cavity step (source at y = 0)'}
  {WSTAR, {'pad', pad},           'C  detached SiN pads beside the first narrow segment, 100 nm gap (source outside the core)'}
  {WSTAR, {'rails', rails},       'D  SiN side rails beside the narrow segments of the first periods (mode matching across the steps, Watts-type)'}
};
f = figure('Color', 'w', 'Position', [40 40 1500 1250]);
tl = tiledlayout(f, 4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:numel(panels)
    p = panels{k}; ax = nexttile(tl); hold(ax, 'on');
    [xe, ye] = edge_line(NAR, WID, p{1}, P, XL);
    fill(ax, [xe fliplr(xe)], [ye -fliplr(ye)], SiN, 'EdgeColor', 'none');
    plot(ax, xe, ye, xe, -ye, 'Color', [0.2 0.35 0.7], 'LineWidth', 1.0);
    if k >= 2, [xs, ys] = edge_line(NAR, WID, W0, P, XL); plot(ax, xs, ys, 'k--', xs, -ys, 'k--', 'LineWidth', 0.7); end
    draw_feature(ax, p{2}, xe, ye, FEAT);
    patch(ax, [-P/4 P/4 P/4 -P/4], YL(1) + [0.02 0.02 0.05 0.05], [0.3 0.3 0.3], 'EdgeColor', 'none');
    axis(ax, 'equal'); xlim(ax, XL); ylim(ax, YL); set(ax, 'FontSize', 9); ylabel(ax, 'y (\mum)');
    title(ax, p{3}, 'Interpreter', 'none', 'FontSize', 9.5);
    if k >= 5, xlabel(ax, 'x along the waveguide (\mum); grey bar = cavity segment; dashed = D0 edge'); end
end
% cross-sections (y-z) at the feature x positions: where each class sits relative to the core
cs = {{'B: slot, cut at x = 0.125 um', 'slot', WSTAR / 2}, {'C (pads) and D (rails), cut at x = 0.25 um', 'side', NAR / 2}};
for k = 1:2
    ax = nexttile(tl); hold(ax, 'on'); c = cs{k}; hw = c{3};
    rectangle(ax, 'Position', [-hw -H/2 2*hw H], 'FaceColor', SiN, 'EdgeColor', [0.2 0.35 0.7]);
    if strcmp(c{2}, 'slot')
        rectangle(ax, 'Position', [-slot(4)/2 -H/2 slot(4) H], 'FaceColor', 'w', 'EdgeColor', 'k');
    else
        for s = [-1 1]
            rectangle(ax, 'Position', [s * pad(2) - pad(4)/2 -H/2 pad(4) H], 'FaceColor', FEAT, 'EdgeColor', 'none');
            rectangle(ax, 'Position', [s * rails(1, 2) - rails(1, 4)/2 -H/2 rails(1, 4) H], 'FaceColor', 'none', 'EdgeColor', 'k', 'LineStyle', '--');
        end
        text(ax, pad(2), H/2 + 0.04, 'C pad', 'HorizontalAlignment', 'center', 'FontSize', 8);
        text(ax, rails(1, 2) + 0.02, -H/2 - 0.05, 'D rail (dashed)', 'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    axis(ax, 'equal'); xlim(ax, [-0.85 0.85]); ylim(ax, [-0.4 0.4]); set(ax, 'FontSize', 9);
    xlabel(ax, 'y (\mum)'); ylabel(ax, 'z (\mum)'); title(ax, ['Cross-section ' c{1} ' (full 350-nm height, oxide around)'], 'Interpreter', 'none', 'FontSize', 9.5);
end
title(tl, 'Plain TE \pi-shift grating (pitch 500, 642/958 nm, h 350 nm): planar feature classes on the best cavity D*  --  sizes are placeholders', 'FontSize', 12);
exportgraphics(f, fullfile(out, 'planar_devices.png'), 'Resolution', 150); savefig(f, fullfile(out, 'planar_devices.fig'));

function draw_feature(ax, F, xe, ye, col)
if isempty(F), return; end
kind = F{1}; g = F{2};
for i = 1:size(g, 1)
    x0 = g(i, 1); lx = g(i, 3); ly = g(i, 4);
    for s = [-1 1]
        switch kind
            case 'shoulder'   % on the wall, drawn x5 in height
                [~, ii] = min(abs(xe - x0)); yw = ye(ii); d = 5 * ly;
                if s > 0, yb = yw; else, yb = -yw - d; end
                rectangle(ax, 'Position', [x0 - lx/2, yb, lx, d], 'FaceColor', col, 'EdgeColor', 'none');
            case 'slot'       % on the axis: one slot, draw once
                if s > 0, rectangle(ax, 'Position', [x0 - lx/2, -ly/2, lx, ly], 'FaceColor', 'w', 'EdgeColor', 'k'); end
            otherwise         % pads / rails, mirrored in y
                rectangle(ax, 'Position', [x0 - lx/2, s * g(i, 2) - ly/2, lx, ly], 'FaceColor', col, 'EdgeColor', 'none');
        end
    end
end
end

function [xe, ye] = edge_line(nar, wid, wcav, P, XL)
% Upper edge every 1 nm (same rule as bragg_device): cavity |x| < P/4; right arm narrow then wide, left arm wide (cavity side) then narrow.
xe = XL(1):0.001:XL(2); ye = zeros(size(xe));
for i = 1:numel(xe)
    s = abs(xe(i)) - P/4;
    if s < 0, ye(i) = wcav / 2; continue; end
    fr = mod(s, P) / P;
    if xe(i) > 0, w = nar * (fr < 0.5) + wid * (fr >= 0.5); else, w = wid * (fr < 0.5) + nar * (fr >= 0.5); end
    ye(i) = w / 2;
end
end
