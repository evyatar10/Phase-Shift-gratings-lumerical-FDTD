% plot_cavity_shape_schematic.m
% Study: inner_shape_study (cavity-shape center study)   |   2026-08-19
% Purpose: to-scale schematic of the pi-shift cavity segment drawn as rect vs
%   barrel vs hourglass, using the builder's own profile
%   w(x) = W_cav +/- depth*sin(pi*u)  (bragg_device.py:1154-1157).
%   Measured rows quoted from results_from_athena/inner_shape_study/results/.

PITCH = 516.83;  W_WIDE = 1000;  W_NARROW = 600;  W_CAV = 800;   % nm
L_CAV = PITCH/2;  DEPTH = 150;                                    % nm
XLIM  = 1.35*PITCH;

col_sin = [0.35 0.62 0.80];  col_sio2 = [0.93 0.93 0.90];
col_hi  = [0.85 0.33 0.24];

% shape, sign, label, measured mode/T/lambda (in-study control = rect)
S = { 'hourglass', -1, 'hourglass, depth 150 nm', 15.370, 0.8583, 1558.50
      'rect',       0, 'rect (control)',          15.532, 0.8864, 1558.62
      'barrel',    +1, 'barrel, depth 150 nm',    15.622, 0.9069, 1558.70 };

fig = figure('Visible', 'off', 'Position', [60 60 900 760]);
tl = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:size(S, 1)
    sgn = S{k, 2};
    ax = nexttile(tl);  hold(ax, 'on');
    patch(ax, XLIM*[-1 1 1 -1], 900*[-1 -1 1 1], col_sio2, 'EdgeColor', 'none');

    % teeth outward from the cavity: left = wide, narrow, ... ; right = narrow, wide, ...
    for m = 0:3
        xL = -L_CAV/2 - (m+1)*PITCH/2;   wL = W_WIDE;   if mod(m,2)==1; wL = W_NARROW; end
        xR =  L_CAV/2 +  m   *PITCH/2;   wR = W_NARROW; if mod(m,2)==1; wR = W_WIDE;   end
        patch(ax, [xL xL+PITCH/2 xL+PITCH/2 xL], wL/2*[-1 -1 1 1], col_sin, 'EdgeColor','k','LineWidth',0.3);
        patch(ax, [xR xR+PITCH/2 xR+PITCH/2 xR], wR/2*[-1 -1 1 1], col_sin, 'EdgeColor','k','LineWidth',0.3);
    end

    % the cavity segment itself — the builder's half-sine profile
    u  = linspace(0, 1, 400);
    xs = -L_CAV/2 + u*L_CAV;
    w  = W_CAV + sgn*DEPTH*sin(pi*u);
    patch(ax, [xs fliplr(xs)], [w/2 -fliplr(w)/2], col_sin, 'EdgeColor', col_hi, 'LineWidth', 1.6);

    % centre-width callout
    wc = W_CAV + sgn*DEPTH;
    plot(ax, [0 0], wc/2*[-1 1], '-', 'Color', col_hi, 'LineWidth', 1.0);
    text(ax, 0, wc/2 + 95, sprintf('%d nm', wc), 'Color', col_hi, 'FontSize', 9, ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold');

    title(ax, sprintf('%s  —  mode %.3f \\mum,  T %.4f,  \\lambda %.2f nm', ...
        S{k,3}, S{k,4}, S{k,5}, S{k,6}), 'FontSize', 10);
    xlim(ax, XLIM*[-1 1]);  ylim(ax, [-820 820]);
    ax.YTick = [-500 0 500];
    if k == 3; xlabel(ax, 'x (nm)'); else; ax.XTickLabel = []; end
    ylabel(ax, 'y (nm)');  box(ax, 'on');
end

title(tl, sprintf(['Cavity-segment shape, \\pi-shift Bragg grating (TM, corr 400, ' ...
    'W_{avg} 800 nm, pitch %.2f nm, cavity %.2f nm)'], PITCH, L_CAV), ...
    'FontSize', 11, 'FontWeight', 'bold');

OUT = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), ...
               'results_from_athena', 'inner_shape_study');
if ~exist(OUT, 'dir'); mkdir(OUT); end
savefig(fig, fullfile(OUT, 'cavity_shape_schematic.fig'));
exportgraphics(fig, fullfile(OUT, 'cavity_shape_schematic.png'), 'Resolution', 180);
disp(fullfile(OUT, 'cavity_shape_schematic.png'));
