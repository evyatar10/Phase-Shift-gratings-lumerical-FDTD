% plot_knob_illustrations.m -- schematic top views of the radiation-cancellation knobs on the 15.5-um Nt80x pi-shift grating.
% Study dir: runners/sweeps/ (farfield_sph_15um_bumps / _contrast) | Jobs: Athena 265111, 265257 | 2026-10-20
% Purpose (user): illustrate the knob families -- (fig 1) mean-width bump vs contrast change on the innermost period pair,
% (fig 2) an ADDED second tooth pair beside the core (the family the user had in mind) vs the in-core contrast knob.
% Geometry = Itai Nt80x tables (pitch 514 nm, cavity segment 950.3 nm x pitch/2); knob amplitudes EXAGGERATED x6 for visibility.
% Output: results_from_athena/farfield_sph_15um_knobs/knobs_{families,second_pair}.{fig,png}
out = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'results_from_athena', 'farfield_sph_15um_knobs');
P = 0.514; Lcav = P/2;                                   % um
nar = [950.3 919.5 898.2 873.3 850.9 829.5 809.2 789.9] * 1e-3;   % per period, innermost first (index 0 = first period)
wid = [950.3 982.6 1007.1 1038.3 1068.8 1100.7 1133.8 1168.1] * 1e-3;
wcav = 0.9503; EX = 10;                                   % exaggeration of the +-5 nm knobs (drawn on the TOP EDGE only, zoomed)
% ---- figure 1: the two measured families on period pair {0,1}
f1 = figure('Color', 'w', 'Position', [100 100 1400 620]);
specs = {struct('name', 'seed (Itai Nt80x)', 'dn', [0 0], 'dw', [0 0]), ...
         struct('name', sprintf('mean-width bump: both teeth of pair {0,1} +5 nm (x%d)', EX), 'dn', [5 5]*EX, 'dw', [5 5]*EX), ...
         struct('name', sprintf('contrast: wide +2.5 / narrow -2.5 nm on pair {0,1} (x%d)', EX), 'dn', -[2.5 2.5]*EX, 'dw', [2.5 2.5]*EX)};
for k = 1:3
    subplot(3, 1, k); hold on; s = specs{k};
    n2 = nar; w2 = wid; n2(1:2) = n2(1:2) + s.dn * 1e-3; w2(1:2) = w2(1:2) + s.dw * 1e-3;
    h1 = draw_grating(n2, w2, wcav, P, Lcav, [0.2 0.45 0.8]);
    if k > 1, h0 = draw_grating(nar, wid, wcav, P, Lcav, [0.3 0.3 0.3], true); end
    xlim([-2.6 2.6]); ylim([0.40 0.60]); set(gca, 'FontSize', 10); ylabel('top edge y (\mum)');
    capt = {'seed: Itai Nt80x, first periods beside the pi-shift segment', ...
            'mean-width bump: both teeth of pair {0,1} +5 nm  --  MEASURED radiation 1.04x, overlap 0.7 % (weak)', ...
            'contrast: wide +2.5 / narrow -2.5 nm on pair {0,1}  --  MEASURED radiation 0.86x, Q_i +15 %, lambda/W held (works)'};
    title(capt{k}, 'Interpreter', 'none', 'FontSize', 11);
    if k > 1, legend([h0 h1], {'seed outline', sprintf('knob, x%d exaggerated', EX)}, 'Location', 'northeast', 'FontSize', 9); end
    if k == 3, xlabel('x along the waveguide (\mum)'); end
end
sgtitle('Top view (upper edge, zoomed), Nt80x: pitch 514 nm, cavity segment 950 nm -- the two measured knob families (Athena 265111)', 'FontSize', 12);
exportgraphics(f1, fullfile(out, 'knobs_families.png'), 'Resolution', 160); savefig(f1, fullfile(out, 'knobs_families.fig'));
% ---- figure 2: the user's idea (an ADDED second pair of teeth beside the core) vs the in-core contrast knob
f2 = figure('Color', 'w', 'Position', [100 100 1400 520]);
subplot(2, 1, 1); hold on;
draw_grating(nar, wid, wcav, P, Lcav, [0.2 0.45 0.8]);
for sgn = [-1 1]                                           % added tooth pair in the cladding, both sides, near the pi shift
    for xc = [-1.0 1.0]
        rectangle('Position', [xc - 0.12, sgn * 0.78 - 0.08, 0.24, 0.16], 'FaceColor', [0.85 0.35 0.2], 'EdgeColor', 'k');
    end
end
xlim([-2.6 2.6]); ylim([-1.0 1.0]); ylabel('y (\mum)'); set(gca, 'FontSize', 10);
title({'Added second pair of teeth beside the core (cladding scatterers, orange) = a DISPLACED source', ...
       'theory: 0.65 um off the axis cancels <= 25 % of the leak; MEASURED: side combs lose on apodized devices'}, 'FontSize', 10);
subplot(2, 1, 2); hold on;
n2 = nar; w2 = wid; n2(1:2) = n2(1:2) - 2.5e-3 * EX; w2(1:2) = w2(1:2) + 2.5e-3 * EX;
h1 = draw_grating(n2, w2, wcav, P, Lcav, [0.2 0.45 0.8]); h0 = draw_grating(nar, wid, wcav, P, Lcav, [0.3 0.3 0.3], true);
xlim([-2.6 2.6]); ylim([0.40 0.60]); ylabel('top edge y (\mum)'); xlabel('x along the waveguide (\mum)'); set(gca, 'FontSize', 10);
legend([h0 h1], {'seed outline', sprintf('contrast knob, x%d exaggerated', EX)}, 'Location', 'northeast', 'FontSize', 9);
title({sprintf('In-core knob that works: contrast of the two innermost periods (upper edge zoomed, x%d) = CO-LOCATED with the leak', EX), ...
       'MEASURED Nt80x: radiation per stored energy 0.86x, Q_i 5.35e5 -> 6.13e5, lambda -0.002 nm, W -0.004 um'}, 'FontSize', 10);
sgtitle('Top view, Nt80x: the "second pair of teeth" idea vs the in-core contrast knob', 'FontSize', 12);
exportgraphics(f2, fullfile(out, 'knobs_second_pair.png'), 'Resolution', 160); savefig(f2, fullfile(out, 'knobs_second_pair.fig'));
fprintf('saved to %s\n', out);

function h = draw_grating(nar, wid, wcav, P, Lcav, col, outline)
% top-view outline of the inner periods: cavity segment at the centre, then periods j=0.. on both arms
% (right arm: narrow half first; left arm: wide half first -- as drawn by bragg_device)
if nargin < 7, outline = false; end
np = numel(nar); xs = []; ys = [];
% right arm, from the cavity edge outward
x = Lcav / 2; xr = []; yr = [];
for j = 1:np
    xr = [xr x x + P/2]; yr = [yr nar(j) nar(j)]; x = x + P/2;        %#ok<AGROW>
    xr = [xr x x + P/2]; yr = [yr wid(j) wid(j)]; x = x + P/2;        %#ok<AGROW>
end
% left arm, mirror with wide half first
x = -Lcav / 2; xl = []; yl = [];
for j = 1:np
    xl = [xl x x - P/2]; yl = [yl wid(j) wid(j)]; x = x - P/2;        %#ok<AGROW>
    xl = [xl x x - P/2]; yl = [yl nar(j) nar(j)]; x = x - P/2;        %#ok<AGROW>
end
xs = [fliplr(xl) -Lcav/2 Lcav/2 xr]; ys = [fliplr(yl) wcav wcav yr] / 2;
if outline
    h = plot(xs, ys, '--', 'Color', col, 'LineWidth', 1.2); plot(xs, -ys, '--', 'Color', col, 'LineWidth', 1.2);
else
    h = fill([xs fliplr(xs)], [ys -fliplr(ys)], col, 'EdgeColor', 'k', 'LineWidth', 0.6, 'FaceAlpha', 0.85);
end
end
