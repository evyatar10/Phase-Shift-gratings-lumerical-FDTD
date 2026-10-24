% plot_nt80x_subteeth.m -- the sub-teeth antenna rows on the Nt80x: layouts (top view) and what each did to the harmonic table.
% Study dir: runners/sweeps/ (farfield_sph_15um_subteeth) | Jobs: Athena 265555 (A/B), 265566 (C/D), 265652 (E/F) | 2026-10-20
% Inputs: the module's site lists (recomputed here from the same formulas), the *_multipoles.csv of control / A / B (and later
% C-F) and the MEASURED scalars. Output: results_from_athena/farfield_sph_15um_subteeth/nt80x_subteeth.{fig,png}
root = fileparts(fileparts(fileparts(mfilename('fullpath')))); out = fullfile(root, 'results_from_athena', 'farfield_sph_15um_subteeth');
P = 0.514; Lc = P/2; nar = [950.3 919.5 898.2 873.3 850.9 829.5 809.2 789.9 770.9 752.3 734.0 716.2 699.0 682.5 666.8]*1e-3;
wid = [950.3 982.6 1007.1 1038.3 1068.8 1100.7 1133.8 1168.1 1203.9 1240.9 1278.9 1317.9 1357.5 1397.6 1438.1]*1e-3;
hw = @(x) half_width(x, nar, wid, P, Lc);
f = figure('Color','w','Position',[60 60 1500 760]);
% --- panel 1: row A layout (periodic 578 nm, wide-half centres), tabs drawn x10 in y
subplot(3,1,1); hold on; draw_outline(nar, wid, P, Lc, 15);
xa = [0.1285 + (1:7)*P + 0.75*P - P, -(0.1285 + (0:6)*P + 0.25*P)];     % realized A sites (right: j=1..7? see module), approximate
xa = [1.028 1.542 2.056 2.570 3.084 4.112 4.626 -0.771 -1.799 -2.313 -2.827 -3.341 -3.855 -4.369];
for x = xa, for sgn = [-1 1], rectangle('Position', [x-0.06, sgn*hw(x) - (sgn<0)*0.05, 0.12, 0.05], 'FaceColor', [0.85 0.35 0.2], 'EdgeColor', 'k'); end, end
xlim([-5 5]); ylim([-0.9 0.9]); ylabel('y (\mum)'); title('Row A: periodic tab array, 578 nm, 7 per arm, 5 nm protrusion (drawn 50 nm) -- MEASURED radiation 0.971, Q_i +1.6 %, \lambda +0.107 nm', 'FontSize', 10);
% --- panel 2: design 2 layout (arms independent)
subplot(3,1,2); hold on; draw_outline(nar, wid, P, Lc, 15);
d2 = [514 10; 1028 10; 2056 2; 3598 10; 4112 10; 6682 10; -257 10; -771 5.5; -3341 10; -3855 10; -4369 10; -4883 10];
for k = 1:size(d2,1), x = d2(k,1)*1e-3; for sgn = [-1 1], rectangle('Position', [x-0.06, sgn*hw(x) - (sgn<0)*d2(k,2)*5e-3, 0.12, d2(k,2)*5e-3], 'FaceColor', [0.2 0.6 0.3], 'EdgeColor', 'k'); end, end
xlim([-5 5]); ylim([-0.9 0.9]); ylabel('y (\mum)'); xlabel('x along the waveguide (\mum)');
title('Row E: model-designed APERIODIC tabs (arms differ, 12 sites, 2-10 nm, drawn x5) -- first-order prediction: radiation 0.86 (queued, runs tonight)', 'FontSize', 10);
% --- panel 3: harmonic table control vs A vs B (share of the control's leak)
subplot(3,1,3);
dirs = {fullfile(root,'results_from_athena','farfield_sph_15um_c_nt80','results'), out+"/results", out+"/results"};
pats = {'*T340ps_ff_multipoles.csv', 'subteeth_A*_multipoles.csv', 'subteeth_B*_multipoles.csv'}; PU = [1 0.971 0.976];
keys = {}; T = containers.Map();
for k = 1:3
    c = dir(fullfile(char(dirs{k}), pats{k})); if isempty(c), continue; end
    M = readmatrix(fullfile(char(dirs{k}), c(1).name));
    for r = 1:size(M,1)
        key = sprintf('%s(%d,%+d)', char('E'*(M(r,3)>=M(r,4)) + 'M'*(M(r,3)<M(r,4))), M(r,1), M(r,2)); v = (M(r,3)+M(r,4))*PU(k);
        if ~isKey(T,key), T(key) = zeros(1,3); keys{end+1} = key; end %#ok<AGROW>
        tmp = T(key); tmp(k) = v; T(key) = tmp;
    end
end
vals = cell2mat(cellfun(@(k) T(k)', keys, 'UniformOutput', false))'; [~, o] = sort(vals(:,1), 'descend'); top = o(1:12);
bar(100*vals(top,:)); set(gca, 'XTick', 1:12, 'XTickLabel', keys(top), 'XTickLabelRotation', 35, 'FontSize', 9); grid on;
ylabel('% of the control''s radiated power'); legend({'control', 'A: tabs 5 nm', 'B: tabs 2.5 nm'}, 'Location', 'northeast');
title('Largest harmonics: the periodic tab array barely moves them (M(3,0), M(2,\pm1) down a little, E(4,\pm4), M(1,0) up a little)', 'FontSize', 10);
sgtitle('Nt80x: added sub-teeth ("second teeth on the teeth") -- periodic array measured, aperiodic designs queued', 'FontSize', 12);
exportgraphics(f, fullfile(out, 'nt80x_subteeth.png'), 'Resolution', 160); savefig(f, fullfile(out, 'nt80x_subteeth.fig')); fprintf('saved %s\n', out);

function h = half_width(x, nar, wid, P, Lc)
s = abs(x) - Lc/2; if s < 0, h = 0.9503/2; return; end
j = min(floor(s/P) + 1, numel(nar)); frac = (s - (j-1)*P)/P; nf = x > 0;
if nf, w = nar(j); if frac >= 0.5, w = wid(j); end, else, w = wid(j); if frac >= 0.5, w = nar(j); end, end
h = w/2;
end
function draw_outline(nar, wid, P, Lc, np)
xs = []; ys = []; x = Lc/2;
for j = 1:np, xs = [xs x x+P/2]; ys = [ys nar(j) nar(j)]; x = x+P/2; xs = [xs x x+P/2]; ys = [ys wid(j) wid(j)]; x = x+P/2; end %#ok<AGROW>
xl = []; yl = []; x = -Lc/2;
for j = 1:np, xl = [xl x x-P/2]; yl = [yl wid(j) wid(j)]; x = x-P/2; xl = [xl x x-P/2]; yl = [yl nar(j) nar(j)]; x = x-P/2; end %#ok<AGROW>
X = [fliplr(xl) -Lc/2 Lc/2 xs]; Y = [fliplr(yl) 0.9503 0.9503 ys]/2;
fill([X fliplr(X)], [Y -fliplr(Y)], [0.2 0.45 0.8], 'EdgeColor', 'k', 'FaceAlpha', 0.85);
end
