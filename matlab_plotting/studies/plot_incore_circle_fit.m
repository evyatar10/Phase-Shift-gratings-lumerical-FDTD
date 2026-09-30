% plot_incore_circle_fit.m — SiO2 holes IN the core vs SiN posts OUTSIDE, 31 posts: fitted curves.
% Study dirs: results_from_athena/{scat_x2_incore_circle, scat_x3_incore_lamscan, scat_x4_incore_below530}
% (jobs 148812 / 149355 / 149982, 2026-09-14..15) + stored rows: scat_x_incore (IGUM), scat_p_antineedle,
% scat_r_aim536, scat_s_refine, scat_aim_extend (IGUM). Plot 2026-09-15.
% 2026-09-15 later: + r 50 phase circle at Lambda 524 (scat_x6_incore_r50 job 150429, scat_x9_incore_r50_phase job 150504).
% Purpose (user): two panels only — (left) peak T vs comb phase at Lambda 536 with a first-harmonic
% sinusoid T = a + b cos(phi - phi0) fitted through the 4 phase points of each family (the stage-Q
% method); (right) peak T vs comb period at 270 deg with a smooth (pchip) curve through each family.
% Device: TM, W 800, corr 400, h 350, pitch 516.83, N 80/side, box y 16, 20 nm / 1501 pts.
% In-core holes r 80 / y +/-250; SiN posts r 110 / y +/-1800; peak T = built-in resonance finder.

root = fullfile(fileparts(mfilename('fullpath')), '..', '..');
res = @(varargin) fullfile(root, varargin{:}, 'results');
ctrl = load(fullfile(res('results_from_athena', 'scat_h_retrocomb'), 'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'), 'resonance_transmission');
Tc = ctrl.resonance_transmission;

incore = [rows(res('results_from_athena', 'scat_x2_incore_circle')); rows(res('results_from_athena', 'scat_x3_incore_lamscan')); ...
          rows(res('results_from_athena', 'scat_x4_incore_below530')); rows(res('results_from_igum', 'scat_x_incore'))];
sin    = [rows(res('results_from_athena', 'scat_p_antineedle')); rows(res('results_from_athena', 'scat_r_aim536')); ...
          rows(res('results_from_athena', 'scat_s_refine')); rows(res('results_from_athena', 'scat_x4_incore_below530')); ...
          rows(res('results_from_igum', 'scat_aim_extend'))];
r50    = [rows(res('results_from_athena', 'scat_x6_incore_r50')); rows(res('results_from_athena', 'scat_x9_incore_r50_phase'))];
incore = incore(incore.n == 31 & incore.r == 80  & incore.y == 250,  :);
sin    = sin(sin.n == 31       & sin.r == 110    & sin.y == 1800,    :);
fam = {sin, incore, r50};  lamsel = [536 536 524];
names = {'SiN posts outside, r 110, \Lambda 536', 'SiO_2 holes in core, r 80, \Lambda 536', 'SiO_2 holes in core, r 50, \Lambda 524'};
col = {[0 0.45 0.74], [0.85 0.33 0.10], [0.47 0.67 0.19]}; mk = {'o', 's', '^'}; grey = [0.45 0.45 0.45];

fig = figure('Visible', 'off', 'Position', [60 60 1400 520]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% (left) phase circle at Lambda 536, first-harmonic sinusoid through the points
ax = nexttile(tl); hold(ax, 'on'); lg = {'no comb'};
yline(ax, Tc, '--', 'Color', grey, 'LineWidth', 1.2);
phi = linspace(0, 360, 361);
for f = 1:3
    s = fam{f}; s = s(round(s.lam) == lamsel(f) & ismember(round(s.phase / 90) * 90, 0:90:270), :);
    s = s(abs(s.phase - round(s.phase / 90) * 90) < 5, :);            % the 4 quadrant points only
    A = [ones(height(s), 1), cosd(s.phase), sind(s.phase)]; c = A \ s.T;  % T = a + b cos(phi) + c sin(phi)
    plot(ax, phi, c(1) + c(2) * cosd(phi) + c(3) * sind(phi), '-', 'Color', col{f}, 'LineWidth', 1.6);
    plot(ax, s.phase, s.T, mk{f}, 'Color', col{f}, 'MarkerFaceColor', col{f}, 'MarkerSize', 8);
    lg{end + 1} = sprintf('%s, fit %.3f + %.3f cos(phi - %.0f deg)', names{f}, c(1), hypot(c(2), c(3)), mod(atan2d(c(3), c(2)), 360)); %#ok<SAGROW>
    lg{end + 1} = 'measured'; %#ok<SAGROW>
end
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); xlim(ax, [0 360]); xticks(ax, 0:90:360); ylim(ax, [0.4 0.95]);
xlabel(ax, 'comb phase \phi (\circ)'); ylabel(ax, 'peak transmission');
legend(ax, lg, 'Location', 'southoutside', 'FontSize', 9);
title(ax, '31 posts: phase circle (\Lambda per family)', 'FontWeight', 'normal');

% (right) period scan at 270 deg, smooth curve through the points
ax = nexttile(tl); hold(ax, 'on'); lg = {'no comb'};
yline(ax, Tc, '--', 'Color', grey, 'LineWidth', 1.2);
for f = 1:2
    s = fam{f}; s = sortrows(s(abs(s.phase - 270) < 5, :), 'lam');
    [lam_u, iu] = unique(round(s.lam)); T_u = s.T(iu);
    lam_f = linspace(min(lam_u), max(lam_u), 200);
    plot(ax, lam_f, pchip(lam_u, T_u, lam_f), '-', 'Color', col{f}, 'LineWidth', 1.6);
    plot(ax, lam_u, T_u, mk{f}, 'Color', col{f}, 'MarkerFaceColor', col{f}, 'MarkerSize', 8);
    lg{end + 1} = [names{f} ', pchip']; lg{end + 1} = 'measured'; %#ok<SAGROW>
end
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); xlim(ax, [505 550]); ylim(ax, [0.4 0.95]);
xlabel(ax, 'comb period \Lambda (nm)'); ylabel(ax, 'peak transmission');
legend(ax, lg, 'Location', 'southoutside', 'FontSize', 9);
title(ax, '270\circ, 31 posts: period scan', 'FontWeight', 'normal');
title(tl, sprintf('Comb in the core vs outside, TM, N 80/side, corr 400 nm, no-comb T %.3f', Tc));

out = fullfile(root, 'results_from_athena', 'scat_x4_incore_below530', 'incore_vs_sin_fit');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150);
fprintf('wrote %s.png\n', out);
disp(sortrows(sin(abs(sin.phase - 270) < 5, {'lam', 'phase', 'T'}), 'lam'));
disp(sortrows(incore(abs(incore.phase - 270) < 5, {'lam', 'phase', 'T'}), 'lam'));

function tb = rows(d)
    % one table row per scatterer result file: period / count / phase parsed from the file name
    L = dir(fullfile(d, 'result_N80_TM_avg_Ybox16p0_Zbox8p8_sc*.mat'));
    lam = []; n = []; phase = []; r = []; y = []; T = [];
    for k = 1:numel(L)
        tok = regexp(L(k).name, 'scR(\d+)_arr(\d+)_X(-?[\d.]+)to(-?[\d.]+)_Y(\d+)to', 'tokens', 'once');
        if isempty(tok), continue, end
        v = str2double(tok); nk = v(2); lam_k = (v(4) - v(3)) / (nk - 1); dx = (v(3) + v(4)) / 2;
        m = load(fullfile(d, L(k).name), 'resonance_transmission');
        lam(end + 1, 1) = lam_k; n(end + 1, 1) = nk; phase(end + 1, 1) = mod(360 * dx / lam_k, 360); %#ok<AGROW>
        r(end + 1, 1) = v(1); y(end + 1, 1) = v(5); T(end + 1, 1) = m.resonance_transmission; %#ok<AGROW>
    end
    tb = table(lam, n, phase, r, y, T);
end
