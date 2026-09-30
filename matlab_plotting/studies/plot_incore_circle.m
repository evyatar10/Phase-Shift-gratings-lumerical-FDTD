% plot_incore_circle.m — SiO2 holes INSIDE the core vs SiN posts OUTSIDE: phase circles overlaid.
% Study dirs: results_from_athena/scat_x2_incore_circle (job 148812) + scat_x3_incore_lamscan (job 149355,
% period scan at 0 / 270 deg), 2026-09-14, plus the stored
% rows: scat_x_incore (IGUM, in-core Lambda 531), scat_p_antineedle (SiN, Lambda 545),
% scat_r_aim536 (SiN, Lambda 536), scat_s_refine (SiN, Lambda 531 at 270 deg).
% Purpose: peak T vs comb phase, one panel per period, both material families on the short
% device (TM, W 800, corr 400, h 350, pitch 516.83, N 80/side, box y 16, 20 nm / 1501 pts);
% fourth panel = amplitude axis (hole count) for the in-core comb at Lambda 531.
% In-core holes: r 80 (minimum renderable; matched amplitude would be r ~ 25), y +/- 250.
% SiN posts: r 110, y +/- 1800. Every file's period / count / phase is parsed from its name.

root = fullfile(fileparts(mfilename('fullpath')), '..', '..');
res = @(varargin) fullfile(root, varargin{:}, 'results');
ctrl_athena = fullfile(res('results_from_athena', 'scat_h_retrocomb'), 'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat');
ctrl_igum   = fullfile(res('results_from_igum', 'scat_q_r80phase'),    'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat');

incore = [rows(res('results_from_athena', 'scat_x2_incore_circle')); rows(res('results_from_athena', 'scat_x3_incore_lamscan')); ...
          rows(res('results_from_igum', 'scat_x_incore'))];
sin    = [rows(res('results_from_athena', 'scat_p_antineedle')); rows(res('results_from_athena', 'scat_r_aim536')); ...
          rows(res('results_from_athena', 'scat_s_refine'))];
sin = sin(sin.r == 110 & sin.n == 31 & sin.y == 1800, :);             % the phase-circle rows only
Tc_a = peakT(ctrl_athena); Tc_i = peakT(ctrl_igum);

blue = [0 0.45 0.74]; red = [0.85 0.33 0.10]; grey = [0.45 0.45 0.45];
fig = figure('Visible', 'off', 'Position', [60 60 1500 1300]);
tl = tiledlayout(fig, 3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for lam = [531 536 545]
    ax = nexttile(tl); hold(ax, 'on');
    yline(ax, Tc_a, '--', 'Color', grey, 'LineWidth', 1.2);
    s = sortrows(sin(round(sin.lam) == lam, :), 'phase');
    plot(ax, s.phase, s.T, 'o-', 'Color', blue, 'MarkerFaceColor', blue, 'LineWidth', 1.4, 'MarkerSize', 7);
    c = sortrows(incore(round(incore.lam) == lam & incore.n == 31, :), 'phase');
    plot(ax, c.phase, c.T, 's-', 'Color', red, 'MarkerFaceColor', red, 'LineWidth', 1.4, 'MarkerSize', 7);
    hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on');
    xlim(ax, [-10 370]); xticks(ax, 0:90:360);
    xlabel(ax, 'comb phase (\circ)'); ylabel(ax, 'peak transmission');
    legend(ax, {'no comb', 'SiN posts outside, r 110', 'SiO_2 holes in core, r 80'}, 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 9);
    ylim(ax, [0.4 0.92]);
    title(ax, ['\Lambda ' num2str(lam) ' nm, 31 posts'], 'FontWeight', 'normal');
end
ax = nexttile(tl); hold(ax, 'on');
yline(ax, Tc_a, '--', 'Color', grey, 'LineWidth', 1.2);
mk = {'o', 's', '^', 'd'}; ph_list = [0 90 180 270]; lg = {'no comb'};
shade = [0.55 0.10 0.05; 0.85 0.33 0.10; 0.95 0.55 0.25; 0.60 0.60 0.60];
for k = 1:4
    c = sortrows(incore(round(incore.lam) == 531 & round(incore.phase / 90) * 90 == ph_list(k), :), 'n');
    if isempty(c), continue, end
    plot(ax, c.n, c.T, [mk{k} '-'], 'Color', shade(k, :), 'MarkerFaceColor', 'w', 'LineWidth', 1.4, 'MarkerSize', 7);
    lg{end + 1} = [num2str(ph_list(k)) '\circ']; %#ok<SAGROW>
end
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on');
set(ax, 'XScale', 'log'); xticks(ax, [3 9 31]); xlim(ax, [2 40]);
xlabel(ax, 'number of holes'); ylabel(ax, 'peak transmission');
legend(ax, lg, 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 9); ylim(ax, [0.4 0.92]);
title(ax, 'SiO_2 holes in core, \Lambda 531 nm: amplitude by hole count', 'FontWeight', 'normal');
for ph = [0 270]
    ax = nexttile(tl); hold(ax, 'on');
    yline(ax, Tc_a, '--', 'Color', grey, 'LineWidth', 1.2);
    s = sortrows(sin(abs(mod(sin.phase - ph + 180, 360) - 180) < 5, :), 'lam');
    plot(ax, s.lam, s.T, 'o-', 'Color', blue, 'MarkerFaceColor', blue, 'LineWidth', 1.4, 'MarkerSize', 7);
    c = sortrows(incore(abs(mod(incore.phase - ph + 180, 360) - 180) < 5 & incore.n == 31, :), 'lam');
    plot(ax, c.lam, c.T, 's-', 'Color', red, 'MarkerFaceColor', red, 'LineWidth', 1.4, 'MarkerSize', 7);
    hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); xlim(ax, [528 553]); ylim(ax, [0.4 0.92]);
    xlabel(ax, 'comb period \Lambda (nm)'); ylabel(ax, 'peak transmission');
    legend(ax, {'no comb', 'SiN posts outside, r 110', 'SiO_2 holes in core, r 80'}, 'Location', 'southoutside', 'Orientation', 'horizontal', 'FontSize', 9);
    title(ax, ['period scan at ' num2str(ph) '\circ, 31 posts'], 'FontWeight', 'normal');
end
title(tl, sprintf('Comb phase circles, TM, N 80/side, corr 400 nm: no-comb T %.3f (Athena) / %.3f (IGUM, stored in-core rows)', Tc_a, Tc_i));

out = fullfile(root, 'results_from_athena', 'scat_x2_incore_circle', 'incore_vs_sin_circles');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150);
fprintf('wrote %s.png\n', out);
disp(sortrows(incore, {'lam', 'n', 'phase'})); disp(sortrows(sin, {'lam', 'phase'}));

function T = peakT(f)
    m = load(f, 'resonance_transmission'); T = m.resonance_transmission;   % built-in finder, never max(T)
end

function tb = rows(d)
    % one table row per scatterer result file: period / count / phase parsed from the file name
    L = dir(fullfile(d, 'result_N80_TM_avg_Ybox16p0_Zbox8p8_sc*.mat'));
    lam = []; n = []; phase = []; r = []; y = []; T = []; file = {};
    for k = 1:numel(L)
        tok = regexp(L(k).name, 'scR(\d+)_arr(\d+)_X(-?[\d.]+)to(-?[\d.]+)_Y(\d+)to', 'tokens', 'once');
        if isempty(tok), continue, end
        v = str2double(tok); nk = v(2); lam_k = (v(4) - v(3)) / (nk - 1); dx = (v(3) + v(4)) / 2;
        lam(end + 1, 1) = lam_k; n(end + 1, 1) = nk; phase(end + 1, 1) = mod(360 * dx / lam_k, 360); %#ok<AGROW>
        r(end + 1, 1) = v(1); y(end + 1, 1) = v(5); T(end + 1, 1) = peakT(fullfile(d, L(k).name)); %#ok<AGROW>
        file{end + 1, 1} = L(k).name; %#ok<AGROW>
    end
    tb = table(lam, n, phase, r, y, T, file);
end
