% plot_incore_r50_summary.m — in-core SiO2 cylinders (r 50) vs SiN cylinders outside (r 110): summary figures.
% Study dirs: results_from_athena/{scat_x4_incore_below530, scat_x5_incore_r110, scat_x6_incore_r50,
% scat_x7_incore_r40_r30, scat_x9_incore_r50_phase, scat_x3_incore_lamscan, scat_x2_incore_circle,
% scat_p_antineedle, scat_r_aim536, scat_s_refine} + results_from_igum/{scat_x_incore, scat_aim_extend}.
% Jobs 148812/149355/149982/150391/150429/150458/150504 (2026-09-14..16). Plot 2026-09-16.
% Fig 1: (left) peak T vs comb period at 270 deg, pchip through the points — SiN r 110 (Lambda 510-545) and
%        in-core r 80 (the only in-core period scan; r 50 exists at Lambda 524 only and is not drawn here).
%        (right) peak T vs comb phase, periodic pchip through the 4 quadrant points, each family at its
%        best-period circle: SiN Lambda 536 (best 531 has only the 270 point stored), in-core r 50 Lambda 524.
% Fig 2: in-core only — peak T (left axis) and spatial mode width (right axis) vs hole radius at 270 deg,
%        Lambda 524 (r 30/40/50/80/110); colours deliberately different from fig 1.
% Device: TM, W 800, corr 400, h 350, pitch 516.83, N 80/side, box y 16, 20 nm / 1501 pts, 31 cylinders.
% Peak T = built-in resonance finder; width = fwhm_m (post_processing convention). No sinusoid fits — interpolation only.
root = fullfile(fileparts(mfilename('fullpath')), '..', '..');
res = @(varargin) fullfile(root, varargin{:}, 'results');
ctrl = load(fullfile(res('results_from_athena', 'scat_h_retrocomb'), 'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'), 'resonance_transmission', 'fwhm_m');
Tc = ctrl.resonance_transmission; wc = ctrl.fwhm_m * 1e6;

inc = [rows(res('results_from_athena', 'scat_x2_incore_circle')); rows(res('results_from_athena', 'scat_x3_incore_lamscan')); ...
       rows(res('results_from_athena', 'scat_x4_incore_below530')); rows(res('results_from_igum', 'scat_x_incore')); ...
       rows(res('results_from_athena', 'scat_x5_incore_r110')); rows(res('results_from_athena', 'scat_x6_incore_r50')); ...
       rows(res('results_from_athena', 'scat_x7_incore_r40_r30')); rows(res('results_from_athena', 'scat_x9_incore_r50_phase'))];
sin = [rows(res('results_from_athena', 'scat_p_antineedle')); rows(res('results_from_athena', 'scat_r_aim536')); ...
       rows(res('results_from_athena', 'scat_s_refine')); rows(res('results_from_athena', 'scat_x4_incore_below530')); ...
       rows(res('results_from_igum', 'scat_aim_extend'))];
inc = inc(inc.n == 31 & inc.y == 250, :);  sin = sin(sin.n == 31 & sin.y == 1800, :);
cs = [0 0.45 0.74]; ci = [0.85 0.33 0.10]; grey = [0.45 0.45 0.45];
hdr = 'TM, W 800, corr 400, N 80/side, 31 cylinders';

%% Figure 1 — period scan at 270 deg (left), phase circle at the best period (right)
fig = figure('Visible', 'off', 'Position', [60 60 1400 520]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on');
yline(ax, Tc, '--', 'Color', grey, 'LineWidth', 1.2, 'DisplayName', sprintf('no cylinders, T %.3f', Tc));
s = sortrows(sin(abs(sin.phase - 270) < 5 & sin.r == 110, :), 'lam'); [~, iu] = unique(s.lam); s = s(iu, :);
curve(ax, s.lam, s.T, 'o', cs, 'SiN cylinders outside, r 110');
s = sortrows(inc(abs(inc.phase - 270) < 5 & inc.r == 80, :), 'lam'); [~, iu] = unique(s.lam); s = s(iu, :);
curve(ax, s.lam, s.T, 's', ci, 'SiO_2 cylinders in core, r 80');
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); ylim(ax, [0.6 0.95]);
xlabel(ax, 'comb period \Lambda (nm)'); ylabel(ax, 'peak transmission');
legend(ax, 'Location', 'southoutside', 'FontSize', 9);
title(ax, 'phase 270\circ: peak T vs comb period', 'FontWeight', 'normal');

ax = nexttile(tl); hold(ax, 'on');
yline(ax, Tc, '--', 'Color', grey, 'LineWidth', 1.2, 'DisplayName', sprintf('no cylinders, T %.3f', Tc));
fam = {sin(sin.r == 110, :), inc(inc.r == 50, :)}; lamsel = [536 524]; col = {cs, ci}; mk = {'o', '^'};
names = {'SiN cylinders outside, r 110, \Lambda 536', 'SiO_2 cylinders in core, r 50, \Lambda 524'};
for f = 1:2
    s = fam{f}; s = s(round(s.lam) == lamsel(f) & abs(s.phase - round(s.phase / 90) * 90) < 5, :);
    [~, iu] = unique(round(s.phase / 90) * 90); s = sortrows(s(iu, :), 'phase');
    curve(ax, [s.phase; 360], [s.T; s.T(1)], mk{f}, col{f}, names{f});      % periodic: the 360 point repeats 0
end
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); xlim(ax, [0 360]); xticks(ax, 0:90:360); ylim(ax, [0.6 0.95]);
xlabel(ax, 'comb phase \phi (\circ)'); ylabel(ax, 'peak transmission');
legend(ax, 'Location', 'southoutside', 'FontSize', 9);
title(ax, 'peak T vs comb phase', 'FontWeight', 'normal');
title(tl, 'Cylinders in the core vs outside - TM, corr 400, N 80/side');
out = fullfile(root, 'results_from_athena', 'scat_x9_incore_r50_phase', 'incore_r50_vs_sin_period_phase');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150); disp(['wrote ' out '.png']);

%% Figure 2 — in-core only: T (left axis) and mode width (right axis) vs hole radius at 270 deg, Lambda 524
cT = [0.15 0.15 0.15]; cW = [0.72 0.05 0.30];                          % not the fig-1 blue/orange
fig = figure('Visible', 'off', 'Position', [60 60 820 560]);
si = sortrows(inc(abs(inc.phase - 270) < 5 & round(inc.lam) == 524, :), 'r'); [~, iu] = unique(si.r); si = si(iu, :);
yyaxis left; hold on
curve(gca, si.r, si.T, 's', cT, 'peak T, SiO_2 cylinders in core');
yline(Tc, '--', 'Color', cT, 'LineWidth', 1.4, 'DisplayName', sprintf('peak T, no holes (%.3f)', Tc));
ylabel('peak transmission'); ylim([0.86 0.94]); set(gca, 'YColor', cT);
yyaxis right; hold on
curve(gca, si.r, si.w, 'o', cW, 'mode width, SiO_2 cylinders in core');
yline(wc, '--', 'Color', cW, 'LineWidth', 1.4, 'DisplayName', ['mode width, no holes (' sprintf('%.1f', wc) ' \mum)']);
ylabel('spatial mode width FWHM (\mum)'); ylim([14 30]); set(gca, 'YColor', cW);
grid on; box on; xlim([20 120]); xlabel('hole radius r (nm)');
legend('Location', 'southoutside', 'NumColumns', 2, 'FontSize', 9);
title('SiO_2 cylinders in core, \Lambda 524, 270\circ - TM, corr 400, N 80/side', 'FontWeight', 'normal');
out = fullfile(root, 'results_from_athena', 'scat_x9_incore_r50_phase', 'incore_radius_T_width');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150); disp(['wrote ' out '.png']);

function curve(ax, x, y, mk, c, name)
    % pchip interpolation through the points + markers, ONE legend entry (line with marker)
    xx = linspace(min(x), max(x), 300);
    plot(ax, xx, pchip(x, y, xx), '-', 'Color', c, 'LineWidth', 1.6, 'HandleVisibility', 'off');
    plot(ax, x, y, mk, 'Color', c, 'MarkerFaceColor', c, 'MarkerSize', 7, 'HandleVisibility', 'off');
    plot(ax, NaN, NaN, ['-' mk], 'Color', c, 'MarkerFaceColor', c, 'MarkerSize', 7, 'LineWidth', 1.6, 'DisplayName', name);
end

function tb = rows(d)
    % one table row per scatterer result file: period / count / phase / radius / y / T / width from the file name + file
    L = dir(fullfile(d, 'result_N80_TM_avg_Ybox16p0_Zbox8p8_sc*_ff.mat'));
    lam = []; n = []; phase = []; r = []; y = []; T = []; w = [];
    for k = 1:numel(L)
        tok = regexp(L(k).name, 'scR(\d+)_arr(\d+)_X(-?[\d.]+)to(-?[\d.]+)_Y(\d+)to', 'tokens', 'once');
        if isempty(tok), continue, end
        v = str2double(tok); nk = v(2); lam_k = (v(4) - v(3)) / (nk - 1); dx = (v(3) + v(4)) / 2;
        m = load(fullfile(d, L(k).name), 'resonance_transmission', 'fwhm_m');
        lam(end + 1, 1) = lam_k; n(end + 1, 1) = nk; phase(end + 1, 1) = mod(360 * dx / lam_k, 360); %#ok<AGROW>
        r(end + 1, 1) = v(1); y(end + 1, 1) = v(5); T(end + 1, 1) = m.resonance_transmission; w(end + 1, 1) = m.fwhm_m * 1e6; %#ok<AGROW>
    end
    tb = table(lam, n, phase, r, y, T, w);
end
