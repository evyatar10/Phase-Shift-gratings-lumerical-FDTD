% plot_axis_r60_summary.m — SiO2 cylinders (one per period on the axis, r 60) vs SiN cylinders outside (r 110): summary figures.
% 2026-09-16 later: phase circle + radius series moved to the r 60 OPTIMUM period Lambda 527 (scat_x21 job 151686 Athena,
% scat_x22 job 90966 IGUM); Lambda 524 rows kept in the data but no longer drawn.
% Study dirs: results_from_athena/{scat_x18_axis_r60_period_c536 (job 151509), scat_x11_incore_axis_r30_40_60 (151333),
% scat_x10_incore_r50_axis (151320)} + results_from_igum/{scat_x19_axis_r60_phase524 (90726), scat_x12_incore_axis_r80_110
% (90593; r 110 via Athena 151353)}; SiN rows: scat_p_antineedle, scat_r_aim536, scat_s_refine, scat_x4_incore_below530,
% IGUM scat_aim_extend. Plot 2026-09-16.
% Fig 1: (left) peak T vs comb period at 270 deg; (right) peak T vs comb phase — axis r 60 at Lambda 524 and 536,
%        SiN r 110 at Lambda 536 (its 531 optimum has only the 270 point). pchip through the points, no fits.
% Fig 2: peak T (left axis) and spatial mode width (right axis) vs hole radius, axis single hole at Lambda 524 / 270.
% Device: TM, W 800, corr 400, h 350, pitch 516.83, N 80/side, box y 16, 20 nm / 1501 pts, 31 sites.
root = fullfile(fileparts(mfilename('fullpath')), '..', '..');
res = @(varargin) fullfile(root, varargin{:}, 'results');
ctrl = load(fullfile(res('results_from_athena', 'scat_h_retrocomb'), 'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'), 'resonance_transmission', 'fwhm_m');
Tc = ctrl.resonance_transmission; wc = ctrl.fwhm_m * 1e6;

ax1 = [rows(res('results_from_athena', 'scat_x18_axis_r60_period_c536')); rows(res('results_from_igum', 'scat_x19_axis_r60_phase524')); ...
       rows(res('results_from_athena', 'scat_x10_incore_r50_axis')); rows(res('results_from_athena', 'scat_x11_incore_axis_r30_40_60')); ...
       rows(res('results_from_igum', 'scat_x12_incore_axis_r80_110')); ...
       rows(res('results_from_athena', 'scat_x21_axis_l527_athena')); rows(res('results_from_igum', 'scat_x22_axis_l527_igum')); ...
       rows(res('results_from_athena', 'scat_x23_axis_l527_r50phase_r30eq'))];
sin = [rows(res('results_from_athena', 'scat_p_antineedle')); rows(res('results_from_athena', 'scat_r_aim536')); ...
       rows(res('results_from_athena', 'scat_s_refine')); rows(res('results_from_athena', 'scat_x4_incore_below530')); ...
       rows(res('results_from_igum', 'scat_aim_extend'))];
ax1 = ax1(ax1.n == 31 & ax1.y == 0, :);  sin = sin(sin.n == 31 & sin.y == 1800 & sin.r == 110, :);
r60 = ax1(ax1.r == 60, :);  r50 = ax1(ax1.r == 50, :);
cs = [0 0.45 0.74]; ci = [0.85 0.33 0.10]; ci2 = [0.93 0.60 0.30]; grey = [0.45 0.45 0.45];

%% Figure 1 — period scan at 270 deg (left), phase circles (right)
fig = figure('Visible', 'off', 'Position', [60 60 1400 520]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on');
yline(ax, Tc, '--', 'Color', grey, 'LineWidth', 1.2, 'DisplayName', sprintf('no cylinders, T %.3f', Tc));
s = uniq(sin(abs(sin.phase - 270) < 5, :), 'lam');  curve(ax, s.lam, s.T, 'o', cs, 'SiN cylinders outside, r 110');
s = uniq(r60(abs(r60.phase - 270) < 5, :), 'lam');  curve(ax, s.lam, s.T, 's', ci, 'SiO_2 cylinders, r 60');
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); ylim(ax, [0.6 0.95]);
xlabel(ax, 'comb period \Lambda (nm)'); ylabel(ax, 'peak transmission');
legend(ax, 'Location', 'southoutside', 'FontSize', 9);
title(ax, 'phase 270\circ: peak T vs comb period', 'FontWeight', 'normal');

ax = nexttile(tl); hold(ax, 'on');
yline(ax, Tc, '--', 'Color', grey, 'LineWidth', 1.2, 'DisplayName', sprintf('no cylinders, T %.3f', Tc));
circ(ax, sin, 536, 'o', cs,  'SiN cylinders outside, r 110, \Lambda 536');
circ(ax, r60, 527, 's', ci,  'SiO_2 cylinders, r 60, \Lambda 527');
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); xlim(ax, [0 360]); xticks(ax, 0:90:360); ylim(ax, [0.6 0.95]);
xlabel(ax, 'comb phase \phi (\circ)'); ylabel(ax, 'peak transmission');
legend(ax, 'Location', 'southoutside', 'FontSize', 9);
title(ax, 'peak T vs comb phase', 'FontWeight', 'normal');
title(tl, 'Cylinders in the core vs outside - TM, corr 400, N 80/side');
out = fullfile(root, 'results_from_athena', 'scat_x21_axis_l527_athena', 'axis_r60_vs_sin_period_phase');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150); disp(['wrote ' out '.png']);

%% Figure 2 — axis single hole: T (left axis) and mode width (right axis) vs radius, Lambda 524 / 270
cT = [0.15 0.15 0.15]; cW = [0.72 0.05 0.30];
fig = figure('Visible', 'off', 'Position', [60 60 820 560]);
si = uniq(ax1(abs(ax1.phase - 270) < 5 & round(ax1.lam) == 527, :), 'r');
yyaxis left; hold on
curve(gca, si.r, si.T, 's', cT, 'peak T, SiO_2 cylinders');
yline(Tc, '--', 'Color', cT, 'LineWidth', 1.4, 'DisplayName', sprintf('peak T, no holes (%.3f)', Tc));
ylabel('peak transmission'); ylim([0.85 0.94]); set(gca, 'YColor', cT);
yyaxis right; hold on
curve(gca, si.r, si.w, 'o', cW, 'mode width, SiO_2 cylinders');
yline(wc, '--', 'Color', cW, 'LineWidth', 1.4, 'DisplayName', ['mode width, no holes (' sprintf('%.1f', wc) ' \mum)']);
ylabel('spatial mode width FWHM (\mum)'); ylim([14 30]); set(gca, 'YColor', cW);
grid on; box on; xlim([20 120]); xlabel('hole radius r (nm)');
legend('Location', 'southoutside', 'NumColumns', 2, 'FontSize', 9);
title('SiO_2 cylinders, \Lambda 527, 270\circ - TM, corr 400, N 80/side', 'FontWeight', 'normal');
out = fullfile(root, 'results_from_athena', 'scat_x21_axis_l527_athena', 'axis_radius_T_width');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150); disp(['wrote ' out '.png']);

function circ(ax, t, lam, mk, c, name)
    s = t(round(t.lam) == lam & abs(t.phase - round(t.phase / 90) * 90) < 5, :);
    [~, iu] = unique(round(s.phase / 90) * 90); s = sortrows(s(iu, :), 'phase');
    curve(ax, [s.phase; 360], [s.T; s.T(1)], mk, c, name);            % periodic: the 360 point repeats 0
end

function s = uniq(t, key)
    t = sortrows(t, key); [~, iu] = unique(t.(key)); s = t(iu, :);
end

function curve(ax, x, y, mk, c, name)
    xx = linspace(min(x), max(x), 300);
    plot(ax, xx, pchip(x, y, xx), '-', 'Color', c, 'LineWidth', 1.6, 'HandleVisibility', 'off');
    plot(ax, x, y, mk, 'Color', c, 'MarkerFaceColor', c, 'MarkerSize', 7, 'HandleVisibility', 'off');
    plot(ax, NaN, NaN, ['-' mk], 'Color', c, 'MarkerFaceColor', c, 'MarkerSize', 7, 'LineWidth', 1.6, 'DisplayName', name);
end

function tb = rows(d)
    L = dir(fullfile(d, 'result_N80_TM_avg_Ybox16p0_Zbox8p8_sc*_ff.mat'));
    lam = []; n = []; phase = []; r = []; y = []; T = []; w = [];
    for k = 1:numel(L)
        tok = regexp(L(k).name, 'scR(\d+)_arr(\d+)_X(-?[\d.]+)to(-?[\d.]+)_Y(-?\d+)to', 'tokens', 'once');
        if isempty(tok), continue, end
        v = str2double(tok); nk = v(2); lam_k = (v(4) - v(3)) / (nk - 1); dx = (v(3) + v(4)) / 2;
        m = load(fullfile(d, L(k).name), 'resonance_transmission', 'fwhm_m');
        lam(end + 1, 1) = lam_k; n(end + 1, 1) = nk; phase(end + 1, 1) = mod(360 * dx / lam_k, 360); %#ok<AGROW>
        r(end + 1, 1) = v(1); y(end + 1, 1) = abs(v(5)); T(end + 1, 1) = m.resonance_transmission; w(end + 1, 1) = m.fwhm_m * 1e6; %#ok<AGROW>
    end
    tb = table(lam, n, phase, r, y, T, w);
end
