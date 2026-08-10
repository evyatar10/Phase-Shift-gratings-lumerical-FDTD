% plot_scat_r_aim536.m
% Study: results_from_athena/scat_r_aim536 (stage R, job 130091, 2026-08-10)
%        + results_from_igum/scat_q_r80phase (stage Q, IGUM job 51285)
% Purpose: verdict figure — phase circles at Lambda=536 (aim-corrected, r=110)
%   vs Lambda=545 (stage P, r=110) vs r=80 IGUM points; T and needle-bin power.
%   Headline: 536/270deg T=0.8928 (+0.0077, 4.3x floor), needle x0.54.

ROOT = fileparts(fileparts(mfilename('fullpath')));
RA = fullfile(ROOT, 'results_from_athena');
ctrl = load(fullfile(RA, 'scat_h_retrocomb', 'results', ...
    'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'));
mask = abs(ctrl.farfield_side.ux) > 0.94;
N0 = sum(ctrl.farfield_side.E2(mask, :), 'all');
T0 = ctrl.resonance_transmission;

% {results dir, tag list (x-range), degrees, label, color}
sets = { ...
  fullfile(RA, 'scat_r_aim536', 'results'), 'scR110_arr31', ...
    {'X-8040to8040','X-7906to8174','X-7772to8308','X-7638to8442'}, [0 90 180 270], ...
    '\Lambda=536 nm, r=110 (aim-corrected)', [0 0.45 0.74]; ...
  fullfile(RA, 'scat_p_antineedle', 'results'), 'scR110_arr31', ...
    {'X-8175to8175','X-8039to8311','X-7902to8448','X-7766to8584'}, [0 90 180 270], ...
    '\Lambda=545 nm, r=110 (stage P)', [0.85 0.33 0.10]; ...
  fullfile(ROOT, 'results_from_igum', 'scat_q_r80phase', 'results'), 'scR80_arr31', ...
    {'X-7766to8584','X-7698to8652'}, [270 315], ...
    '\Lambda=545 nm, r=80 (IGUM)', [0.47 0.67 0.19]};

fig = figure('Visible', 'off', 'Position', [80 80 900 640]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile(tl); hold(ax1, 'on');
ax2 = nexttile(tl); hold(ax2, 'on');
for s = 1:size(sets, 1)
    [dirp, rtag, tags, degs, lab, col] = sets{s, :};
    [T, ndl] = deal(zeros(1, numel(tags)));
    for k = 1:numel(tags)
        fl = dir(fullfile(dirp, sprintf('*%s_%s_*_ff.mat', rtag, tags{k})));
        d = load(fullfile(fl(1).folder, fl(1).name));
        T(k) = d.resonance_transmission;
        ndl(k) = sum(d.farfield_side.E2(mask, :), 'all') / N0;
    end
    plot(ax1, degs, T, 'o-', 'Color', col, 'LineWidth', 1.4, 'DisplayName', lab);
    plot(ax2, degs, ndl, 'o-', 'Color', col, 'LineWidth', 1.4, 'DisplayName', lab);
end
yline(ax1, T0, 'k:', 'control T=0.885', 'FontSize', 8, 'HandleVisibility', 'off');
ylabel(ax1, 'peak T');
title(ax1, 'A — transmission vs comb phase: aim-corrected circle crosses ABOVE control at 270\circ');
legend(ax1, 'Location', 'southwest', 'FontSize', 8);
yline(ax2, 1, 'k:', 'control', 'FontSize', 8, 'HandleVisibility', 'off');
ylabel(ax2, 'grazing-lobe power (\times ctrl)');
title(ax2, 'B — side-lobe power vs comb phase');
for ax = [ax1 ax2]
    grid(ax, 'on');
    xlabel(ax, 'comb phase 2\pi\deltax/\Lambda  [deg]');
    xlim(ax, [-10 325]);
end
title(tl, {'Anti-needle comb phase circles — measured (jobs 129989 / 130091 / IGUM 51285)'; ...
    'TM corr 400, W800, N=80, 31 SiN posts h350, d=1.8 \mum — best: \Lambda=536, 270\circ: T 0.8928 (+0.0077), lobes \times0.54'});

set(fig, 'Visible', 'on');
savefig(fig, fullfile(RA, 'scat_r_aim536', 'scat_r_aim536.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(RA, 'scat_r_aim536', 'scat_r_aim536.png'), 'Resolution', 150);
disp(fullfile(RA, 'scat_r_aim536', 'scat_r_aim536.png'));
