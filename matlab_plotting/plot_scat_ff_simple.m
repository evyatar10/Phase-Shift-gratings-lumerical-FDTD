% plot_scat_ff_simple.m
% Study: scat_c_response (job 120976) + air_trench_dscan (job 124379)
% Date: 2026-08-09
% Purpose: few representative far-field curves (side monitor, 1D vs ux), simple:
%   panel 1 = bare device vs best/worst single scatterer (own control, box y=6.8);
%   panel 2 = bare device vs air trench d=1.8um (own control, box y=16).

ROOT = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results_from_athena');
OUT_DIR = fullfile(ROOT, 'scat_c_response');
DB_FLOOR = -40;

sc = @(t) fullfile(ROOT, 'scat_c_response', 'results', ...
    ['result_N80_TM_avg_Ybox6p8_Zbox8p8' t '_ff.mat']);
tr = @(t) fullfile(ROOT, 'air_trench_dscan', 'results', ...
    ['result_N80_TM_avg_Ybox16p0_Zbox8p8' t '_ff.mat']);

panels = { ...
  {sc(''), 'no scatterer'; ...
   sc('_scR80_arr1_X135to135_Y700_pair'),   'scatterer, best position (x=+135 nm)'; ...
   sc('_scR80_arr1_X-945to-945_Y700_pair'), 'scatterer, worst position (x=-945 nm)'}, ...
   'Single SiN scatterer pair (r=80 nm, y=\pm0.7 \mum): only the middle changes'; ...
  {tr(''), 'no trench'; ...
   tr('_scRECT_L84000xW800_X0_Y1800_pair_hole'), 'air trench at d=1.8 \mum'}, ...
   'Air trench (extended along the device): the side lobes are removed'};

fig = figure('Visible', 'off', 'Position', [80 80 880 680]);
tl = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for p = 1:2
    ax = nexttile(tl);
    hold(ax, 'on');
    files = panels{p, 1};
    cols = [0 0 0; 0.00 0.45 0.74; 0.85 0.33 0.10];
    ref = [];
    for k = 1:size(files, 1)
        d = load(files{k, 1}, 'farfield_side', 'resonance_transmission');
        cut = sum(d.farfield_side.E2, 2);          % E2 rows = ux, sum over uy
        if k == 1; ref = max(cut); end
        plot(ax, d.farfield_side.ux, 10*log10(max(cut/ref, 10^(DB_FLOOR/10))), ...
            'Color', cols(k, :), 'LineWidth', 1.2 + 0.5*(k==1), ...
            'DisplayName', sprintf('%s  (T=%.2f)', files{k, 2}, d.resonance_transmission));
    end
    grid(ax, 'on');
    xlabel(ax, 'u_x  (emission direction: 0 = sideways, \pm1 = along the waveguide)');
    ylabel(ax, 'radiated |E|^2  [dB]');
    ylim(ax, [DB_FLOOR 3]);
    title(ax, panels{p, 2});
    legend(ax, 'Location', 'south', 'FontSize', 9);
    text(ax, -0.98, 2, 'side lobe', 'FontSize', 8, 'HorizontalAlignment', 'center');
    text(ax,  0.98, 2, 'side lobe', 'FontSize', 8, 'HorizontalAlignment', 'center');
end
title(tl, 'Representative far fields (side monitor) — TM corr 400, W800, N=80, \lambda_{res} 1558.6 nm');

set(fig, 'Visible', 'on');
savefig(fig, fullfile(OUT_DIR, 'scat_ff_simple.fig'));
set(fig, 'Visible', 'off');
exportgraphics(fig, fullfile(OUT_DIR, 'scat_ff_simple.png'), 'Resolution', 150);
disp(fullfile(OUT_DIR, 'scat_ff_simple.png'));
