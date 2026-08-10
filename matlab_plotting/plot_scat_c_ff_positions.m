% plot_scat_c_ff_positions.m
% Study: results_from_athena/scat_c_response + scat_c3_ygrid + scat_e_validate
%        (stage-C response matrix job 120976; C3 y-grid 121525/121614; round-10 121929)
% Date: 2026-08-09
% Purpose: far-field comparison of single SiN scatterer-pair configs on TM corr-400
%   W800 N=80. Two figure sets:
%   'pos'  — x-position sweep at y=700 nm (r=80): best/inert/enhancer/far.
%   'dist' — standoff-distance ladder: r=80 best site per row y=0.97-1.51 um +
%            giant r=400 at y=1.0/1.7 um (trench-like distances).
%   Each set: 1D |E|^2 vs ux (sum over uy) overlaid for both monitors + 2D
%   direction-cosine maps, one subplot per config, shared dB scale.

ROOT = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results_from_athena');
OUT_DIR = fullfile(ROOT, 'scat_c_response');
DB_FLOOR = -60;      % 2D map colormap floor, dB rel. ctrl peak
DB_FLOOR_1D = -30;   % 1D-cut y-axis floor

crsp = @(t) fullfile(ROOT, 'scat_c_response', 'results', ...
    ['result_N80_TM_avg_Ybox6p8_Zbox8p8' t '_ff.mat']);
c3   = @(t) fullfile(ROOT, 'scat_c3_ygrid', 'results', ...
    ['result_N80_TM_avg_Ybox6p8_Zbox8p8' t '_ff.mat']);
ev   = @(t) fullfile(ROOT, 'scat_e_validate', 'results', ...
    ['result_N80_TM_W800_Ybox6p8_Zbox8p8' t '_ff.mat']);
sh   = @(t) fullfile(ROOT, 'scat_h_retrocomb', 'results', ...
    ['result_N80_TM_avg_Ybox16p0_Zbox8p8' t '_ff.mat']);
so   = @(t) fullfile(ROOT, 'scat_o_comb1800', 'results', ...
    ['result_N80_TM_avg_Ybox16p0_Zbox8p8' t '_ff.mat']);
sp   = @(t) fullfile(ROOT, 'scat_p_antineedle', 'results', ...
    ['result_N80_TM_avg_Ybox16p0_Zbox8p8' t '_Y1800to1800_C400_pair_ff.mat']);
su   = @(t) fullfile(ROOT, 'scat_u_flushcomb', 'results', ...
    ['result_N80_TM_avg_Ybox16p0_Zbox8p8' t '_ff.mat']);
sw   = @(t) fullfile(ROOT, 'scat_w_dscan', 'results', ...
    ['result_N80_TM_avg_Ybox16p0_Zbox8p8' t '_ff.mat']);

% Each set: ctrl file + config files (short labels; T and dP_FF appended from data)
sets = struct( ...
    'suffix', {'', '_dist', '_ray', '_comb', '_stagep', '_final3'}, ...
    'tag', {'x-position sweep, SiN pillar pair r=80 nm, y=\pm700 nm', ...
            'standoff ladder: r=80 best-x per row + giant r=400 at x=0', ...
            'on-ray sites (\theta\approx11-12\circ), r=80, y=\pm700 nm', ...
            'SiN comb \Lambda=551 nm (angle-matched), r=110, 151 posts/row', ...
            'stage-P anti-needle comb: 31 posts h350, d=1.8 \mum (job 129989)', ...
            'final devices: flush air trench vs best comb (jobs 130184/130179)'}, ...
    'ctrl', {crsp(''), c3(''), crsp(''), sh(''), sh(''), sh('')}, ...
    'files', { ...
      {crsp('_scR80_arr1_X135to135_Y700_pair'),   'x=+135 nm'; ...
       crsp('_scR80_arr1_X-135to-135_Y700_pair'), 'x=-135 nm'; ...
       crsp('_scR80_arr1_X-270to-270_Y700_pair'), 'x=-270 nm'; ...
       crsp('_scR80_arr1_X-945to-945_Y700_pair'), 'x=-945 nm'; ...
       crsp('_scR80_arr1_X5535to5535_Y700_pair'), 'x=+5535 nm'}, ...
      {c3('_scR80_arr1_X675to675_Y970_pair'),     'r80, y=0.97\mum'; ...
       c3('_scR80_arr1_X540to540_Y1240_pair'),    'r80, y=1.24\mum'; ...
       c3('_scR80_arr1_X1080to1080_Y1510_pair'),  'r80, y=1.51\mum'; ...
       ev('_scR400_arr1_X0to0_Y1700to1700_pair'), 'r400, y=1.7\mum'; ...
       ev('_scR400_arr1_X0to0_Y1000to1000_pair'), 'r400, y=1.0\mum'}, ...
      {crsp('_scR80_arr1_X3240to3240_Y700_pair'),   'x=+3.24\mum (12.2\circ)'; ...
       crsp('_scR80_arr1_X3375to3375_Y700_pair'),   'x=+3.38\mum (11.7\circ, on ray)'; ...
       crsp('_scR80_arr1_X3510to3510_Y700_pair'),   'x=+3.51\mum (11.3\circ, on ray)'; ...
       crsp('_scR80_arr1_X-3375to-3375_Y700_pair'), 'x=-3.38\mum (backward ray)'; ...
       crsp('_scR80_arr1_X4050to4050_Y700_pair'),   'x=+4.05\mum (9.8\circ, off ray)'}, ...
      {sh('_scR110_arr151_X-41325to41325_Y3000to3000_C400_pair'),  '1 row, d=3.0\mum'; ...
       sh('_scR110_arr151_X-41325to41325_Y5100to5100_C400_pair'),  '1 row, d=5.1\mum'; ...
       sh('_scR110_arr302_X-41325to41325_Y3000to5680_C400_pair'),  '2 rows, d=3.0+5.68\mum'; ...
       so('_scR110_arr151_X-41325to41325_Y1800to1800_C400_pair'),  '1 row, d=1.8\mum (core height)'; ...
       so('_scR110_arr151_X-41325to41325_Y1800to1800_C400_pair_H12000'), '1 row, d=1.8\mum, full depth'}, ...
      {sp('_scR110_arr31_X-8175to8175'),  '\Lambda=545, \deltax=0'; ...
       sp('_scR110_arr31_X-8039to8311'),  '\Lambda=545, \deltax=136 (90\circ)'; ...
       sp('_scR110_arr31_X-7902to8448'),  '\Lambda=545, \deltax=273 (180\circ)'; ...
       sp('_scR110_arr31_X-7766to8584'),  '\Lambda=545, \deltax=409 (270\circ)'; ...
       sp('_scR80_arr31_X-8175to8175'),   '\Lambda=545, \deltax=0, r=80'}, ...
      {su('_scRECT_L84000xW800_arr1_X0to0_Y1800to1800_C400_pair_hole_Zminm3975'), ...
           'flush air trench (BOX\rightarrowSiN top)'; ...
       sw('_scR82_arr31_X-7582to8348_Y1500to1500_C400_pair'), ...
           'comb d=1.5 \mum, r=82 (best pillars)'}});

mons = {'farfield_side', 'Side monitor', 'farfield_top', 'Top monitor'};

for s = 1:numel(sets)
    ctrl = load(sets(s).ctrl);
    n = size(sets(s).files, 1);
    D = cell(1, n);
    for k = 1:n
        D{k} = load(sets(s).files{k, 1});
    end
    all_d = [{ctrl}, D];

    P0 = sum(ctrl.farfield_side.E2(:)) + sum(ctrl.farfield_top.E2(:));
    labels = cell(1, n + 1);
    labels{1} = sprintf('no scatterer (T=%.3f)', ctrl.resonance_transmission);
    short = [{'no scatterer'}, sets(s).files(:, 2)'];
    for k = 1:n
        Pk = sum(D{k}.farfield_side.E2(:)) + sum(D{k}.farfield_top.E2(:));
        labels{k+1} = sprintf('%s (T=%.3f, \\DeltaP_{FF} %+.0f%%)', ...
            sets(s).files{k, 2}, D{k}.resonance_transmission, 100*(Pk/P0 - 1));
    end
    dev_line = sprintf('TM corr 400, W800, N=80/side — %s, \\lambda_{res} %.2f nm', ...
        sets(s).tag, ctrl.resonance_wavelength_nm);
    cols = [0 0 0; lines(n)];

    % ---- 1D cuts vs ux, both monitors (E2 rows = ux, cols = uy; verified) ----
    fig1 = figure('Visible', 'off', 'Position', [80 80 900 700]);
    tl = tiledlayout(fig1, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    for m = 1:2
        ax = nexttile(tl);
        hold(ax, 'on');
        fld = mons{2*m-1};
        ref = max(sum(ctrl.(fld).E2, 2));             % ctrl 1D-cut peak = 0 dB
        top_db = 0;
        for k = 1:n+1
            cut = sum(all_d{k}.(fld).E2, 2);
            top_db = max(top_db, 10*log10(max(cut/ref)));
            plot(ax, all_d{k}.(fld).ux, 10*log10(max(cut/ref, 10^(DB_FLOOR_1D/10))), ...
                'Color', cols(k,:), 'LineWidth', 1.1 + 0.7*(k==1));
        end
        grid(ax, 'on');
        xlabel(ax, 'u_x');
        ylabel(ax, '\Sigma_{u_y} |E|^2  [dB rel. ctrl peak]');
        ylim(ax, [DB_FLOOR_1D ceil(top_db) + 3]);
        title(ax, mons{2*m});
        if m == 1
            legend(ax, labels, 'Location', 'south', 'FontSize', 8, 'NumColumns', 2);
        end
    end
    title(tl, {'Far field vs u_x — single scatterer-pair configs'; dev_line});

    % ---- 2D maps, one subplot per config, per monitor -----------------------
    figs = {fig1, [], []};
    for m = 1:2
        fld = mons{2*m-1};
        n_tiles = n + 1;
        rows2 = ceil(n_tiles / 3);
        cols2 = min(3, n_tiles);
        fig = figure('Visible', 'off', 'Position', [80 80 60 + 370*cols2, 80 + 280*rows2]);
        tl2 = tiledlayout(fig, rows2, cols2, 'TileSpacing', 'compact', 'Padding', 'compact');
        ref = max(ctrl.(fld).E2(:));                  % ctrl map peak = 0 dB
        top_db = 0;
        for k = 1:n+1
            top_db = max(top_db, 10*log10(max(all_d{k}.(fld).E2(:))/ref));
        end
        for k = 1:n+1
            ax = nexttile(tl2);
            ff = all_d{k}.(fld);
            % rows of E2 = ux -> transpose so ux is horizontal (project convention)
            imagesc(ax, ff.ux, ff.uy, 10*log10(max(ff.E2'/ref, 10^(DB_FLOOR/10))));
            axis(ax, 'xy', 'square');
            clim(ax, [DB_FLOOR ceil(top_db)]);
            xlabel(ax, 'u_x');
            ylabel(ax, 'u_y');
            title(ax, short{k}, 'FontSize', 9);
        end
        cb = colorbar(ax);
        cb.Layout.Tile = 'east';
        cb.Label.String = '|E|^2 [dB rel. ctrl peak]';
        title(tl2, {sprintf('%s far field — scatterer-pair comparison', mons{2*m}); dev_line});
        figs{1 + m} = fig;
    end

    % ---- Save (.fig must be saved Visible on, else it opens blank) ----------
    names = {['scat_c_ff' sets(s).suffix '_1dcuts'], ...
             ['scat_c_ffmap' sets(s).suffix '_side'], ...
             ['scat_c_ffmap' sets(s).suffix '_top']};
    for k = 1:3
        set(figs{k}, 'Visible', 'on');
        savefig(figs{k}, fullfile(OUT_DIR, [names{k} '.fig']));
        set(figs{k}, 'Visible', 'off');
        exportgraphics(figs{k}, fullfile(OUT_DIR, [names{k} '.png']), 'Resolution', 150);
    end
end
disp('saved to:');
disp(OUT_DIR);
