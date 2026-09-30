% plot_proposal_fieldmaps_n100.m -- research-proposal figure: |E|^2 maps, uniform vs
% inverse-designed TM corr-325 pi-shift Bragg grating, short device (N=100/side).
% Study: runners/sweeps/proposal_fieldmaps_n100.py | Athena job 165464 | 2026-09-30.
% Purpose: show the uniform device radiating and the optimized one radiating less,
% at the same mode width. Two figures (user picks): Top view (XY plane, z=0) and
% Side view (XZ plane, y=0). Each = uniform / optimized maps on ONE shared absolute
% dB scale + a small T(lambda) panel. Planes taken at the recorded point nearest
% each run's own resonance_wavelength_nm. Outlines come from the built scenes
% (outlines_n100.mat, same directory as the results).
% Run: matlab -batch "plot_proposal_fieldmaps_n100"   (env RES_DIR overrides the
% results folder, used for rendering on Athena).

res_dir = getenv('RES_DIR');
if isempty(res_dir)
    res_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_athena', ...
                       'proposal_fieldmaps_n100', 'results');
end
out_dir = fullfile(res_dir, '..');

FILES = {'result_N100_TM_avg_C325_Ybox8p0_Zbox8p8.mat', ...
         ['result_N100_TM_W961_C325_dsh25S66s3_ptw25W964to981_ptn25W640to620_Ybox8p0_' ...
          'Zbox8p8_scR80_arr57_X-14467to15269_Y1900to1900_C325_pair.mat']};
NAMES = {'uniform grating', 'optimized design'};
KEYS  = {'bare', 'design'};
DB_RANGE = 60;                                  % dB range below the (shared or own) peak
ol = load(fullfile(res_dir, 'outlines_n100.mat'));

% ── load both runs: plane at resonance + spectrum ─────────────────────────────
for k = 1:2
    d = load(fullfile(res_dir, FILES{k}));
    lam_res = d.resonance_wavelength_nm;
    R(k).lam = lam_res;
    R(k).T   = d.resonance_transmission;
    R(k).Q   = lam_res / abs(d.spectral_fwhm_nm);
    R(k).W   = d.fwhm_m * 1e6;
    R(k).wl  = d.wl_nm;  R(k).Tspec = d.T;
    for v = {'xy', 'xz_side'}
        f = d.(['field_' v{1}]);
        [~, i] = min(abs(f.lambda_3d * 1e9 - lam_res));
        E = squeeze(f.E_res(:, :, :, i, :));            % [nx n2 3]
        R(k).(v{1}).E2  = sum(abs(E).^2, 3);
        R(k).(v{1}).x   = f.x * 1e6;
        R(k).(v{1}).lam = f.lambda_3d(i) * 1e9;
        if strcmp(v{1}, 'xy'), R(k).(v{1}).v = f.y * 1e6; else, R(k).(v{1}).v = f.z * 1e6; end
    end
    f = d.field_yz_cross;                               % E_res [ny nz nf 3], x = +pitch/4
    [~, i] = min(abs(f.lambda_3d * 1e9 - lam_res));
    E = squeeze(f.E_res);                               % drop the leading x singleton first
    R(k).yz.E2 = sum(abs(squeeze(E(:, :, i, :))).^2, 3);
    R(k).yz.y = f.y * 1e6;  R(k).yz.z = f.z * 1e6;  R(k).yz.x0 = f.x * 1e6;
    fprintf('%s: lam_res %.3f nm, planes at %.3f nm, T %.4f, Q %.0f, W %.2f um\n', ...
            NAMES{k}, lam_res, R(k).xy.lam, R(k).T, R(k).Q, R(k).W);
end

% ── top + side view, 4 variants each (user 2026-09-30): {dB, linear} x
%    {shared normalization (one peak for both devices), own (each device on its own peak)}.
%    dB/shared = the original look (60 dB below the shared peak).
VIEWS = {'xy', 'Top view', 'y (\mum)', 'topview'; 'xz_side', 'Side view', 'z (\mum)', 'sideview'};
for vv = 1:2
  v = VIEWS{vv, 1};
  pk_shared = max([max(R(1).(v).E2(:)), max(R(2).(v).E2(:))]);
  for scl = {'db', 'lin'}
    for nrm = {'shared', 'own'}
      fig = figure('Visible', 'off', 'Position', [40 40 1500 900]);
      tl = tiledlayout(fig, 8, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
      for k = 1:2
        pk = pk_shared;
        if strcmp(nrm{1}, 'own'), pk = max(R(k).(v).E2(:)); end
        A = R(k).(v).E2' / pk;
        if strcmp(scl{1}, 'db')
            img = 10*log10(max(A, 1e-30));  lim = [-DB_RANGE 0];  lab = '|E|^2 (dB)';
        else
            img = A;                        lim = [0 1];          lab = '|E|^2 (linear)';
        end
        ax = nexttile(tl, [3 1]);
        imagesc(ax, R(k).(v).x, R(k).(v).v, img);
        axis(ax, 'xy'); clim(ax, lim); colormap(ax, turbo);
        ylabel(ax, VIEWS{vv, 3}); hold(ax, 'on');
        if strcmp(v, 'xy')
            x = ol.([KEYS{k} '_x']) * 1e6;  hw = ol.([KEYS{k} '_hw']) * 1e6;
            plot(ax, x, hw, 'w-', x, -hw, 'w-', 'LineWidth', 0.4);
            p = ol.([KEYS{k} '_posts']);
            if ~isempty(p), plot(ax, p(:,1)*1e6, p(:,2)*1e6, 'wo', 'MarkerSize', 2); end
        else
            yline(ax, [-0.175 0.175], 'w-', 'LineWidth', 0.4);   % 350 nm core
        end
        title(ax, sprintf('%s:  T %.3f  (radiation loss %.1f%%),  Q %.0f,  mode FWHM %.1f \\mum', ...
              NAMES{k}, R(k).T, 100*(1 - R(k).T), R(k).Q, R(k).W), 'FontWeight', 'normal');
        if k == 2, xlabel(ax, 'x (\mum)'); end
        cb = colorbar(ax); cb.Label.String = lab;
      end
      ax = nexttile(tl, [2 1]); hold(ax, 'on');
      for k = 1:2
          plot(ax, R(k).wl - R(k).lam, R(k).Tspec, 'LineWidth', 1.4, ...
               'DisplayName', sprintf('%s (\\lambda_0 %.2f nm)', NAMES{k}, R(k).lam));
      end
      xlim(ax, [-3 3]); ylim(ax, [0 1]); grid(ax, 'on'); box(ax, 'on');
      xlabel(ax, '\lambda - \lambda_0 (nm)'); ylabel(ax, 'T');
      legend(ax, 'Location', 'northwest', 'Box', 'off');
      nlab = 'shared scale';  if strcmp(nrm{1}, 'own'), nlab = 'each on its own peak'; end
      title(tl, sprintf(['%s, |E|^2 at resonance (%s): TM ' char(960) '-shift Bragg grating, ' ...
            'corrugation 325 nm, N = 100 periods/side (short device)'], VIEWS{vv, 2}, nlab));
      base = fullfile(out_dir, sprintf('proposal_fieldmaps_n100_%s_%s_%s', VIEWS{vv, 4}, scl{1}, nrm{1}));
      savefig(fig, [base '.fig']);
      exportgraphics(fig, [base '.png'], 'Resolution', 200);
      fprintf('saved %s.{fig,png}\n', base);
      close(fig);
    end
  end
end

% ── cross section (YZ plane at x = +pitch/4, inside the pi-shift cavity) ────────
top = 10*log10(max([max(R(1).yz.E2(:)), max(R(2).yz.E2(:))]));
fig = figure('Visible', 'off', 'Position', [40 40 1300 560]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:2
    ax = nexttile(tl);
    imagesc(ax, R(k).yz.y, R(k).yz.z, 10*log10(max(R(k).yz.E2', 1e-30)));
    axis(ax, 'xy', 'equal', 'tight'); clim(ax, [top - DB_RANGE, top]); colormap(ax, turbo);
    x = ol.([KEYS{k} '_x']) * 1e6;  hw = ol.([KEYS{k} '_hw']) * 1e6;
    w = hw(find(x <= R(k).yz.x0, 1, 'last'));        % core half-width at the cut
    rectangle(ax, 'Position', [-w -0.175 2*w 0.35], 'EdgeColor', 'w', 'LineWidth', 0.8);
    xlabel(ax, 'y (\mum)'); ylabel(ax, 'z (\mum)');
    title(ax, sprintf('%s: radiation loss %.1f%%, Q %.0f', NAMES{k}, 100*(1 - R(k).T), R(k).Q), ...
          'FontWeight', 'normal');
end
cb = colorbar(ax); cb.Label.String = '|E|^2 (dB)';
title(tl, sprintf(['Cross section at the cavity (x = %.2f \\mum), |E|^2 at resonance: ' ...
      'TM \x03c0-shift Bragg grating, corrugation 325 nm, N = 100/side'], R(1).yz.x0));
base = fullfile(out_dir, 'proposal_fieldmaps_n100_crosssection');
savefig(fig, [base '.fig']);
exportgraphics(fig, [base '.png'], 'Resolution', 200);
fprintf('saved %s.{fig,png}\n', base);
