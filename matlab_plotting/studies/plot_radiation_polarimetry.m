% plot_radiation_polarimetry.m — Phase-1 gating diagnostic readout.
% Loads results_from_athena/tm_radiation_polarimetry/results/result_*.mat and
% answers, in order (docs/tm_loss_program_phase0_2026-07-05.md):
%   G1 energy audit: 2*(P_side + P_top) flux vs 1-T-R  (>=60% or STOP)
%   G2 direction:    in-plane share = side/(side+top)  (P1: >=70% in-plane)
%   G3 polarization: f_TE on the side monitor          (reflector design gate)
%   G4 attribution:  |prof_total(x)| arm vs cavity     (vs k-space ~30/70)
%   G5 lobe angle:   far-field E2(ux) side-monitor cut
% Rows: 1 W800 map / 2 champion 1050 / 3 off-peak ctl / 4 y8.0 box ctl /
%       5 W1400 / 6 accurate. Headless-safe.
clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
d_res = fullfile(proj, 'results_from_athena', 'tm_radiation_polarimetry', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'tm_radiation_polarimetry');

files = { ...
    'result_N80_TM_avg_Ybox6p8_Zbox8p8.mat', ...                 % 0 control (no ff)
    'result_N80_TM_W800_Ybox6p8_Zbox8p8_ff.mat', ...             % 1 baseline map
    'result_N80_TM_W1050_Ybox6p8_Zbox8p8_ff.mat', ...            % 2 champion
    'result_N80_D0p02_TM_avg_Ybox6p8_Zbox8p8_ff.mat', ...        % 3 off-peak ctl
    'result_N80_TM_avg_Ybox8p0_Zbox8p8_ff.mat', ...              % 4 big-box ctl
    'result_N80_TM_W1400_Ybox6p8_Zbox8p8_ff.mat', ...            % 5 ladder reversal
    'result_N80_D0p04_TM_avg_Ybox6p8_Zbox8p8_ff.mat'};           % 6 accurate
lb = {'control (ff off)', 'W800 baseline', 'W1050 champion', 'off-peak ctl', ...
      'y=8.0 box ctl', 'W1400 reversal', 'accurate mesh'};

n = numel(files);
r = cell(1, n);
for k = 1:n
    fp = fullfile(d_res, files{k});
    if exist(fp, 'file'), r{k} = load(fp); else, fprintf('MISSING: %s\n', files{k}); end
end

fprintf('\n=== G1/G2/G3 table (symmetry factor 2 on side and top) ===\n');
fprintf('%-18s %8s %8s %8s %8s %8s %8s\n', 'row', 'loss', '2*side', '2*top', ...
        'audit', 'inplane', 'f_TE');
for k = 1:n
    d = r{k}; if isempty(d), continue; end
    [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
    loss = 1 - d.resonance_transmission - d.R(i);
    if isfield(d, 'polarimetry_side') && isfield(d.polarimetry_side, 'flux_norm')
        ps = 2 * double(d.polarimetry_side.flux_norm);
        pt = 2 * double(d.polarimetry_top.flux_norm);
        fte = double(d.polarimetry_side.P_te) / ...
              (double(d.polarimetry_side.P_te) + double(d.polarimetry_side.P_tm));
        fprintf('%-18s %8.4f %8.4f %8.4f %7.0f%% %7.0f%% %8.2f\n', lb{k}, loss, ...
                ps, pt, 100 * (ps + pt) / loss, 100 * ps / (ps + pt), fte);
    else
        fprintf('%-18s %8.4f   (no farfield data)\n', lb{k}, loss);
    end
end

% G4: arm-vs-cavity attribution profiles + G5 far-field lobes
fig = figure('Visible', 'off', 'Position', [40 40 1200 800]);
rows_plot = [2 3 6];   % W800 / W1050 / W1400 (1-based file idx: 2,3,6)
subplot(2, 2, 1); hold on; grid on;
for k = rows_plot
    d = r{k}; if isempty(d) || ~isfield(d, 'polarimetry_side'), continue; end
    x_um = double(d.polarimetry_side.x) * 1e6;
    plot(x_um, double(d.polarimetry_side.prof_total), 'DisplayName', lb{k});
end
xlabel('x (\mum)'); ylabel('dP/dx through side plane');
title('G4: side-radiation origin along the device'); legend('Location', 'best');

subplot(2, 2, 2); hold on; grid on;
d = r{2};
if ~isempty(d) && isfield(d, 'polarimetry_side')
    x_um = double(d.polarimetry_side.x) * 1e6;
    plot(x_um, double(d.polarimetry_side.prof_tm), 'DisplayName', 'TM-pol part');
    plot(x_um, double(d.polarimetry_side.prof_te), 'DisplayName', 'TE-pol part');
end
xlabel('x (\mum)'); ylabel('dP/dx');
title('G3: polarization split along x (W800)'); legend('Location', 'best');

subplot(2, 2, 3); hold on; grid on;
for k = rows_plot
    d = r{k}; if isempty(d) || ~isfield(d, 'farfield_side'), continue; end
    ff = d.farfield_side; if ~isfield(ff, 'E2'), continue; end
    E2 = double(ff.E2); ux = double(ff.ux);
    [~, j0] = min(abs(double(ff.uy)));            % uy~0 cut
    plot(ux, E2(:, j0) / max(E2(:, j0)), 'DisplayName', lb{k});
end
xlabel('ux (direction cosine along x)'); ylabel('E^2 (norm.)');
title('G5: side far-field lobe cut (uy=0)'); legend('Location', 'best');

subplot(2, 2, 4); hold on; grid on;
d = r{2};
if ~isempty(d) && isfield(d, 'farfield_top') && isfield(d.farfield_top, 'E2')
    imagesc(double(d.farfield_top.ux), double(d.farfield_top.uy), ...
            double(d.farfield_top.E2).');
    axis tight; colorbar;
end
xlabel('ux'); ylabel('uy'); title('top-monitor far-field E^2 (W800)');

sgtitle(sprintf(['TM \\pi-shift radiation polarimetry, W800/corr400/pitch516.83/h350, ' ...
    'N=80 — direction + polarization diagnostic']));

exportgraphics(fig, fullfile(out_dir, 'radiation_polarimetry_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'radiation_polarimetry_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'radiation_polarimetry_summary.png'));
