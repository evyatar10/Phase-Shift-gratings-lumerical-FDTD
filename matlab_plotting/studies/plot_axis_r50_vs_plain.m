% plot_axis_r50_vs_plain.m — plain device vs SiO2 cylinders r 50 (one per period on the axis, Lambda 527) only.
% Study dirs: results_from_athena/{scat_x21_axis_l527_athena (270 deg row, job 151686), scat_x23_axis_l527_r50phase_r30eq
% (0/90/180 deg, job 151719)} + results_from_igum/scat_x24_axis_l527_r50eq (corr 456 equal-width row, job 91008);
% control scat_h_retrocomb. Plot 2026-09-17.
% (left) peak T vs comb phase, r 50, corr 400, periodic pchip; (right) T(lambda): no cylinders, r 50 at corr 400,
% r 50 at corr 456 (mode width rescaled back to the plain device's 15.5 um).
root = fullfile(fileparts(mfilename('fullpath')), '..', '..');
res = @(varargin) fullfile(root, varargin{:}, 'results');
fc = fullfile(res('results_from_athena', 'scat_h_retrocomb'), 'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat');
ctrl = load(fc, 'resonance_transmission', 'fwhm_m'); Tc = ctrl.resonance_transmission;
grey = [0.45 0.45 0.45]; ci = [0.85 0.33 0.10]; ci2 = [0.49 0.18 0.56];

% phase points (r 50, Lambda 527, corr 400)
L = [dir(fullfile(res('results_from_athena', 'scat_x21_axis_l527_athena'), '*scR50_*Y0to0_C400*.mat')); ...
     dir(fullfile(res('results_from_athena', 'scat_x23_axis_l527_r50phase_r30eq'), '*Ybox16p0_Zbox8p8_scR50_*Y0to0_C400*.mat'))];
ph = []; T = [];
for k = 1:numel(L)
    tok = str2double(regexp(L(k).name, '_X(-?[\d.]+)to(-?[\d.]+)_Y', 'tokens', 'once'));
    m = load(fullfile(L(k).folder, L(k).name), 'resonance_transmission');
    ph(end + 1, 1) = mod(360 * mean(tok) / 527, 360); T(end + 1, 1) = m.resonance_transmission; %#ok<AGROW>
end
ph = round(ph / 90) * 90; [ph, iu] = unique(ph); T = T(iu);

fig = figure('Visible', 'off', 'Position', [60 60 1400 520]);
tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on');
yline(ax, Tc, '--', 'Color', grey, 'LineWidth', 1.2, 'DisplayName', sprintf('no cylinders, T %.3f', Tc));
x = [ph; 360]; y = [T; T(1)]; xx = linspace(0, 360, 361);
plot(ax, xx, pchip(x, y, xx), '-', 'Color', ci, 'LineWidth', 1.6, 'HandleVisibility', 'off');
plot(ax, x, y, 's', 'Color', ci, 'MarkerFaceColor', ci, 'MarkerSize', 7, 'HandleVisibility', 'off');
plot(ax, NaN, NaN, '-s', 'Color', ci, 'MarkerFaceColor', ci, 'MarkerSize', 7, 'LineWidth', 1.6, 'DisplayName', 'SiO_2 cylinders, r 50, \Lambda 527');
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); xlim(ax, [0 360]); xticks(ax, 0:90:360); ylim(ax, [0.7 0.95]);
xlabel(ax, 'comb phase \phi (\circ)'); ylabel(ax, 'peak transmission');
legend(ax, 'Location', 'southoutside', 'FontSize', 9); title(ax, 'peak T vs comb phase', 'FontWeight', 'normal');

% spectra
ax = nexttile(tl); hold(ax, 'on');
f50  = dir(fullfile(res('results_from_athena', 'scat_x21_axis_l527_athena'), '*scR50_*X-7510to8300*Y0to0_C400*.mat'));
f50c = dir(fullfile(res('results_from_igum', 'scat_x24_axis_l527_r50eq'), '*scR50_*C456*.mat'));
files = {fc, fullfile(f50.folder, f50.name), fullfile(f50c.folder, f50c.name)};
names = {'no cylinders', 'SiO_2 cylinders r 50, corr 400', 'SiO_2 cylinders r 50, corr 456 (width rescaled)'}; col = {grey, ci, ci2};
for k = 1:3
    m = load(files{k}, 'wl_nm', 'T', 'resonance_wavelength_nm', 'resonance_transmission', 'spectral_fwhm_nm', 'fwhm_m');
    plot(ax, m.wl_nm, m.T, '-', 'Color', col{k}, 'LineWidth', 1.6, 'DisplayName', [names{k} ': ' ...
        sprintf('%.2f nm, T %.3f, Q %.0f, width %.1f', m.resonance_wavelength_nm, m.resonance_transmission, ...
        m.resonance_wavelength_nm / abs(m.spectral_fwhm_nm), m.fwhm_m * 1e6) ' \mum']);
end
hold(ax, 'off'); grid(ax, 'on'); box(ax, 'on'); xlim(ax, [1552 1562]); ylim(ax, [0 1]);
xlabel(ax, 'wavelength (nm)'); ylabel(ax, 'transmission');
legend(ax, 'Location', 'southoutside', 'FontSize', 9); title(ax, 'transmission spectra', 'FontWeight', 'normal');
title(tl, 'Plain device vs SiO_2 cylinders r 50 - TM, N 80/side');
out = fullfile(root, 'results_from_athena', 'scat_x23_axis_l527_r50phase_r30eq', 'r50_vs_plain');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150); disp(['wrote ' out '.png']);
