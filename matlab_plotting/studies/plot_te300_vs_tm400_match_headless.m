% Side-by-side: TE@300 nm vs pitch+kappa-matched TM@400 nm.
%   LEFT  : transmission T(lambda) of both (Q-factor in the legend) -- expected to differ.
%   RIGHT : normalized spatial mode-energy envelope of both (matched width).
% Big title only ("Pitch and kappa matching TM to TE"); clean legends, no clutter.
%
% Data: results_from_athena/tm_match_corr/results/ (corrugation-match job 113571).
% Headless render -> PNG next to this script.

clear; close all;

here     = fileparts(mfilename('fullpath'));
repo     = fileparts(here);
resdir   = fullfile(repo, 'results_from_athena', 'tm_match_corr', 'results');
te_file  = fullfile(resdir, 'result_corrmatch_te_C300000.mat');
% TM re-trimmed to pitch 516.83 nm so its resonance sits back on the TE line at
% corrugation 400 nm (corrugation 300->400 detuned it ~1.7 nm; job 113807).
tm_file  = fullfile(repo, 'results_from_athena', 'tm_te', 'results', ...
                    'result_N80_TM_avg_tm_P516p8_smp.mat');

TE = load(te_file);
TM = load(tm_file);

% --- Q = lambda_res / FWHM, with FWHM measured by half-max interpolation on the
%     T(lambda) curve. The stored spectral_fwhm_nm is grid-snapped and inflates the
%     linewidth on coarse scans (e.g. the 25 pm run_tm window read 1.31 nm vs the
%     true ~1.23 nm), so interpolating recovers the correct Q from the same data.
Q_te = peak_Q(TE.wl_nm(:), TE.T(:), double(TE.resonance_wavelength_nm));
Q_tm = peak_Q(TM.wl_nm(:), TM.T(:), double(TM.resonance_wavelength_nm));

% Pitches (anchored): TE 500 nm; TM 516.83 nm (co-resonant at corr 400, file _P516p8).
TE_PITCH_NM = 500;
TM_PITCH_NM = 516.83;

c_te = [0.85 0.33 0.10];   % orange
c_tm = [0.00 0.45 0.74];   % blue
FS   = 14; LW = 1.8;

fig = figure('Color', 'w', 'Position', [100 100 1180 470]);

% ===== LEFT: transmission (changes between TE and TM) =====
ax1 = subplot(1, 2, 1); hold(ax1, 'on'); box(ax1, 'on');
plot(ax1, TE.wl_nm(:), TE.T(:), '-', 'Color', c_te, 'LineWidth', LW);
plot(ax1, TM.wl_nm(:), TM.T(:), '-', 'Color', c_tm, 'LineWidth', LW);
xlabel(ax1, 'Wavelength (nm)', 'FontSize', FS);
ylabel(ax1, 'Transmission', 'FontSize', FS);
title(ax1, 'Transmission', 'FontSize', FS);
lam_c = 0.5 * (double(TE.resonance_wavelength_nm) + double(TM.resonance_wavelength_nm));
xlim(ax1, [lam_c - 12, lam_c + 12]);
ylim(ax1, [0 1]);
lg1 = legend(ax1, {sprintf('TE: pitch %g nm, Q = %.0f', TE_PITCH_NM, Q_te), ...
                   sprintf('TM: pitch %.2f nm, Q = %.0f', TM_PITCH_NM, Q_tm)}, ...
       'Location', 'south', 'FontSize', FS);
lg1.ItemTokenSize = [12 18];   % shorter line sample -> narrower box
grid(ax1, 'on'); set(ax1, 'FontSize', FS - 2);

% ===== RIGHT: normalized mode-energy envelope (matched width) =====
ax2 = subplot(1, 2, 2); hold(ax2, 'on'); box(ax2, 'on');
te_env = double(TE.field_envelope_1D(:)); te_env = te_env / max(te_env);
tm_env = double(TM.field_envelope_1D(:)); tm_env = tm_env / max(tm_env);
plot(ax2, double(TE.field_x(:)) * 1e6, te_env, '-', 'Color', c_te, 'LineWidth', LW);
plot(ax2, double(TM.field_x(:)) * 1e6, tm_env, '-', 'Color', c_tm, 'LineWidth', LW);
xlabel(ax2, 'x (\mum)', 'FontSize', FS);
ylabel(ax2, 'Normalized energy', 'FontSize', FS);
title(ax2, 'Mode envelope', 'FontSize', FS);
xlim(ax2, [-45 45]);
lg2 = legend(ax2, {'TE: 300 nm corrugation', 'TM: 400 nm corrugation'}, ...
       'Location', 'north', 'FontSize', FS);
lg2.ItemTokenSize = [12 18];   % shorter line sample -> narrower box
grid(ax2, 'on'); set(ax2, 'FontSize', FS - 2);

sgtitle('Pitch and \kappa matching TM to TE, 80 periods, 350 nm depth', ...
        'FontSize', FS + 4, 'FontWeight', 'bold');

out_png = fullfile(here, 'plot_te300_vs_tm400_match.png');
exportgraphics(fig, out_png, 'Resolution', 150);
fprintf('wrote %s\n', out_png);

out_fig = fullfile(here, 'plot_te300_vs_tm400_match.fig');
savefig(fig, out_fig);
fprintf('wrote %s\n', out_fig);
fprintf('Q_TE = %.1f   Q_TM = %.1f\n', Q_te, Q_tm);


% ---- local function: FWHM by half-max interpolation around the resonance ----
function Q = peak_Q(wl, T, lam_res)
    [wl, o] = sort(wl); T = T(o);
    [~, ip] = min(abs(wl - lam_res));
    half = T(ip) / 2;
    l = ip; while l > 1 && T(l) > half, l = l - 1; end
    r = ip; while r < numel(T) && T(r) > half, r = r + 1; end
    wl_l = wl(l)   + (half - T(l))   * (wl(l+1) - wl(l))   / (T(l+1) - T(l));
    wl_r = wl(r-1) + (half - T(r-1)) * (wl(r)   - wl(r-1)) / (T(r)   - T(r-1));
    Q = lam_res / (wl_r - wl_l);
end
