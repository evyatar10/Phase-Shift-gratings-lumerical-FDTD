% plot_width_lightline.m — TM light-line-margin study (job 116979).
% Loss and spatial mode width vs core width, for fixed corrugation (400 nm)
% and proportionally scaled corrugation (corr = W/2). Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
res_dir = fullfile(proj, 'results_from_athena', 'tm_width_lightline', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'tm_width_lightline');

fl = dir(fullfile(res_dir, 'result_*.mat'));
n = numel(fl);
W = nan(1,n); C = nan(1,n); loss = nan(1,n); fw = nan(1,n); T = nan(1,n); lam = nan(1,n);
for k = 1:n
    d = load(fullfile(fl(k).folder, fl(k).name));
    tok = regexp(fl(k).name, 'Wavg(\d+)_C(\d+)', 'tokens');
    if isempty(tok), W(k) = 800; C(k) = 400;
    else, W(k) = str2double(tok{1}{1}); C(k) = str2double(tok{1}{2}); end
    lam(k) = d.resonance_wavelength_nm;
    T(k) = d.resonance_transmission;
    [~, i] = min(abs(d.wl_nm - lam(k)));
    loss(k) = 1 - T(k) - d.R(i);
    fw(k) = d.fwhm_m * 1e6;
end

fixed  = (C == 400);                         % includes the 800/400 control
scaled = (C == W / 2);                       % 800/400 control belongs to both
[Wf, if_] = sort(W(fixed));  Lf = loss(fixed);  Lf = Lf(if_);  Ff = fw(fixed);  Ff = Ff(if_);
[Ws, is_] = sort(W(scaled)); Ls = loss(scaled); Ls = Ls(is_); Fs = fw(scaled); Fs = Fs(is_);

fig = figure('Visible', 'off', 'Position', [80 80 1100 430]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile; hold on; grid on;
plot(Wf, Lf, 'o-', 'Color', [0.80 0.33 0.20], 'LineWidth', 1.4, 'DisplayName', 'corr fixed 400 nm (\kappa drops)');
plot(Ws, Ls, 's-', 'Color', [0.19 0.45 0.72], 'LineWidth', 1.4, 'DisplayName', 'corr = W/2 (\kappa \approx const)');
xlabel('avg core width W (nm)'); ylabel('resonant loss 1 - T - R');
title('Radiation loss vs width');
legend('Location', 'northeast');

nexttile; hold on; grid on;
plot(Wf, Ff, 'o-', 'Color', [0.80 0.33 0.20], 'LineWidth', 1.4, 'DisplayName', 'corr fixed 400 nm');
plot(Ws, Fs, 's-', 'Color', [0.19 0.45 0.72], 'LineWidth', 1.4, 'DisplayName', 'corr = W/2');
xlabel('avg core width W (nm)'); ylabel('spatial mode width fwhm_m (\mum)');
title('Mode-width cost');
legend('Location', 'northwest');

sgtitle(sprintf(['TM \\pi-shift, h 350 nm, pitch 516.83 nm, N=80 — light-line margin study\n' ...
    'control W800/corr400: \\lambda_{res}=1558.6 nm, T=0.884, loss=0.112, fwhm=15.5 \\mum; ' ...
    'W1000/corr500: loss=0.086 (-23%%), fwhm +5%%']));

exportgraphics(fig, fullfile(out_dir, 'width_lightline_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'width_lightline_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'width_lightline_summary.png'));
