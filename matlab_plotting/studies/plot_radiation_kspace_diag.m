% Light-cone (leaky-component) diagnostic of the anchored TM pi-shift device.
% Data: results_from_athena/radiation_kspace_diag/kspace_diag_N80_TM.mat
% (Nakamura/Noda-style: FFT the resonant Ez along x, keep |kx| < n_clad*k0,
%  inverse-FFT -> real-space map of the radiating components.)

clear; close all;
root = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\radiation_kspace_diag';
m = load(fullfile(root, 'kspace_diag_N80_TM.mat'));

x  = m.x_um(:);  y = m.y_um(:);
kx = m.kx_um(:); spec = m.spec_kx(:);
kc = m.kc_um; beta = m.beta_um; pitch = m.pitch_um;

f = figure('Position', [80 80 1150 820], 'Color', 'w');
tl = tiledlayout(f, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% (a) longitudinal spectrum with the light cone
nexttile;
semilogy(kx, spec / max(spec), 'LineWidth', 1.1); hold on;
yl = [1e-7 2];
patch([-kc kc kc -kc], [yl(1) yl(1) yl(2) yl(2)], [1 0.85 0.7], ...
    'FaceAlpha', 0.35, 'EdgeColor', 'none');
xline( beta, '--', '+\beta'); xline(-beta, '--', '-\beta');
xline( kc, ':'); xline(-kc, ':');
ylim(yl); xlim([-10 10]);
xlabel('k_x (rad/\mum)'); ylabel('|E_z(k_x)|^2 (norm.)');
title('(a) Mode spectrum; shaded = light cone (radiating)');
set(gca, 'Layer', 'top');

% (b) radiating-component map near the defect
nexttile;
sel = abs(x) <= 5;
imagesc(x(sel), y, m.E_leak_abs2(sel, :).' / max(m.E_leak_abs2, [], 'all'));
axis xy; colorbar;
xlabel('x (\mum)'); ylabel('y (\mum)');
title('(b) |E_{leaky}|^2 — where the radiation originates');
hold on;
for xt = (-4:4) * pitch          % pitch grid, defect at 0
    xline(xt, 'w:', 'Alpha', 0.4);
end

% (c) radiating vs total profile along x
nexttile;
semilogy(x, m.prof_leak / max(m.prof_leak), 'LineWidth', 1.2); hold on;
semilogy(x, m.prof_tot / max(m.prof_tot), 'LineWidth', 0.8);
xlim([-15 15]); ylim([1e-4 2]);
xlabel('x (\mum)'); ylabel('norm. weight');
legend({'radiating (in-cone)', 'total |E_z|^2'}, 'Location', 'northeast');
title('(c) Defect lobe \sim100\times the per-tooth background, on a distributed floor');

% (d) cumulative attribution
sh = @(n) interp1(m.cum_absx_um / pitch, m.cum_share, n);
nexttile;
plot(m.cum_absx_um / pitch, m.cum_share, 'LineWidth', 1.2); hold on;
xline(1, ':', '\pm1 pitch'); xline(2, ':', '\pm2'); xline(3, ':', '\pm3');
yline(sh(1), ':'); yline(sh(2), ':'); yline(sh(3), ':');
xlim([0 40]); ylim([0 1.02]);
xlabel('|x| from defect (pitches)'); ylabel('cumulative share of radiating weight');
title(sprintf('(d) %.0f%% within \\pm1 pitch, %.0f%% within \\pm2, %.0f%% within \\pm3', ...
    100 * sh(1), 100 * sh(2), 100 * sh(3)));

title(tl, sprintf(['TM \\pi-shift, pitch %.2f nm, corr 400 nm, W 800 nm, h 350 nm — ', ...
    'light-cone diagnostic at \\lambda_{res} = %.2f nm'], pitch * 1e3, m.lres_nm), ...
    'FontSize', 12);

savefig(f, fullfile(root, 'radiation_kspace_diag.fig'));
exportgraphics(f, fullfile(root, 'radiation_kspace_diag.png'), 'Resolution', 200);
disp('saved fig+png');
