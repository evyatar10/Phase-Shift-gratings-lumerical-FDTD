% Transmission/reflection of Itai's HH apodization at our Q3dB point.
% Study: runners/sweeps/itai_hh_apod.py | IGUM 63237 task 0 | 2026-08-25
% Panel 1 = full scan window (is the peak we picked THE defect resonance?).
% Panel 2 = resonance zoom with T+R, which exceeds 1 by up to 4% (under
% investigation -- energy conservation is violated only across the resonance).
clear; close all;

f = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_igum', ...
             'itai_hh_apod_task0_slice.mat');
d = load(f);
wl = d.wl_nm(:); T = d.T(:); R = d.R(:);
[wl, k] = sort(wl); T = T(k); R = R(k);
lam0 = d.resonance_wavelength_nm; fw = abs(d.spectral_fwhm_nm);
Q = lam0 / fw; Tpk = interp1(wl, T, lam0);

fig = figure('Position', [100 100 1000 780], 'Color', 'w');

subplot(2,1,1);
semilogy(wl, T, 'b-', 'LineWidth', 1.3); hold on;
semilogy(wl, R, 'r-', 'LineWidth', 1.0);
xline(lam0, 'k--', 'LineWidth', 1.0);
ylim([1e-4 3]); grid on; xlim([min(wl) max(wl)]);
xlabel('\lambda (nm)'); ylabel('T, R');
legend({'T','R','\lambda_{res}'}, 'Location', 'east');
title({'HH apod, TE, 98 periods/side, pitch 490.09 nm, avg width 1.0 um', ...
       ['\lambda_{res} = ' sprintf('%.3f', lam0) ' nm,  peak T = ' sprintf('%.3f', Tpk) ...
        ',  Q = ' sprintf('%.0f', Q)]});

subplot(2,1,2);
m = abs(wl - lam0) < 0.6;
plot(wl(m), T(m), 'b-', 'LineWidth', 1.5); hold on;
plot(wl(m), R(m), 'r-', 'LineWidth', 1.2);
plot(wl(m), T(m)+R(m), 'k-', 'LineWidth', 1.5);
yline(1, 'k:', 'LineWidth', 1.0);
xline(lam0, 'k--', 'LineWidth', 1.0);
grid on; xlim([lam0-0.6 lam0+0.6]); ylim([0 1.1]);
xlabel('\lambda (nm)'); ylabel('T, R, T+R');
legend({'T','R','T+R','unity'}, 'Location', 'east');
title(sprintf('Resonance zoom: FWHM %.1f pm, max(T+R) = %.4f (energy conservation violated)', ...
              fw*1000, max(T(m)+R(m))));

out = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_igum');
savefig(fig, fullfile(out, 'itai_hh_apod_task0_T.fig'));
exportgraphics(fig, fullfile(out, 'itai_hh_apod_task0_T.png'), 'Resolution', 150);
fprintf('lam_res %.4f nm | peak T %.4f | Q %.0f | max(T+R) %.5f\n', lam0, Tpk, Q, max(T+R));
