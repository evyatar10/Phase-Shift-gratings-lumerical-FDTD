% plot_incore_r50_spectrum.m — T(lambda) of the in-core r 50 cylinder comb (Lambda 524, 270 deg, 31 holes) vs no holes.
% Study dir: results_from_athena/scat_x6_incore_r50 (job 150429) + control scat_h_retrocomb. Plot 2026-09-16.
% Purpose (user): the transmission spectrum of the r 50 device, its resonance and the shift vs the plain device.
% Device: TM, W 800, corr 400, h 350, pitch 516.83, N 80/side, box y 16, 20 nm / 1501 pts. Peak = built-in finder.
root = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results_from_athena');
f = {fullfile(root, 'scat_h_retrocomb', 'results', 'result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'), ...
     fullfile(root, 'scat_x6_incore_r50', 'results', 'result_N80_TM_avg_Ybox16p0_Zbox8p8_scR50_arr31_X-7467to8253_Y250to250_C400_pair_hole_ff.mat')};
names = {'no holes', 'SiO_2 cylinders in core, r 50'}; col = {[0.45 0.45 0.45], [0.85 0.33 0.10]};
fig = figure('Visible', 'off', 'Position', [60 60 820 520]); hold on
for k = 1:2
    m = load(f{k}, 'wl_nm', 'T', 'resonance_wavelength_nm', 'resonance_transmission', 'spectral_fwhm_nm');
    plot(m.wl_nm, m.T, '-', 'Color', col{k}, 'LineWidth', 1.6, 'DisplayName', ...
        [names{k} ': ' sprintf('%.2f nm, T %.3f, FWHM %.2f nm', m.resonance_wavelength_nm, m.resonance_transmission, abs(m.spectral_fwhm_nm))]);
    plot(m.resonance_wavelength_nm, m.resonance_transmission, 'o', 'Color', col{k}, 'MarkerFaceColor', col{k}, 'MarkerSize', 6, 'HandleVisibility', 'off');
    lam(k) = m.resonance_wavelength_nm; %#ok<SAGROW>
end
hold off; grid on; box on; xlim([1548.5 1568.5]); ylim([0 1]);
xlabel('wavelength (nm)'); ylabel('transmission'); legend('Location', 'southoutside', 'FontSize', 9);
title(sprintf('\\pi-shift Bragg grating, TM, corr 400, N 80/side - resonance shift %+.2f nm', lam(2) - lam(1)), 'FontWeight', 'normal');
out = fullfile(root, 'scat_x6_incore_r50', 'incore_r50_spectrum');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150); disp(['wrote ' out '.png']);
