% plot_incore_radius_T_over_width.m
% Study: in-core oxide hole comb, Lambda 524 / 270 deg / 31 holes at y=+/-250, radius series.
% Jobs: Athena 149982 (r80), 150391 (r110), 150429 (r50), 150458 (r40, r30); ctrl scat_h_retrocomb.
% Date: 2026-09-15.  Purpose: peak T divided by spatial mode width vs hole radius.
% Peak T = stored resonance_transmission (built-in finder); width = fwhm_m (post_processing convention).
root = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results_from_athena');
files = { ...
  0,   fullfile(root,'scat_h_retrocomb','results','result_N80_TM_avg_Ybox16p0_Zbox8p8_ff.mat'); ...
  30,  fullfile(root,'scat_x7_incore_r40_r30','results','result_N80_TM_avg_Ybox16p0_Zbox8p8_scR30_arr31_X-7467to8253_Y250to250_C400_pair_hole_ff.mat'); ...
  40,  fullfile(root,'scat_x7_incore_r40_r30','results','result_N80_TM_avg_Ybox16p0_Zbox8p8_scR40_arr31_X-7467to8253_Y250to250_C400_pair_hole_ff.mat'); ...
  50,  fullfile(root,'scat_x6_incore_r50','results','result_N80_TM_avg_Ybox16p0_Zbox8p8_scR50_arr31_X-7467to8253_Y250to250_C400_pair_hole_ff.mat'); ...
  80,  fullfile(root,'scat_x4_incore_below530','results','result_N80_TM_avg_Ybox16p0_Zbox8p8_scR80_arr31_X-7467to8253_Y250to250_C400_pair_hole_ff.mat'); ...
  110, fullfile(root,'scat_x5_incore_r110','results','result_N80_TM_avg_Ybox16p0_Zbox8p8_scR110_arr31_X-7467to8253_Y250to250_C400_pair_hole_ff.mat')};
n = size(files,1); r = zeros(n,1); T = r; w = r; lam = r;
for k = 1:n
    m = load(files{k,2}, 'resonance_transmission', 'fwhm_m', 'resonance_wavelength_nm');
    r(k) = files{k,1}; T(k) = m.resonance_transmission; w(k) = m.fwhm_m*1e6; lam(k) = m.resonance_wavelength_nm;
end
ratio = T ./ w;

fig = figure('Color','w','Position',[100 100 760 520]);
plot(r(2:end), ratio(2:end), '-o', 'Color',[0.85 0.33 0.1], 'LineWidth',1.8, 'MarkerFaceColor',[0.85 0.33 0.1], 'MarkerSize',7); hold on
yline(ratio(1), '--', 'Color',[0.3 0.3 0.3], 'LineWidth',1.4);
plot(0, ratio(1), 's', 'Color',[0.3 0.3 0.3], 'MarkerFaceColor',[0.3 0.3 0.3], 'MarkerSize',8);
for k = 1:n
    text(r(k)+2, ratio(k)+0.0008, ['T ' sprintf('%.3f', T(k)) ' / ' sprintf('%.1f', w(k)) ' \mum'], 'FontSize',9);
end
grid on; xlim([-5 120]);
xlabel('hole radius r (nm)'); ylabel('peak T / spatial FWHM (\mum^{-1})');
legend({'in-core SiO_2 hole comb (\Lambda 524, 270^\circ, 31 holes)', 'no holes (control)'}, 'Location','northeast');
title({'\pi-shift Bragg grating, TM h350, pitch 516.83, corr 400, W800, N=80', ...
       ['control: \lambda_{res} ' sprintf('%.2f', lam(1)) ' nm, T ' sprintf('%.4f', T(1)) ', width ' sprintf('%.1f', w(1)) ' \mum']});
out = fullfile(root, 'scat_x7_incore_r40_r30', 'incore_radius_T_over_width');
savefig(fig, [out '.fig']); exportgraphics(fig, [out '.png'], 'Resolution', 150);
