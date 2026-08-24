% plot_best_spectrum_and_mode.m — T(lambda) + mode profile of the current best.
% Study: runners/lumopt2_design | job 136303 (spectrum re-read from the solved
% iter-0 .fsp of campaign 136248) | 2026-08-23
% Device: pi-shift Bragg grating, re-trimmed best (BEST_T9635 + 52.5 nm),
% N=100/side surrogate, pitch-locked mesh dx = pitch/10 = 51.683 nm, PVA.
% Data written by runners/lumopt2_design/extract_spectrum.py (CSV bridge).

OUT = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\lumopt2_c325_logs\';
S = readmatrix([OUT 'best_spectrum.csv']);     % wl_nm, T, R
P = readmatrix([OUT 'best_modeprofile.csv']);  % x_um, I
wl = S(:,1); T = S(:,2); R = S(:,3);
x  = P(:,1); I = P(:,2);

[tpk, ipk] = max(T); lampk = wl(ipk);
half = tpk/2; il = find(T(1:ipk) <= half, 1, 'last'); ir = ipk - 1 + find(T(ipk:end) <= half, 1, 'first');
if isempty(il) || isempty(ir)
    fw_nm = NaN;
else
    xl = interp1(T([il il+1]), wl([il il+1]), half);
    xr = interp1(T([ir-1 ir]), wl([ir-1 ir]), half);
    fw_nm = xr - xl;
end
Q = lampk / fw_nm;

% envelope through the standing-wave peaks (same construction as the engine)
pk = find(I(2:end-1) >= I(1:end-2) & I(2:end-1) >= I(3:end)) + 1;
xe = x(pk); Ie = I(pk);
flo = 0.5*(mean(Ie(1:3)) + mean(Ie(end-2:end)));
thr = flo + 0.5*(max(Ie) - flo);
ia = find(Ie >= thr); modefw = xe(ia(end)) - xe(ia(1));

f = figure('Visible','off','Position',[60 60 1150 780],'Color','w');
tl = tiledlayout(f,2,1,'TileSpacing','compact','Padding','compact');
title(tl, sprintf(['\\pi-shift Bragg grating, re-trimmed best  —  ' ...
    'T_{pk} %.4f,  \\lambda_{res} %.3f nm,  Q_L %.0f,  mode FWHM %.2f \\mum ' ...
    '(N=100 surrogate, dx = pitch/10)'], tpk, lampk, Q, modefw));

ax1 = nexttile; hold(ax1,'on');
plot(ax1, wl, T, 'LineWidth', 1.8, 'Color', [0.20 0.45 0.75]);
plot(ax1, wl, R, 'LineWidth', 1.0, 'Color', [0.75 0.35 0.20]);
plot(ax1, lampk, tpk, 'kv', 'MarkerFaceColor','k','MarkerSize',6);
yline(ax1, half, ':', 'Color',[0.5 0.5 0.5]);
legend(ax1, {sprintf('T  (peak %.4f)',tpk), 'R', ...
             sprintf('\\lambda_{res} %.3f nm',lampk), ...
             sprintf('half max  (spectral FWHM %.3f nm)',fw_nm)}, ...
       'Location','northwest','Box','off');
xlabel(ax1,'wavelength (nm)'); ylabel(ax1,'transmission / reflection');
xlim(ax1,[min(wl) max(wl)]); grid(ax1,'on'); box(ax1,'on');
title(ax1,'Resonance');

ax2 = nexttile; hold(ax2,'on');
plot(ax2, x, I, 'LineWidth', 0.6, 'Color', [0.65 0.75 0.88]);
plot(ax2, xe, Ie, 'LineWidth', 1.8, 'Color', [0.20 0.45 0.75]);
yline(ax2, thr, '--', 'Color',[0.75 0.35 0.20], 'LineWidth',1.2);
plot(ax2, [xe(ia(1)) xe(ia(end))], [thr thr], 'o', 'Color',[0.75 0.35 0.20], ...
     'MarkerFaceColor',[0.75 0.35 0.20],'MarkerSize',6);
xline(ax2, [-13.18 13.18], ':', 'Color',[0.45 0.45 0.45]);
legend(ax2, {'|E|^2 along x (standing wave)','envelope through peaks', ...
             sprintf('half max  (mode FWHM %.2f \\mum)',modefw), ...
             'FWHM crossings','free/frozen tooth boundary'}, ...
       'Location','northeast','Box','off');
xlabel(ax2,'x  (\mum)   — propagation'); ylabel(ax2,'energy density (a.u.)');
xlim(ax2,[-30 30]); grid(ax2,'on'); box(ax2,'on');
title(ax2,'Mode profile at resonance');

exportgraphics(f,[OUT 'best_spectrum_and_mode.png'],'Resolution',180);
savefig(f,[OUT 'best_spectrum_and_mode.fig']);
close(f); disp('written');
