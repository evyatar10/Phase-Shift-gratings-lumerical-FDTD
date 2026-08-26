% plot_invdesign_q3db_20um - the INVERSE-DESIGNED pi-shift TM device taken to the
% -3 dB / 20 um-mode operating point: T(N), Q(N), the interpolated crossing, and
% T(lambda) of the device at that crossing.
% Study: runners/sweeps/invdesign_q3db_20um.py  |  2026-08-26
% Jobs: IGUM 63202 (N=100), 63423 (N=150) | Athena 137322 (N=180/200/220)
% Rungs live on BOTH clusters, so both result dirs are scanned.
% Reference points are STORED family measurements at identical numerics, never
% re-run: bare N=100 T 0.9104 / Q 1760 ; ctrl N=165 T 0.4906 (-3.09 dB) / Q 13930 ;
% winner comb N=169 -3.04 dB / Q 16203.

root = fileparts(fileparts(mfilename('fullpath')));
dirs = { fullfile(root, '..', 'results_from_athena', 'invdesign_q3db_20um', 'results'), ...
         fullfile(root, '..', 'results_from_igum',   'invdesign_q3db_20um', 'results') };

N = []; T = []; Q = []; W = []; M = {}; outdir = '';
for d = 1:numel(dirs)
    if ~isfolder(dirs{d}); continue; end
    if isempty(outdir); outdir = fullfile(dirs{d}, '..'); end   % figures land beside
end                                                             % the first dir that EXISTS
for d = 1:numel(dirs)
    if ~isfolder(dirs{d}); continue; end
    files = dir(fullfile(dirs{d}, 'result_N*.mat'));
    for k = 1:numel(files)
        m = load(fullfile(dirs{d}, files(k).name));
        N(end+1) = double(m.n_periods_each_side);                       %#ok<*SAGROW>
        T(end+1) = m.resonance_transmission;
        Q(end+1) = m.resonance_wavelength_nm / abs(m.spectral_fwhm_nm);
        W(end+1) = m.fwhm_m * 1e6;
        M{end+1} = m;
    end
end
assert(~isempty(N), 'no result_N*.mat found in either cluster results dir');
[Ns, si] = sort(N); Ts = T(si); Qs = Q(si); Ws = W(si); Ms = M(si);
dB = 10*log10(Ts);

% -3 dB crossing by linear interpolation in dB vs N (the curve is smooth in dB).
% If the ladder does not straddle -3 dB this EXTRAPOLATES — say so on the figure
% rather than quoting a crossing that was never bracketed.
straddles = any(dB > -3) && any(dB < -3);
if straddles
    Ncross = interp1(dB, Ns, -3, 'linear');
    Qcross = interp1(Ns, Qs, Ncross, 'linear');
else
    p = polyfit(log(Ns), log(-dB), 1);          % dB ~ N^alpha
    Ncross = exp((log(3) - p(2)) / p(1));
    Qcross = NaN;      % DELIBERATE: Q climbs ~N^4 near the crossing, so extrapolating
end                    % it off the end of the ladder is meaningless. No number is
                       % better than a wrong one.
if straddles
    lbl = sprintf('N = %.0f, Q = %.0f', Ncross, Qcross);
else
    lbl = sprintf('N ~ %.0f (EXTRAPOLATED - not bracketed); Q NOT AVAILABLE', Ncross);
end
fprintf('-3 dB crossing: %s | rungs N = %s\n', lbl, mat2str(Ns));
fprintf('mode FWHM across the ladder: %.3f - %.3f um\n', min(Ws), max(Ws));

c0 = [0.00 0.45 0.74]; c1 = [0.85 0.33 0.10];
fig1 = figure('Position', [100 100 720 640]);
ax1 = subplot(2,1,1); hold(ax1,'on'); grid(ax1,'on');
ax2 = subplot(2,1,2); hold(ax2,'on'); grid(ax2,'on');
plot(ax1, Ns, Ts, 'o-', 'Color', c0, 'MarkerFaceColor', c0, 'DisplayName', 'inverse design');
plot(ax1, [165 169], [0.4906 0.4966], 's', 'Color', c1, 'MarkerFaceColor', c1, ...
    'DisplayName', 'stored family (ctrl N165, comb N169)');
yline(ax1, 0.5, 'k--', 'T = 0.5 (-3 dB)', 'HandleVisibility', 'off');
xline(ax1, Ncross, 'k:', 'HandleVisibility', 'off');
plot(ax2, Ns, Qs, 'o-', 'Color', c0, 'MarkerFaceColor', c0, 'DisplayName', 'inverse design');
plot(ax2, [165 169], [13930 16203], 's', 'Color', c1, 'MarkerFaceColor', c1, ...
    'DisplayName', 'stored family');
if ~isnan(Qcross)
    plot(ax2, Ncross, Qcross, 'kp', 'MarkerSize', 14, 'HandleVisibility', 'off');
end
text(ax2, Ncross, max(Qs), ['  ' lbl], 'VerticalAlignment', 'top', 'Interpreter', 'none');
ylabel(ax1, 'Peak transmission'); set(ax1, 'XTickLabel', []);
ylabel(ax2, 'Loaded Q'); xlabel(ax2, 'N periods per side');
legend(ax1, 'Location', 'southwest'); legend(ax2, 'Location', 'northwest');
title(ax1, {'Inverse-designed \pi-shift TM at -3 dB', ...
    ['corr 325 nm outer / apodized inner 25, height 350 nm, pitch 516.83 nm, mode ' num2str(Ws(1), '%.2f') ' \mum']});
out = fullfile(outdir, 'invdesign_q3db_20um_T_Q');
savefig(fig1, [out '.fig']); exportgraphics(fig1, [out '.png'], 'Resolution', 150);
fprintf('saved %s.png\n', out);

% Device at (or nearest) the crossing: T(lambda) in dB.
[~, ib] = min(abs(Ns - Ncross)); mb = Ms{ib};
fig2 = figure('Position', [100 100 820 480]); hold on; grid on;
plot(mb.wl_nm, 10*log10(mb.T), '-', 'Color', c0, 'DisplayName', ...
    ['inverse design, N=' num2str(Ns(ib)) ':  {\bf Q = ' num2str(Qs(ib), '%.0f') '}   (' ...
     '\lambda_{res} ' num2str(mb.resonance_wavelength_nm, '%.2f') ' nm, peak ' ...
     num2str(dB(ib), '%.2f') ' dB, mode ' num2str(Ws(ib), '%.2f') ' \mum)']);
yline(-3, 'k--', '-3 dB', 'HandleVisibility', 'off');
xlabel('Wavelength [nm]'); ylabel('Transmission [dB]');
xlim([min(mb.wl_nm) max(mb.wl_nm)]); ylim([-45 2]);
legend('Location', 'south');
title({'Inverse-designed \pi-shift TM: device at the -3 dB point', ...
    sprintf('height 350 nm, pitch 516.83 nm, cavity 961 nm, 57-post SiN comb')});
out2 = fullfile(outdir, 'invdesign_q3db_20um_final_T_dB');
savefig(fig2, [out2 '.fig']); exportgraphics(fig2, [out2 '.png'], 'Resolution', 150);
fprintf('saved %s.png\n', out2);
