% plot_inner_shape_study.m — center-shape study (job 117000): reshape only the
% defect region. Bars of dLoss vs the same-run control, mode-width cost as text.
% Headless-safe; saves .fig + .png next to the data.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
res_dir = fullfile(proj, 'results_from_athena', 'inner_shape_study', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'inner_shape_study');

fl = dir(fullfile(res_dir, 'result_*.mat'));
n = numel(fl);
lab = cell(1, n); dloss = nan(1, n); fwp = nan(1, n); isctrl = false(1, n);
loss = nan(1, n); fw = nan(1, n);
for k = 1:n
    d = load(fullfile(fl(k).folder, fl(k).name));
    lam = d.resonance_wavelength_nm;
    [~, i] = min(abs(d.wl_nm - lam));
    loss(k) = 1 - d.resonance_transmission - d.R(i);
    fw(k) = d.fwhm_m * 1e6;
    ish = strtrim(string(d.inner_tooth_shape));
    csh = strtrim(string(d.cavity_shape));
    if ish == "rect" && csh == "rect"
        lab{k} = 'control'; isctrl(k) = true;
    elseif csh ~= "rect"
        lab{k} = sprintf('cavity %s %d', csh, round(d.cavity_shape_depth_m * 1e9));
    else
        lab{k} = sprintf('%s n=%d', ish, d.n_shaped_inner_teeth);
    end
end
k0 = find(isctrl, 1);
dloss = (loss - loss(k0)) * 1e3;            % in 1e-3 units
fwp = (fw / fw(k0) - 1) * 100;              % mode-width change (%)

keep = ~isctrl;
[dl, ord] = sort(dloss(keep));
labs = lab(keep); labs = labs(ord);
fwk = fwp(keep); fwk = fwk(ord);

fig = figure('Visible', 'off', 'Position', [80 80 1150 520]);
hold on; grid on;
cols = zeros(numel(dl), 3);
cols(dl < 0, :) = repmat([0.13 0.55 0.33], nnz(dl < 0), 1);
cols(dl >= 0, :) = repmat([0.75 0.32 0.20], nnz(dl >= 0), 1);
b = bar(dl, 'FaceColor', 'flat');
b.CData = cols;
yline(0, 'k-');
for k = 1:numel(dl)
    yy = dl(k) + sign(dl(k)) * 1.2;
    text(k, yy, sprintf('fwhm %+.1f%%', fwk(k)), 'HorizontalAlignment', 'center', ...
        'FontSize', 8, 'Interpreter', 'none');
end
set(gca, 'XTick', 1:numel(dl), 'XTickLabel', labs, 'XTickLabelRotation', 25, ...
    'TickLabelInterpreter', 'none');
ylabel('\Delta loss vs control (\times10^{-3})');
title(sprintf(['Center-shape study — TM \\pi-shift, pitch 516.83 nm, corr 400 nm, h 350 nm\n' ...
    'control: \\lambda_{res}=1558.62 nm, loss=%.4f, fwhm=%.1f \\mum \\cdot green = less loss'], ...
    loss(k0), fw(k0)));

exportgraphics(fig, fullfile(out_dir, 'inner_shape_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'inner_shape_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'inner_shape_summary.png'));
