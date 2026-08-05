% plot_autoshutoff_qspan - auto-shutoff convergence: Q error vs threshold & Q.
% Study: runners/sweeps/autoshutoff_qspan.py | IGUM 49537+49718 | 2026-08-04
% Verdict: error = f(Q) only (~Q^0.7); production 1e-7 for all; 1e-8 unreachable.
% KEEP-FOREVER convergence data (CLAUDE.md section 7).

root = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..');
dirs = {fullfile(root, 'results_from_igum', 'autoshutoff_qspan', 'results'), ...
        fullfile(root, 'results_from_igum', 'trench_q3db_20um', 'results')};

% device key -> {display name}; rows collected as (shutoff, T, Q) per device.
data = containers.Map();
for d = 1:2
    files = dir(fullfile(dirs{d}, 'result_*.mat'));
    for k = 1:numel(files)
        nm = files(k).name;
        as = regexp(nm, '_AS1em(\d)', 'tokens', 'once');
        if d == 2   % trench study: only the two corr-325 1e-7 anchors
            if ~isempty(as) || ~contains(nm, '_C325'), continue; end
            if ~(contains(nm, 'N165') && ~contains(nm, 'scRECT')) && ...
               ~(contains(nm, 'N185') && contains(nm, 'scRECT')), continue; end
            s = 1e-7;
        else
            if isempty(as), s = 1e-7; else, s = 10^(-str2double(as{1})); end
        end
        m = load(fullfile(files(k).folder, nm));
        n = double(m.n_periods_each_side);
        pol = 'TM'; if ~contains(nm, '_TM'), pol = 'TE'; end
        fam = 'plain';
        if contains(nm, 'scRECT'), fam = 'trench';
        elseif ~isempty(regexp(nm, '_A\d+_', 'once')), fam = 'apod'; end
        key = sprintf('%s N%d %s', pol, n, fam);
        Q = m.resonance_wavelength_nm / abs(m.spectral_fwhm_nm);
        row = [s, m.resonance_transmission, Q];
        if isKey(data, key), data(key) = [data(key); row];
        else, data(key) = row; end
    end
end

fig = figure('Position', [80 80 980 430]);
ax1 = subplot(1, 2, 1); hold on; grid on; set(ax1, 'XScale', 'log', 'YScale', 'log');
ax2 = subplot(1, 2, 2); hold on; grid on; set(ax2, 'XScale', 'log', 'YScale', 'log');
keys = data.keys; Qref = []; err6 = [];
cols = lines(numel(keys));
for i = 1:numel(keys)
    rows = sortrows(data(keys{i}), 1);
    ref = rows(rows(:,1) == 1e-7, :);
    if isempty(ref), continue; end
    coarse = rows(rows(:,1) > 1e-7, :);
    eQ = 100 * abs(coarse(:,3) - ref(3)) / ref(3);
    lbl = sprintf('%s  (Q_{ref} %.0f)', keys{i}, ref(3));
    plot(ax1, coarse(:,1), eQ, 'o-', 'Color', cols(i,:), ...
        'MarkerFaceColor', cols(i,:), 'DisplayName', lbl);
    i6 = coarse(:,1) == 1e-6;
    if any(i6), Qref(end+1) = ref(3); err6(end+1) = eQ(i6); end %#ok<SAGROW>
end
yline(ax1, 2, 'k--', '2% tolerance', 'HandleVisibility', 'off');
xlabel(ax1, 'auto shutoff min'); ylabel(ax1, '|\DeltaQ|/Q  [%]  (vs 1e-7)');
legend(ax1, 'Location', 'northwest', 'FontSize', 8);
title(ax1, 'Q error vs shutoff threshold');

p = polyfit(log(Qref), log(err6), 1);
qq = logspace(log10(min(Qref)*0.8), log10(max(Qref)*1.3), 50);
plot(ax2, Qref, err6, 'ks', 'MarkerFaceColor', [0.3 0.3 0.3], ...
    'DisplayName', 'devices @ 1e-6');
plot(ax2, qq, exp(polyval(p, log(qq))), 'r-', ...
    'DisplayName', sprintf('\\propto Q^{%.2f}', p(1)));
yline(ax2, 2, 'k--', '2% tolerance', 'HandleVisibility', 'off');
xlabel(ax2, 'device Q at 1e-7'); ylabel(ax2, '|\DeltaQ|/Q at 1e-6  [%]');
legend(ax2, 'Location', 'northwest');
title(ax2, 'Error collapses on Q alone (TE+TM, all families)');
sgtitle('\pi-shift auto-shutoff convergence — verdict: 1e-7 for all devices');

out = fullfile(dirs{1}, '..', 'autoshutoff_qspan_verdict');
savefig(fig, [out '.fig']);
exportgraphics(fig, [out '.png'], 'Resolution', 150);
fprintf('saved %s.fig/.png\n', out);
