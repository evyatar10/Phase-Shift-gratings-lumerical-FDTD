% plot_shape_study.m — whole-device tooth-profile study (job 117042), TM + TE.
% dLoss vs the matching control (sym rows vs sym control; wall-phase rows vs
% the nosym control), with mode-width change annotated. Headless-safe.

clear; close all;

proj = 'c:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes';
res_dir = fullfile(proj, 'results_from_athena', 'shape_study', 'results');
out_dir = fullfile(proj, 'results_from_athena', 'shape_study');

fl = dir(fullfile(res_dir, 'result_*.mat'));
S = struct('pol', {}, 'prof', {}, 'wp', {}, 'nosym', {}, 'loss', {}, 'fw', {}, 'lam', {});
for k = 1:numel(fl)
    d = load(fullfile(fl(k).folder, fl(k).name));
    [~, i] = min(abs(d.wl_nm - d.resonance_wavelength_nm));
    S(end+1) = struct( ...
        'pol',  ternary(contains(fl(k).name, '_TM_'), "TM", "TE"), ...
        'prof', strtrim(string(d.corrugation_profile)), ...
        'wp',   d.wall_phase_offset_deg, ...
        'nosym', contains(fl(k).name, '_nosym'), ...
        'loss', 1 - d.resonance_transmission - d.R(i), ...
        'fw',   d.fwhm_m * 1e6, ...
        'lam',  d.resonance_wavelength_nm); %#ok<SAGROW>
end

fig = figure('Visible', 'off', 'Position', [80 80 1150 470]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for pol = ["TM", "TE"]
    sub = S(strcmp([S.pol], pol));
    csym = sub(strcmp([S(strcmp([S.pol], pol)).prof], "rect") & [sub.wp] == 0 & ~[sub.nosym]);
    csym = csym(1);
    cns_idx = find([sub.wp] == 0 & [sub.nosym] & strcmp([sub.prof], "rect"), 1);
    labs = {}; dl = []; fwp = [];
    for k = 1:numel(sub)
        r = sub(k);
        if r.prof == "rect" && r.wp == 0, continue; end
        if r.wp ~= 0
            ctrl = sub(cns_idx);
            labs{end+1} = sprintf('wall-phase %d%c', round(r.wp), char(176)); %#ok<SAGROW>
        else
            ctrl = csym;
            labs{end+1} = sprintf('%s teeth', r.prof); %#ok<SAGROW>
        end
        dl(end+1) = (r.loss - ctrl.loss) * 1e3;      %#ok<SAGROW>
        fwp(end+1) = (r.fw / ctrl.fw - 1) * 100;     %#ok<SAGROW>
    end
    [dl, ord] = sort(dl); labs = labs(ord); fwp = fwp(ord);

    nexttile; hold on; grid on;
    b = bar(dl, 'FaceColor', 'flat');
    cols = zeros(numel(dl), 3);
    cols(dl < 0, :) = repmat([0.13 0.55 0.33], nnz(dl < 0), 1);
    cols(dl >= 0, :) = repmat([0.75 0.32 0.20], nnz(dl >= 0), 1);
    viol = abs(fwp) > 15;                      % mode-width constraint violated
    cols(viol, :) = repmat([0.62 0.62 0.62], nnz(viol), 1);
    b.CData = cols;
    yline(0, 'k-');
    for k = 1:numel(dl)
        s = sprintf('fwhm %+.0f%%', fwp(k));
        if viol(k), s = [s ' (violates constraint)']; end %#ok<AGROW>
        text(k, dl(k) / 2, s, 'HorizontalAlignment', 'center', ...
            'FontSize', 8, 'Interpreter', 'none', 'Rotation', 90 * viol(k));
    end
    set(gca, 'XTick', 1:numel(dl), 'XTickLabel', labs, 'XTickLabelRotation', 20, ...
        'TickLabelInterpreter', 'none');
    ylabel('\Delta loss vs control (\times10^{-3})');
    title(sprintf('%s — control: \\lambda_{res}=%.2f nm, loss=%.4f, fwhm=%.1f \\mum', ...
        pol, csym.lam, csym.loss, csym.fw));
end
sgtitle(sprintf(['Tooth-profile study, \\kappa-matched depths (\\pi-shift, h 350 nm, N=80)\n' ...
    'green = less loss; wall-phase rows vs their own symmetry-off control']));

exportgraphics(fig, fullfile(out_dir, 'shape_study_summary.png'), 'Resolution', 200);
savefig(fig, fullfile(out_dir, 'shape_study_summary.fig'));
fprintf('saved: %s\n', fullfile(out_dir, 'shape_study_summary.png'));

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end
