% plot_scatterer_greens.m — response-matrix (Green's) scatterer program overview.
% Study dir: runners/scatterers/ | results_from_athena/scat_c_response (+ scat_b_gates,
% scat_e_validate) | Created 2026-07-12 | Job(s): TBD (fill in after dispatch).
% Purpose: panel 1 = measured |response| per candidate position with the numerics
% noise floor and the chosen combination; panel 2 = measured dT per validated
% combination vs the stage-D predicted far-field reduction.
% Inputs are produced by runners/scatterers/solve_response_matrix.py (gates/solve).

function plot_scatterer_greens()
base_c = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_athena', ...
                  'scat_c_response', 'results');
base_b = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_athena', ...
                  'scat_b_gates', 'results');
base_e = fullfile(fileparts(mfilename('fullpath')), '..', 'results_from_athena', ...
                  'scat_e_validate', 'results');

rep_file = fullfile(base_c, 'greens_report.json');
assert(exist(rep_file, 'file') == 2, ...
    'greens_report.json not found — run: python runners/scatterers/solve_response_matrix.py solve');
rep = jsondecode(fileread(rep_file));

floor_norm = NaN;
gates_file = fullfile(base_b, 'gates_report.json');
if exist(gates_file, 'file') == 2
    g = jsondecode(fileread(gates_file));
    floor_norm = g.floor_cross_norm;
end

% ── Stage-E measured results (optional — panel 2 appears when downloaded) ──────
e_files = dir(fullfile(base_e, 'result_*.mat'));
meas = struct('n', {}, 'dT', {}, 'dloss', {}, 'label', {});
if ~isempty(e_files)
    T0 = NaN; L0 = NaN;
    runs = cell(1, numel(e_files));
    for i = 1:numel(e_files)
        d = load(fullfile(e_files(i).folder, e_files(i).name), ...
                 'scatterer_r_m', 'scatterer_x_list_m', 'scatterer_n_sites', ...
                 'resonance_transmission', 'resonance_wavelength_nm', ...
                 'wl_nm', 'loss', 'T', 'field_envelope_1D', 'field_x', 'fwhm_m');
        [~, ii] = min(abs(d.wl_nm(:) - d.resonance_wavelength_nm));
        d.loss_res = d.loss(ii);
        runs{i} = d;
        if d.scatterer_r_m == 0, T0 = d.resonance_transmission; L0 = d.loss_res; end
    end
    for i = 1:numel(runs)
        d = runs{i};
        if d.scatterer_r_m == 0, continue; end
        k = numel(meas) + 1;
        meas(k).n     = double(d.scatterer_n_sites);
        meas(k).dT    = d.resonance_transmission - T0;
        meas(k).dloss = d.loss_res - L0;
        meas(k).label = ['[' num2str(round(d.scatterer_x_list_m(:)' * 1e9)) '] nm'];
    end
end

% ── Figure ─────────────────────────────────────────────────────────────────────
fig = figure('Visible', 'off', 'Position', [80 80 900 640], 'Color', 'w');
n_panels = 1 + double(~isempty(meas));
tl = tiledlayout(fig, n_panels, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1 — |response| vs candidate position + chosen combination.
ax1 = nexttile(tl);
x_um = rep.positions_nm(:) / 1000;
resp_sc = rep.response_norm(:) * 1e8;   % scale into the ylabel, no axis exponent
stem(ax1, x_um, resp_sc, 'filled', 'MarkerSize', 3, ...
     'DisplayName', '|response_j|');
hold(ax1, 'on');
if isfinite(floor_norm)
    yline(ax1, floor_norm * 1e8, 'r--', 'DisplayName', 'numerics floor');
end
if isfield(rep, 'greedy') && ~isempty(rep.greedy.positions_nm)
    gx = rep.greedy.positions_nm(:) / 1000;
    [~, gi] = ismember(round(gx * 1000, 1), round(rep.positions_nm(:), 1));
    plot(ax1, gx, resp_sc(gi), 'ks', 'MarkerSize', 9, 'LineWidth', 1.2, ...
         'DisplayName', sprintf('chosen combination (%d)', numel(gx)));
end
hold(ax1, 'off');
grid(ax1, 'on');
xlabel(ax1, 'scatterer position x (\mum)');
ylabel(ax1, '|response| (far-field norm, \times10^{-8})');
legend(ax1, 'Location', 'best', 'FontSize', 8);
title(ax1, sprintf(['\\pi-shift TM corr-400 — single-scatterer far-field responses', ...
    '   \\lambda_{res} = %.2f nm,  T_0 = %.3f,  LS ceiling = %.1f%%'], ...
    rep.lambda_res_nm, rep.baseline_T_res, 100 * rep.ls_ceiling_frac));

% Panel 2 — measured dT per validated combination vs predicted reduction.
if ~isempty(meas)
    ax2 = nexttile(tl);
    bar(ax2, [meas.dT]);
    set(ax2, 'XTick', 1:numel(meas), 'XTickLabel', {meas.label});
    grid(ax2, 'on');
    ylabel(ax2, '\DeltaT at resonance (measured)');
    pred = '';
    if isfield(rep, 'greedy')
        pred = sprintf('predicted far-field reduction: greedy %.1f%%', ...
                       100 * rep.greedy.power_reduction_frac);
        if isfield(rep, 'periodic_best') && ~isempty(rep.periodic_best)
            pred = sprintf('%s, periodic %.1f%%', pred, ...
                           100 * rep.periodic_best.power_reduction_frac);
        end
    end
    title(ax2, sprintf('Validated combinations — %s', pred));
end

out_png = fullfile(base_c, '..', 'scatterer_greens_overview.png');
out_fig = fullfile(base_c, '..', 'scatterer_greens_overview.fig');
exportgraphics(fig, out_png, 'Resolution', 200);
set(fig, 'Visible', 'on');   % else the saved .fig opens as an invisible window
savefig(fig, out_fig);
close(fig);
fprintf('Saved:\n  %s\n  %s\n', char(java.io.File(out_png).getCanonicalPath()), ...
        char(java.io.File(out_fig).getCanonicalPath()));

% ── Comparison figures: regular vs best measured combination (2 sites) ─────────
if ~isempty(meas)
    ctrl = []; win = [];
    for i = 1:numel(runs)
        d = runs{i};
        if d.scatterer_r_m == 0, ctrl = d;
        elseif double(d.scatterer_n_sites) == 2, win = d; end
    end
    if ~isempty(ctrl) && ~isempty(win)
        win_lbl = ['pillar pairs [' num2str(round(win.scatterer_x_list_m(:)' * 1e9)) '] nm'];

        f2 = figure('Visible', 'off', 'Position', [80 80 880 430], 'Color', 'w');
        ax = axes(f2);
        plot(ax, ctrl.wl_nm, ctrl.T, 'LineWidth', 1.1, 'DisplayName', ...
             sprintf('regular:  T_{res} = %.3f @ %.2f nm', ...
                     ctrl.resonance_transmission, ctrl.resonance_wavelength_nm));
        hold(ax, 'on');
        plot(ax, win.wl_nm, win.T, 'LineWidth', 1.1, 'DisplayName', ...
             sprintf('%s:  T_{res} = %.3f @ %.2f nm', win_lbl, ...
                     win.resonance_transmission, win.resonance_wavelength_nm));
        hold(ax, 'off');
        grid(ax, 'on');
        xlabel(ax, 'wavelength (nm)');
        ylabel(ax, 'transmission T');
        legend(ax, 'Location', 'southeast', 'FontSize', 8);
        title(ax, '\pi-shift TM corr-400 (h 350 nm, \Lambda 516.83 nm) — regular vs scatterer-recycled');
        out2p = fullfile(base_e, '..', 'scat_e_T_compare.png');
        out2f = fullfile(base_e, '..', 'scat_e_T_compare.fig');
        exportgraphics(f2, out2p, 'Resolution', 200);
        set(f2, 'Visible', 'on');
        savefig(f2, out2f);
        close(f2);

        f3 = figure('Visible', 'off', 'Position', [80 80 880 430], 'Color', 'w');
        ax = axes(f3);
        plot(ax, ctrl.field_x * 1e6, ctrl.field_envelope_1D, 'LineWidth', 1.1, ...
             'DisplayName', sprintf('regular:  FWHM = %.2f \\mum', abs(ctrl.fwhm_m) * 1e6));
        hold(ax, 'on');
        plot(ax, win.field_x * 1e6, win.field_envelope_1D, 'LineWidth', 1.1, ...
             'DisplayName', sprintf('%s:  FWHM = %.2f \\mum', win_lbl, abs(win.fwhm_m) * 1e6));
        hold(ax, 'off');
        grid(ax, 'on');
        xlabel(ax, 'x (\mum)');
        ylabel(ax, 'field envelope (a.u.)');
        legend(ax, 'Location', 'northeast', 'FontSize', 8);
        title(ax, '\pi-shift TM corr-400 — cavity-mode envelope, regular vs scatterer-recycled');
        out3p = fullfile(base_e, '..', 'scat_e_envelope_compare.png');
        out3f = fullfile(base_e, '..', 'scat_e_envelope_compare.fig');
        exportgraphics(f3, out3p, 'Resolution', 200);
        set(f3, 'Visible', 'on');
        savefig(f3, out3f);
        close(f3);

        fprintf('Saved:\n  %s\n  %s\n  %s\n  %s\n', ...
            char(java.io.File(out2p).getCanonicalPath()), char(java.io.File(out2f).getCanonicalPath()), ...
            char(java.io.File(out3p).getCanonicalPath()), char(java.io.File(out3f).getCanonicalPath()));
    end
end
end
