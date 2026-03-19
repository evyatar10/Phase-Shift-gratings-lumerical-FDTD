% plot_convergence.m
% ------------------
% Loads convergence_farfield.mat and produces 3 figures:
%   Figure 1: Top monitor grid  (near-field + far-field at each Z distance)
%   Figure 2: Side monitor grid (near-field + far-field at each Y distance)
%   Figure 3: FWHM half-angle vs monitor distance (convergence plot)

clear; clc;

% ── CONFIG ─────────────────────────────────────────────────────────────────
RESULTS_DIR = 'C:\Users\evyat\Lumerical\long_bragg_grating_newer_results\radiation_angles_8\results';
MAT_FILE    = fullfile(RESULTS_DIR, 'convergence_farfield.mat');
NF_CROP_X_UM = 30;  % crop near-field X to +/- N um ([] = no crop)
% ───────────────────────────────────────────────────────────────────────────

data = load(MAT_FILE);
n = double(data.n_steps);
lambda_res = double(data.lambda_res);
core_h = double(data.core_h);
w_wide = double(data.w_wide);

top_ux = double(data.top_ux(:));
top_uy = double(data.top_uy(:));
side_ux = double(data.side_ux(:));
side_uy = double(data.side_uy(:));

top_z = double(data.top_z_positions(:));
side_y = double(data.side_y_positions(:));

% Distance from waveguide surface (for labels)
top_dist_um = (top_z - core_h / 2) * 1e6;
top_dist_lam = (top_z - core_h / 2) / lambda_res;
side_dist_um = (side_y - w_wide / 2) * 1e6;
side_dist_lam = (side_y - w_wide / 2) / lambda_res;

% Near-field axes
top_nf_x = double(data.top_nf_x(:));
top_nf_y = double(data.top_nf_y(:));
side_nf_x = double(data.side_nf_x(:));
side_nf_z = double(data.side_nf_z(:));


%% ═══════════════════════════════════════════════════════════════════════════
%  FIGURE 1: Top Monitor Grid
%  ═══════════════════════════════════════════════════════════════════════════
fig1 = figure('Name', 'Top Monitor Convergence', 'Position', [30 80 1800 700]);
sgtitle('Top Monitor (Z-normal) — Distance Convergence', ...
    'FontWeight', 'bold', 'FontSize', 15);

[UX_top, UY_top] = meshgrid(top_ux, top_uy);
UX_top = UX_top'; UY_top = UY_top';
valid_top = (UX_top.^2 + UY_top.^2) <= 1.0;

for i = 1:n
    % ── Near-field (top row) ──
    ax_nf = subplot(2, n, i);
    E2_nf = squeeze(data.top_nf_E2(i, :, :));

    % Crop X
    v_axis = top_nf_x;
    h_axis = top_nf_y;
    if ~isempty(NF_CROP_X_UM)
        mask = abs(v_axis) <= NF_CROP_X_UM * 1e-6;
        v_axis = v_axis(mask);
        E2_nf = E2_nf(mask, :);
    end

    imagesc(ax_nf, h_axis * 1e6, v_axis * 1e6, E2_nf);
    set(ax_nf, 'YDir', 'normal');
    axis(ax_nf, 'tight');
    colormap(ax_nf, jet);
    title(ax_nf, sprintf('z = %.2f \\mum\n(%.2f\\lambda from core)', ...
        top_z(i) * 1e6, top_dist_lam(i)), 'FontSize', 9);
    if i == 1
        ylabel(ax_nf, 'x [\\mum]');
    end
    xlabel(ax_nf, 'y [\\mum]');

    % ── Far-field (bottom row) ──
    ax_ff = subplot(2, n, n + i);
    E2_ff_raw = squeeze(data.top_E2(i, :, :));
    E2_ff = zeros(size(E2_ff_raw));
    E2_ff(valid_top) = E2_ff_raw(valid_top);

    imagesc(ax_ff, top_uy, top_ux, E2_ff);
    set(ax_ff, 'YDir', 'normal');
    axis(ax_ff, 'equal');
    xlim(ax_ff, [-1.05 1.05]); ylim(ax_ff, [-1.05 1.05]);
    colormap(ax_ff, jet);
    hold(ax_ff, 'on');
    phi_c = linspace(0, 2*pi, 360);
    plot(ax_ff, cos(phi_c), sin(phi_c), 'k-', 'LineWidth', 0.6);
    hold(ax_ff, 'off');
    if i == 1
        ylabel(ax_ff, 'u_x');
    end
    xlabel(ax_ff, 'u_y');
end


%% ═══════════════════════════════════════════════════════════════════════════
%  FIGURE 2: Side Monitor Grid
%  ═══════════════════════════════════════════════════════════════════════════
fig2 = figure('Name', 'Side Monitor Convergence', 'Position', [50 60 1800 700]);
sgtitle('Side Monitor (Y-normal) — Distance Convergence', ...
    'FontWeight', 'bold', 'FontSize', 15);

[UX_side, UY_side] = meshgrid(side_ux, side_uy);
UX_side = UX_side'; UY_side = UY_side';
valid_side = (UX_side.^2 + UY_side.^2) <= 1.0;

for i = 1:n
    % ── Near-field (top row) ──
    ax_nf = subplot(2, n, i);
    E2_nf = squeeze(data.side_nf_E2(i, :, :));

    v_axis = side_nf_x;
    h_axis = side_nf_z;
    if ~isempty(NF_CROP_X_UM)
        mask = abs(v_axis) <= NF_CROP_X_UM * 1e-6;
        v_axis = v_axis(mask);
        E2_nf = E2_nf(mask, :);
    end

    imagesc(ax_nf, h_axis * 1e6, v_axis * 1e6, E2_nf);
    set(ax_nf, 'YDir', 'normal');
    axis(ax_nf, 'tight');
    colormap(ax_nf, jet);
    title(ax_nf, sprintf('y = %.2f \\mum\n(%.2f\\lambda from edge)', ...
        side_y(i) * 1e6, side_dist_lam(i)), 'FontSize', 9);
    if i == 1
        ylabel(ax_nf, 'x [\\mum]');
    end
    xlabel(ax_nf, 'z [\\mum]');

    % ── Far-field (bottom row) ──
    ax_ff = subplot(2, n, n + i);
    E2_ff_raw = squeeze(data.side_E2(i, :, :));
    E2_ff = zeros(size(E2_ff_raw));
    E2_ff(valid_side) = E2_ff_raw(valid_side);

    imagesc(ax_ff, side_uy, side_ux, E2_ff);
    set(ax_ff, 'YDir', 'normal');
    axis(ax_ff, 'equal');
    xlim(ax_ff, [-1.05 1.05]); ylim(ax_ff, [-1.05 1.05]);
    colormap(ax_ff, jet);
    hold(ax_ff, 'on');
    phi_c = linspace(0, 2*pi, 360);
    plot(ax_ff, cos(phi_c), sin(phi_c), 'k-', 'LineWidth', 0.6);
    hold(ax_ff, 'off');
    if i == 1
        ylabel(ax_ff, 'u_x');
    end
    xlabel(ax_ff, 'u_z');
end


%% ═══════════════════════════════════════════════════════════════════════════
%  FIGURE 3: FWHM vs Distance (Convergence Plot)
%  ═══════════════════════════════════════════════════════════════════════════
fig3 = figure('Name', 'FWHM Convergence', 'Position', [100 100 1000 450]);
sgtitle('Far-Field FWHM Half-Angle vs Monitor Distance', ...
    'FontWeight', 'bold', 'FontSize', 14);

% ── Top monitors ──
top_fwhm_half = zeros(1, n);
for i = 1:n
    E2_i = squeeze(data.top_E2(i, :, :));
    E2_m = E2_i;
    E2_m(~valid_top) = 0;
    [~, peak_lin] = max(E2_m(:));
    [pi_ux, ~] = ind2sub(size(E2_m), peak_lin);

    slice_uy = E2_i(pi_ux, :);
    fwhm_deg = compute_fwhm_deg(slice_uy, top_uy');
    top_fwhm_half(i) = fwhm_deg / 2;
    fprintf('Top monitor %d (z=%.2f um, %.2f lam): FWHM half-angle = %.2f deg\n', ...
        i, top_z(i)*1e6, top_dist_lam(i), top_fwhm_half(i));
end

subplot(1, 2, 1);
plot(top_dist_lam, top_fwhm_half, '-o', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.2 0.5 0.8], 'Color', [0.2 0.5 0.8]);
xlabel('Distance from core surface [\lambda]', 'FontSize', 11);
ylabel('FWHM half-angle [deg]', 'FontSize', 11);
title('Top Monitor (Z-normal)', 'FontSize', 12);
grid on;
% Add secondary X axis in um
ax1 = gca;
ax1_pos = ax1.Position;
ax1b = axes('Position', ax1_pos, 'XAxisLocation', 'top', 'Color', 'none', ...
    'YTick', [], 'XLim', [top_dist_um(1) top_dist_um(end)]);
xlabel(ax1b, 'Distance from core surface [\mum]', 'FontSize', 10);

% ── Side monitors ──
side_fwhm_half = zeros(1, n);
for i = 1:n
    E2_i = squeeze(data.side_E2(i, :, :));
    E2_m = E2_i;
    E2_m(~valid_side) = 0;
    [~, peak_lin] = max(E2_m(:));
    [pi_ux, ~] = ind2sub(size(E2_m), peak_lin);

    slice_uy = E2_i(pi_ux, :);
    fwhm_deg = compute_fwhm_deg(slice_uy, side_uy');
    side_fwhm_half(i) = fwhm_deg / 2;
    fprintf('Side monitor %d (y=%.2f um, %.2f lam): FWHM half-angle = %.2f deg\n', ...
        i, side_y(i)*1e6, side_dist_lam(i), side_fwhm_half(i));
end

subplot(1, 2, 2);
plot(side_dist_lam, side_fwhm_half, '-s', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.8 0.4 0.2], 'Color', [0.8 0.4 0.2]);
xlabel('Distance from waveguide edge [\lambda]', 'FontSize', 11);
ylabel('FWHM half-angle [deg]', 'FontSize', 11);
title('Side Monitor (Y-normal)', 'FontSize', 12);
grid on;
% Secondary X axis in um
ax2 = gca;
ax2_pos = ax2.Position;
ax2b = axes('Position', ax2_pos, 'XAxisLocation', 'top', 'Color', 'none', ...
    'YTick', [], 'XLim', [side_dist_um(1) side_dist_um(end)]);
xlabel(ax2b, 'Distance from waveguide edge [\mum]', 'FontSize', 10);


%% ── HELPER: compute_fwhm_deg ─────────────────────────────────────────────
function fwhm = compute_fwhm_deg(profile, axis_uc)
    profile = profile(:)'; axis_uc = axis_uc(:)';
    half    = max(profile) / 2;
    above   = profile >= half;
    cross   = find(diff(above) ~= 0);
    if numel(cross) < 2; fwhm = NaN; return; end
        function theta = interp_c(i)
            x0 = axis_uc(i); x1 = axis_uc(i+1);
            y0 = profile(i); y1 = profile(i+1);
            uc = x0 + (half - y0) / (y1 - y0) * (x1 - x0);
            theta = asind(min(abs(uc), 1)) * sign(uc);
        end
    fwhm = abs(interp_c(cross(end)) - interp_c(cross(1)));
end
