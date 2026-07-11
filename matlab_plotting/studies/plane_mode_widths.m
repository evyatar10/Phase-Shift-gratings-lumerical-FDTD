function [h_um, v_um] = plane_mode_widths(matpath)
% PLANE_MODE_WIDTHS  Longitudinal cavity-mode FWHM (along x) in two planes.
%
%   [h_um, v_um] = plane_mode_widths(matpath)
%
%   h_um : "horizontal" / in-plane width  — |E|^2 integrated over Y from the
%          top (XY / yx) monitor field_xy, FWHM of the x-envelope [um].
%   v_um : "vertical" / out-of-plane width — |E|^2 integrated over Z from the
%          side (XZ / zx) monitor field_xz_side, FWHM of the x-envelope [um].
%
%   Returns NaN for a plane whose 2D monitor was not recorded in the file.
%
%   This is the MATLAB twin of the Python algorithm in sim_helpers.py
%   (extract_and_process_field_profile + extract_envelope_peaks +
%   calculate_fwhm_relative): integrate |E|^2 over the transverse axis at the
%   resonance wavelength, crop to the grating extent (|x| <= L_device/2),
%   connect standing-wave peaks into an envelope, then take the FWHM relative
%   to the envelope floor. Recomputing the horizontal value here (rather than
%   reusing the stored scalar fwhm_m, which comes from the separate 1D monitor)
%   keeps H and V on an identical footing so the two planes are comparable.

    h_um = NaN;  v_um = NaN;
    d = load(matpath);
    if ~isfield(d, 'resonance_wavelength_nm') || ~isfield(d, 'L_device')
        return;
    end
    res_wl_m = double(d.resonance_wavelength_nm) * 1e-9;
    x_limit  = double(d.L_device) / 2.0;

    if isfield(d, 'field_xy')
        h_um = plane_width(d.field_xy, 'y', res_wl_m, x_limit);
    end
    if isfield(d, 'field_xz_side')
        v_um = plane_width(d.field_xz_side, 'z', res_wl_m, x_limit);
    end
end

% ── one plane ────────────────────────────────────────────────────────────────
function w_um = plane_width(s, transverse, res_wl_m, x_limit)
    w_um = NaN;
    if isempty(s) || ~isfield(s, 'E_res'); return; end

    x   = double(s.x(:));
    lam = double(s.lambda_3d(:));
    [~, idx] = min(abs(lam - res_wl_m));        % resonance lambda slice

    % E_res dims: (x, y, z, lambda, component). |E|^2 summed over components.
    E = s.E_res;
    Ei = squeeze(E(:, :, :, idx, :));           % (x, y, z, comp)  [a singleton
    I  = sum(abs(Ei).^2, ndims(Ei));            %  transverse dim survives squeeze]

    % Integrate |E|^2 over the transverse axis -> I(x).
    if transverse == 'y'
        coord = double(s.y(:));                  % field_xy: y is the long axis
        Ix = trapz_axis(I, coord, 2);            % I is (x, y)
    else
        coord = double(s.z(:));                  % field_xz_side: z is the long axis
        Ix = trapz_axis(I, coord, 2);            % I squeezes to (x, z)
    end
    Ix = Ix(:);

    % Crop to grating extent.
    m  = abs(x) <= x_limit;
    xc = x(m);  Ic = Ix(m);
    if numel(xc) < 3; return; end

    env  = envelope_peaks(xc, Ic);
    w_um = fwhm_relative(xc, env) * 1e6;
end

% trapezoid integration of A along DIM using non-uniform sample points c.
function out = trapz_axis(A, c, dim)
    if isscalar(c)              % single transverse sample -> nothing to integrate
        out = A;
    else
        out = trapz(c, A, dim);
    end
end

% Envelope = cubic-spline through the standing-wave peaks, flat-extrapolated.
function e = envelope_peaks(x, y)
    [pks, locs] = findpeaks(y);
    if numel(pks) < 2
        e = y; return;
    end
    xp = x(locs);
    e  = interp1(xp, pks, x, 'spline');
    e(x < xp(1))   = pks(1);          % match scipy fill_value=(first,last)
    e(x > xp(end)) = pks(end);
end

% FWHM relative to the signal floor, linear-interpolated half-max crossings.
function w = fwhm_relative(x, y)
    w = 0.0;
    ymax = max(y);  ymin = min(y);
    tgt  = ymin + 0.5 * (ymax - ymin);
    s    = sign(y - tgt);
    zc   = find(diff(s) ~= 0);
    if numel(zc) < 2; return; end
    xL = interp_cross(x, y, zc(1),   tgt);
    xR = interp_cross(x, y, zc(end), tgt);
    w  = xR - xL;
end

function xc = interp_cross(x, y, i, tgt)
    slope = (y(i+1) - y(i)) / (x(i+1) - x(i));
    if slope == 0
        xc = x(i);
    else
        xc = x(i) + (tgt - y(i)) / slope;
    end
end
