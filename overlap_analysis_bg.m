% MATLAB Script: calculate_overlap_profile.m

% --- USER CONFIGURATION ---


% Paths (Copied directly from your snippet)
filename_short = 'C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_v6_3d_profiles\results\result_60_periods_CONST_3D_crop.mat';
filename_long = 'C:\Users\evyat\Lumerical\long_bragg_grating_interconnect\bragg_fdtd_elements_v6_3d_profiles\results\result_100_periods_CONST_3D_crop.mat';

% Run Analysis
calculate_overlap_profile(filename_long, filename_short);


% --- MAIN FUNCTION ---
function calculate_overlap_profile(file_long, file_short)
    % Calculates the field overlap integral between two devices as a function of X.
    % Uses all 3 field components (Ex, Ey, Ez).
    % Forces alignment to the Short Device's grid.

    % Check if files exist
    if exist(file_long, 'file') ~= 2
        fprintf('ERROR: Long file not found: %s\n', file_long);
        return;
    end
    if exist(file_short, 'file') ~= 2
        fprintf('ERROR: Short file not found: %s\n', file_short);
        return;
    end

    [~, nameL, ~] = fileparts(file_long);
    fprintf('Loading Long Device: %s\n', nameL);
    % Load .mat file. Struct access is equivalent to simplify_cells=True
    data_L = load(file_long);

    [~, nameS, ~] = fileparts(file_short);
    fprintf('Loading Short Device: %s\n', nameS);
    data_S = load(file_short);

    try
        [xL, yL, zL, EL] = unpack_3d(data_L, 'Long');
        [xS, yS, zS, ES] = unpack_3d(data_S, 'Short');
    catch ME
        fprintf('Data Unpacking Error: %s\n', ME.message);
        return;
    end

    % --- 1. SETUP INTERPOLATORS ---
    fprintf('Building interpolators for Long Field (Ex, Ey, Ez)...\n');
    
    % Safety Check for Dimensions: Ensure EL is (Nx, Ny, Nz, 3)
    if ndims(EL) == 5
        EL = squeeze(EL);
    end
    
    % Create griddedInterpolant objects for 3D interpolation
    % Inputs are grid vectors {x, y, z} and values.
    % 'linear' interpolation, 'none' for extrapolation (returns NaN outside domain)
    F_Ex = griddedInterpolant({xL, yL, zL}, double(EL(:,:,:,1)), 'linear', 'none');
    F_Ey = griddedInterpolant({xL, yL, zL}, double(EL(:,:,:,2)), 'linear', 'none');
    F_Ez = griddedInterpolant({xL, yL, zL}, double(EL(:,:,:,3)), 'linear', 'none');

    % --- 2. DEFINE THE COMMON GRID ---
    x_min = max(min(xL), min(xS));
    x_max = min(max(xL), max(xS));

    valid_mask = (xS >= x_min) & (xS <= x_max);
    x_common = xS(valid_mask);
    
    % Get indices of x_common relative to xS
    idx_map = find(valid_mask);

    overlap_vals = zeros(length(x_common), 1);
    fprintf('Calculating Overlap for %d slices...\n', length(x_common));

    % Create 2D grid for Y-Z slices (Short device grid)
    % ndgrid is used to match the dimension order of the loaded arrays (y, z)
    [YY, ZZ] = ndgrid(yS, zS);

    for i = 1:length(x_common)
        x_val = x_common(i);
        idx_S = idx_map(i);

        % A. Short Field Slice
        % Extract slice at idx_S. Result is (1, Ny, Nz, 3) -> squeeze to (Ny, Nz, 3)
        E_S_slice = squeeze(ES(idx_S, :, :, :)); 

        % B. Long Field Slice (Interpolated)
        % Create query points for X (constant x_val across the Y-Z plane)
        XX_query = repmat(x_val, size(YY));
        
        % Interpolate
        Ex_L = F_Ex(XX_query, YY, ZZ);
        Ey_L = F_Ey(XX_query, YY, ZZ);
        Ez_L = F_Ez(XX_query, YY, ZZ);
        
        % Replace NaNs with 0j (equivalent to bounds_error=False, fill_value=0j)
        Ex_L(isnan(Ex_L)) = 0;
        Ey_L(isnan(Ey_L)) = 0;
        Ez_L(isnan(Ez_L)) = 0;

        % Stack components back into (Ny, Nz, 3) matrix
        E_L_slice = cat(3, Ex_L, Ey_L, Ez_L);

        % C. Overlap Integral
        % Dot product: sum(E_L * conj(E_S)) over component dimension (3)
        dot_prod = sum(E_L_slice .* conj(E_S_slice), 3);

        % Integration using trapezoidal rule
        % First integrate over Z (dimension 2 of the slice)
        integ_z = trapz(zS, dot_prod, 2);
        % Then integrate over Y (dimension 1 of the slice)
        integ_overlap = trapz(yS, integ_z, 1);

        % Norm L
        dens_L = sum(abs(E_L_slice).^2, 3);
        norm_L = trapz(yS, trapz(zS, dens_L, 2), 1);

        % Norm S
        dens_S = sum(abs(E_S_slice).^2, 3);
        norm_S = trapz(yS, trapz(zS, dens_S, 2), 1);

        if (norm_L > 0) && (norm_S > 0)
            O_x = (abs(integ_overlap)^2) / (norm_L * norm_S);
        else
            O_x = 0.0;
        end
        overlap_vals(i) = O_x;
    end

    % Plot
    % figure;
    plot(x_common * 1e6, overlap_vals, 'LineWidth', 2);
    xlabel('Position X [um]');
    ylabel('Mode Overlap Factor (0-1)');
    title({ 'Overlap Profile', sprintf('%s vs %s', nameL, nameS) }, 'Interpreter', 'none');
    grid on;
    ylim([0, 1.05]);
end

% --- HELPER: UNPACK 3D DATA SAFELY ---
function [x, y, z, E_res] = unpack_3d(data, name)
    if ~isfield(data, 'field_3d')
        error('File %s is missing "field_3d". Did you run with record_3d_fields=True?', name);
    end

    f3d = data.field_3d;

    % Access fields directly
    if isfield(f3d, 'x')
        x = f3d.x;
        y = f3d.y;
        z = f3d.z;
        E_res = f3d.E_res;
    else
        fprintf('Debug - Available keys: %s\n', strjoin(fieldnames(f3d)', ', '));
        error('Key Error: Standard keys (x, y, z, E_res) not found in field_3d');
    end
    
    % Ensure data types are double for interpolation
    x = double(x);
    y = double(y);
    z = double(z);
end