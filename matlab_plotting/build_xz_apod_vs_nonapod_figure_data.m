% build_xz_apod_vs_nonapod_figure_data.m
% Study: xz_apod_vs_nonapod_n80 (Athena job 175019, 2026-10-10).
% Purpose: pack the two re-simulated N=80 devices (non-apodized / 10-period
% linear apodization) behind the 2026-04-02 "XZ Zoomed Side View" figure into
% ONE .mat with the XZ |E|^2 map at the resonance bin, Poynting components, the
% spectrum, the 1D profile and the device parameters, then re-plot both panels
% as a check. Input = the *_slim.mat files reduced on Athena by
% athena/scripts/slim_xz_result.py (field_xz_side kept at ONE lambda bin).
%
% Usage: run as-is, or set `in_files` / `out_mat` before calling.

if ~exist('in_files', 'var')
    res_dir  = 'C:\Users\evyat\Lumerical\phase_shift_grating_FTDT_codes\results_from_athena\xz_apod_vs_nonapod_n80\results';
    in_files = {fullfile(res_dir, 'result_N80_avg_Ybox8p8_Zbox8p2_slim.mat'), ...
                fullfile(res_dir, 'result_N80_A10_M4_avg_Ybox8p8_Zbox8p2_slim.mat')};   % 5λ box tag from generate_file_tag
end
if ~exist('out_mat', 'var')
    out_mat = 'C:\Users\evyat\Lumerical\xz_N80_apod_vs_nonapod_figure_data.mat';
end
check_png = strrep(out_mat, '.mat', '_check.png');
crop_um   = 25;      % stored window |x| <= 25 um (what the figure shows is ±12)

% ---- device parameters (user spec 2026-10-10 = runner BASE/SPEC values) ----
N = 80; pitch_nm = 500; w_avg_nm = 800; depth_nm = 300; h_nm = 350;
d = (1:N)';
base = struct('polarization','TE', 'n_periods_each_side',N, 'pitch_nm',pitch_nm, ...
    'avg_width_nm',w_avg_nm, 'corrugation_depth_nm',depth_nm, ...
    'narrow_width_nm',w_avg_nm-depth_nm/2, 'wide_width_nm',w_avg_nm+depth_nm/2, ...
    'core_height_nm',h_nm, 'core_material','Si3N4', ...
    'cavity_length_nm',pitch_nm/2, 'cavity_width_nm',w_avg_nm, 'cavity_width_option','avg', ...
    'tooth_shift_nm',0, 'grating_length_um',80.25, 'port_width_nm',1000, 'substrate_um',10, ...
    'n_core',1.977, 'n_clad',1.44, 'index_model','constant', ...
    'mesh','optimization (dx = 50 nm)', 'box_span_multiplier',5.0, ...
    'runner','runners/sweeps/xz_apod_vs_nonapod_n80.py', 'athena_job',175019);
p_non = base; p_non.apodization = 'none'; p_non.n_apod = 0; p_non.center_mod_depth_nm = depth_nm;
dep = depth_nm*ones(N,1);
p_non.teeth = struct('index',d, 'depth_nm',dep, 'narrow_width_nm',w_avg_nm-dep/2, 'wide_width_nm',w_avg_nm+dep/2, ...
    'note','d=1 is the tooth next to the cavity; same on both sides');
p_apo = base; p_apo.apodization = 'linear'; p_apo.n_apod = 10; p_apo.center_mod_depth_nm = 4;
dep = depth_nm*ones(N,1); k = d <= 10; dep(k) = 4 + (depth_nm-4)*(d(k)-1)/10;
p_apo.teeth = struct('index',d, 'depth_nm',dep, 'narrow_width_nm',w_avg_nm-dep/2, 'wide_width_nm',w_avg_nm+dep/2, ...
    'note','depth = 4 + (300-4)*(d-1)/10 nm for d<=10, 300 nm beyond; d=1 next to the cavity; same on both sides');

keys   = {'nonapodized', 'apodized'};
params = {p_non, p_apo};
S = struct();
for k = 1:2
    data = load(in_files{k});
    fprintf('--- %s\n', in_files{k});

    % ---- file-vs-spec consistency (the file wins; report mismatches) ----
    chk = {'n_periods_each_side', N; 'pitch_m', pitch_nm*1e-9; 'core_height_m', h_nm*1e-9; ...
           'avg_corrugation_width_m', w_avg_nm*1e-9; 'corrugation_depth_m', depth_nm*1e-9};
    for c = 1:size(chk,1)
        if isfield(data, chk{c,1})
            v = double(data.(chk{c,1}));
            if abs(v - chk{c,2}) > 1e-9*max(1,abs(chk{c,2}))
                fprintf('  MISMATCH %s: file %g vs spec %g\n', chk{c,1}, v, chk{c,2});
            end
        end
    end

    % ---- XZ plane at the resonance bin (exactly what plot_field_poynting_zoom.m does) ----
    f  = data.field_xz_side;
    x = double(f.x(:)); y = double(f.y(:)); z = double(f.z(:)); lam = double(f.lambda_3d(:));
    Nx = numel(x); Ny = numel(y); Nz = numel(z); Nlam = numel(lam);
    wl_res = double(data.resonance_wavelength_nm)*1e-9;
    [~, il] = min(abs(lam - wl_res));
    [~, iy] = min(abs(y));
    E5 = reshape(f.E_res, [Nx, Ny, Nz, Nlam, 3]);
    E  = squeeze(E5(:, iy, :, il, :));            % Nx x Nz x 3, complex
    I  = sum(abs(E).^2, 3);
    xz = struct();
    xz.x_um = x'*1e6; xz.z_um = z'*1e6; xz.y_plane_um = y(iy)*1e6;
    xz.lambda_plot_nm = lam(il)*1e9;
    xz.lambda_bins_available_nm = lam'*1e9;
    if isfield(f, 'lambda_3d_full'); xz.lambda_bins_in_run_nm = double(f.lambda_3d_full(:))'*1e9; end
    xz.E = E; xz.I = I; xz.I_dB = 10*log10(I);
    if isfield(f, 'P_res') && ~isempty(f.P_res)      % time-averaged Poynting vector (monitor "P" result)
        P5 = reshape(f.P_res, [Nx, Ny, Nz, Nlam, 3]);
        xz.Px = squeeze(P5(:, iy, :, il, 1)); xz.Pz = squeeze(P5(:, iy, :, il, 3));
        xz.P_units = 'W/m^2, time-averaged Poynting vector P = 0.5*Re(E x H*) from the XZ monitor at lambda_plot_nm; Px along the waveguide, Pz vertical';
    end
    ix = abs(xz.x_um) <= crop_um;
    xz.crop = struct('x_um', xz.x_um(ix), 'z_um', xz.z_um, 'I_dB', xz.I_dB(ix,:), 'E', E(ix,:,:));
    if isfield(xz, 'Px'); xz.crop.Px = xz.Px(ix,:); xz.crop.Pz = xz.Pz(ix,:); end
    xz.note = 'E/I/I_dB/Px/Pz are Nx x Nz (x 3) on the y=0 plane at lambda_plot_nm; crop = |x| <= 25 um';

    % ---- spectrum ----
    sp = struct();
    for fn = {'wl_nm','T','R','loss','S21_complex','S11_complex','resonance_wavelength_nm', ...
              'resonance_transmission','spectral_fwhm_nm'}
        if isfield(data, fn{1}); sp.(fn{1}) = data.(fn{1}); end
    end
    sp.spectral_fwhm_abs_nm = abs(double(data.spectral_fwhm_nm));   % stored sign is a convention
    sp.Q = double(data.resonance_wavelength_nm) / sp.spectral_fwhm_abs_nm;

    % ---- 1D profile + stored geometry ----
    pr = struct();
    for fn = {'field_x','field_energy_density_1D','field_envelope_1D','fwhm_m'}
        if isfield(data, fn{1}); pr.(fn{1}) = data.(fn{1}); end
    end
    pr.x_um = double(data.field_x(:))'*1e6;
    pr.fwhm_um = double(data.fwhm_m)*1e6;
    pr.method = ['1D mode profile along the waveguide (x): a thin z=0 (XY, top-view) field monitor spanning the whole grating ' ...
        '(x span = 2*x_grating_end + 2 um, y span = 1.5*wide width, 501 wavelength bins over the scan) is read at the bin nearest ' ...
        'the resonance; |E|^2 = |Ex|^2+|Ey|^2+|Ez|^2 is integrated over y (trapezoid) to give field_energy_density_1D(x) ' ...
        '[arb. units], cropped to |x| <= x_grating_end. field_envelope_1D = cubic interpolation through the peaks of that ' ...
        'standing-wave pattern (scipy find_peaks + interp1d, edge values held). fwhm_m = width of the envelope at half of ' ...
        '(max - min), half-max crossings linearly interpolated (sim_helpers.extract_and_process_field_profile / ' ...
        'extract_envelope_peaks / calculate_fwhm_relative). This is the "spatial mode width" quoted everywhere.'];
    geo = struct();
    for fn = {'L_device','pitch_m','core_height_m','avg_corrugation_width_m','corrugation_depth_m','n_periods_each_side'}
        if isfield(data, fn{1}); geo.(fn{1}) = data.(fn{1}); end
    end
    pk = params{k}; pk.stored_in_file = geo;

    % ---- one-glance summary of every number (DERIVED from the fields above) ----
    [~, ir] = min(abs(double(data.wl_nm) - double(data.resonance_wavelength_nm)));
    sm = struct();
    sm.device = sprintf('pi-shift Bragg grating, TE, N=%d periods/side, pitch %d nm, avg width %d nm, corrugation %d nm (%d/%d nm), Si3N4 core %d nm, cavity %d nm, %s', ...
        N, pitch_nm, w_avg_nm, depth_nm, w_avg_nm-depth_nm/2, w_avg_nm+depth_nm/2, h_nm, pitch_nm/2, ...
        ternary(k==1, 'no apodization', 'linear apodization over 10 teeth/side, 4 nm centre depth'));
    sm.resonance_wavelength_nm   = double(data.resonance_wavelength_nm);
    sm.plotted_wavelength_nm     = xz.lambda_plot_nm;
    sm.peak_transmission         = double(data.resonance_transmission);
    sm.reflection_at_resonance   = double(data.R(ir));
    sm.loss_at_resonance         = double(data.loss(ir));
    sm.spectral_fwhm_nm          = sp.spectral_fwhm_abs_nm;
    sm.Q                         = sp.Q;
    sm.spatial_mode_fwhm_um      = double(data.fwhm_m)*1e6;
    sm.grating_length_um         = double(data.L_device)*1e6;
    sm.scan_window_nm            = [min(double(data.wl_nm)) max(double(data.wl_nm))];
    sm.n_wavelength_points       = numel(data.wl_nm);
    sm.xz_monitor_bins_nm        = [min(xz.lambda_bins_in_run_nm) max(xz.lambda_bins_in_run_nm) numel(xz.lambda_bins_in_run_nm)];
    sm.xz_monitor_size_um        = [max(xz.x_um)-min(xz.x_um), max(xz.z_um)-min(xz.z_um)];
    sm.I_dB_max                  = max(xz.I_dB(:));
    sm.units = 'wavelengths nm; widths/lengths um; T/R/loss are power fractions (0-1); Q = lambda_res / |spectral FWHM|; spatial mode FWHM from the 1D energy-density envelope along x';

    S.(keys{k}) = struct('summary',sm, 'description','', 'params',pk, 'xz',xz, 'spectrum',sp, 'profile1d',pr, 'source_file',in_files{k});
    fprintf('  resonance %.4f nm | plotted bin %.4f nm | T_res %.4f | FWHM %.4f nm | Q %.0f | mode width %.2f um\n', ...
        double(data.resonance_wavelength_nm), xz.lambda_plot_nm, double(data.resonance_transmission), ...
        sp.spectral_fwhm_abs_nm, sp.Q, double(data.fwhm_m)*1e6);
end
% ---- physics descriptions (what the two XZ maps show) ----
a = S.nonapodized.summary; b = S.apodized.summary;
S.nonapodized.description = sprintf(['NON-APODIZED pi-shift Bragg grating. All 80 teeth per side have the full 300 nm corrugation ' ...
    '(650/950 nm), right up to the pi-shift cavity (a 250 nm = pitch/2 segment at the 800 nm average width). ' ...
    'The XZ map (|E|^2 in dB on the y=0 plane, side view, x along the waveguide, z out of the chip plane) at the ' ...
    'resonance bin %.3f nm shows the cavity mode standing wave in the core plus clear radiation lobes leaving the ' ...
    'core near the defect (x ~ 0): the abrupt grating termination at the pi-shift acts like an antenna and scatters ' ...
    'guided light into the cladding (dipole-like leak; the Poynting arrows Px/Pz point out of the core there). This vertical loss limits the ' ...
    'peak transmission to T = %.4f (R = %.4f, loss = %.4f at resonance). Resonance %.3f nm, spectral FWHM %.3f nm, ' ...
    'Q = %.0f, spatial mode FWHM %.2f um along x.'], a.plotted_wavelength_nm, a.peak_transmission, ...
    a.reflection_at_resonance, a.loss_at_resonance, a.resonance_wavelength_nm, a.spectral_fwhm_nm, a.Q, a.spatial_mode_fwhm_um);
S.apodized.description = sprintf(['10-PERIOD LINEAR APODIZATION, same grating otherwise. The corrugation depth of the 10 teeth ' ...
    'nearest the cavity on each side ramps linearly from 4 nm (tooth 1, next to the cavity) to 300 nm (tooth 11 onward): ' ...
    'depth(d) = 4 + 296*(d-1)/10 nm. Removing the abrupt index step at the defect suppresses the radiation lobes: the ' ...
    'XZ map at %.3f nm shows only the guided standing wave, no leak into the cladding. Peak transmission rises to ' ...
    'T = %.4f (R = %.4f, loss = %.4f at resonance) and the resonance shifts to %.3f nm (weaker effective grating near ' ...
    'the cavity). The price is a wider cavity mode: spatial FWHM %.2f um vs %.2f um non-apodized (+%.0f%%), because the ' ...
    'weak inner teeth let the field penetrate deeper. Spectral FWHM %.3f nm, Q = %.0f.'], b.plotted_wavelength_nm, ...
    b.peak_transmission, b.reflection_at_resonance, b.loss_at_resonance, b.resonance_wavelength_nm, ...
    b.spatial_mode_fwhm_um, a.spatial_mode_fwhm_um, 100*(b.spatial_mode_fwhm_um/a.spatial_mode_fwhm_um-1), b.spectral_fwhm_nm, b.Q);

% ---- simulation numerics (identical for both devices) ----
S.simulation = struct( ...
    'engine', 'Lumerical FDTD 2026 R1.3 (build 4572), Athena cluster container, 1 GPU', ...
    'job', 'Athena SLURM 175019, tasks 0 (non-apod, 2146 s solve) and 1 (apod, 1317 s solve), 2026-10-10', ...
    'runner', 'runners/sweeps/xz_apod_vs_nonapod_n80.py (SweepSpec, 2 rows)', ...
    'materials', 'constant index: Si3N4 core n = 1.977, SiO2 cladding/substrate n = 1.44 (April-2026 values, NOT the later project default 1.97/1.444)', ...
    'mesh', 'simulation_mode = "optimization": dx = 50 nm along x (0.0497 um grid in the stored map), auto non-uniform mesh elsewhere', ...
    'domain', 'y span 8.75 um, z span 8.15 um = core + 5.0 * lambda_center (span_multiplier_override = 5.0, same box as the original April run; today''s default is 1.8 lambda)', ...
    'symmetry', 'y = 0 anti-symmetric, z = 0 symmetric (TE)', ...
    'source_and_ports', 'port 1 excitation, 1000 nm access waveguides, 20 periods from grating edge to port, 5 wavelengths port-to-PML', ...
    'scan', '3001 points over 1550.1-1570.1 nm (S-parameters, T/R/loss)', ...
    'field_monitors', '2D XZ side (y = 0), XY top (z = 0), YZ cross (x = cavity centre); 51 wavelength bins over the scan; this file keeps the XZ plane at the one bin nearest the resonance (reduced on the server by athena/scripts/slim_xz_result.py)', ...
    'resonance_finder', 'built-in Bragg-defect peak finder (sim_helpers.find_bragg_resonance), NOT max(T) of the scan', ...
    'spatial_mode_width', 'fwhm_m from the 1D core energy-density envelope (sim_helpers.extract_and_process_field_profile)', ...
    'auto_shutoff', '1e-7 (project production value)');

save(out_mat, '-struct', 'S', '-v7.3');
fprintf('saved %s\n', out_mat);

% ---- re-plot check: hot, 60 dB, x in [-12 12] ----
fig = figure('Visible','off','Position',[100 100 1500 520]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact');
ttl = {'N=80 periods', 'N=80 periods, 10 apodized'};
for k = 1:2
    D = S.(keys{k}).xz; ax = nexttile(tl);
    imagesc(ax, D.x_um, D.z_um, D.I_dB'); set(ax,'YDir','normal'); colormap(ax,'hot');
    mx = max(D.I_dB(:)); clim(ax,[mx-60 mx]); colorbar(ax); xlim(ax,[-12 12]);
    if isfield(D, 'Px')                       % Poynting arrows, plot_field_poynting_zoom.m grid recipe
        hold(ax, 'on');
        iw = abs(D.x_um) <= 12;
        sk = max(2, round([nnz(iw) numel(D.z_um)] / 25));
        draw_poynting_quiver(ax, D.x_um(iw), D.z_um, D.Px(iw,:), D.Pz(iw,:), sk(1), sk(2), 0.55, 'c', 0.9, ...
            D.I_dB(iw,:), mx, 55, 10, 0.60);
        hold(ax, 'off');
    end
    title(ax, sprintf('XZ Side View | %s | \\lambda = %.3f nm', ttl{k}, D.lambda_plot_nm));
    xlabel(ax,'Position X [\mum]'); ylabel(ax,'Position Z [\mum]');
end
exportgraphics(fig, check_png, 'Resolution', 110);
fprintf('check figure: %s\n', check_png);

% ---- deliverable bundle: zip = .mat + description.txt + PNG panels ----
bundle_dir = fullfile(fileparts(out_mat), 'xz_N80_apod_vs_nonapod');
if ~exist(bundle_dir, 'dir'); mkdir(bundle_dir); end
copyfile(out_mat, fullfile(bundle_dir, 'xz_N80_apod_vs_nonapod_figure_data.mat'));
copyfile(check_png, fullfile(bundle_dir, 'xz_side_view_nonapodized_vs_apodized_resim.png'));

a = S.nonapodized.summary; b = S.apodized.summary; sim = S.simulation;
L = {};
L{end+1} = 'XZ SIDE-VIEW FIELD MAPS: NON-APODIZED vs 10-PERIOD APODIZED PI-SHIFT BRAGG GRATING (N = 80)';
L{end+1} = sprintf('Built %s. Re-simulation of the two devices behind the April-2026 meeting figure (original result files were lost).', char(datetime('now','Format','yyyy-MM-dd HH:mm')));
L{end+1} = '';
L{end+1} = 'FILES IN THIS BUNDLE';
L{end+1} = '  xz_N80_apod_vs_nonapod_figure_data.mat          all data (see STRUCTURE below)';
L{end+1} = '  xz_side_view_nonapodized_vs_apodized_resim.png  the two XZ panels re-plotted from the .mat (|E|^2 dB, hot, 60 dB, x in [-12 12] um)';
L{end+1} = '';
L{end+1} = 'DEVICE (both): pi-shift Bragg grating, TE, 80 periods each side, pitch 500 nm, average width 800 nm, corrugation 300 nm';
L{end+1} = '  (narrow 650 / wide 950 nm), Si3N4 core 350 nm, cavity = pitch/2 = 250 nm at 800 nm width, no tooth shift, grating length 80.25 um,';
L{end+1} = '  port width 1000 nm, substrate 10 um, constant index n_core 1.977 / n_clad 1.44.';
L{end+1} = '  NON-APODIZED: 300 nm corrugation on every tooth.';
L{end+1} = '  APODIZED: linear over the 10 teeth nearest the cavity on each side, depth(d) = 4 + 296*(d-1)/10 nm for d <= 10 (d = 1 next to';
L{end+1} = '  the cavity), 300 nm beyond; narrow = 800 - depth/2, wide = 800 + depth/2. Per-tooth arrays are in params.teeth.';
L{end+1} = '';
L{end+1} = 'RESULTS (measured in this run)';
L{end+1} = sprintf('  %-28s %14s %14s', '', 'non-apodized', 'apodized');
rows = {'resonance wavelength [nm]', 'resonance_wavelength_nm', '%.3f'; 'plotted XZ bin [nm]', 'plotted_wavelength_nm', '%.3f'; ...
        'peak transmission T', 'peak_transmission', '%.4f'; 'R at resonance', 'reflection_at_resonance', '%.4f'; ...
        'loss at resonance', 'loss_at_resonance', '%.4f'; 'spectral FWHM [nm]', 'spectral_fwhm_nm', '%.3f'; ...
        'Q = lambda/FWHM', 'Q', '%.0f'; 'spatial mode FWHM [um]', 'spatial_mode_fwhm_um', '%.2f'};
for r = 1:size(rows,1)
    L{end+1} = sprintf(['  %-28s %14' rows{r,3}(2:end) ' %14' rows{r,3}(2:end)], rows{r,1}, a.(rows{r,2}), b.(rows{r,2}));
end
L{end+1} = '';
L{end+1} = 'WHAT THE MAPS SHOW';
L{end+1} = ['  ' S.nonapodized.description];
L{end+1} = '';
L{end+1} = ['  ' S.apodized.description];
L{end+1} = '';
L{end+1} = '1D MODE PROFILE (profile1d) - how it is calculated';
L{end+1} = ['  ' S.nonapodized.profile1d.method];
L{end+1} = sprintf('  Result: spatial mode FWHM %.2f um (non-apodized) vs %.2f um (apodized). profile1d.x_um / field_x [m] is the position along the', a.spatial_mode_fwhm_um, b.spatial_mode_fwhm_um);
L{end+1} = '  waveguide (0 = cavity centre), field_energy_density_1D the y-integrated |E|^2, field_envelope_1D its peak envelope, fwhm_m / fwhm_um the width.';
L{end+1} = '  Plot:  p = nonapodized.profile1d; plot(p.x_um, p.field_energy_density_1D, p.x_um, p.field_envelope_1D); xlabel(''x [um]'')';
L{end+1} = '';
L{end+1} = 'POYNTING VECTOR (xz.Px, xz.Pz) and the arrows in the image - how they were made';
L{end+1} = '  Data: Lumerical''s "P" result of the XZ side monitor = time-averaged Poynting vector P = 0.5*Re(E x H*) [W/m^2], read at the same';
L{end+1} = '  wavelength bin as the field map, on the y = 0 plane. Px is the component along the waveguide, Pz the vertical (out-of-plane)';
L{end+1} = '  component; Py is zero by symmetry on this plane and is not stored. These arrays are quantitative - use them for any analysis.';
L{end+1} = '  Arrows (image only, same recipe as matlab_plotting/plot_field_poynting_zoom.m, grid algorithm): the (Px,Pz) field inside the';
L{end+1} = '  shown window is sub-sampled uniformly to ~25 arrows per axis; each arrow points along the local unit vector (Px,Pz)/|P| and';
L{end+1} = '  its length is NOT proportional to |P| - it is 0.6 + 0.4 * log(1 + 10*|P|/|P|max)/log(11), so weak flow stays visible. Arrows are';
L{end+1} = '  drawn only where the intensity is within 55 dB of the maximum, where the direction agrees with its 3x3 neighbourhood';
L{end+1} = '  (cosine > 0.72, at least 2 neighbours), and head-on converging neighbour pairs are thinned to the stronger one. Cyan, quiver';
L{end+1} = '  scale 0.55. Read the arrows as flow DIRECTION; read magnitude from Px/Pz.';
L{end+1} = '';
L{end+1} = 'SIMULATION';
fn = fieldnames(sim);
for i = 1:numel(fn); L{end+1} = sprintf('  %-18s %s', [fn{i} ':'], sim.(fn{i})); end
L{end+1} = '';
L{end+1} = 'STRUCTURE OF THE .MAT  (structs nonapodized and apodized have identical fields; simulation = the block above)';
L{end+1} = '  summary     every number above';
L{end+1} = '  description the paragraph above';
L{end+1} = '  params      design values + teeth.(index, depth_nm, narrow_width_nm, wide_width_nm) for d = 1..80 + stored_in_file (geometry read back from the result)';
L{end+1} = '  xz          x_um (1xNx), z_um (1xNz), y_plane_um (=0), lambda_plot_nm, E (Nx x Nz x 3 complex, Ex Ey Ez), I = sum|E|^2, I_dB = 10*log10(I),';
L{end+1} = '              Px, Pz (time-averaged Poynting vector components, W/m^2, same grid), crop = the same restricted to |x| <= 25 um,';
L{end+1} = '              lambda_bins_in_run_nm = the 51 monitor bins';
L{end+1} = '  spectrum    wl_nm, T, R, loss, S21_complex, S11_complex, resonance_wavelength_nm, resonance_transmission, spectral_fwhm_nm (stored sign is a';
L{end+1} = '              convention; spectral_fwhm_abs_nm is the magnitude), Q';
L{end+1} = '  profile1d   field_x, field_energy_density_1D, field_envelope_1D (along the core), fwhm_m (spatial mode width)';
L{end+1} = '  source_file the reduced Athena result the struct was built from';
L{end+1} = '  Plot the map:  d = nonapodized.xz; imagesc(d.x_um, d.z_um, d.I_dB''); axis xy; colormap hot; clim([max(d.I_dB(:))-60 max(d.I_dB(:))]); xlim([-12 12])';
fid = fopen(fullfile(bundle_dir, 'description.txt'), 'w', 'n', 'UTF-8'); fprintf(fid, '%s\n', L{:}); fclose(fid);
zip_path = [bundle_dir '.zip'];
zip(zip_path, bundle_dir);
fprintf('bundle: %s\n', zip_path);

function out = ternary(c, a, b)
if c; out = a; else; out = b; end
end

function draw_poynting_quiver(ax, coord1, coord2, P1, P2, skip1, skip2, scale, color, lw, I_dB, max_dB, threshold_dB, log_k, min_frac)
% Verbatim copy of the local helper in plot_field_poynting_zoom.m (grid algorithm),
% so the bundle image carries exactly the arrows the project's figures use.
    n1 = numel(coord1); n2 = numel(coord2);
    idx1 = (1:skip1:n1)'; idx2 = (1:skip2:n2)';
    c1_q = coord1(idx1); c2_q = coord2(idx2);
    [C1q, C2q] = meshgrid(c1_q, c2_q);
    P1_q = P1(idx1, idx2).'; P2_q = P2(idx1, idx2).';
    Pmag = sqrt(P1_q.^2 + P2_q.^2); Pmag_max = max(Pmag(:));
    if Pmag_max == 0; return; end
    Pmag_norm = Pmag / Pmag_max;
    log_norm = log1p(Pmag_norm * log_k) / log1p(log_k);
    vis_scale = min_frac + (1 - min_frac) * log_norm;
    I_dB_q = I_dB(idx1, idx2).';
    mask = I_dB_q >= (max_dB - threshold_dB);
    c1_range = max(c1_q) - min(c1_q); c2_range = max(c2_q) - min(c2_q);
    mask = mask & (C1q >= min(c1_q) + 0.01*c1_range) & (C1q <= max(c1_q) - 0.01*c1_range) ...
                & (C2q >= min(c2_q) + 0.01*c2_range) & (C2q <= max(c2_q) - 0.01*c2_range);
    P1_unit = P1_q ./ (Pmag + 1e-30); P2_unit = P2_q ./ (Pmag + 1e-30);
    w = double(mask);
    local_mean1 = conv2(P1_unit .* w, ones(3,3)/9, 'same');
    local_mean2 = conv2(P2_unit .* w, ones(3,3)/9, 'same');
    local_mag   = sqrt(local_mean1.^2 + local_mean2.^2) + 1e-30;
    coherence   = (P1_unit .* local_mean1 + P2_unit .* local_mean2) ./ local_mag;
    neighbor_cnt = conv2(w, ones(3,3), 'same') - w;
    mask = mask & (coherence > 0.72) & (neighbor_cnt >= 2);
    rmv = false(size(mask));
    valid_x = mask(:,1:end-1) & mask(:,2:end);
    conv_x  = (P1_unit(:,1:end-1) > 0.25) & (P1_unit(:,2:end) < -0.25) & valid_x;
    keep_left = Pmag(:,1:end-1) >= Pmag(:,2:end);
    rmv(:,1:end-1) = rmv(:,1:end-1) | (conv_x & ~keep_left);
    rmv(:,2:end)   = rmv(:,2:end)   | (conv_x &  keep_left);
    mask = mask & ~rmv;
    rmv = false(size(mask));
    valid_z = mask(1:end-1,:) & mask(2:end,:);
    conv_z  = (P2_unit(1:end-1,:) > 0.25) & (P2_unit(2:end,:) < -0.25) & valid_z;
    keep_top = Pmag(1:end-1,:) >= Pmag(2:end,:);
    rmv(1:end-1,:) = rmv(1:end-1,:) | (conv_z & ~keep_top);
    rmv(2:end,:)   = rmv(2:end,:)   | (conv_z &  keep_top);
    mask = mask & ~rmv;
    vis_scale = vis_scale .* mask;
    quiver(ax, C1q, C2q, (P1_q ./ (Pmag + 1e-30)) .* vis_scale, (P2_q ./ (Pmag + 1e-30)) .* vis_scale, ...
        scale, 'Color', color, 'LineWidth', lw);
end
