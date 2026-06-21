%% Plot Transmission (T, R, Loss, Phase, Q-factor)
% Loads .mat files with complex S-parameters (or T/R) and plots Transmission /
% Reflection / Loss / Phase. On the Combined Transmission plot it locates the
% cavity resonance with a dedicated sharpness x dip-depth scorer (the sharp peak
% inside the stopband, NOT the global passband maximum) and reports the Q-factor.
% Resonance finder requires the Signal Processing Toolbox (findpeaks); falls back
% to the global max if no interior peaks are found.
addpath(fileparts(mfilename('fullpath')));              % Add matlab_plotting to path
addpath(fileparts(fileparts(mfilename('fullpath'))));   % Add project root to path
clear; clc;
%close all;
%% User settings
% Filter range in microns for the combined loss and transmission plots5
LAMBDA_MIN_UM = 1.553;
LAMBDA_MAX_UM = 1.590;

% LAMBDA_MIN_UM = 1.5;
% LAMBDA_MAX_UM = 1.8;

% Convert to nm
LAMBDA_MIN_NM = 1000.0 * LAMBDA_MIN_UM;
LAMBDA_MAX_NM = 1000.0 * LAMBDA_MAX_UM;

HIGHLIGHT_PEAKS     = true; % Only applies to Combined Loss plot
CALC_Q_FACTOR       = true; % Toggle calculating Q-factor on Transmission plot
PLOT_IN_DB          = false; % Plot T and R in dB (10*log10) on Combined Transmission and Individual plots
DB_FLOOR            = -60;  % Lower y-axis limit when PLOT_IN_DB is true
FONT_SIZE           = 13;  % Font size for xlabel, ylabel, title
LEGEND_FONT_SIZE    = 11;  % Font size for legend

to_db = @(x) 10*log10(max(x, 10^(DB_FLOOR/10)));

%% Select .mat files
prefs_file = fullfile(fileparts(mfilename('fullpath')), 'plot_prefs.mat');
start_path = '*.mat';
use_saved_files = false;
if exist(prefs_file, 'file')
    p = load(prefs_file);
    has_folder = isfield(p, 'fdtd_last_folder') && isfolder(p.fdtd_last_folder);
    has_files  = isfield(p, 'fdtd_last_files') && ~isempty(p.fdtd_last_files);
    if has_folder
        if has_files
            % Build a file list string for the dialog
            saved_files = p.fdtd_last_files;
            if ischar(saved_files), saved_files = {saved_files}; end
            file_list_str = strjoin(saved_files, ', ');
            msg = ['Last used:' newline p.fdtd_last_folder newline ...
                   'Files: ' file_list_str];
            answer = questdlg(msg, 'Reuse Previous Selection', ...
                'Same Files', 'Same Folder', 'Browse...', 'Same Files');
        else
            msg = ['Use last folder?' newline p.fdtd_last_folder];
            answer = questdlg(msg, 'Reuse Previous Selection', ...
                'Same Folder', 'Browse...', 'Same Folder');
        end
        if strcmp(answer, 'Same Files')
            use_saved_files = true;
            files  = p.fdtd_last_files;
            folder = p.fdtd_last_folder;
        elseif strcmp(answer, 'Same Folder')
            start_path = fullfile(p.fdtd_last_folder, '*.mat');
        end
    end
end

if ~use_saved_files
    [files, folder] = uigetfile(start_path, 'Select .mat files', 'MultiSelect', 'on');
    if isequal(files, 0)
        disp('No files selected.');
        return;
    end
end

% Save last folder and files
fdtd_last_folder = folder;
fdtd_last_files  = files;
if exist(prefs_file, 'file')
    save(prefs_file, 'fdtd_last_folder', 'fdtd_last_files', '-append');
else
    save(prefs_file, 'fdtd_last_folder', 'fdtd_last_files');
end

% Ensure files is a cell array
if ischar(files)
    files = {files};
end
nFiles = numel(files);

%% Plot selection dialog
plot_options = {'Combined Loss', 'Combined Transmission', ...
                'Individual T/R/Loss (per file)', ...
                'Combined Phase', 'Phase Derivative'};
% Load saved checkbox state or use defaults
defaults = [true, true, true, false, false];
if exist(prefs_file, 'file')
    p = load(prefs_file);
    if isfield(p, 'fdtd_plot_checks') && numel(p.fdtd_plot_checks) == numel(defaults)
        defaults = p.fdtd_plot_checks;
    end
end

% Build checkbox dialog
fig_sel = figure('Name', 'Select Plots', 'NumberTitle', 'off', ...
    'MenuBar', 'none', 'ToolBar', 'none', ...
    'Resize', 'off', 'WindowStyle', 'modal');
nOpts = numel(plot_options);
fig_h = 30 * nOpts + 70;
fig_sel.Position = [fig_sel.Position(1:2) 320 fig_h];

cb = gobjects(nOpts, 1);
for i = 1:nOpts
    cb(i) = uicontrol(fig_sel, 'Style', 'checkbox', ...
        'String', plot_options{i}, 'Value', defaults(i), ...
        'Position', [20 fig_h - 30 - 30*(i-1) 280 25], ...
        'FontSize', 10);
end

uicontrol(fig_sel, 'Style', 'pushbutton', 'String', 'OK', ...
    'Position', [110 10 100 30], 'FontSize', 11, ...
    'Callback', @(~,~) uiresume(fig_sel));

uiwait(fig_sel);

if ~isvalid(fig_sel)
    disp('Plot selection cancelled.');
    return;
end
PLOT_COMBINED_LOSS  = cb(1).Value == 1;
PLOT_COMBINED_TRANS = cb(2).Value == 1;
PLOT_INDIVIDUAL     = cb(3).Value == 1;
PLOT_PHASE          = cb(4).Value == 1;
PLOT_PHASE_DERIV    = cb(5).Value == 1;
close(fig_sel);

% Save checkbox state for next run
fdtd_plot_checks = [PLOT_COMBINED_LOSS, PLOT_COMBINED_TRANS, ...
                    PLOT_INDIVIDUAL, PLOT_PHASE, PLOT_PHASE_DERIV];
if exist(prefs_file, 'file')
    save(prefs_file, 'fdtd_plot_checks', '-append');
else
    save(prefs_file, 'fdtd_plot_checks');
end

% Derived flag: relative derivative only if phase deriv is shown
PLOT_RELATIVE_DERIV = PLOT_PHASE_DERIV;

%% Load data from each .mat file
% Preallocate struct with new fields for filtered T
datasets(nFiles) = struct('name', [], 'wl_nm', [], 'T', [], 'R', [], ...
    'loss', [], 'wl_loss', [], 'loss_f', [], 'T_f', [], ...
    'phase_T', [], 'has_phase_T', false, ...
    'phase_R', [], 'has_phase_R', false, ...
    'dPhi_T', [], 'dPhi_R', [], 'dPhi_Rel', []);

for k = 1:nFiles
    fp = fullfile(folder, files{k});

    % Load .mat file
    fileData = load(fp);

    % 1. Extract Wavelength
    if isfield(fileData, 'wl_nm')
        wl_nm = double(fileData.wl_nm);
    elseif isfield(fileData, 'wl_m')
        wl_nm = double(fileData.wl_m)*1e9;
    elseif isfield(fileData, 'lambda')
        wl_nm = double(fileData.lambda) * 1e9;
    else
        error('Variable wl_nm, wl, or lambda not found in %s', files{k});
    end

    % 2. Extract Complex S-Parameters (to get t and r)
    S11 = [];
    S21 = [];

    if isfield(fileData, 'S11_complex')
        S11 = fileData.S11_complex;
    elseif isfield(fileData, 'S11')
        S11 = fileData.S11;
    end

    if isfield(fileData, 'S21_complex')
        S21 = fileData.S21_complex;
    elseif isfield(fileData, 'S12_complex')
        S21 = fileData.S12_complex;
    elseif isfield(fileData, 'S21')
        S21 = fileData.S21;
    end

    % 3. Calculate T and R (Power)
    if ~isempty(S21)
        T = abs(S21).^2;
    elseif isfield(fileData, 'T')
        T = double(fileData.T);
    else
        error('Could not find S21 (t) or T in %s', files{k});
    end

    if ~isempty(S11)
        R = abs(S11).^2;
    elseif isfield(fileData, 'R')
        R = double(fileData.R);
    else
        error('Could not find S11 (r) or R in %s', files{k});
    end

    % 4. Calculate Loss
    if isfield(fileData, 'loss')
        loss = double(fileData.loss);
    else
        loss = 1 - R - T;
    end

    % 5. Extract Phases and Derivatives
    phaseVal_T = []; has_phase_T = false; dPhi_T = [];
    phaseVal_R = []; has_phase_R = false; dPhi_R = [];
    dPhi_Rel   = []; % Derivative of Relative Phase

    if PLOT_PHASE
        % Transmission Phase & Derivative
        if ~isempty(S21)
            phaseVal_T = unwrap(angle(S21)) / pi;
            has_phase_T = true;
            % Derivative: d(UnwrappedPhase/pi) / d(Lambda)
            uPhase_T = unwrap(angle(S21));
            dPhi_T = diff(uPhase_T / pi) ./ diff(wl_nm);
        end

        % Reflection Phase & Derivative
        if ~isempty(S11)
            phaseVal_R = (angle(S11)) / pi;
            has_phase_R = true;
            uPhase_R = unwrap(angle(S11));
            dPhi_R = diff(uPhase_R / pi) ./ diff(wl_nm);
        end

        % Relative Phase Derivative: d(angle(S21/S11))/dLambda
        if ~isempty(S21) && ~isempty(S11)
            % Complex Division
            S_rel = S11 ./ S21;

            % Unwrap angle of the ratio
            uPhase_Rel = unwrap(angle(S_rel));

            % Derivative w.r.t wavelength (normalized by pi)
            dPhi_Rel = diff(uPhase_Rel / pi) ./ diff(wl_nm);
        end
    end

    % Mask for combined plots (Loss and T)
    mask      = (wl_nm >= LAMBDA_MIN_NM) & (wl_nm <= LAMBDA_MAX_NM);
    wl_filt   = wl_nm(mask);
    loss_filt = loss(mask);
    T_filt    = T(mask); % Extract Filtered T

    rawName = erase(files{k}, '.mat');
    cleanName = strrep(rawName, 'result_', '');
    cleanName = strrep(cleanName, '_CONST', '');
    cleanName = regexprep(cleanName, '(?i)innersize', 'size');
    datasets(k).name      = cleanName;
    datasets(k).wl_nm     = wl_nm;
    datasets(k).T         = T;
    datasets(k).R         = R;
    datasets(k).loss      = loss;
    datasets(k).wl_loss   = wl_filt; % Reusing this field name for the x-axis of zoomed plots
    datasets(k).loss_f    = loss_filt;
    datasets(k).T_f       = T_filt;  % Store Filtered T

    datasets(k).phase_T   = phaseVal_T;
    datasets(k).has_phase_T = has_phase_T;
    datasets(k).phase_R   = phaseVal_R;
    datasets(k).has_phase_R = has_phase_R;

    datasets(k).dPhi_T    = dPhi_T;
    datasets(k).dPhi_R    = dPhi_R;
    datasets(k).dPhi_Rel  = dPhi_Rel;
end

%% Color palette (shared across plots)
colorPalette = [
    0.00  0.45  0.74;  % blue
    0.85  0.33  0.10;  % orange-red
    0.47  0.67  0.19;  % green
    0.49  0.18  0.56;  % purple
    0.93  0.69  0.13;  % yellow
    0.30  0.75  0.93;  % light blue
    0.64  0.08  0.18;  % dark red
    0.00  0.50  0.50;  % teal
    0.55  0.27  0.07;  % brown
    0.85  0.53  0.60;  % pink
    0.40  0.40  0.40;  % gray
    0.00  0.25  0.50;  % dark blue
    0.60  0.80  0.20;  % lime green
    0.80  0.60  0.00;  % amber
    0.30  0.00  0.50;  % dark purple
    0.90  0.20  0.50;  % magenta
    0.10  0.60  0.40;  % emerald
    0.60  0.20  0.80;  % violet
    0.80  0.80  0.10;  % yellow-green
    0.20  0.80  0.80;  % aqua
];
if nFiles <= size(colorPalette, 1)
    colors = colorPalette(1:nFiles, :);
else
    colors = hsv(nFiles);  % fallback for very many files
end

%% Plot 1: Combined Filtered Loss
if PLOT_COMBINED_LOSS
    figure;
    hold on;
    for k = 1:nFiles
        ds = datasets(k);

        % Main line
        plot(ds.wl_loss, ds.loss_f, 'DisplayName', ds.name, ...
            'Color', colors(k,:), 'LineWidth', 1.5);

        % Highlight Peak
        if HIGHLIGHT_PEAKS && ~isempty(ds.wl_loss)
            [peakVal, idxPeak] = max(ds.loss_f);
            xPeak = ds.wl_loss(idxPeak);
            yPeak = peakVal;

            plot(xPeak, yPeak, 'o', 'Color', colors(k,:), ...
                'HandleVisibility', 'off');
            text(xPeak, yPeak, ...
                sprintf('  (%0.2f, %0.3f)', xPeak, yPeak), ...
                'FontSize', 8, ...
                'Color', 'k', ...
                'HorizontalAlignment', 'left', ...
                'VerticalAlignment', 'bottom');
            xline(xPeak, '--', 'Color', 'g', ...
                'HandleVisibility', 'off', 'Alpha', 0.5);
        end
    end
    hold off;
    xlabel('Wavelength [nm]', 'FontSize', FONT_SIZE);
    ylabel('Loss (1 - R - T)', 'FontSize', FONT_SIZE);
    title('Combined Loss (Zoomed)', 'FontSize', FONT_SIZE);
    grid on;
    legend('show', 'Location', 'best', 'Interpreter', 'none', 'FontSize', LEGEND_FONT_SIZE);
    set(gcf, 'Name', 'Combined Loss');
end

%% Plot 2: Combined Filtered Transmission
if PLOT_COMBINED_TRANS
    figure;
    hold on;
    for k = 1:nFiles
        ds = datasets(k);

        displayName = ds.name;

        if CALC_Q_FACTOR && ~isempty(ds.T_f)
            % --- Resonance finding: dedicated sharpness x dip-depth scorer ---
            % Ported from sim_helpers.find_bragg_resonance. The cavity resonance
            % is the sharpest peak (high prominence / narrow width) that ALSO sits
            % in the deepest dip (stopband floor ~ 0) — NOT the global maximum,
            % which for a pi-shift grating is just a passband ripple at T ~ 1.
            % score = (prominence / (width+1)) * (1 - base_level),  base ~ pk - prom.
            Tf = ds.T_f(:);
            [pks, locs, w, p] = findpeaks(Tf, 'WidthReference', 'halfprom');
            if isempty(pks)
                [T_max, idx_max] = max(Tf);   % fallback: no interior peaks
            else
                base_level = pks - p;
                score = (p ./ (w + 1)) .* (1 - base_level);
                [~, ibest] = max(score);
                idx_max = locs(ibest);
                T_max   = Tf(idx_max);
            end
            lambda_res = ds.wl_loss(idx_max);

            % Half max value
            half_max = T_max / 2;

            % Find FWHM: walk OUTWARD from the peak to the FIRST half-max
            % crossing on each side (local crossings only), then interpolate
            % linearly between the two bracketing samples for sub-grid accuracy.
            %
            % BUGFIX: the previous version searched the WHOLE zoom window for the
            % half-max crossing (find(... ,1,'first'/'last')). For a sharp cavity
            % peak the window ends sit in the passband (T > half_max), so those
            % finds latched onto window-edge points and the interpolation grossly
            % inflated the FWHM (TE 0.96 -> 7.5 nm), collapsing Q (1640 -> 208)
            % and even flipping the TE/TM ordering. Walking outward from the peak
            % uses only the local crossing, matching the Python find_resonance.
            lambda_left = NaN;
            iL = idx_max;
            while iL > 1 && ds.T_f(iL) > half_max
                iL = iL - 1;
            end
            if iL < idx_max          % ds.T_f(iL) <= half_max < ds.T_f(iL+1)
                x1 = ds.wl_loss(iL);   y1 = ds.T_f(iL);
                x2 = ds.wl_loss(iL+1); y2 = ds.T_f(iL+1);
                if y1 ~= y2
                    lambda_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1);
                else
                    lambda_left = x2;
                end
            end

            lambda_right = NaN;
            iR = idx_max;
            while iR < numel(ds.T_f) && ds.T_f(iR) > half_max
                iR = iR + 1;
            end
            if iR > idx_max          % ds.T_f(iR) <= half_max < ds.T_f(iR-1)
                x1 = ds.wl_loss(iR-1); y1 = ds.T_f(iR-1);
                x2 = ds.wl_loss(iR);   y2 = ds.T_f(iR);
                if y1 ~= y2
                    lambda_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1);
                else
                    lambda_right = x1;
                end
            end

            if ~isnan(lambda_left) && ~isnan(lambda_right)
                FWHM = abs(lambda_right - lambda_left);
                Q_factor = lambda_res / FWHM;

                exponent = floor(log10(Q_factor));
                mantissa = Q_factor / 10^exponent;
                displayName = sprintf('%s (Q ~ %.2f x 10^%d)', ds.name, mantissa, exponent);

                % Plot resonance peak marker
                T_peak_marker = T_max;
                if PLOT_IN_DB, T_peak_marker = to_db(T_max); end
                plot(lambda_res, T_peak_marker, 'v', 'Color', colors(k,:), ...
                    'HandleVisibility', 'off', 'MarkerFaceColor', colors(k,:));
            end
        end

        % Main line (Zoomed T)
        T_plot = ds.T_f;
        if PLOT_IN_DB, T_plot = to_db(T_plot); end
        plot(ds.wl_loss, T_plot, 'DisplayName', displayName, ...
            'Color', colors(k,:), 'LineWidth', 1.5);
    end
    hold off;
    xlabel('Wavelength [nm]', 'FontSize', FONT_SIZE);
    if PLOT_IN_DB
        ylabel('Transmission (T) [dB]', 'FontSize', FONT_SIZE);
        ylim([DB_FLOOR 0]);
    else
        ylabel('Transmission (T)', 'FontSize', FONT_SIZE);
    end
    title('Combined Transmission (Zoomed)', 'FontSize', FONT_SIZE);
    grid on;
    legend('show', 'Location', 'best', 'Interpreter', 'none', 'FontSize', LEGEND_FONT_SIZE);
    set(gcf, 'Name', 'Combined Transmission');
end

%% Plot 3..N: Individual File Plots (T, R, Loss)
if PLOT_INDIVIDUAL
    for k = 1:nFiles
        ds = datasets(k);
        figure;
        hold on;

        if PLOT_IN_DB
            plot(ds.wl_nm, to_db(ds.T),    'DisplayName', 'T',    'LineWidth', 1.5);
            plot(ds.wl_nm, to_db(ds.R),    'DisplayName', 'R',    'LineWidth', 1.5);
            plot(ds.wl_nm, to_db(ds.loss), 'DisplayName', 'Loss', 'LineWidth', 1.5);
        else
            plot(ds.wl_nm, ds.T,   'DisplayName', 'T',    'LineWidth', 1.5);
            plot(ds.wl_nm, ds.R,   'DisplayName', 'R',    'LineWidth', 1.5);
            plot(ds.wl_nm, ds.loss,'DisplayName', 'Loss', 'LineWidth', 1.5);
        end

        hold off;
        xlabel('Wavelength [nm]', 'FontSize', FONT_SIZE);
        if PLOT_IN_DB
            ylabel('Normalized Power [dB]', 'FontSize', FONT_SIZE);
            ylim([DB_FLOOR 0]);
        else
            ylabel('Normalized Power', 'FontSize', FONT_SIZE);
            ylim([-0.1 1.1]);
        end
        title(sprintf('%s : Power (T, R)', ds.name), 'Interpreter', 'none', 'FontSize', FONT_SIZE);
        grid on;
        legend('show', 'FontSize', LEGEND_FONT_SIZE);
        set(gcf, 'Name', sprintf('%s_Power', ds.name));
    end
end

%% Plot: Combined Phase (t & r)
if PLOT_PHASE
    figure;
    hold on;
    hasPhaseData = false;
    for k = 1:nFiles
        ds = datasets(k);

        % Transmission Phase (t)
        if ds.has_phase_T
            plot(ds.wl_nm, ds.phase_T, ...
                'DisplayName', [ds.name ' (t Phase)'], ...
                'Color', colors(k,:), ...
                'LineWidth', 1.5, ...
                'LineStyle', '-');
            hasPhaseData = true;
        end
        % Reflection Phase (r)
        if ds.has_phase_R
            plot(ds.wl_nm, ds.phase_R, ...
                'DisplayName', [ds.name ' (r Phase)'], ...
                'Color', colors(k,:), ...
                'LineWidth', 1.5, ...
                'LineStyle', '--');
            hasPhaseData = true;
        end
    end
    hold off;
    xlabel('Wavelength [nm]', 'FontSize', FONT_SIZE);
    ylabel('Wrapped Phase [\pi rad]', 'FontSize', FONT_SIZE);
    title('Combined Phase (Solid=t, Dashed=r)', 'FontSize', FONT_SIZE);
    grid on;
    if hasPhaseData
        legend('show', 'Interpreter', 'none');
    end
    set(gcf, 'Name', 'Combined Phase');
end

%% Plot: Derivatives of Phase (T, R, and Relative)
if PLOT_PHASE_DERIV
    figure;
    hold on;
    hasDerivData = false;
    for k = 1:nFiles
        ds = datasets(k);

        % Note: diff() reduces array size by 1.
        % We use the first N-1 wavelength points for x-axis.
        if length(ds.wl_nm) > 1
            wl_deriv = ds.wl_nm(1:end-1);

            % 1. Plot Derivative T (Solid)
            if ~isempty(ds.dPhi_T)
                plot(wl_deriv, ds.dPhi_T, ...
                    'DisplayName', [ds.name ' (d\phi_t/d\lambda)'], ...
                    'Color', colors(k,:), ...
                    'LineWidth', 1.5, ...
                    'LineStyle', '-');
                hasDerivData = true;
            end

            % 2. Plot Derivative R (Dashed)
            if ~isempty(ds.dPhi_R)
                plot(wl_deriv, ds.dPhi_R, ...
                    'DisplayName', [ds.name ' (d\phi_r/d\lambda)'], ...
                    'Color', colors(k,:), ...
                    'LineWidth', 1.5, ...
                    'LineStyle', '--');
                hasDerivData = true;
            end

            % 3. Plot Derivative Relative (Dotted) - Optional
            if PLOT_RELATIVE_DERIV && ~isempty(ds.dPhi_Rel)
                plot(wl_deriv, ds.dPhi_Rel, ...
                    'DisplayName', [ds.name ' (d(\phi_{t/r})/d\lambda)'], ...
                    'Color', colors(k,:), ...
                    'LineWidth', 2.0, ...
                    'LineStyle', ':');
                hasDerivData = true;
            end
        end
    end
    hold off;
    xlabel('Wavelength [nm]', 'FontSize', FONT_SIZE);
    ylabel('d(Phase)/d\lambda [\pi rad / nm]', 'FontSize', FONT_SIZE);
    title('Derivative of Phase w.r.t Wavelength', 'FontSize', FONT_SIZE);
    grid on;
    if hasDerivData
        legend('show', 'Interpreter', 'none');
    end
    set(gcf, 'Name', 'Phase Derivative');
end