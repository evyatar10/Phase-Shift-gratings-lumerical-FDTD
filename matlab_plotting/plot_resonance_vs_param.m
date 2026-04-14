%% Plot Resonance Peak Wavelength / Peak Transmission vs. Sweep Parameter
% Manually select .mat files. Parses the sweep parameter from the filename
% and plots resonance wavelength and/or peak transmission vs. that parameter.
%
% Supported sweep parameters (auto-detected via dialog):
%   Shift : filename contains  _shift_<value>nm
%   Size  : filename contains  _innersize_<value>nm

clear; clc;
close all;

%% User settings
FONT_SIZE = 13;    % Font size for xlabel, ylabel, title

%% Select .mat files
prefs_file = fullfile(fileparts(mfilename('fullpath')), 'plot_prefs.mat');
start_path = '*.mat';
use_saved_files = false;
if exist(prefs_file, 'file')
    p = load(prefs_file);
    has_folder = isfield(p, 'res_last_folder') && isfolder(p.res_last_folder);
    has_files  = isfield(p, 'res_last_files') && ~isempty(p.res_last_files);
    if has_folder
        if has_files
            saved_files = p.res_last_files;
            if ischar(saved_files), saved_files = {saved_files}; end
            file_list_str = strjoin(saved_files, ', ');
            msg = ['Last used:' newline p.res_last_folder newline ...
                   'Files: ' file_list_str];
            answer = questdlg(msg, 'Reuse Previous Selection', ...
                'Same Files', 'Same Folder', 'Browse...', 'Same Files');
        else
            msg = ['Use last folder?' newline p.res_last_folder];
            answer = questdlg(msg, 'Reuse Previous Selection', ...
                'Same Folder', 'Browse...', 'Same Folder');
        end
        if strcmp(answer, 'Same Files')
            use_saved_files = true;
            files  = p.res_last_files;
            folder = p.res_last_folder;
        elseif strcmp(answer, 'Same Folder')
            start_path = fullfile(p.res_last_folder, '*.mat');
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
res_last_folder = folder;
res_last_files  = files;
if exist(prefs_file, 'file')
    save(prefs_file, 'res_last_folder', 'res_last_files', '-append');
else
    save(prefs_file, 'res_last_folder', 'res_last_files');
end

% Ensure files is a cell array
if ischar(files)
    files = {files};
end
nFiles = numel(files);

%% Plot selection dialog
plot_options = {'Peak Transmission', 'Resonance Wavelength'};
sweep_modes  = {'Shift', 'Size'};

% Load saved state or use defaults
defaults_checks = [true, false];
default_sweep   = 1; % 1 = Shift, 2 = Size
if exist(prefs_file, 'file')
    p = load(prefs_file);
    if isfield(p, 'res_plot_checks') && numel(p.res_plot_checks) == numel(defaults_checks)
        defaults_checks = p.res_plot_checks;
    end
    if isfield(p, 'res_sweep_mode') && ismember(p.res_sweep_mode, [1 2])
        default_sweep = p.res_sweep_mode;
    end
end

% Build dialog
nOpts = numel(plot_options);
fig_h = 30 * nOpts + 140;
fig_w = 320;
fig_sel = figure('Name', 'Select Plots', 'NumberTitle', 'off', ...
    'MenuBar', 'none', 'ToolBar', 'none', ...
    'Resize', 'off', 'WindowStyle', 'modal', 'Units', 'pixels');
fig_sel.Position = [fig_sel.Position(1:2) fig_w fig_h];

% Sweep parameter radio buttons
uicontrol(fig_sel, 'Style', 'text', 'String', 'Sweep parameter:', ...
    'Position', [20 fig_h - 35 130 20], 'FontSize', 10, ...
    'HorizontalAlignment', 'left');
bg = uibuttongroup(fig_sel, 'Units', 'pixels', ...
    'Position', [155 fig_h - 40 155 30], 'BorderType', 'none');
rb_shift = uicontrol(bg, 'Style', 'radiobutton', 'String', 'Shift', ...
    'Position', [0 2 70 25], 'FontSize', 10);
rb_size  = uicontrol(bg, 'Style', 'radiobutton', 'String', 'Size', ...
    'Position', [75 2 70 25], 'FontSize', 10);
if default_sweep == 2
    bg.SelectedObject = rb_size;
else
    bg.SelectedObject = rb_shift;
end

% Separator line
y_sep = fig_h - 55;
uicontrol(fig_sel, 'Style', 'text', 'String', '', ...
    'Position', [10 y_sep fig_w-20 1], 'BackgroundColor', [0.7 0.7 0.7]);

% Plot checkboxes
cb = gobjects(nOpts, 1);
for i = 1:nOpts
    cb(i) = uicontrol(fig_sel, 'Style', 'checkbox', ...
        'String', plot_options{i}, 'Value', defaults_checks(i), ...
        'Position', [20 y_sep - 10 - 30*i 280 25], ...
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
PLOT_PEAK_T       = cb(1).Value == 1;
PLOT_RESONANCE_WL = cb(2).Value == 1;
SWEEP_BY_SIZE     = (bg.SelectedObject == rb_size);
close(fig_sel);

% Save state for next run
res_plot_checks = [PLOT_PEAK_T, PLOT_RESONANCE_WL];
res_sweep_mode  = 1 + SWEEP_BY_SIZE; % 1 = Shift, 2 = Size
if exist(prefs_file, 'file')
    save(prefs_file, 'res_plot_checks', 'res_sweep_mode', '-append');
else
    save(prefs_file, 'res_plot_checks', 'res_sweep_mode');
end

% Set labels based on sweep mode
if SWEEP_BY_SIZE
    param_label = 'Inner Tooth Size [nm]';
    param_short = 'Size';
else
    param_label = 'Innermost Teeth Shift [nm]';
    param_short = 'Shift';
end

%% Preallocate result arrays
param_vals = zeros(1, nFiles);
lambda_res = nan(1, nFiles);
T_peak     = nan(1, nFiles);

%% Loop: load each file, parse parameter, find resonance peak
for k = 1:nFiles
    fname = files{k};
    fp    = fullfile(folder, fname);

    % --- Parse sweep parameter from filename ---
    if SWEEP_BY_SIZE
        tok = regexp(fname, '_innersize_([\d.]+)nm', 'tokens');
    else
        tok = regexp(fname, '_shift_([\d.]+)nm', 'tokens');
    end
    if ~isempty(tok)
        param_vals(k) = str2double(tok{1}{1});
    else
        param_vals(k) = 0;
    end

    % --- Load .mat file ---
    fileData = load(fp);

    % Extract wavelength
    if isfield(fileData, 'wl_nm')
        wl_nm = double(fileData.wl_nm);
    elseif isfield(fileData, 'wl_m')
        wl_nm = double(fileData.wl_m) * 1e9;
    elseif isfield(fileData, 'lambda')
        wl_nm = double(fileData.lambda) * 1e9;
    else
        error('No wavelength variable found in %s', fname);
    end

    % Extract T
    if isfield(fileData, 'S21_complex')
        T = abs(fileData.S21_complex).^2;
    elseif isfield(fileData, 'S12_complex')
        T = abs(fileData.S12_complex).^2;
    elseif isfield(fileData, 'S21')
        T = abs(fileData.S21).^2;
    elseif isfield(fileData, 'T')
        T = double(fileData.T);
    else
        error('No transmission variable found in %s', fname);
    end

    % Read resonance wavelength directly from file
    if isfield(fileData, 'resonance_wavelength_nm')
        lambda_res(k) = double(fileData.resonance_wavelength_nm);
    else
        error('resonance_wavelength_nm not found in %s', fname);
    end

    % Read peak transmission at resonance wavelength
    wl_nm_arr = wl_nm(:);
    T_arr     = T(:);
    [~, idx_res] = min(abs(wl_nm_arr - lambda_res(k)));
    T_peak(k) = T_arr(idx_res);
end

%% Sort by parameter value
[param_sorted, sort_idx] = sort(param_vals);
lambda_res_sorted = lambda_res(sort_idx);
T_peak_sorted     = T_peak(sort_idx);

%% Plot 1: Peak Transmission
if PLOT_PEAK_T
    figure;
    plot(param_sorted, T_peak_sorted, 's-', 'LineWidth', 1.5, ...
        'MarkerSize', 7, 'MarkerFaceColor', 'auto');
    xlabel(param_label, 'FontSize', FONT_SIZE);
    ylabel('Peak Transmission (T)', 'FontSize', FONT_SIZE);
    title(['Peak Transmission vs. ' param_short], 'FontSize', FONT_SIZE);
    grid on;
    set(gcf, 'Name', ['Peak T vs ' param_short]);
end

%% Plot 2: Resonance Wavelength
if PLOT_RESONANCE_WL
    figure;
    plot(param_sorted, lambda_res_sorted, 'o-', 'LineWidth', 1.5, ...
        'MarkerSize', 7, 'MarkerFaceColor', 'auto');
    xlabel(param_label, 'FontSize', FONT_SIZE);
    ylabel('Resonance Wavelength [nm]', 'FontSize', FONT_SIZE);
    title(['Resonance Wavelength vs. ' param_short], 'FontSize', FONT_SIZE);
    grid on;
    set(gcf, 'Name', ['Resonance vs ' param_short]);
end
