%% Plot Peak Transmission and Mode Width vs. Apodization (TE & TM)
% Apodization sweep for the pi-shift Bragg grating. Plots, for each polarization:
%   Fig 1 : peak (resonance) transmission   vs. # apodized teeth per side
%   Fig 2 : spatial mode width fwhm_m [um]   vs. # apodized teeth per side
%
% Each .mat file's apodization is parsed from its filename:
%   _A<n>  -> n apodized teeth per side  (no _A token => 0, i.e. no apodization)
%   _TM    -> TM polarization            (otherwise TE)
% so you can mix the apodized results (result_N80_A*_M4*[_TM]_avg.mat from the
% tm_te_apod study) with the reused 0-teeth baseline files
% (result_N80_avg_te.mat / result_N80_TM_avg_tm.mat from run_tm_vs_te).
%
% Because the apodized and baseline files live in DIFFERENT folders, the picker
% asks twice: first the apodized results, then the 0-teeth baseline files
% (Cancel the second dialog to skip the baseline).
addpath(fileparts(fileparts(mfilename('fullpath'))));  % project root on path

clear; clc;
close all;

%% User settings
FONT_SIZE = 13;

%% Select .mat files — two picks (apodized, then baseline) since they live in
%  different folders and uigetfile browses one folder at a time.
prefs_file = fullfile(fileparts(mfilename('fullpath')), 'plot_prefs.mat');
start_path = '*.mat';
if exist(prefs_file, 'file')
    p = load(prefs_file);
    if isfield(p, 'apod_last_folder') && isfolder(p.apod_last_folder)
        start_path = fullfile(p.apod_last_folder, '*.mat');
    end
end

paths = {};

% Pick 1: apodized result files (result_N80_A*_..._avg.mat)
[sel, folder] = uigetfile(start_path, ...
    'Select APODIZED result files (result_N80_A*..._avg.mat)', 'MultiSelect', 'on');
if ~isequal(sel, 0)
    if ischar(sel), sel = {sel}; end
    paths = [paths, fullfile(folder, sel)];   % fullfile is vectorized over the cell
    apod_last_folder = folder;
    if exist(prefs_file, 'file')
        save(prefs_file, 'apod_last_folder', '-append');
    else
        save(prefs_file, 'apod_last_folder');
    end
    start_path = fullfile(folder, '*.mat');
end

% Pick 2: 0-teeth baseline files (result_N80_avg_te / result_N80_TM_avg_tm) — optional
[sel, folder] = uigetfile(start_path, ...
    'Select 0-teeth BASELINE files (run_tm_vs_te) — or Cancel to skip', 'MultiSelect', 'on');
if ~isequal(sel, 0)
    if ischar(sel), sel = {sel}; end
    paths = [paths, fullfile(folder, sel)];
end

if isempty(paths)
    disp('No files selected.');
    return;
end
nFiles = numel(paths);

%% Parse + load each file
n_apod = zeros(1, nFiles);
is_tm  = false(1, nFiles);
T_peak = nan(1, nFiles);
width_um = nan(1, nFiles);

for k = 1:nFiles
    fp = paths{k};
    [~, base, ext] = fileparts(fp);
    fname = [base ext];

    % --- apodized teeth per side from filename (_A<n>; absent => 0) ---
    tok = regexp(fname, '_A(\d+)', 'tokens', 'once');
    if isempty(tok)
        n_apod(k) = 0;
    else
        n_apod(k) = str2double(tok{1});
    end

    % --- polarization (_TM present => TM, else TE) ---
    is_tm(k) = contains(fname, '_TM');

    % --- load ---
    d = load(fp);

    % --- peak (resonance) transmission ---
    if isfield(d, 'resonance_transmission')
        T_peak(k) = double(d.resonance_transmission);
    elseif isfield(d, 'T') && isfield(d, 'wl_nm') && isfield(d, 'resonance_wavelength_nm')
        wl = double(d.wl_nm(:));  T = double(d.T(:));
        [~, ir] = min(abs(wl - double(d.resonance_wavelength_nm)));
        T_peak(k) = T(ir);
    else
        warning('No transmission info in %s', fname);
    end

    % --- spatial mode width (fwhm_m) in micrometers ---
    if isfield(d, 'fwhm_m')
        width_um(k) = double(d.fwhm_m) * 1e6;
    else
        warning('No fwhm_m in %s', fname);
    end

    fprintf('%-44s  pol=%s  n_apod=%2d  T=%.4f  width=%.3f um\n', ...
        fname, ternary(is_tm(k),'TM','TE'), n_apod(k), T_peak(k), width_um(k));
end

%% Split by polarization and sort by n_apod
[te_x, te_T, te_w] = collect(~is_tm, n_apod, T_peak, width_um);
[tm_x, tm_T, tm_w] = collect( is_tm, n_apod, T_peak, width_um);

all_x = unique(n_apod);

%% Fig 1: Peak transmission vs apodization
figure('Name', 'Peak T vs apodization');
hold on;
if ~isempty(te_x), plot(te_x, te_T, 'o-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TE'); end
if ~isempty(tm_x), plot(tm_x, tm_T, 's-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TM'); end
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Peak transmission (T)', 'FontSize', FONT_SIZE);
title('Resonance transmission vs. apodization', 'FontSize', FONT_SIZE);
legend('Location', 'best'); grid on; box on;
if ~isempty(all_x), xticks(all_x); end

%% Fig 2: Spatial mode width vs apodization
figure('Name', 'Mode width vs apodization');
hold on;
if ~isempty(te_x), plot(te_x, te_w, 'o-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TE'); end
if ~isempty(tm_x), plot(tm_x, tm_w, 's-', 'LineWidth', 1.6, 'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', 'DisplayName', 'TM'); end
hold off;
xlabel('Apodized teeth per side', 'FontSize', FONT_SIZE);
ylabel('Mode width, FWHM [\mum]', 'FontSize', FONT_SIZE);
title('Spatial mode width vs. apodization', 'FontSize', FONT_SIZE);
legend('Location', 'best'); grid on; box on;
if ~isempty(all_x), xticks(all_x); end

%% ── local helpers ───────────────────────────────────────────────────────────
function [x, y1, y2] = collect(mask, n_apod, T_peak, width_um)
    x  = n_apod(mask);
    y1 = T_peak(mask);
    y2 = width_um(mask);
    [x, si] = sort(x);
    y1 = y1(si);
    y2 = y2(si);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
