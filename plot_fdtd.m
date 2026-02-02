%% Plot T, R, Loss, Phase, and Phase Derivatives
% This script loads .mat files containing complex S-parameters.
% It calculates T and R from them, and plots Loss, Phase, and Phase Derivatives.
clear; clc;

%% User settings
% Filter range in microns for the combined loss and transmission plots
LAMBDA_MIN_UM = 1.54; 
LAMBDA_MAX_UM = 1.59;

% LAMBDA_MIN_UM = 1.5; 
% LAMBDA_MAX_UM = 1.8; 

% Convert to nm
LAMBDA_MIN_NM = 1000.0 * LAMBDA_MIN_UM;
LAMBDA_MAX_NM = 1000.0 * LAMBDA_MAX_UM;

HIGHLIGHT_PEAKS     = true; % Only applies to Combined Loss plot
PLOT_PHASE          = true; 
PLOT_RELATIVE_DERIV = true; % Toggle plotting the derivative of angle(t/r)

%% Select .mat files
[files, folder] = uigetfile('*.mat', 'Select .mat files', ...
                            'MultiSelect', 'on');
if isequal(files, 0)
    disp('No files selected.');
    return;
end

% Ensure files is a cell array
if ischar(files)
    files = {files};
end
nFiles = numel(files);

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
    
    datasets(k).name      = erase(files{k}, '.mat');
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

%% Plot 1: Combined Filtered Loss
figure;
hold on;
colors = lines(nFiles);
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
xlabel('Wavelength [nm]');
ylabel('Loss (1 - R - T)');
title('Combined Loss (Zoomed)');
grid on;
legend('show', 'Location', 'best', 'Interpreter', 'none');
set(gcf, 'Name', 'Combined Loss');

%% Plot 2: Combined Filtered Transmission
figure;
hold on;
for k = 1:nFiles
    ds = datasets(k);
    
    % Main line (Zoomed T)
    plot(ds.wl_loss, ds.T_f, 'DisplayName', ds.name, ...
         'Color', colors(k,:), 'LineWidth', 1.5);
end
hold off;
xlabel('Wavelength [nm]');
ylabel('Transmission (T)');
title('Combined Transmission (Zoomed)');
grid on;
legend('show', 'Location', 'best', 'Interpreter', 'none');
set(gcf, 'Name', 'Combined Transmission');

%% Plot 3..N: Individual File Plots (T, R, Loss)
for k = 1:nFiles
    ds = datasets(k);
    figure;
    hold on;
    
    plot(ds.wl_nm, ds.T,   'DisplayName', 'T',    'LineWidth', 1.5); 
    plot(ds.wl_nm, ds.R,   'DisplayName', 'R',    'LineWidth', 1.5);
    plot(ds.wl_nm, ds.loss,'DisplayName', 'Loss', 'LineWidth', 1.5);
    
    hold off;
    xlabel('Wavelength [nm]');
    ylabel('Normalized Power');
    title(sprintf('%s : Power (T, R)', ds.name), 'Interpreter', 'none');
    grid on;
    legend('show');
    ylim([-0.1 1.1]);
    set(gcf, 'Name', sprintf('%s_Power', ds.name));
end

%% Plot: Combined Phase (t & r)
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
xlabel('Wavelength [nm]');
ylabel('Wrapped Phase [\pi rad]');
title('Combined Phase (Solid=t, Dashed=r)');
grid on;
if hasPhaseData
    legend('show', 'Interpreter', 'none');
end
set(gcf, 'Name', 'Combined Phase');

%% Plot: Derivatives of Phase (T, R, and Relative)
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
xlabel('Wavelength [nm]');
ylabel('d(Phase)/d\lambda [\pi rad / nm]');
title('Derivative of Phase w.r.t Wavelength');
grid on;
if hasDerivData
    legend('show', 'Interpreter', 'none');
end
set(gcf, 'Name', 'Phase Derivative');