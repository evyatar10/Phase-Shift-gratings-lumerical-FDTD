% startup.m - Run once to add project folders to MATLAB path
project_root = fileparts(mfilename('fullpath'));
addpath(project_root);
addpath(fullfile(project_root, 'matlab_plotting'));
addpath(fullfile(project_root, 'matlab_analysis'));
