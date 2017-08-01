%% Main example

clear all; 
close all;
clc;

%% Make sure you have Manopt installed (see www.manopt.org)
if isempty(which('spherefactory'))
    warning('Please download Manopt from www.manopt.org and add it to your Matlab path.');
end

%% Choose the simulation parameters

% Signal length
N = 53;

% Window length. This script uses a rectangular window
W = 6;

% Separation in time between adjacent window sections
L = 2;

% Frequency cut-off
low_pass = false;
if low_pass
    fc = 18; %#ok
else
    fc = (N-1)/2;
end

%% Generating the signal and the measurements


% Ground truth signal
x = randn(N, 1);
    
% Rectangular window
g = zeros(N, 1);
g(1:W) = 1; 

% Compute the STFT of x
X = my_stft(x, L, g);

% Noiseless measurements
Yc = abs(X).^2;


% Noisy measurements
snr = 30;
noise = randn(size(Yc));
sigma = norm(Yc, 'fro')*10^(-snr/20)/norm(noise, 'fro');
Y = Yc + sigma*noise;

% Alternative noise model:    
% Y = poissrnd(Yc);

% low-pass
if low_pass
	Y(:, fc+2:N-fc) = 0; %#ok<UNRCH>
end



%% Initialization algorithm

T = tic();
Yhat = fft(Y')' / N;
x0 = LS_init(Yhat, W, L);
fprintf('Initialization\nTime: %.3g [s]\nRelative error: %.3g\n', toc(T), norm(x-(x0*sign(x0'*x)))/norm(x));

%% Empirical risk minimization

T = tic();

% Ground truth x passed along to compute error metrics
[x_hat_er, elapse, err, obj, x0] = PR_STFT_empirical_risk_minimization(Y, W, L, fc, x0, x);

% Align estimator to ground truth
x_hat_er = x_hat_er*sign(x_hat_er'*x);

fprintf('Empirical risk\nTime: %.3g [s]\nRelative error: %.3g\n', toc(T), norm(x-x_hat_er)/norm(x));

%% Non-convex PhaseCut formulation

T = tic();

x_hat_ncpc = PR_STFT_nonconvex_phasecut(Y, N, L, g, x0);
x_hat_ncpc = x_hat_ncpc*sign(x_hat_ncpc'*x);

fprintf('Non-convex phase cut\nTime: %.3g [s]\nRelative error: %.3g\n', toc(T), norm(x-x_hat_ncpc)/norm(x));

%% GLA

T = tic();

x_hat_gla = PR_STFT_GLA(Y, N, L, g, x0); 
x_hat_gla = x_hat_gla*sign(x_hat_gla'*x);

fprintf('Griffin-Lim Algorithm (GLA)\nTime: %.3g [s]\nRelative error: %.3g\n', toc(T), norm(x-x_hat_gla)/norm(x));
