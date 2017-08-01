function xx = PR_STFT_nonconvex_phasecut(Y, N, L, g, x0, handle_negative_b)
% Phase retrieval for STFT via non-convex optimization on the PhaseCut objective
% This code requires the toolbox Manopt.
%
% Inputs:
% Y  - STFT magnitudes
% N - signal's length
% L - spacing between adjacent windows.
% g - stft window
% x0 - initial estimation
% handle_negative_b - 1 or 2 depending on how nonpositive values in the measurements should be handled (2 by default)
%
% Authors: Tamir Bendory and Nicolas Boumal, July 2017

    vec = @(X) X(:);
    
    A = zeros(numel(Y), N);
    I = eye(N);
    for kk = 1 : N
        A(:, kk) = vec(my_stft(I(:, kk), L, g));
    end
    % The matrix A is such that abs(vec(y)) == abs(A*x)
    
    b2 = vec(Y);
   
    % Want to get rid of bad measurements 
    if ~exist('handle_negative_b', 'var') || isempty(handle_negative_b)
        handle_negative_b = 2;
    end
    switch handle_negative_b
        case 1
            b = sqrt(max(0, b2));
        case 2    
            badguys = (b2 <= 0);
            b = sqrt(b2(~badguys));
            A = A(~badguys, :);
        otherwise
            error('handle_negative_b must be 1 or 2.');
    end
    
	% The magic happens here
    xx = PR_manopt(A, b, x0);

end
