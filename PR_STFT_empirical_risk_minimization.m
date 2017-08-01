function [xhat, elapse, err, obj, x0] = PR_STFT_empirical_risk_minimization(Y, W, L, fc, x0, x)
% STFT Phase retrieval via empirical risk minimization
%
% This implementation uses the trust-regions algorithm in the Manopt toolbox.
%
% Authors: Tamir Bendory and Nicolas Boumal, July 2017


%% -------------------INPUT ---------------------

% Y        - the measurements of the squared magnitudes of the STFT
% W        - the length of the rectangular window. This code works for rectangular windows defined as g = zeros(N,1); g(1:W) = 1;
% L        - spacing between adjacent windows.
% x0       - initial guess for the signal (optional)
% x        - the underlying signal; used merely for comparison purposes (optional)


%% ------------------ OUTPUT -----------

% xhat   - estimation of the signal
% elapse - elapsed time
% err    - error curve with respect to the underlying signal x (if x is provided)
% obj    - objective function curve
% x0     - the intial guess that was used


	total_time = tic();

	%% ----------------- GENERATING VARIABLES  -------------------------

	% the signal's length
	N = size(Y, 2);
	vec = (0:N-1)';

	% the rectangular window
	g = zeros(N,1);
	g(1:W) = 1;


	%% ------------- INITIALIZATION-------------------------
	
	if ~exist('x0', 'var') || isempty(x0)
		Yhat = fft(Y')' / N;
		x0 = LS_init(Yhat, W, L);
	end


	%% ---------------MAIN ITERATION -------------------------


	% TODO: improve implementation to avoid the need to precompute the DFT matrix
	F = dftmtx(N);


	problem.M = euclideanfactory(N, 1);
	problem.costgrad = @costgrad;
	function [obj, grad] = costgrad(xx)

		obj = 0;
		grad = zeros(N, 1);
		
		for mm = 0 : ceil(N/L)-1
			
			gm = g(mod(mm*L-vec, N)+1);
			
			for kk = -fc:fc
				fk = F(mod(kk,N)+1,:);
				a = gm .* fk';
				atxx = a'*xx;
				Axx = a*atxx;
				xxtAxx = abs(atxx)^2;
				grad = grad + real(4*(xxtAxx - Y(mm+1, mod(kk,N)+1) )*Axx);
				obj = obj + (Y(mm+1, mod(kk,N)+1) - xxtAxx)^2;
			end
			
		end
	end


	% Ask Manopt to compute a distance to the ground truth at each iteration (for analysis)
	if exist('x', 'var')
		metrics.err = @(xx) norm(real(xx*sign(xx'*x))-x)/norm(x);
		options.statsfun = statsfunhelper(metrics);
	end

	options.tolgradnorm = 1e-5 * N^2;
	options.verbosity = 0;

	% This part uses Manopt to solve the optimization problem
	warning('off', 'manopt:getHessian:approx');
	[xhat, ~, stats] = trustregions(problem, x0, options);
	warning('on', 'manopt:getHessian:approx');

	obj = [stats.cost];
	if exist('x', 'var')
		err = [stats.err];
	else
		err = [];
	end

	elapse = toc(total_time);

end
