function [x_est, problem] = PR_manopt(A, b, x0)
% Generic implementation of phase retrieval in Manopt, via the PhaseCut cost function.
% x0 is the initial guess; b contains the moduli of the measurements A*x.
% Returns x_est : an estimate of the signal x.
%
% Authors: Tamir Bendory and Nicolas Boumal, July 2017

	[M, N] = size(A);

	[Q, R] = qr(A, 0);

	
	%% initial guess for u
	u0 = sign(A*x0);

	%% Manopt

	% We aim to solve max u' * H * u   s.t.   norm(u_i) = 1 for all i,
	% where H = diag(b) * A * pinv(A) *diag(b) is a linear operator.
	% u0 can be used to initialize the algorithm.
	% Once u is esimated via manopt, we can recover x through
	% a least-squares computation: x = pinv(A)*diag(b)*u

	% Pick the manifold of phases and define the cost function and its derivatives here
	problem.M = complexcirclefactory(M);
	problem.cost = @(u) -.5*norm(Q'*(b.*u))^2;
	problem.egrad = @(u) -b.*(Q*(Q'*(b.*u)));
	problem.ehess = @(u, udot) -b.*(Q*(Q'*(b.*udot)));
	
	opts.verbosity = 0; % increase to see iteration information
	opts.tolgradnorm = 1e-7;
	
	% Optimization happens here
	u_est = trustregions(problem, u0, opts);
	
	% Solve the least-squares problem
	x_est = R\(Q'*(b.*u_est));

end
