function [out1, out2] = gpr_multi_sl(logtheta, covfunc, x, y, xstar)

% This version implements multioutput as in Python's scikit-learn. It is similar
% to our multioutput version (gpr_multi), but here the true log likelihood is
% computed, assuming that each output variable is independent and then the total
% likelihood is the sum of the individual likelihoods.
% This is very similar to what we do in gpr_multi, just removing the square from
% the likelihood and the 2 * factor in the partial derivatives.
% A good thing about this version is that it is the same as the single-output
% version when y is 1-D. That is, the single-output version is a particular case
% of this one.

% gpr - Gaussian process regression, with a named covariance function. Two
% modes are possible: training and prediction: if no test data are given, the
% function returns minus the log likelihood and its partial derivatives with
% respect to the hyperparameters; this mode is used to fit the hyperparameters.
% If test data are given, then (marginal) Gaussian predictions are computed,
% whose mean and variance are returned. Note that in cases where the covariance
% function has noise contributions, the variance returned in S2 is for noisy
% test targets; if you want the variance of the noise-free latent function, you
% must substract the noise variance.
%
% usage: [nlml dnlml] = gpr(logtheta, covfunc, x, y)
%    or: [mu S2]  = gpr(logtheta, covfunc, x, y, xstar)
%
% where:
%
%   logtheta is a (column) vector of log hyperparameters
%   covfunc  is the covariance function
%   x        is a n by D matrix of training inputs
%   y        is a (column) vector (of size n) of targets
%   xstar    is a nn by D matrix of test inputs
%   nlml     is the returned value of the negative log marginal likelihood
%   dnlml    is a (column) vector of partial derivatives of the negative
%                 log marginal likelihood wrt each log hyperparameter
%   mu       is a (column) vector (of size nn) of prediced means
%   S2       is a (column) vector (of size nn) of predicted variances
%
% For more help on covariance functions, see "help covFunctions".
%
% (C) copyright 2006 by Carl Edward Rasmussen (2006-03-20).

if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed
[n,D] = size(x);  %#ok<NASGU>
if eval(feval(covfunc{:})) ~= size(logtheta, 1)
    error('Error: Number of parameters do not agree with covariance function')
end

K = feval(covfunc{:}, logtheta, x);  % compute training set covariance matrix

L = chol(K)';                        % cholesky factorization of the covariance
alpha = L'\(L\y);

if nargin == 4 % if no test cases, compute the negative log marginal likelihood

    % Original single-output likelihood
    % out1 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);
    % scikit-learn likelihood
    out1 = sum( 0.5*diag(y'*alpha) + sum(log(diag(L))) + 0.5*n*log(2*pi) );

    if nargout == 2                     % ... and if requested, its partial derivatives
        out2 = zeros(length(logtheta), size(alpha,2));  % set the size of the derivative vector
        Wp = L' \ (L \ eye(n));                         % precompute for convenience
        for d = 1:size(alpha,2)
            W = Wp - alpha(:,d) * alpha(:,d)';          % precompute for convenience
            for i = 1:length(out2)
                out2(i,d) = sum(sum(W .* feval(covfunc{:}, logtheta, x, i))) / 2;
            end
        end
        out2 = sum(out2,2);
    end

else                    % ... otherwise compute (marginal) test predictions ...

    [Kss, Kstar] = feval(covfunc{:}, logtheta, x, xstar);     % test covariances

    out1 = Kstar' * alpha;                                    % predicted means

    if nargout == 2
        v = L \ Kstar;
        out2 = Kss - sum(v.*v)';
    end

end
