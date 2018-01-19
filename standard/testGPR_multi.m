function [out1, out2] = testGPR_multi(model, xstar)

logtheta = model.loghyper;
covfunc  = model.K;
x        = model.Xtrain;
y        = model.Ytrain;

% GPR - Gaussian process regression.
% With test data are given, then (marginal) Gaussian predictions are computed,
% whose mean and variance are returned. Note that in cases where the covariance
% function has noise contributions, the variance returned in S2 is for noisy
% test targets; if you want the variance of the noise-free latent function, you
% must substract the noise variance.
%
% usage: [nlml dnlml] = gpr(model, xstar)
%
% where:
%
%   model    is a the model returned by trainGPR_multi.m
%   xstar    is a nn by D matrix of test inputs
%   nlml     is the returned value of the negative log marginal likelihood
%   dnlml    is a (column) vector of partial derivatives of the negative
%                 log marginal likelihood wrt each log hyperparameter
%
% For more help on covariance functions, see "help covFunctions".
%
% (C) copyright 2006 by Carl Edward Rasmussen (2006-03-20).
%               2017 by Gus, Valero & Jordi

if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed
[n,D] = size(x);
if eval(feval(covfunc{:})) ~= size(logtheta, 1)
    error('Error: Number of parameters do not agree with covariance function')
end

K = feval(covfunc{:}, logtheta, x);    % compute training set covariance matrix

L = chol(K)';                        % cholesky factorization of the covariance
alpha = L'\(L\y);

% compute (marginal) test predictions ...

[Kss, Kstar] = feval(covfunc{:}, logtheta, x, xstar);     %  test covariances

out1 = Kstar' * alpha;                                      % predicted means

if nargout == 2
    v = L\Kstar;
    out2 = Kss - sum(v.*v)';
end
