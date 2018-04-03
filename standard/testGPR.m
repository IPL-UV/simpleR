function [out1, out2] = testGPR(model, xstar)

logtheta = model.loghyper;
covfunc  = model.K;
x        = model.Xtrain;
y        = model.Ytrain;
L        = model.L;
alpha    = model.alpha;

if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed
D = size(x,2);
if eval(feval(covfunc{:})) ~= size(logtheta, 1)
    error('Error: Number of parameters do not agree with covariance function')
end

% Compute (marginal) test predictions ...
[Kss, Kstar] = feval(covfunc{:}, logtheta, x, xstar);     %  test covariances

out1 = Kstar' * alpha;                                    % predicted means

if nargout == 2
    v = L \ Kstar;
    out2 = Kss - sum(v.*v)';
end
