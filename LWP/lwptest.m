function [MSE, RMSE, RRMSE, R2] = lwptest(Xtr, Ytr, Xtst, Ytst, p, alfa)
% LWPTEST
% Tests the LWP approximation on a test data set (Xtst, Ytst)
%
% Call
%   [MSE, RMSE, RRMSE, R2] = lwptest(Xtr, Ytr, Xtst, Ytst, p, alfa)
%
% Input
% Xtr, Ytr  : Training data points (Xtr(i,:), Ytr(i)), i = 1,...,n
% Xtst, Ytst: Test data points (Xtst(i,:), Ytst(i)), i = 1,...,ntst
% p         : Degree of the polynomials
% alfa      : Coefficient of the Gaussian weight function (alfa >= 0)
%
% Output
% MSE       : Mean Squared Error
% RMSE      : Root Mean Squared Error
% RRMSE     : Relative Root Mean Squared Error
% R2        : Coefficient of Determination

% Copyright (C) 2009-2010  Gints Jekabsons

if nargin < 6
    error('Too few input arguments.');
end
if (size(Xtr, 1) ~= size(Ytr, 1)) || (size(Xtst, 1) ~= size(Ytst, 1))
    error('The number of rows in matrices and vectors should be equal.');
end
MSE = mean((lwppredict(Xtr, Ytr, Xtst, p, alfa) - Ytst) .^ 2);
RMSE = sqrt(MSE);
if size(Ytst, 1) > 1
    RRMSE = RMSE / std(Ytst, 1);
    R2 = 1 - MSE / var(Ytst, 1);
else
    RRMSE = Inf;
    R2 = Inf;
end
return
