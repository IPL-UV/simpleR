function [MSE, RMSE, RRMSE, R2] = arestest(model, Xtst, Ytst)
% arestest
% Tests an ARES model on a test data set (Xtst, Ytst).
%
% Call:
%   [MSE, RMSE, RRMSE, R2] = arestest(model, Xtst, Ytst)
%
% Input:
%   model         : ARES model
%   Xtst, Ytst    : Test data cases (Xtst(i,:), Ytst(i)), i = 1,...,ntst
%
% Output:
%   MSE           : Mean Squared Error
%   RMSE          : Root Mean Squared Error
%   RRMSE         : Relative Root Mean Squared Error
%   R2            : Coefficient of Determination

% =========================================================================
% ARESLab: Adaptive Regression Splines toolbox for Matlab/Octave
% Author: Gints Jekabsons (gints.jekabsons@rtu.lv)
% URL: http://www.cs.rtu.lv/jekabsons/
%
% Copyright (C) 2009-2011  Gints Jekabsons
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% =========================================================================

% Last update: November 9, 2009

if nargin < 3
    error('Too few input arguments.');
end
if (isempty(Xtst)) || (isempty(Ytst))
    error('Data is empty.');
end
if (size(Xtst, 1) ~= size(Ytst, 1))
    error('The number of rows in the matrix and the vector should be equal.');
end
if size(Ytst,2) ~= 1
    error('The vector Ytst should have one column.');
end
MSE = mean((arespredict(model, Xtst) - Ytst) .^ 2);
RMSE = sqrt(MSE);
if size(Ytst, 1) > 1
    RRMSE = RMSE / std(Ytst, 1);
    R2 = 1 - MSE / var(Ytst, 1);
else
    RRMSE = Inf;
    R2 = Inf;
end
return
