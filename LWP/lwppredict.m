function [Yq] = lwppredict(Xtr, Ytr, Xq, p, alfa)
% LWPPREDICT
% Predicts output values for the given query data points Xq using a
% Locally Weighted Polynomial (LWP) approximation (also called Moving Least
% Squares) on the training data set (Xtr, Ytr) using Gaussian weight
% function
%
% Call
%   [Yq] = lwppredict(Xtr, Ytr, Xq, p, alfa)
%
% Input
% Xtr, Ytr  : Training data points (Xtr(i,:), Ytr(i)), i = 1,...,n
% Xq        : Inputs of query data points (Xq(i,:)), i = 1,...,nq
% p         : Degree of the polynomials
% alfa      : Coefficient of the Gaussian weight function (alfa >= 0)
%
% Output
% Yq        : Predicted outputs of query data points (Yq(i)), i = 1,...,nq
%
% Reference for the particular LWP implementation:
% Kalnins K., Ozolins O., Jekabsons G. Metamodels in design of GFRP
% composite stiffened deck structure. Proceedings of 7th ASMO-UK/ISSMO
% International Conference on Engineering Design Optimization, Association
% for Structural and Multidisciplinary Optimization in the UK (ASMO-UK),
% ISBN: 978-0853162728, Bath, UK, 2008, pp. 263-273
%
% Or give a reference to the software web page, e.g. like this
% Jekabsons G. Locally Weighted Polynomials for Matlab, 2010, available at
% http://www.cs.rtu.lv/jekabsons/

% This source code is tested with Matlab version 7.1 (R14SP3).

% =========================================================================
% LWP
% Version: 1.3
% Date: February 10, 2010
% Author: Gints Jekabsons (gints.jekabsons@rtu.lv)
% URL: http://www.cs.rtu.lv/jekabsons/
%
% Copyright (C) 2009-2010  Gints Jekabsons
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
% =========================================================================

% The "mlist" code (mlist.cpp) and dll (mlist.dll) is taken from ENTOOL
% (http://www.j-wichard.de/entool/) which also falls under the GNU GPL
% license. The authors of ENTOOL are Christian Merkwirth and Joerg Wichard.

if nargin < 5
    error('Too few input arguments.');
end

Yq = zeros(size(Xq, 1), 1);
[n, d] = size(Xtr);
nq = size(Xq, 1);

%calculate weights
u = zeros(n, nq);
for t = 1 : nq
    %calculate the numerators of u
    u(:, t) = sum((repmat(Xq(t, :),n,1) - Xtr) .^ 2, 2);
    %find the farthest points
    maxidx = find(u(:,t) == max(u(:,t)),1);
    %calculate the final values of u
    dn = sum((Xq(t, :) - Xtr(maxidx, :)) .^ 2);
    u(:, t) = u(:, t) / dn;
end
w = exp(-alfa * u);

%create matrix of linear, interaction, and squared terms
if p <= 2

    ncols = 1 + d * p;
    if (p > 1)
        ncols = ncols + d*(d-1)/2;
    end
    Vals = ones(n, ncols);
    Valsq = ones(nq, ncols);
    prevcol = 1;
    if p > 0
        cols = prevcol+1 : prevcol+d;
        Vals(:, cols) = Xtr;
        Valsq(:, cols) = Xq;
        prevcol = prevcol + d;
        if p > 1
            first = repmat(1:d, d, 1);
            second = first';
            t = first < second;
            cols = prevcol+1 : prevcol+d*(d-1)/2;
            Vals(:, cols) = Xtr(:, first(t)) .* Xtr(:, second(t));
            Valsq(:, cols) = Xq(:, first(t)) .* Xq(:, second(t));
            prevcol = prevcol + d*(d-1)/2;

            cols = prevcol+1 : prevcol+d;
            Vals(:, cols) = Xtr .^ 2;
            Valsq(:, cols) = Xq .^ 2;
        end
    end

else

    r = mlist(d, p)';
    k = size(r, 1);
    Vals = ones(n, k);
    Valsq = ones(nq, k);
    for idx = 2 : k
        bf = r(idx, :);
        t = bf > 0;
        tmp = Xtr(:, t) .^ bf(ones(n, 1), t);
        tmpq = Xq(:, t) .^ bf(ones(nq, 1), t);
        Vals(:, idx) = prod(tmp, 2);
        Valsq(:, idx) = prod(tmpq, 2);
    end

end

%calculate coefs and predict the output values
ws = warning('off');
for t = 1 : nq
    Vals_wd = Vals' * diag(w(:, t));
    coefs = (Vals_wd * Vals) \ (Vals_wd * Ytr);
    Yq(t) = Valsq(t,:) * coefs;
end
warning(ws);
return
