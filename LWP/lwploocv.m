function [alfa, loocvSSE, time] = lwploocv(Xtr, Ytr, p, alfas, maxBadSteps, dpercent, stepSize, verbose)
% LWPLOOCV
% Finds the "best" coefficient value of the Gaussian weight function for
% the LWP using Leave-One-Out Cross-Validation Sum of Squared Error either
% from a provided vector of values or using a simple search algorithm
%
% Call
%   [alfa, loocvSSE, time] = lwploocv(Xtr, Ytr, p, alfas)
%   [alfa, loocvSSE, time] = lwploocv(Xtr, Ytr, p)
%   [alfa, loocvSSE, time] = lwploocv(Xtr, Ytr, p, [], MaxBadSteps, dpercent, stepSize)
%   [alfa, loocvSSE, time] = lwploocv(Xtr, Ytr, p, [], MaxBadSteps, dpercent)
%   [alfa, loocvSSE, time] = lwploocv(Xtr, Ytr, p, [], MaxBadSteps)
%
% Input
% Xtr, Ytr    : Training data points (Xtr(i,:), Ytr(i)), i = 1,...,n
% p           : Degree of the polynomials
% alfas       : A vector of coefficients to try for the Gaussian weight
%               function (>= 0) (if omitted or empty, a simple search
%               algorithm will be used instead)
% maxBadSteps : Maximum number of "bad" steps in which the SSE in the
%               search algorithm did not improve (default = 5); the
%               argument is ignored if alfas is not empty
% dpercent    : Used to stop the search (0...1; default = 0.05 (meaning
%               that the new SSE value should be at least 5% smaller than
%               the so-far best SSE value)); the argument is ignored if
%               alfas is not empty
% stepSize    : A value by which alfa is increased in each step of the 
%               optimization algorithm (default = 10); the argument is
%               ignored if alfas is not empty
% verbose     : Set to 0 for no verbose (default = 1)
%
% Output
% alfa      : The best found value of the coefficient according to LOOCV
% loocvSSE  : LOOCV SSE value at alfa
% time      : Execution time
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

if nargin < 3
    error('Too few input arguments.');
end

[n, d] = size(Xtr);
[ny, dy] = size(Ytr);
if (n < 1) || (d < 1) || (ny ~= n) || (dy ~= 1)
    error('Wrong training data sizes.');
end
if nargin < 5
    maxBadSteps = 5;
end
if nargin < 6
    dpercent = 0.05;
end
if nargin < 7
    stepSize = 10;
end
if nargin < 8
    verbose = 1;
end
if verbose
    fprintf('Searching for the "best" alfa value...\n');
end

ws = warning('off');
tic;

if (nargin >= 4) && (length(alfas) > 0)

    alfas = alfas(:);
    numvals = size(alfas);
    SSE = zeros(numvals);
    inds = true(n, 1);
    for i = 1 : n
        inds(i) = false;
        Xtr2 = Xtr(inds,:);
        Ytr2 = Ytr(inds);
        Xtr3 = Xtr(~inds,:);
        Ytr3 = Ytr(~inds);
        for j = 1 : numvals
            SSE(j) = SSE(j) + (lwppredict(Xtr2, Ytr2, Xtr3, p, alfas(j)) - Ytr3) .^ 2;
        end
        inds(i) = true;
    end
    loocvSSE = min(SSE);
    alfa = alfas(SSE == loocvSSE);

else

    alfa = 0;
    bestAlfa = 0;
    bestSSE = Inf;
    numBadSteps = 0;
    while numBadSteps <= maxBadSteps
        SSE = 0;
        inds = true(n, 1);
        for i = 1 : n
            inds(i) = false;
            SSE = SSE + (lwppredict(Xtr(inds,:), Ytr(inds), Xtr(~inds,:), p, alfa) - Ytr(~inds)) .^ 2;
            inds(i) = true;
        end
        if (SSE / bestSSE < 1 - dpercent) || (bestSSE == Inf)
            bestSSE = SSE;
            bestAlfa = alfa;
            numBadSteps = 0;
        else
            numBadSteps = numBadSteps + 1;
        end
        alfa = alfa + stepSize;
    end
    loocvSSE = bestSSE;
    alfa = bestAlfa;

end

time = toc;
if verbose
    fprintf('The "best" alfa value: %0.2f\n', alfa);
    fprintf('Execution time: %0.2f seconds\n', time);
end
warning(ws);
return
