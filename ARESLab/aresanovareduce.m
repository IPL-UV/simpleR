function [modelReduced usedBasis] = aresanovareduce(model, vars, exact)
% aresanovareduce
% Deletes all the basis functions from an ARES model in which at least one
% used variable is not in the given list of allowed variables. This can be
% used to perform ANOVA decomposition as well as for investigation of
% individual and joint contributions of variables in the model, i.e.,
% the reduced model can be plotted using function aresplot to visualize
% the contributions.
%
% Call:
%   [modelReduced usedBasis] = aresanovareduce(model, vars, exact)
%
% Input:
%   model         : ARES model
%   vars          : A vector of indexes for input variables to stay in the
%                   model. The size of the vector should be between 1 and
%                   d, where d is the total number of input variables.
%   exact         : Set this to true to get a model with only those basis
%                   functions where the exact combination of variables is
%                   present (default value = false). This is used from
%                   function aresanova.
%
% Output:
%   modelReduced  : The reduced model
%   usedBasis     : The list of original indexes for the used basis
%                   functions

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

% Last update: June 2, 2011

if nargin < 2
    error('Too few input arguments.');
end
if (nargin < 3) || (isempty(exact))
    exact = false;
end

len = length(model.knotdims);
notvars = setdiff(1:length(model.minX), vars);
stay = true(len,1);

for i = 1 : len
    if exact && (length(unique(model.knotdims{i})) ~= length(vars))
        stay(i) = false;
        continue;
    end
    for j = 1 : length(notvars)
        if any(model.knotdims{i} == notvars(j))
            stay(i) = false;
            break;
        end
    end
end

usedBasis = find(stay);

modelReduced = model;
for i = len : -1 : 1
    if ~stay(i)
        modelReduced.coefs(i+1) = [];
        modelReduced.knotdims(i) = [];
        modelReduced.knotsites(i) = [];
        modelReduced.knotdirs(i) = [];
        modelReduced.parents(i) = [];
        modelReduced.parents = updateParents(modelReduced.parents, i);
        if modelReduced.trainParams.cubic
            modelReduced.t1(i,:) = [];
            modelReduced.t2(i,:) = [];
        end
    end
end

return

function parents = updateParents(parents, deletedInd)
% Updates direct parent indexes after deletion of a basis function.
parents(parents == deletedInd) = 0;
tmp = parents > deletedInd;
parents(tmp) = parents(tmp) - 1;
return
