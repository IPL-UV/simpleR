function model = trainCCF(X, Y, nTrees)

% Canonical correlation forests
% https://arxiv.org/abs/1507.05444
% https://github.com/twgr/ccfs/

if ~exist('nTrees', 'var')
    nTrees = 500;
end

model = genCCF(nTrees, X, Y, true);
