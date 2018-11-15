function Yp = testCCF(model, X)

% Canonical correlation forests
% https://arxiv.org/abs/1507.05444
% https://github.com/twgr/ccfs/

Yp = predictFromCCF(model, X);
