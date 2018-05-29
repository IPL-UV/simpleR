function yt = testXGB(model, xt)

% function y = testXGB(model, xt)
%
% Predict using previously trained XGB in model

yt = SQBMatrixPredict(model, single(xt));
