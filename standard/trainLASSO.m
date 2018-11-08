function model_LASSO = trainLASSO(Xtrain, Ytrain)

[b, s]= lasso(Xtrain, Ytrain, 'CV', 5);
% lassoPlot(b, fitinfo);
model_LASSO.B = b;
model_LASSO.S = s;

% Weights and offsets
lam = s.Index1SE; % index of suggested lambda
model_LASSO.W = b(:,lam);
model_LASSO.O = s.Intercept(lam);
