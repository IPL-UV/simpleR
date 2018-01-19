function Yp = testMSVR(model,Xtest)

% model.Beta = Beta;
% model.NSV = NSV;
% model.sigma = sigma(bestGamma);
% model.C = C(bestC);
% model.Xtrain = X;

Ktest = kernelmatrix('rbf', Xtest', model.Xtrain', model.sigma);
Yp = Ktest * model.Beta;
