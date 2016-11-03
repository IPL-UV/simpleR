function Yp = testSVR(model,Xtest)

% Xtrain = model.Xtrain;
% sigma  = model.sigma;

% Kt = kernelmatrix('rbf',Xtest',Xtrain',sigma);
Yp = svmpredict(zeros(size(Xtest,1),1), Xtest, model); % Kt, rmfield(model,{'Xtrain','sigma','C'}));
