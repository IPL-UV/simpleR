function Yp = testSVR(model,Xtest)

% Xtrain = model.Xtrain;
% sigma  = model.sigma;

% Kt = kernelmatrix('rbf', Xtest', Xtrain', sigma);
Yp = mysvmpredict(zeros(size(Xtest,1),1), Xtest, model); % Kt, rmfield(model,{'Xtrain','sigma','C'}));

% This is the code to obtain predictions directly in MATLAB without svmpredict.
% However, in my speed tests I found svmpredict to be faster than this.
% K = kernelmatrix('rbf', Xtest', model.SVs', 1/(sqrt(2*model.Parameters(4))));
% Yp = K * model.sv_coef - model.rho;
