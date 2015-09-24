function model = trainTGP(X,Y)

[n d] = size(X);
rate = 0.66;      % Use 2/3 - 1/3 for xvalidation
ntrain = round(rate*n);
r = randperm(n);
Xtrain = X(r(1:ntrain),:);
Ytrain = Y(r(1:ntrain),:);
Xtest = X(r(ntrain+1:end),:);
Ytest = Y(r(ntrain+1:end),:);

% TGP code
% Initialization
Param.kparam1 = 1e-5/3;
Param.kparam2 = 5*1e-6;
Param.kparam3 = Param.kparam2;
Param.lambda = 1e-3;
Param.tradeoff = 20;
Param.knn = 100;
[InvIK, InvOK] = TGPTrain(Xtrain, Ytrain, Param);

% Ypred = TGPTest(Xtest, model.Xtrain, model.Ytrain, model.Param, model.InvIK, model.InvOK);

% Save model
model.Xtrain = Xtrain;
model.Ytrain = Ytrain;
model.Param  = Param;
model.InvIK  = InvIK;
model.InvOK  = InvOK;
