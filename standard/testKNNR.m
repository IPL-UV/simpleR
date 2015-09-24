function Ypred = testKNNR(model,X)

[n d] = size(X);
[IDX,D] = knnsearch(model.Xtrain,X,'K',model.k);
Ypred   = mean(model.Ytrain(IDX),2);
