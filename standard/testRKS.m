function Ypred = testRKS(model,Xtest)

Xtest = Xtest';
Ztest = exp(1i*model.w*Xtest);
Ypred = real(model.alpha'*Ztest)';
