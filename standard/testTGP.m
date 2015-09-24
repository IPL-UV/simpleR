function Ypred = testTGP(model,Xtest)

Ypred = TGPTest(Xtest, model.Xtrain, model.Ytrain, model.Param, model.InvIK, model.InvOK);