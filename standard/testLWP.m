function Yp = testLWP(model,Xtest)

Yp = lwppredict(model.X, model.Y, Xtest, model.p, model.alpha);
