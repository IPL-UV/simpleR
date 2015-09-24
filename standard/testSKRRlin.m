function yp = testSKRRlinear(model,Xtest)

K = kernelmatrix('rbf',Xtest',model.X',model.BestSigma1);
yp = K * model.alpha;

