function [out1, out2] = testSSGPR(model, xstar)

optimizeparams = model.loghyper;
x              = model.Xtrain;
y              = model.Ytrain;

[out1,out2] = ssgprfixed(optimizeparams, x, y, xstar);