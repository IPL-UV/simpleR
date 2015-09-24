function model_ARES = trainARES(X,Y)

maxBasis = 20;  % reasonable number
vfold    = 3;

for b=1:maxBasis
    params = aresparams(b, [], [], [], [], 2);
    MSE(b) = arescv(X, Y, params, [], vfold,[],[],[],0);
end
[val bestBasis] = min(MSE);
params = aresparams(bestBasis, [], [], [], [], 2);
model_ARES.model  = aresbuild(X, Y, params)
model_ARES.bestBasis = bestBasis;
model_ARES.vfold = vfold;
