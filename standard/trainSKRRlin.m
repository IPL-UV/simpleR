function model = trainSKRRlinear(X,Y)

[n,~]  = size(X);
rate   = 0.66;      % Use 2/3 - 1/3 for xvalidation
ntrain = round(rate*n);
r      = randperm(n);
Xtrain = X(r(1:ntrain),:);
Ytrain = Y(r(1:ntrain),:);
Xvalid = X(r(ntrain+1:end),:);
Yvalid = Y(r(ntrain+1:end),:);

n = size(Xtrain,1);  % numero de muestras

% First guess for the sigma_x parameter
Distancias = pdist([Xtrain;Xvalid]);
medianSigma = median(Distancias(:));
sigmaMin = log10(medianSigma*0.1);
sigmaMax = log10(medianSigma*5);
SIGMAS1 = logspace(sigmaMin,sigmaMax,10);
LAMBDAS1 = [0 logspace(-8,4,10)];
LAMBDAS2 = [0 logspace(-8,4,10)];

KtrainY = Ytrain*Ytrain';

i=0;
for sigma1 = SIGMAS1
    Ktrain   = kernelmatrix('rbf',Xtrain',Xtrain',sigma1);
    Kvalid   = kernelmatrix('rbf',Xvalid',Xtrain',sigma1);
    for lambda1 = LAMBDAS1
        for lambda2 = LAMBDAS2
            i=i+1;
            if lambda2==0
                gamma = (Ktrain + lambda1*eye(n))\Ytrain;
            else
                gamma = ((Ktrain + lambda1*eye(n)) \ KtrainY) * ((KtrainY+lambda2*eye(n))\Ytrain);
            end
            Ypred = Kvalid*gamma;
            RESULTS(i,:) = [sigma1 lambda1 lambda2 norm(Yvalid-Ypred,'fro')];
        end
    end
end

% Best model:
[val idx]   = min(RESULTS(:,4));
BestSigma1  = RESULTS(idx,1);
BestLambda1 = RESULTS(idx,2);
BestLambda2 = RESULTS(idx,3);

XX = [Xtrain;Xvalid];
YY = [Ytrain;Yvalid];
[ntrain d] = size(XX);
K = kernelmatrix('rbf',XX',XX',BestSigma1);
Ky = YY*YY';

if BestLambda2==0
    alpha = (K + BestLambda1*eye(ntrain))\YY;
else
    alpha = ((K + BestLambda1*eye(ntrain))\Ky) * ((Ky + BestLambda2*eye(ntrain))\YY);
end

model.BestSigma1 = BestSigma1;
model.BestLambda1 = BestLambda1;
model.BestLambda2 = BestLambda2;
model.alpha = alpha;
model.X     = XX;

