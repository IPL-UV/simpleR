function model = trainRKS(X,Y)

[n d] = size(X);
rate = 0.66;      % Use 2/3 - 1/3 for xvalidation
ntrain = round(rate*n);
r = randperm(n);
Xtrain = X(r(1:ntrain),:)';
Ytrain = Y(r(1:ntrain),:);
Xtest = X(r(ntrain+1:end),:)';
Ytest = Y(r(ntrain+1:end),:);

DMAX = 2000; %% impact on memory consumption!
DD = round(linspace(5,DMAX,20));

dd=0;
for D = DD;          % number of random features
    dd=dd+1;
    lambda = 1e-3;    % regularization parameter
    w = randn(D,d);
    Z = exp(1i*w*Xtrain);
    Ztest = exp(1i*w*Xtest);
    alpha = (Z*Z' + lambda*ntrain*eye(D) )\(Z*Ytrain);  
    % Testing
    Ypred = real(alpha'*Ztest)';
    error(dd) = norm(Ytest-Ypred,'fro');
end
% figure,plot(DD,error,'ro-'), 
[val idx] = min(error);
D = DD(idx);
w = randn(D,d);
Z = exp(1i*w*Xtrain);
Ztest = exp(1i*w*Xtest);
alpha = (Z*Z' + lambda*ntrain*eye(D) )\(Z*Ytrain);  

% Save model
model.alpha = alpha;
model.w = w;
model.basis = 'fourier';
model.lambda = lambda;
