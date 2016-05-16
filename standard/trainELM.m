function model = trainELM(X,Y)

[N,di] = size(X);

% Set to non-zero to use it (for instance, vfold = 4)
vfold = 0;

% Cross validation
rand('seed',0)
r = randperm(size(Y,1)); % random index
Ntrain = round(size(Y,1) * 0.66);

% Training
Xtrain = X(r(1:Ntrain),:);
Ytrain = Y(r(1:Ntrain),:);
Ntrain = size(Xtrain,1);

% Validation
Xvalid = X(r((Ntrain+1):end),:);
Yvalid = Y(r((Ntrain+1):end),:);
Nvalid = size(Xvalid,1);

% Row-wise
X = X';
Y = Y';
Xtrain = Xtrain';
Xvalid = Xvalid';
Ytrain = Ytrain';
Yvalid = Yvalid';

NHmax = min(Ntrain,3000);
c=0;
for h = 1:NHmax;
    
    % Build and train the net:
    W1  = rand(h,di)*2-1;
    BH  = rand(h,1);
    tempH = W1*Xtrain;
    ind = ones(1,Ntrain);
    B   = BH(:,ind);
    tempH = tempH+B;
    H  = tempH;
    for lambda=logspace(-10,0,20);
        c=c+1;
    %     W2 = pinv(H') * Ytrain';
        W2 = (lambda*eye(size(H,1)) + H * H') \ H * Ytrain'; % regularized
        Ypred = (H' * W2)';
        RMSEtr = mean(sqrt(mean((Ytrain'-Ypred').^2)));
        
        % Validation
        tempH = W1*Xvalid;
        ind   = ones(1,Nvalid);
        B2    = BH(:,ind);
        Hvalid = tempH+B2;
        Ypredvalid = (Hvalid' * W2)';
        RMSEval = mean(sqrt(mean((Yvalid'-Ypredvalid').^2)));
        results(c,:) = [h lambda RMSEtr RMSEval];
    end
    
end

% Optimal structure in xval
% figure, semilogy(results(:,3),'b'),hold on, semilogy(results(:,4),'r')
[val idx] = min(results(:,4));
h      = results(idx,1);
lambda = results(idx,2);

% Build and train the net:
W1    = rand(h,di)*2-1;
BH    = rand(h,1);
tempH = W1*X;
ind   = ones(1,N);
B     = BH(:,ind);
H     = tempH+B;
% W2    = pinv(H') * Y';
W2 = (lambda*eye(size(H,1)) + H * H') \ H * Y'; % regularized
% Ypred = H' * W2;

% The model
model.hopt = h;
model.W1 = W1;
model.W2 = W2;
model.BH = BH;

% r = assessment(Ytest,Ypredtest, 'regress');
% figure,plot(Ytest,Ypredtest,'k.')
