function Ypredtest = testELM(model,Xtest)

[Ntest,d] = size(Xtest);

Xtest = Xtest';

% The model
h  = model.hopt;
W1 = model.W1;
W2 = model.W2;
BH = model.BH;

% Test the net
tempH = W1*Xtest;
ind   = ones(1,Ntest);
B2    = BH(:,ind);
Htest = tempH+B2;
Ypredtest = Htest' * W2;
