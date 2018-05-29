function model = trainWGPR(X1,Y1)

% rescale targets
tmax = max(Y1); tmin = min(Y1);
Y    = (Y1-tmin)/(tmax-tmin) - 0.5;

num = 5; % set number of tanh functions in warping
[n,D] = size(X1);
Xw = randn(D+2+num*3,1); % randomly initialise parameters for gpw01lik
% Xw = randn(D+3+num*3,1); % randomly initialise parameters for gpw03lik
			 
% Alternate between 'slow' full gradient optimization and 'fast'
% warping parameter only optimization
for j = 1:5
    [Xw,fX,i] = minimize(Xw,'gpw01lik',-100,X1,Y);
    [Xw,fX,i] = minimize(Xw,'gpw01fastlik',-200,X1,Y);
end

% lengthscales = log((max(X1)-min(X1))/2)'
% SignalPower  = var(Y1)
% Xw = [lengthscales; 0.5*log(SignalPower); 0.5*log(SignalPower); rand(num*3,1)];
% [Xw fX i] = minimize(Xw, 'gpw01lik', -200, X1, Y);

model.Xw = Xw;
model.fX = fX;
model.i = i;
model.functions = num;
model.Xtrain = X1;
model.Ytrain = Y1;
model.Y = Y;
model.tmax = tmax;
model.tmin = tmin;
