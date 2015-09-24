function model = trainSSGPR(X1,Y1)

% Regression assumes zero-mean functions, substract mean
meanp=mean(Y1);
Y1=Y1-meanp;

[n,D]=size(X1);

% -- Initial hyperparameters setting

% use a reasonable initial guess
lengthscales=log((max(X1)-min(X1))'/2);
lengthscales(lengthscales<-1e2)=-1e2;
covpower=0.5*log(var(Y1,1));
noisepower=0.5*log(var(Y1,1)/4);

% the spectral points must be initialized at random
nlml=inf;
optimizeparams = [lengthscales; covpower; noisepower];
m = round(n); % FIXED NUMBER OF BASIS TO 1-TENH OF THE NUMBER OF SAMPLES. (length(optimizeparams)-D-2)/D;  % number of basis
for k=1:100 % try several initializations and use the best one
    otherparams = randn(m*D,1);
    nlmlc=ssgprfixed([optimizeparams; otherparams], X1, Y1);
    if nlmlc<nlml
        w_save=otherparams;
        nlml=nlmlc;
    end
end
otherparams = w_save;

iteropt=-100;

optimizeparams = [optimizeparams; otherparams];
[optimizeparams, convergence] = minimize(optimizeparams, 'ssgprfixed', iteropt, X1, Y1);

model.loghyper = optimizeparams;
model.Xtrain = X1;
model.Ytrain = Y1;
model.meanY  = meanp;
