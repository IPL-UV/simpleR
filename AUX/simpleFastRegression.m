dir

%% Setup
clear;clc;close all;

fontname = 'Bookman';
fontsize = 14;
fontunits = 'points';
set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
    'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
    'DefaultLineLineWidth',3,'DefaultLineMarkerSize',10,'DefaultLineColor',[0 0 0]);

% Paths
addpath('./vhgpr')                  % VHGPR [Lázaro-Gredilla, 2011]
addpath('./standard')               % Standard statistical regression: LR, LASSO, TREES, SVR, KRR, GPR

clear;clc;close all;

%% Training with 1 million data points
N     = 1e5;
X     = [sin(1:N)', cos(1:N)', tanh(1:N)'] + 0.1*randn(N,3);
Y     = sin(1:N)';
VARIABLES = {'SIN', 'COS', 'TANH'};

%% Split training-testing data
rate = 0.1; %[0.05 0.1 0.2 0.3 0.4 0.5 0.6]
% Fix seed random generator (important: disable when doing the 100 realizations loop!)
rand('seed',12345);
randn('seed',12345);
[n d] = size(X);                 % samples x bands
r = randperm(n);                 % random index
ntrain = round(rate*n);          % #training samples
Xtrain = X(r(1:ntrain),:);       % training set
Ytrain = Y(r(1:ntrain),:);       % observed training variable
Xtest  = X(r(ntrain+1:end),:);   % test set
Ytest  = Y(r(ntrain+1:end),:);   % observed test variable

%% Remove the mean of Y for training only
my      = mean(Ytrain);
Ytrain  = Ytrain - my;


% METHODS = {'RLR' 'LASSO' 'ELASTICNET' 'TREE' 'BAGTREE' 'BOOST' 'NN' 'ELM' 'SVR' 'KRR' 'RVM' 'GPR' 'VHGPR'}

%% FAST TRAINING ...
% method = 'RLR'
% method = 'KRR'
% method = 'GPR'
% method = 'VHGPR'
model = fastTrain(method,Xtrain,Ytrain);
Yp    = fastTest(method,model,Xtest);
corrcoef(Yp,Ytest)


