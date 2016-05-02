%
% "simpleR: A simple educational Matlab toolbox for statistical regression"
%
% [simpleR 3.0] 
%        Version: 3.0
%        Date   : 15-May-2016
%
% This demo shows the training and testing of several state-of-the-art 
% statistical models for regression. Please read the README file for more 
% details. 
% 
% If you find this toolbox useful, cite it!
%
% @misc{simpler,
%  author = {Camps-Valls, G. and G\'omez-Chova, L. and Mu{\~n}oz-Mar\'i, J. and L\'azaro-Gredilla, M. and Verrelst, J.},
%  title = {{simpleR}: A simple educational Matlab toolbox for statistical regression},
%  month = {3},
%  year = {2016},
%  note = {V3.0},
%  url = {http://www.uv.es/gcamps/},
% }
%
% ------------------------------
% AVAILABLE METHODS
% ------------------------------
%
% LINEAR MODELS
%    * Regularized Least squares Linear regression (RLR) 
%    * Least Absolute Shrinkage and Selection Operator (LASSO).
%    * Elastic Net (ELASTICNET).
%
% SPLINES and POLYNOMIALS
%    * Adaptive Regression Splines (ARES)
%    * Locally Weighted Polynomials (LWP)
%
% NEIGHBORS
%    * k-nearest neighbors regression (KNNR)
%    * Weighted k-nearest neighbors regression (WKNNR)
%
% TREE MODELS
%    * Decision trees (TREE)
%    * Bagging trees (BAGTREE)
%    * Boosting trees (BOOST)
%    * Random forests (RF1)
%    * Boosting random trees (RF2)
%
% NEURAL NETWORS
%    * Neural networks (NN)
%    * Extreme Learning Machines (ELM)
%
% KERNEL METHODS
%    * Support Vector Regression (SVR)
%    * Kernel Ridge Regression (KRR), aka Least Squares SVM
%    * Relevance Vector Machine (RVM)
%    * Kernel signal to noise regression (KSNR)
%    * Structured KRR (SKRR)
%    * Random Kitchen Sinks Regression (RKS)
%
% GAUSSIAN PROCESSES
%    * Gaussian Process Regression (GPR)
%    * Variational Heteroscedastic Gaussian Process Regression (VHGPR)
%    * Warped Gaussian Processes (WGPR)
%    * Sparse Spectrum Gaussian Process Regression (SSGPR)
%    * Twin Gaussian Processes (TGP)
%
% Copyright (c) 2016 by Gustau Camps-Valls
% gustavo.camps@uv.es
% http://isp.uv.es/
% http://www.uv.es/gcamps
%

%% Setup
clear;clc;close all;

fontname = 'Bookman';
fontsize = 20;
fontunits = 'points';
set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
    'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
    'DefaultLineLineWidth',3,'DefaultLineMarkerSize',10,'DefaultLineColor',[0 0 0]);

% Paths
addpath('./AUX')        % Auxiliary functions for visualization, results analysis, plots, etc.
addpath('./DATA')       % Put your data here
addpath('./FIGURES')    % All figures are saved here
addpath('./RESULTS')    % All files with results are saved here

% Paths for the methods
addpath('./standard')   % Train-Test functions for all methods
addpath('./SVM')        % libsvm code and kernel matrix
addpath('./MRVM')       % Relevance vector machine (RVM)
addpath('./VHGPR')      % Variational Heteroscedastic Gaussian Process regression [Lázaro-Gredilla, 2011]
addpath('./ARES')       % ARESLab -- Adaptive Regression Splines toolbox for Matlab/Octave, ver. 1.5.1, by Gints Jekabsons
addpath('./LWP')        % Locally-Weighted Polynomials, Version 1.3, by Gints Jekabsons
addpath('./WGP')        % Warped GPs
addpath('./SSGP')       % Sparse Spectrum Gaussian Process (SSGP)  [Lázaro-Gredilla, 2008]
addpath('./TGP')        % Twin Gaussian Process (TGP) [Liefeng Bo and Cristian Sminchisescu]  http://www.maths.lth.se/matematiklth/personal/sminchis/code/TGP.html

%% Load data: 
%   X: Input data of size n x d
%   Y: Output/target/observation of size n x do
%   n: number of samples/examples/patterns (in rows)
%   d: input data dimensionality/features (in columns)
%   do: output data dimensionality (variables, observations). 
load SeaBAM.mat

%% Split training-testing data
rate = 0.05; %[0.05 0.1 0.2 0.3 0.4 0.5 0.6]
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
[ntest do] = size(Ytest);
VARIABLES = {'b1' 'b2' 'b3' 'b4' 'b5'};

%% Input data normalization, either between 0-1 or standardization (zero mean, unit variance)
% [Xtrain a b] = scale(Xtrain);
% Xtest        = scale(Xtest,a,b);
% [Xtrain a b] = scalestd(Xtrain);
% Xtest        = scalestd(Xtest,a,b);

%% Remove the mean of Y for training only
my      = mean(Ytrain);
Ytrain  = Ytrain - repmat(my,ntrain,1);

%% SELECT METHODS FOR COMPARISON
% METHODS = {'KRR'}
% METHODS = {'RLR' 'LASSO' 'ENET'} % LINEAR
% METHODS = {'LWP' 'ARES'} % SPLINES
% METHODS = {'KNNR' 'WKNNR'} % NEIGHBORS
% METHODS = {'TREE' 'BAGTREE' 'BOOST' 'RF1' 'RF2'}   % TREES
% METHODS = {'NN' 'RBFNET' 'ELM'}  % NEURAL NETS
% METHODS = {'SVR' 'KRR' 'RVM' 'KSNR' 'SKRRrbf' 'SKRRlin' 'RKS'}   % KERNELS
% METHODS = {'KRR' 'SKRRrbf' 'SKRRlin'}   % KERNELS
% METHODS = {'GPR' 'VHGPR' 'WGPR' 'SSGPR' 'TGP'}  % GPs

%%%% ALL!
METHODS = {'RLR' 'LASSO' 'ENET' 'LWP' 'ARES' 'KNNR' 'WKNNR', ...
    'TREE' 'BAGTREE' 'BOOST' 'RF1' 'RF2', ...
    'NN' 'ELM', 'SVR' 'KRR' 'RVM' 'KSNR' 'SKRRrbf' 'SKRRlin' 'RKS', ...
    'GPR' 'VHGPR' 'WGPR' 'SSGPR' 'TGP'}

%%%% REPRESENTATIVE PER FAMILY
%  METHODS = {'RLR' 'LASSO' ,...
%             'LWP' 'ARES', ...
%             'KNNR', ...
%             'TREE' 'RF1', ...
%             'NN', ...
%             'SVR' 'KRR', ...
%             'GPR' 'VHGPR' 'WGPR' 'TGP'}

%%%% MULTIOUTPUT ONLY
% METHODS = {'RLR' 'NN' 'KRR' 'KSNR' 'SKRRrbf' 'SKRRlin' 'RKS' 'TGP'}

%% TRAIN ALL MODELS
numModels = numel(METHODS);

for m=1:numModels
    fprintf(['Training ' METHODS{m} '... \n'])
    t=cputime;
    eval(['model = train' METHODS{m} '(Xtrain,Ytrain);']); % Train the model
    eval(['Yp = test' METHODS{m} '(model,Xtest);']);       % Test the model
    Yp = Yp + repmat(my,ntest,1);
    RESULTS(m) = assessregres(Ytest,Yp);
    CPUTIMES(m) = cputime-t;
    MODELS{m} = model;
    YPREDS(:,m) = Yp;
end

% % Fast training (divide and conquer strategy, nice for kernel machines)
% for m=1:numModels
%     fprintf(['Fast Training ' METHODS{m} '... \n'])
%     t=cputime;
%     eval(['model2 = fastTrain(''' METHODS{m} ''',Xtrain,Ytrain);']); % fast Train the model
%     eval(['Yp = fastTest(''' METHODS{m} ''',model2,Xtest);']);       % fast Test the model
%     Ypred(:,m)     = Yp + my;
%     results2(m)     = assessment(Ypred(:,m),Ytest,'regress');
%     CPUTIMES2(m) = cputime-t;
% end

%% Save results
save('RESULTS/results.mat','RESULTS','CPUTIMES','MODELS','METHODS','VARIABLES','X','Y','Xtrain','Ytrain','Xtest','Ytest','YPREDS')
