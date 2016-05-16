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
addpath('./AUXF')       % Auxiliary functions for visualization, results analysis, plots, etc.
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
load SPARC.mat

% Y=Y(:,1) % R: [0.9093 0.8298 0.6353]

%% Split training-testing data
rate = 0.2; %[0.05 0.1 0.2 0.3 0.4 0.5 0.6]
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
VARIABLES = {'b1','b2','b3','b4','b5','b6','b7','b8','b9','b10','b11','b12','b13','b14','b15','b16','b17','b18','b19','b20','b21','b22','b23','b24','b25','b26','b27','b28','b29','b30','b31','b32','b33','b34','b35','b36','b37','b38','b39','b40','b41','b42','b43','b44','b45','b46','b47','b48','b49','b50','b51','b52','b53','b54','b55','b56','b57','b58','b59','b60','b61','b62'};

%% Input data normalization, either between 0-1 or standardization (zero mean, unit variance)
% [Xtrain a b] = scale(Xtrain);
% Xtest        = scale(Xtest,a,b);
% [Xtrain a b] = scalestd(Xtrain);
% Xtest        = scalestd(Xtest,a,b);

%% Remove the mean of Y for training only
my      = mean(Ytrain);
Ytrain  = Ytrain - repmat(my,ntrain,1);

%% SELECT METHODS FOR COMPARISON: MULTIOUTPUT ONLY
% METHODS = {'RLR' 'RF1' 'ELM' 'NN' 'KRR'}
METHODS = {'NN'} % 'KRR'}

%% TRAIN ALL MODELS
numModels = numel(METHODS);

for m=1:numModels
    fprintf(['Training ' METHODS{m} '... \n'])
    t=cputime;
    eval(['model = train' METHODS{m} '(Xtrain,Ytrain);']); % Train the model
    eval(['Yp = test' METHODS{m} '(model,Xtest);']);       % Test the model
    Yp = Yp + repmat(my,ntest,1);
    RESULTS{m} = assessregres(Ytest,Yp);
    CPUTIMES{m} = cputime-t;
    MODELS{m} = model;
    YPREDS{m} = Yp;
end
RESULTS{1}
%% Save results
save('RESULTS/results.mat','RESULTS','CPUTIMES','MODELS','METHODS','VARIABLES','X','Y','Xtrain','Ytrain','Xtest','Ytest','YPREDS')
