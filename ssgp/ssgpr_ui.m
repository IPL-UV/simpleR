% 
%   Front-end for SSGP (ssgpr_ui.m)
%   
%   Automatically selects a set of hyperparameters using joint minimization
%   of the negative log-evidence on the training set. 
%   Estimates predictive means and variances on the test set. 
%   Error measures are computed.
%
%   Usage:
%
%   [NMSE, mu, S2, NMLP, loghyper, convergence] = 
%             ssgpr_ui(x_tr, y_tr, x_tst, y_tst, m, iteropt, loghyper)
%
%   The first five arguments are mandatory:
% 
%   x_tr:  nxD matrix consisting of n D-dimensional training inputs.
%   y_tr:  nx1 vector of training targets.
%   x_tst: ntstxD matrix consisting of ntst D-dimensional test inputs.
%   y_tst: ntstx1 vector of test targets (used to compute error measures, 
%          can be zeros).
%   m:     number of harmonic basis used for the approximation.
%
%   The other two are for advanced usage:
%
%   iteropt: It will be passed to minimize.m as the number of iterations.
%   If this number is negative, it refers to number of evaluations,
%   otherwise to number of line searches. (See minimize.m for help).
%
%   loghyper: column vector of (log) hyperparameters.
%             The first D+2 the hyperparameters are interpreted as:
%             - Log of the D lengthscale hyperparameters.
%             - Log of the square root of the signal power.
%             - Log of the square root of the noise power.
%             The rest of the hyperparameters (which are not mandatory) are
%             the m x D real values that define the spectral points, in
%             column format.
%
%   Returns:
%
%   NMSE:           Normalized Mean Square Error for the test set.
%   mu:             Predictive mean for the test set.
%   S2:             Predictive variance for the test set.
%   NMLP:           Negative Mean Log Probability
%   loghyper:       Values for all the selected hyperparameters.
%   convergence:    Evolution of the training evidence as the optimization
%                   progresses.
%
%   See also: ssgpr
%
%   This code corresponds to the algorithm developed in
%   "Sparse Spectrum Gaussian Process Regression",
%   check the paper and the online tutorial for further reference.
%
%   Copyright (C) 2007 Miguel Lazaro Gredilla (Nov/2008).

function [NMSE, mu, S2, NMLP, loghyper, convergence] ...
                = ssgpr_ui(x_tr, y_tr, x_tst, y_tst, m, iteropt, loghyper)

% Check the arguments
if nargin > 7 || nargin < 5
    error('Incorrect number of arguments')
end

% Regression assumes zero-mean functions, substract mean 
meanp=mean(y_tr);                                                           
y_tr=y_tr-meanp;

[n,D]=size(x_tr);


% -- Initial hyperparameters setting

if nargin < 7 || (nargin == 7 && length(loghyper) == D+2)
    % some hyperparameters haven't been provided
    if nargin == 7
        % but the lengthscales and powers have
        lengthscales = loghyper(1:D);
        covpower = loghyper(D+1);
        noisepower = loghyper(D+2);
    else
        % not even the lengthscales and powers
        % use a reasonable initial guess
        lengthscales=log((max(x_tr)-min(x_tr))'/2);
        lengthscales(lengthscales<-1e2)=-1e2;
        covpower=0.5*log(var(y_tr,1));
        noisepower=0.5*log(var(y_tr,1)/4);
    end

    % the spectral points must be initialized at random
    nlml=inf;
    optimizeparams = [lengthscales; covpower; noisepower];
    for k=1:100 % try several initializations and use the best one
        otherparams = randn(m*D,1);
        nlmlc=ssgpr([optimizeparams; otherparams], x_tr, y_tr);
        if nlmlc<nlml
            w_save=otherparams;
            nlml=nlmlc;
        end
    end
    otherparams = w_save;
elseif nargin == 7
    % all parameters have been provided, check they are as many as expected
    if length(loghyper) ~= D+2+D*m
        error('Incorrect number of hyperparameters: expected %d, but received %d.',  ...
            D+2+D*m, length(loghyper))
    end
    optimizeparams = loghyper(1:D+2);
    otherparams = loghyper(D+3:D+2+D*m);
end
% set number of iterations at each stage if it hasn't been specified
if nargin == 5 || isempty(iteropt)
    iteropt=-1000;
end

optimizeparams=[optimizeparams; otherparams];
[optimizeparams, convergence] = minimize(optimizeparams, 'ssgpr', iteropt, x_tr, y_tr);
loghyper=optimizeparams;

% -- Prediction with selected hyperparameters

if nargout < 3
    mu = ssgpr(optimizeparams, x_tr, y_tr, x_tst);
else
    [mu, S2] = ssgpr(optimizeparams, x_tr, y_tr, x_tst);
end
% add back the mean
mu=mu+meanp;

% -- Compute error measures for test set

NMSE=mean((mu-y_tst).^2)/mean((meanp-y_tst).^2);

if nargout > 3
    NMLP=-0.5*mean(-(mu-y_tst).^2./S2-log(2*pi)-log(S2));
end