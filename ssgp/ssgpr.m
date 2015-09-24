%
%   SSGP - Sparse Spectrum Gaussian Processes 
%  
%   Performs efficient regression. Two usage modes are possible: training 
%   and prediction. If no test data are given, the function returns minus 
%   the log likelihood (nlml) and its partial derivatives (deriv_nlml) wrt
%   the hyperparameters being optimized (optimizeparams); this mode is 
%   used to select the hyperparameters. If test data are given, the 
%   predictive mean and variance are computed.
%
%   - Training
%
%    [nlml, deriv_nlml] = 
%               ssgpr(optimizeparams, x_tr, y_tr)
%   
%   where the inputs are:
%                   - Log of the D lengthscale hyperparameters,
%                   - Log of the square root of the signal power, and
%                   - Log of the square root of the noise power.
%                   - The m x D real values that define the spectral
%                   points.
%                   In both cases, they must be provided as a vector.
%   x_tr:           nxD matrix consisting of n D-dimensional training 
%                   inputs.
%   y_tr:           nx1 vector of training targets.
%
%   - Prediction
%
%   [mu, S2] = ssgpr(optimizeparams, x_tr, y_tr, x_tst)
%   
%   where x_tst is a ntstxD matrix consisting of ntst D-dimensional test
%   inputs and the rest of the inputs is as before. The ouputs are:
%
%   mu:             Predictive mean for the test set.
%   S2:             Predictive variance for the test set.
%
%
%   ***NOTE***
%   In most cases you will prefer to use the front end ssgpr_ui.m, that
%   adequately handles initialization and provides error measures. 
%
%   See also: ssgpr_ui
%   
%   This code corresponds to the algorithm developed in
%   "Sparse Spectrum Gaussian Process Regression",
%   check the paper and the online tutorial for further reference.
%
%   Copyright (C) 2007 Miguel Lazaro Gredilla (Nov/2008).

function [out1, out2] = ssgpr(optimizeparams, x_tr, y_tr, x_tst)

[n, D] = size(x_tr);                                                    % number of training samples, dimension

m=(length(optimizeparams)-D-2)/D;                                       % number of basis
ell  = exp(optimizeparams(1:D));                                        % characteristic lengthscale
sf2  = exp(2*optimizeparams(D+1));                                      % signal power
sn2  = exp(2*optimizeparams(D+2));                                      % noise power
w = reshape(optimizeparams(D+3:end), [m, D]);                           % unscaled model angular frequencies
w = w./repmat(ell',[m,1]);                                              % scaled model angular frequencies

phi = x_tr*w';
phi = [cos(phi) sin(phi)];                                              % design matrix

R = chol((sf2/m)*(phi'*phi) + sn2*eye(2*m));                            % calculate some often-used constants
PhiRi=phi/R;
RtiPhit = PhiRi';
Rtiphity=RtiPhit*y_tr;

if nargin < 3
    error('Not enough parameters!')
elseif nargin == 3                                                          
    % output NLML
    out1=0.5/sn2*(sum(y_tr.^2)-sf2/m*sum(Rtiphity.^2))+ ...
    +sum(log(diag(R)))+(n/2-m)*log(sn2)+n/2*log(2*pi);

if nargout == 2
    % also output derivatives
    %------------------------------- Begining of derivatives calculation
    out2=zeros(D+2+D*m,1);

    A=[y_tr/sn2-PhiRi*((sf2/sn2/m)*Rtiphity) ...
        sqrt(sf2/sn2/m)*PhiRi];                                             % O(nm)
    diagfact=-1/sn2+sum(A.^2,2);                                            % O(nm)
    Aphi=A'*phi;                                                            % O(nm^2)
    B=A*Aphi(:,1:m).*phi(:,m+1:end)-A*Aphi(:,m+1:end).*phi(:,1:m);          % O(nm^2)

    clear A
    clear PhiRi
    clear RtiPhit

    % derivatives wrt the lenghtscales
    for d = 1:D
        out2(d)=-0.5*2*sf2/m*(x_tr(:,d)'*B*w(:,d));                         % O(nm)
    end
    % derivative wrt signal power hyperparameter
    out2(D+1)=+0.5*2*(sf2/m)*(n*m/sn2-sum(sum(Aphi.^2)));                   % O(nm)
    % derivative wrt noise power hyperparameter
    out2(D+2)=-0.5*sum(diagfact)*2*sn2;                                     % O(n)

    % derivatives wrt the representative frequencies
    for d = 1:D
        out2(D+2+(d-1)*m+1:D+2+d*m)=+0.5*2*sf2/m*(x_tr(:,d)'*B)/ell(d);     % O(nm)
    end
    %------------------------------------ End of derivatives calculation
end
elseif nargin == 4
    
    clear phi
    
    % output predictive mean
    ns = size(x_tst, 1);                                                    % number of testing points
    out1 = zeros(ns, 1);                                                    % initialize outputs
    out2 = zeros(ns, 1);
    
    alfa=sf2/m*(R\Rtiphity);                                                % cosines/sines coefficients
    
    chunksize = 5000;                                                       % calculate the output in chunks of this size
    allxstar = x_tst;
    
    for beg_chunk = 1:chunksize:ns
        % do testing in chunks not to run out of memory
        end_chunk = min(beg_chunk + chunksize - 1, ns);
        x_tst = allxstar(beg_chunk:end_chunk, :);                           % testing points in this chunk
        
        phistar = x_tst*w';
        phistar = [cos(phistar) sin(phistar)];                              % test design matrix
        out1(beg_chunk:end_chunk) = phistar*alfa;                           % Predictive mean

        if nargout == 2                                                     
            % also output predictive variance
            out2(beg_chunk:end_chunk) = sn2*(1+sf2/m*sum((phistar/R).^2,2));% Predictive variance
        end
    end
end