% function K = kernelmatrix(ker,X1,X2,sigma,b,d)
%
% Inputs:
%	ker:    'lin','poly','rbf','sam'
%	X1: data matrix with training samples in rows and features in columns
%	X2:	data matrix with test samples in rows and features in columns
%	sigma: width of the RBF kernel
% 	b:     bias in the linear and polinomial kernel
%	d:     degree in the polynomial kernel
%
% Output:
%	K: kernel matrix

% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Gustavo Camps-Valls, 2006
% Jordi (jordi@uv.es),
%   2010-04: RBF can be computed now also on vectors with only one feature (ie: scalars)
%   2007-11: if/then -> switch, and fixed RBF kernel
%   2014-11: added (missing) b,d input parameters. Faster RBF kernel :-).
%   2016-01: added exponential, Laplace, tanh, ANOVA, rq (rational quadratic),
%            mq (multiquadratic) kernels

function K = kernelmatrix(ker,X1,X2,varargin) % ,sigma,b,d)

% Check input arguments
if ~exist('X2','var')
    X2 = X1;
end

switch ker
    case 'poly'
        if size(varargin,2) < 2,
            error('RBF kernel needs ''bias'' and ''degree'' parameters')
        end
        bias = varargin{1};
        degree = varargin{2};

    case {'tanh','rq','mq'}
        if size(varargin,2) < 1,
            error('tanh kernel needs ''bias'' parameter')
        end
        bias = varargin{1};

    case {'pow','power'}
        if size(varargin,2) < 1,
            error('power kernel needs ''degree'' parameter')
        end
        degree = varargin{1};

    case  {'rbf','sam','exp','exponential','laplace','anova'}
        if size(varargin,2) < 1,
            error('RBF kernel needs ''sigma'' parameter')
        end
        sigma = varargin{1};
end

% Compute requested kernel
switch ker
    case {'lin','sam'}
        K = X1' * X2;
        if strcmp(ker,'sam')
            K = exp(-acos(K).^2/(2*sigma^2));
        end

    case {'poly','tanh'}
        K = (X1' * X2 + bias);
        if strcmp(ker, 'poly')
            K = K .^ degree;
        else
            K = tanh(K);
        end

    case 'rq'
        K = norm2mat(X1,X2);
        K = 1 - K./(K + bias);

    case 'mq'
        K = norm2mat(X1,X2);
        K = sqrt( K + bias^2 );

    case {'rbf','exp','exponential','laplace'}
        K = norm2mat(X1,X2);
        switch ker
            case 'rbf'
                K = exp(-K/(2*sigma^2));
            case {'exp','exponential'}
                K = exp(-sqrt(K)/(2*sigma^2));
            case 'laplacian'
                K = exp(-sqrt(K)/sigma);
        end

    case 'power'
        K = norm2mat(X1,X2);
        K = K .^ (degree/2);

    case 'anova'
        % For each dimension compute a RBF kernel and sum them all
        K = 0;
        d = size(X1,1);
        for k = 1:d
            K = K + d * kernelmatrix('rbf', X1(k,:), X2(k,:), sigma);
        end

    case {'chi-square','chi'}
        d = size(X1,1);
        n1 = size(X1,2);
        n2 = size(X2,2);
        if (size(X2,1) ~= d)
            error('X1 and X2 have different dimensions')
        end
        K = 0;
        for k = 1:d
            %N = 2 * X1(k,:)' * X2(k,:);
            %D = X1(k,:)' * ones(1,n2) + ones(n1,1) * X2(k,:);
            %K = K + N./D;
            K = K + (2 * X1(k,:)' * X2(k,:)) ./ (X1(k,:)' * ones(1,n2) + ones(n1,1) * X2(k,:));
        end

        % Version lenta pero segura (para comprobar)
        %K = zeros(n1,n2);
        %for i = 1:n1
        %    for j = 1:n2
        %        for k = 1:d
        %            K(i,j) = K(i,j) + (2 * X1(k,i) .* X2(k,j)) / ( X1(k,i) + X2(k,j) );
        %        end
        %    end
        %end

    otherwise
        error(['Unsupported kernel ' ker])
end

function D = norm2mat(X1,X2)
    D = - 2 * (X1' * X2);
    D = bsxfun(@plus, D, sum(X1.^2,1)');
    D = bsxfun(@plus, D, sum(X2.^2,1));
end

end
