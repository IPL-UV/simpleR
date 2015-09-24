function K = EvalKernel(samples1, samples2, kernel, kernelparam)
% Fastly evaluate kernel function

if (size(samples1,2)~=size(samples2,2))
    error('sample1 and sample2 differ in dimensionality!!');
end
[L1, dim] = size(samples1);
[L2, dim] = size(samples2);

switch kernel
case 'no'
    K = samples1;
case 'dist2'
    a = sum(samples1.*samples1,2);
    b = sum(samples2.*samples2,2);
    dist2 = a*ones(1,L2);
    dist2 = dist2 + ones(L1,1)*b';
    K = dist2 - 2*samples1*samples2';
case 'linear'
    K = samples1*samples2';
case 'poly'
    K = (1 + samples1*samples2'/dim).^kernelparam;
case 'rbf'
    % If sigle parammeter, expand it.
    if length(kernelparam) < dim
        a = sum(samples1.*samples1,2);
        b = sum(samples2.*samples2,2);
        dist2 = a*ones(1,L2);
        dist2 = dist2 + ones(L1,1)*b';
        dist2 = dist2 - 2*samples1*samples2';
        K = exp(-kernelparam*dist2);
    else
        kernelparam = kernelparam(:);
        a = sum(samples1.*samples1.*repmat(kernelparam',L1,1),2);
        b = sum(samples2.*samples2.*repmat(kernelparam',L2,1),2);
        dist2 = a*ones(1,L2);
        dist2 = dist2 + ones(L1,1)*b';
        dist2 = dist2 - 2*(samples1.*repmat(kernelparam',L1,1))*samples2';
        K = exp(-dist2);
    end
case 'convrbf'
    a = sum(samples1.*samples1,2);
    b = sum(samples2.*samples2,2);
    fftsamples1 = fft(fliplr(samples1),[],2);
    fftsamples2 = fft(samples2,[],2);
    dist2 = zeros(L1,L2);
    for i = 1:L1
        dist2(i,:) = (a(i)*ones(L2,1) + b - 2*max(ifft(repmat(fftsamples1(i,:),L2,1).*fftsamples2,[],2),[],2))';
    end
    K = exp(-dist2);
case 'kfrbf'
    K = zeros(L1,L2);
    for i = 1:L1
        tsamples = ones(L2,1)*samples1(i,:);
        K(i,:) = (sum((tsamples - samples2).^2./( tsamples + samples2 + eps),2))';
    end
    K = exp(-kernelparam*K);
case 'laplace'
    K = zeros(L1,L2);
    for i = 1:L1
        K(i,:) = (sum(abs(repmat(samples1(i,:),L2,1) - samples2),2))';
    end
    K = exp(-kernelparam*K);
case 'wave'
    K = zeros(L1,L2);
    for i = 1:L1
        temp = repmat(samples1(i,:),L2,1);
        K(i,:) = exp(-kernelparam*sum((temp-samples2).^2,2)).*prod(cos(1.75*sqrt(2*kernelparam)*(temp-samples2)),2);
    end
case 'cauchy'
    a = sum(samples1.*samples1,2);
    b = sum(samples2.*samples2,2);
    dist2 = a*ones(1,L2) + ones(L1,1)*b' - 2*samples1*samples2';
    K = kernelparam./(kernelparam + dist2);
case 'nn'
    K = tanh(samples1*samples2');
otherwise
    error('Unknown kernel function');
end