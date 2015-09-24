function TGPTarget = TGPKNN(TestInput, Input, Target, Param,InitEstimation)
% Make the prediction using Twin Gaussian Process with k nearest neighbors

T = size(TestInput,1);
TGPTarget = zeros(T,size(Target,2));
Weight = LinearRegressor(Input, Target);
for frame = 1:T
    % Initialize
    OneTestInput = TestInput(frame,:);
    EOneTestInput = [1 OneTestInput];
    InitTarget = (EOneTestInput*Weight)';

    % find k nearest neighbors
    dist2 = EvalKernel(Input, OneTestInput, 'dist2');
%     dist2 = sum(abs(repmat(OneTestInput,size(Input,1),1)-Input),2);
    [ttt,index] = sort(dist2);
    knnindex = index(1:Param.knn);
    TInput = Input(knnindex,:);
    TTarget = Target(knnindex,:);

    % compute the inverse kernel matrix of input and output
    IK = EvalKernel(TInput, TInput, 'rbf', Param.kparam1);
    InvIK = inv(IK + Param.lambda*eye(size(IK)));
    OK = EvalKernel(TTarget, TTarget, 'rbf', Param.kparam2);
    InvOK = inv(OK + Param.lambda*eye(size(OK)));

    % optimize
    IR = EvalKernel(TInput,OneTestInput,'rbf',Param.kparam1);
    alpha = InvIK*IR;
    beta = 1 + Param.lambda - IR'*InvIK*IR;
    if nargin < 5
        Y = ComputeOutput(InitTarget, TTarget, Param.kparam2, Param.lambda, alpha, beta, InvOK);
    else
        Y = ComputeOutput(InitEstimation(frame,:)', TTarget, Param.kparam2, Param.lambda, alpha, beta, InvOK);
    end
    TGPTarget(frame,:) = Y';
end

function [Y, fval] = ComputeOutput(Y, Target, kernelparam, lambda, alpha, beta, InvOK)

%% Compute the output

options = optimset('GradObj','on');
options = optimset(options,'LargeScale','off');
options = optimset(options,'DerivativeCheck','off');
options = optimset(options,'Display','off');
options = optimset(options,'MaxIter',50);
options = optimset(options,'TolFun',1e-6);
options = optimset(options,'TolX',1e-6);
options = optimset(options,'LineSearchType','cubicpoly');
aaa = sum(Target.^2,2);

% Optimization
[Y, fval] = fminunc(@correlation,Y, options, Target, kernelparam, lambda, alpha, beta, InvOK, aaa);

% Cost function of twin Gaussian processes and its derivatives.
function [FY, DFY] = correlation(Y,Target,kernelparam, lambda, alpha, beta, InvOK, aaa)

bbb = sum(Y.^2);
kvec = exp(-kernelparam*(aaa + bbb - 2*Target*Y));
InvOKkvec = InvOK*kvec;
ybeta = 1+lambda - kvec'*InvOKkvec;
FY = -2*alpha'*kvec - beta*log(ybeta);
TTT = 2*(beta/ybeta*InvOKkvec - alpha);
DFY = 2*kernelparam.*((TTT.*kvec)'*Target - (TTT'*kvec)*Y')';

function Weight = LinearRegressor(Input, Target, Lambda)
%% Linear regression

[N, d] = size(Input);
BiasVec = ones(N,1);
Hessian = [BiasVec'*BiasVec BiasVec'*Input; Input'*BiasVec Input'*Input];
InputTarget = [sum(Target); Input'*Target];

if nargin < 3
    Lambda = 1e-5*mean(diag(Hessian));
else
    Lambda = Lambda*min(diag(Hessian));
end

Weight = (Hessian + Lambda*eye(d+1))\InputTarget;

