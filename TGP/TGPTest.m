function TGPTarget = TGPTest(TestInput, Input, Target, Param, InvIK, InvOK)
% Make the prediction using Twin Gaussian Processess

T = size(TestInput,1);
TGPTarget = zeros(T,size(Target,2));
Weight = LinearRegressor(Input, Target);
for frame = 1:T
    % Initialize
    OneTestInput = TestInput(frame,:);
    EOneTestInput = [1 OneTestInput];
    InitTarget = (EOneTestInput*Weight)';

    % optimize
    IR = EvalKernel(Input,OneTestInput,'rbf',Param.kparam1);
    alpha = InvIK*IR;
    beta = EvalKernel(OneTestInput,OneTestInput,'rbf',Param.kparam1) + Param.lambda - IR'*InvIK*IR;
    Y = ComputeOutput(InitTarget, Target, Param.kparam2, Param.lambda, alpha, beta, InvOK);
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

