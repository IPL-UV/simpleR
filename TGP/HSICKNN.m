function HSICTarget = HSICKNN(TestInput, Input, Target, Param)    
% Hilbert-Schmidt Independent Criterion with K Nearest Neighbors

n = size(TestInput,1);
Weight = LinearRegressor(Input, Target);
HSICTarget = zeros(n,size(Target,2));
for Frame = 1:size(TestInput,1)
    TTestInput = TestInput(Frame,:);
    
    % find k nearest neighbors
%     dist2 = EvalKernel(Input, OneTestInput, 'dist2');
    dist2 = sum(abs(repmat(TTestInput,size(Input,1),1)-Input),2);
    [ttt,index] = sort(dist2);
    knnindex = index(1:Param.knn);
    TInput = Input(knnindex,:);
    TTarget = Target(knnindex,:);
    
    IK = EvalKernel(TInput,TInput,'rbf',Param.kparam1);
    sumIK = sum(IK)';
    sumsumIK = sum(sumIK);
    IR = EvalKernel(TInput,TTestInput,'rbf',Param.kparam1);
    alpha = IR - (sum(IR)+1)/(n+1)-(sumIK+IR)/(n+1) + (sumsumIK+2*sum(IR)+1)/(n+1)/(n+1);
    ETestInput = [1 TTestInput];
    Y = (ETestInput*Weight)';
    Y = ComputeTarget(Y, TTarget, Param.kparam2, alpha);
    HSICTarget(Frame,:) = Y';        
end

function [Y, fval] = ComputeTarget(Y, Target, kernelparam, alpha)

%% Improved Kernel Dependency Estimation
options = optimset('GradObj','on');
options = optimset(options,'LargeScale','off');
options = optimset(options,'DerivativeCheck','off');
options = optimset(options,'Display','off');
options = optimset(options,'MaxIter',50);
options = optimset(options,'TolFun',1e-6);
options = optimset(options,'TolX',1e-6);
options = optimset(options,'LineSearchType','cubicpoly');
aaa = sum(Target.^2,2);
%% Optimization
[Y, fval] = fminunc(@correlation,Y, options, Target, kernelparam, alpha, aaa);

%% Cost function of improved kernel dependency estimation and its derivatives.
function [FY, DFY] = correlation(Y,Target,kernelparam, alpha, aaa)

bbb = sum(Y.^2);
kvec = exp(-kernelparam*(aaa + bbb - 2*Target*Y));
FY = -2*alpha'*kvec;
DFY = -2*kernelparam.*((2*alpha.*kvec)'*Target + FY*Y')';

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