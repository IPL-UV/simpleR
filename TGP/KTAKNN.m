function KTATarget = KTAKNN(TestInput, Input, Target, Param)    
% Kernel Target Alignment with K Nearest Neighbors

n = size(TestInput,1);
Weight = LinearRegressor(Input, Target);
KTATarget = zeros(n,size(Target,2));
for Frame = 1:size(TestInput,1)
    TTestInput = TestInput(Frame,:);
    
    % find k nearest neighbors
    dist2 = sum(abs(repmat(TTestInput,size(Input,1),1)-Input),2);
    [ttt,index] = sort(dist2);
    knnindex = index(1:Param.knn);
    TInput = Input(knnindex,:);
    TTarget = Target(knnindex,:);
    
    IK = EvalKernel(TInput,TInput,'rbf',Param.kparam1);
    OK = EvalKernel(TTarget,TTarget,'rbf',Param.kparam2);
    TrIKOK = sum(sum(IK.*OK));
    TrOKOK = sum(sum(OK.*OK));
    alpha = EvalKernel(TInput,TTestInput,'rbf',Param.kparam1);
    ETestInput = [1 TTestInput];
    Y = (ETestInput*Weight)';
    Y = ComputeTarget(Y, TTarget, Param.kparam2, alpha, TrIKOK, TrOKOK);
    KTATarget(Frame,:) = Y';        
end

function [Y, fval] = ComputeTarget(Y, Target, kernelparam, alpha, TrIKOK, TrOKOK)

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
[Y, fval] = fminunc(@correlation,Y, options, Target, kernelparam, alpha, TrIKOK, TrOKOK, aaa);

%% Cost function of improved kernel dependency estimation and its derivatives.
function [FY, DFY] = correlation(Y,Target,kernelparam, alpha, TrIKOK, TrOKOK, aaa)

bbb = sum(Y.^2);
kvec = exp(-kernelparam*(aaa + bbb - 2*Target*Y));
FY = (-TrIKOK-2*alpha'*kvec-1)/sqrt(TrOKOK+2*kvec'*kvec+1);
DFY = -2*kernelparam.*((2*alpha.*kvec)'*Target - 2*alpha'*kvec*Y')'/sqrt(TrOKOK+2*kvec'*kvec+1)-...
0.5*(-TrIKOK-2*alpha'*kvec-1)*(TrOKOK+2*kvec'*kvec+1).^(-1.5)*(4*kernelparam.*((2*kvec.*kvec)'*Target - 2*kvec'*kvec*Y')');

% FY = -2*alpha'*kvec;
% DFY = -2*kernelparam.*((2*alpha.*kvec)'*Target - 2*alpha'*kvec*Y')';


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

