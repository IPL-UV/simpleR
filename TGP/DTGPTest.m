function DTGPTarget = DTGPTest(TestInput,Input,Target,AutoInput,AutoTarget,Param,...
    InvIK,InvOK,InvAIK,InvAOK,InvAIKAOK,InitAutoInput,T)
% Make the prediction using dynamic Twin Gaussian Processes

N = size(TestInput,1);
DTGPTarget = [];
index = partition(N,T);
for i = 1:length(index);
    TestSeq = TestInput(index{i},:);
    TTarget = DTGPSeq(TestSeq,Input,Target,AutoInput,AutoTarget,Param,...
        InvIK,InvOK,InvAIK,InvAOK,InvAIKAOK,InitAutoInput);
    DTGPTarget = [DTGPTarget; TTarget];
    InitAutoInput = DTGPTarget(end,:);
end

function TargetSeq = DTGPSeq(TestInput,Input,Target,AutoInput,AutoTarget,Param,...
    InvIK,InvOK,InvAIK,InvAOK,InvAIKAOK,InitAutoInput)
%% Twin Gaussian Process with Dynamic

T = size(TestInput,1);
D = size(Target,2);
TGPTarget = zeros(D,T);
Weight = LinearRegressor(Input, Target);
for frame = 1:T
    % Initialize
    OneTestInput = TestInput(frame,:);
    EOneTestInput = [1 OneTestInput];
    InitTarget = (EOneTestInput*Weight)';

    % optimize
    IR = EvalKernel(Input,OneTestInput,'rbf',Param.kparam1);
    Alpha(:,frame) = InvIK*IR;
    Beta(:,frame) = 1 + Param.lambda - IR'*InvIK*IR;
    Y = ComputeOutput(InitTarget, Target, Param.kparam2, Param.lambda, Alpha(:,frame), Beta(:,frame), InvOK);
    TGPTarget(:,frame) = Y;
end

IR = EvalKernel(AutoInput,InitAutoInput,'rbf',Param.kparam3);
Initalpha = InvAIK*IR;
Initbeta = 1 + Param.lambda - IR'*InvAIK*IR;
TargetSeq = ComputeDynamicalOutput(TGPTarget(:)+1,Target,AutoInput,AutoTarget,Param.kparam2,Param.kparam3,...
    Param.lambda,Param.tradeoff,Alpha,Beta,InvOK,InvAIK,InvAOK,InvAIKAOK,Initalpha,Initbeta);
TargetSeq = reshape(TargetSeq,D,T)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y, fval] = ComputeDynamicalOutput(Y,Target,AutoInput,AutoTarget,kernelparam,autokernelparam,...
    lambda, tradeoff, Alpha, Beta, InvOK, InvAIK, InvAOK, InvAIKAOK,Initalpha,Initbeta)

%% Compute the output

options = optimset('GradObj','on');
options = optimset(options,'LargeScale','off');
options = optimset(options,'DerivativeCheck','off');
options = optimset(options,'Display','off');
options = optimset(options,'MaxIter',500);
options = optimset(options,'TolFun',1e-8);
options = optimset(options,'TolX',1e-8);
options = optimset(options,'LineSearchType','cubicpoly');
aaa = sum(Target.^2,2);
iaaa = sum(AutoInput.^2,2);
oaaa = sum(AutoTarget.^2,2);
% Optimization
[Y, fval] = fminunc(@sumcorrelation,Y,options,Target,AutoInput,AutoTarget,kernelparam,autokernelparam,...
    lambda,tradeoff,Alpha,Beta,InvOK,InvAIK,InvAOK,InvAIKAOK,Initalpha,Initbeta,aaa,iaaa,oaaa);

function [FY, DFY] = sumcorrelation(Y,Target,AutoInput,AutoTarget,kernelparam,autokernelparam,...
    lambda,tradeoff,Alpha,Beta,InvOK,InvAIK,InvAOK,InvAIKAOK,Initalpha,Initbeta,aaa,iaaa,oaaa)

D = size(Target,2);
T = length(Y)/D;

Y = reshape(Y,D,T);
FY = 0;
DFY = [];
% input-output regression
for frame = 1:T
    [TFY, TDFY] = correlation(Y(:,frame),Target,kernelparam,lambda,...
        Alpha(:,frame),Beta(:,frame),InvOK,aaa);
    FY = FY + TFY;
    DFY = [DFY; TDFY];
end

% autoregression
[TFY, TDFY] = correlation(Y(:,1),AutoTarget,autokernelparam,lambda,...
    Initalpha,Initbeta,InvAOK,oaaa);
FY = FY + tradeoff*TFY;
DFY(1:D) = DFY(1:D) + tradeoff*TDFY;

for frame = 2:T
    [TFY, TDFY] = autocorrelation(Y(:,frame-1),Y(:,frame),AutoInput,AutoTarget, ...
        autokernelparam,lambda,InvAIK,InvAOK,InvAIKAOK,iaaa,oaaa);
    FY = FY + tradeoff*TFY;
    DFY((frame-2)*D+1:frame*D) = DFY((frame-2)*D+1:frame*D) + tradeoff*TDFY;
end

function [FY, DFY] = autocorrelation(IY,OY,Input,Target,kernelparam,lambda,InvIK,InvOK,InvAIKAOK,iaaa,oaaa)

ibbb = sum(IY.^2);
ikvec = exp(-kernelparam*(iaaa + ibbb - 2*Input*IY));
obbb = sum(OY.^2);
okvec = exp(-kernelparam*(oaaa + obbb - 2*Target*OY));
InvIKikvec = InvIK*ikvec;
InvOKokvec = InvOK*okvec;
ibeta = 1 + lambda - ikvec'*InvIKikvec;
obeta = 1 + lambda - okvec'*InvOKokvec;

% compute the objectve function
FY = -2*okvec'*InvIKikvec + ikvec'*InvAIKAOK*ikvec - ibeta*log(obeta) + ibeta*log(ibeta);

% compute the derivative
ITTT = 2*(-InvIK*okvec + InvAIKAOK*ikvec + log(obeta)*InvIKikvec - log(ibeta)*InvIKikvec - InvIKikvec);
DFY = 2*kernelparam.*((ITTT.*ikvec)'*Input - (ITTT'*ikvec)*IY')';
OTTT = 2*(-InvIKikvec + ibeta/obeta*InvOKokvec);
DFY = [DFY; 2*kernelparam.*((OTTT.*okvec)'*Target - (OTTT'*okvec)*OY')'];


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
TTT = 2*(-alpha + beta/ybeta*InvOKkvec);
DFY = 2*kernelparam.*((TTT.*kvec)'*Target - (TTT'*kvec)*Y')';

function Weight = LinearRegressor(Input, Target, Lambda)

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
