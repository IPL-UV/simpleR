function [InvIK, InvOK] = TGPTrain(Input, Target, Param)
% Train Twin Gaussian Processes

IK = EvalKernel(Input, Input, 'rbf', Param.kparam1);
InvIK = inv(IK + Param.lambda*eye(size(IK)));

OK = EvalKernel(Target, Target, 'rbf', Param.kparam2);
InvOK = inv(OK + Param.lambda*eye(size(OK)));
