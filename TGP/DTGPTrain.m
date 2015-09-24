function [InvIK, InvOK, InvAIK, InvAOK, InvAIKAOK] = DTGPTrain(Input, Target, AutoInput, AutoTarget, Param)
% Train dynamic Twin Gaussian Processes

IK = EvalKernel(Input, Input, 'rbf', Param.kparam1);
InvIK = inv(IK + Param.lambda*eye(size(IK)));

OK = EvalKernel(Target, Target, 'rbf', Param.kparam2);
InvOK = inv(OK + Param.lambda*eye(size(OK)));

AIK = EvalKernel(AutoInput, AutoInput, 'rbf', Param.kparam3);
InvAIK = inv(AIK + Param.lambda*eye(size(AIK)));

AOK = EvalKernel(AutoTarget, AutoTarget, 'rbf', Param.kparam3);
InvAOK = inv(AOK + Param.lambda*eye(size(AOK)));

InvAIKAOK = InvAIK*AOK*InvAIK;