% Demonstrate TGP on Walking Motion of Subject 1, 
% and HMAX features are extracted from three color cameras

clear;
% load HMAX features
load('./data/HMAX_S1_Walking_TrainValidation_C1C2C3.mat');

% Initialization
Param.kparam1 = 1e-5/3;
Param.kparam2 = 5*1e-6;
Param.kparam3 = Param.kparam2;
Param.lambda = 1e-3;
Param.tradeoff = 20;
Param.knn = 100;

% Roughly separate training set and validation set
tag = floor(size(hmax,1)/2);
TestInput = hmax(1:tag,:);
Input = hmax(tag+1:end,:);
TestTarget = pose(1:tag,:);
Target = pose(tag+1:end,:);

disp('Joint Position Error (HMAX features)');
% Twin Gaussian Processes
[InvIK, InvOK] = TGPTrain(Input, Target, Param);
TGPPred = TGPTest(TestInput, Input, Target, Param, InvIK, InvOK);
[TGPError, TGPErrorvec] = JointError(TGPPred, TestTarget);
disp(['TGP: ' num2str(TGPError)]);

% Twin Gaussian Processes with K Nearest Neighbors
TGPKNNPred = TGPKNN(TestInput, Input, Target, Param);
[TGPKNNError, TGPKNNErrorvec] = JointError(TGPKNNPred, TestTarget);
disp(['TGPKNN: ' num2str(TGPKNNError)]);
    
% Weighted K-Nearest Neighbor Regression
K = 15;
WKNNPred = WKNNRegressor(TestInput,Input,Target,K);
[WKNNError, WKNNErrorvec] = JointError(WKNNPred, TestTarget);
disp(['WKNN: ' num2str(WKNNError)]);

% Gaussian Process Regression
kparam = 1e-5;
lambda = 1e-4;
K = EvalKernel(Input,Input,'rbf',kparam);
alpha = (K+lambda*eye(size(K)))\Target;
testK = EvalKernel(TestInput,Input,'rbf',kparam);
GPPred = testK*alpha;
[GPError, GPErrorvec] = JointError(GPPred, TestTarget);
disp(['GP: ' num2str(GPError)]);

% Hilbert-Schmidt Independent Criterion with K Nearest Neighbors
Param.kparam1 = 1e-3;
Param.kparam2 = 2*1e-5;
HSICKNNPred = HSICKNN(TestInput, Input, Target, Param);
[HSICKNNError, HSICKNNErrorvec] = JointError(HSICKNNPred, TestTarget);
disp(['HSICKNN: ' num2str(HSICKNNError)]);

% Kernel Target Alignment with K Nearest Neighbors
Param.kparam1 = 1e-3;
Param.kparam2 = 2*1e-5;
KTAKNNPred = KTAKNN(TestInput, Input, Target, Param);
[KTAKNNError, KTAKNNErrorvec] = JointError(KTAKNNPred, TestTarget);
disp(['KTAKNN: ' num2str(KTAKNNError)]);

