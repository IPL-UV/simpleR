function model = trainXGB(X,Y)

opts = [];
opts.loss = 'squaredloss'; % can be logloss or exploss

% this has to be not too high (max 1.0)
opts.shrinkageFactor = 0.01;
opts.subsamplingFactor = 0.2;
opts.maxTreeDepth = uint32(2);  % this was the default before customization
opts.randSeed = uint32(rand() * 1000);

numIters = 600;
model = SQBMatrixTrain(single(X), Y, uint32(numIters), opts);
