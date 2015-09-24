% Demonstrate TGP on all motions for all three subjects
% and HoG features are extracted from three color cameras

clear;
% load HoG features
load('./HoG/HoG_All_TrainValidation_C1C2C3.mat');

% Initialization
Param.kparam1 = 0.2;
Param.kparam2 = 2*1e-6;
Param.kparam3 = Param.kparam2;
Param.lambda = 1e-3;
Param.knn = 800;

hogtrain = [];
posetrain = [];
for i = 1:length(motionname)
    % Roughly separate training set and validation set
    index = find(motionframe(:,1) == i);
    tag = floor(length(index)/2);
    hogtrain  = [hogtrain; hog(index(tag+1:end),:)];
    hogtest{i} = hog(index(1:tag),:);
    posetrain  = [posetrain; pose(index(tag+1:end),:)];
    posetest{i} = pose(index(1:tag),:);
end

disp('Joint Position Error (HoG features)');
for i = 1:length(motionname)
    % Twin Gaussian Processes with K Nearest Neighbors
    if size(hogtest{i},1)
       TGPKNNPred = TGPKNN(hogtest{i}, hogtrain, posetrain, Param);
       [TGPKNNError, TGPKNNErrorvec] = JointError(TGPKNNPred, posetest{i});
       disp(['TGPKNN on:' motionname{i} ' ' num2str(TGPKNNError)]); 
    end
end

