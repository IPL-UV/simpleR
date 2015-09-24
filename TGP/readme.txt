
This is sample implementation of Twin Gaussian Processes. 
Please acknowledge the use of code and cite as:  

    Liefeng Bo and Cristian Sminchisescu, 
    Twin Gaussian Processes for Structured Prediction
    International Journal of Computer Vision, vol. 87, pp. 28-52, 2010.
    http://ttic.uchicago.edu/~blf0218/paper/ijcv10.pdf

Table of Contents

% BME.m - demonstrate fast BME on a toy example
% BMECreate.m - customize your own BME
% BMEInit.m - initialize BME
% BMETrain.m - train BME
% BMETest.m - test BME with new inputs.
% DTGPTest.m - make the prediction using dynamic Twin Gaussian Processes
% DTGPTrain.m - train dynamic Twin Gaussian Processes
% EvalKernel.m - fastly evaluate kernel function
% HSICKNN.m - Hilbert-Schmidt independent criterion with K nearest neighbors
% JointError.m - compute the 3d joint position error
% KTAKNN.m - kernel target alignment with K nearest neighbors
% LinearRegressor.m - linear regression
% TGPDemoHMAX.m - demonstrate TGP on Walking Motion of Subject 1 with HMAX features
% TGPDemoHoG.m - demonstrate TGP on Walking Motion of Subject 1 with HoG features
% TGPDemoToy.m - demonstrate TGP on S data
% TGPKNN.m - make the prediction using Twin Gaussian Process with k nearest neighbors
% TGPKNN_HE_HMAX_Multi_Test.m - predict 3d poses on test set and exports the pose vector to XML file with HMAX features
% TGPKNN_HE_HoG_Multi_Test.m - predict 3d poses on test set and exports the pose vector to XML file with HoG features
% TGPTest.m - make the prediction using Twin Gaussian Processes
% TGPTrain.m - train Twin Gaussian Processes
% WKNNRegressor.m - weighted nearest neighbor regression