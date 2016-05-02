clear;clc;

load('RESULTS/results.mat')
 
%% NUMERICAL COMPARISON
numModels = length(MODELS);
clc
fprintf('----------------------------------------------------------------------------------- \n')
fprintf('#\t METHOD\t & ME\t & RMSE\t & nMSE \t & MAE\t & R \\\\ \n')
fprintf('----------------------------------------------------------------------------------- \n')
for m=1:numModels
    fprintf([num2str(m) '\t' METHODS{m} '\t & %3.3f\t & %3.3f\t & %3.3f\t & %3.3f\t & %3.3f \\\\ \n'],abs(RESULTS(m).ME),RESULTS(m).RMSE,RESULTS(m).nMSE,RESULTS(m).MAE,RESULTS(m).R)
end
fprintf('----------------------------------------------------------------------------------- \n')

%% The best method is ...
[val idx] = min([RESULTS.RMSE]);
disp(['The best method in RMSE terms is: ' METHODS{idx}])
[val idx] = min(abs([RESULTS.ME]));
disp(['The best method in ME terms is: ' METHODS{idx}])
[val idx] = max([RESULTS.R]);
disp(['The best method in correlation terms is: ' METHODS{idx}])

%% THE ERROR BOXPLOTS
figure,
ERRORS = YPREDS - repmat(Ytest,1,size(YPREDS,2));
% boxplot(ERRORS,'labels',METHODS)
% h = findobj(gca, 'type', 'text');
% set(h, 'Interpreter', 'tex');
% ylabel('Residuals')

%% STATISTICAL ANALYSIS OF THE BIAS
anova1(ERRORS)

%% STATISTICAL ANALYSIS OF THE ACCURACY OF THE RESIDUALS
anova1(abs(ERRORS))

%% THE CPU TIMES
figure,
barh([CPUTIMES])
set(gca,'Ytick',1:length(METHODS),'YTickLabel',METHODS);
xlabel('CPU Time [s]')
ylabel('Methods')
grid

%% SCATTER PLOTS OF THE BEST METHOD IN RMSE

figure,
plot(Ytest,YPREDS(:,idx),'k.'),
xlabel('Observed signal')
ylabel('Predicted signal')
grid

figure,
plot(Ytest,YPREDS(:,idx)-Ytest,'k.'),
xlabel('Observed signal')
ylabel('Residuals')
grid

%% RMSE vs #predictions
[ntest do] = size(Ytest);
REALIZ = 100;
RMSEvsNumPredictions = zeros(REALIZ,ntest,size(YPREDS,2));
for realiza=1:REALIZ
    r=randperm(ntest);
    for i=1:ntest
        for m=1:size(YPREDS,2)
            RMSEvsNumPredictions(realiza,i,m) = sqrt( mean(  ( Ytest(r(1:i)) - YPREDS(r(1:i),m) ).^2  ));
        end
    end
end

M =  squeeze(mean(RMSEvsNumPredictions,1));
S =  1.96/sqrt(REALIZ)*squeeze(std(RMSEvsNumPredictions,1));

figure,
myeb(M,S);
xlabel('# Predictions')
ylabel('RMSE')
grid
axis tight
legend(METHODS)


%% Arrange figures on the desktop

tile

