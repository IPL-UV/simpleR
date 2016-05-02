%
% Analysis of the models stored in RESULTS/results.mat
% 
% Gustau Camps-Valls, 2016(c)
% gustau.camps@uv.es
%

clear;clc;close all;

%% SETUP of FIGURES
fontname = 'AvantGarde';
fontsize = 16;
fontunits = 'points';
set(0,'DefaultAxesFontName',fontname,'DefaultAxesFontSize',fontsize,'DefaultAxesFontUnits',fontunits,...
    'DefaultTextFontName',fontname,'DefaultTextFontSize',fontsize,'DefaultTextFontUnits',fontunits,...
    'DefaultLineLineWidth',2,'DefaultLineMarkerSize',10,'DefaultLineColor',[0 0 0]);

rand('seed',1234)
randn('seed',1234)

%% LOAD RESULTS
load('RESULTS/results.mat')

d = size(Xtest,2);

numMethods = length(METHODS);
for m = 1:numMethods
    model = MODELS{m};

    % 1- feature ranking with a permutation analysis
    prank = permutation(METHODS{m},model,Xtest,Ytest);
    MRANK(m,:) = mean(prank);
    SRANK(m,:) = std(prank);

    % 2- Partial plots
    [XPLOTS(m,:,:) PPLOTS(m,:,:)] = partialplots(METHODS{m},model,Xtest);
    
end

%% Permutation analysis 
figure,
bar(MRANK)
ylabel('Relevance'),xlabel('Methods')
title('RMSE Permutation analysis')
set(gca,'XtickLabel',METHODS)
grid

%% Partial plots for all methods
% figure,
% g = round(sqrt(d));
% for i=1:d
%     for m=1:numMethods
%         subplot(g,g,i),
%         plot(squeeze(XPLOTS(m,i,:)),squeeze(PPLOTS(m,i,:)),'color',[1/m,0,0])
%         hold on,
%         ylabel('\Delta y'),
%         xlabel(VARIABLES{i})
%         grid
%         drawnow
%     end
% end
% legend(METHODS)

%% Partial plots for the first model only

figure,
g = round(sqrt(d));
for i=1:d
    for m=1%:numMethods
        subplot(g,g,i),
        plot(squeeze(XPLOTS(m,i,:)),squeeze(PPLOTS(m,i,:)),'color',[1/m,0,0])
        hold on,
        ylabel('\Delta y'),
        xlabel(VARIABLES{i})
        grid
        drawnow
    end
end
