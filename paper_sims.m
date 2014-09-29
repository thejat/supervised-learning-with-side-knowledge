%Generate multiple random fullsamples (training, test, side knowledge) with
%the same coefficients and run regression with side knowledge while learning.

%% Step 1: Generate simulated data

clc;close all;clear all;

s = RandStream('mcg16807','Seed',0) ; RandStream.setGlobalStream(s);
nRuns       = 1;
nBeta       = 100; %fixed dimension. Higher means, more data is required.
betaTrue    = randn(nBeta,1);
nTrainArray = floor(nBeta*[1.1]);
nTest       = mean(nTrainArray);%can be anything.
nKnowledge  = 100;
noiseSigma  = .4;

%The features need not be random here. Only the residuals need to be random.

% featureSigma = randn(nBeta); %+1 for covariance with residuals.
% featureSigma = featureSigma'*featureSigma;
featureSigma = gallery('randcorr',nBeta);
sampleTestX = randn(nTest,nBeta)*chol(featureSigma); %kept same for a given beta
sampleTestY = sampleTestX*betaTrue + noiseSigma*randn(nTest,1);
sampleKnowledgeX = randn(nKnowledge,nBeta)*chol(featureSigma); %kept same for a given beta
sampleKnowledgeY = sampleKnowledgeX*betaTrue + noiseSigma*randn(nKnowledge,1);

for i=1:nRuns
    sampleTrainX = randn(max(nTrainArray),nBeta)*chol(featureSigma);
    sampleTrainY = sampleTrainX*betaTrue + noiseSigma*randn(max(nTrainArray),1);
end

%% Step 2a: Ordinary least squares

betaEst = sampleTrainX\sampleTrainY;
figure(1); plot(betaEst,betaTrue,'.')
title(['norm(betaEst-betaTrue): ' num2str(norm(betaEst-betaTrue,2))])

%% Step 2: Fit model without side Knowledge: Ridge Regression

nFoldsCV   = 5;
nRepeatsCV = 3;
coeffRange = 2.^([-3:1:0]);
[Y_hat_val,betaCVX,regularize_coeff,cv_matrix,Y_hat_trn] = ...
    cvalidated_model('Ridge',coeffRange,nfolds,nrepeats,sampleTrainX,sampleTrainY,X_val,model_disable_CV,0.0001);
metrics = metric_of_success(sampleTestY,Y_hat_val,length(sampleTrainX(1,:)),'Val',str_dependent,'Ridge',plot_enable);


%%




%{

plot of performance (measured using RMSE) as a function of training sample
size.

For st plot performance for without knowledge case with error bars.

%}
